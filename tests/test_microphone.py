"""The controller's microphone: the Ogg stream the recording is, the PCM beside it, the switch,
the endpoint and push to talk, against the lab controller. The tests that decode sound need
ffmpeg and skip without it; the others run everywhere."""

from __future__ import annotations

import array
import asyncio
import io
import itertools
import json
import os
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from stadiacast import microphone, receiver
from stadiacast.audio import ffmpeg_path
from stadiacast.cli import parse_args
from stadiacast.controller import MIC_TRIES
from stadiacast.errors import SetupError
from stadiacast.gamepad import Button
from stadiacast.http_server import HttpRequest
from stadiacast.microphone import (
    MIC_PAYLOAD_TYPE,
    QUEUE_PAGES,
    MicrophoneCapture,
    OggOpusWriter,
    capture_path,
    ogg_crc,
    opus_samples,
    pcm_level,
)
from stadiacast.protos import PKG_API, Protos
from stadiacast.signaling import build_answer
from tests.fake_controller import MIC_FRAME, MIC_SSRC, FakeController
from tests.helpers import connect_firmware_flow, make_receiver, no_log, single, wait_for

needs_ffmpeg = pytest.mark.skipif(ffmpeg_path() is None, reason="needs ffmpeg (with libopus) on PATH")
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="the stand-in decoder is a shell script")

ASSISTANT = int(Button.ASSISTANT)
#: What stands for a lost frame of the microphone's kind: its first byte, and no data behind it.
CONCEALED = b"\xf8"
#: What a controller sends for 20 ms of digital silence.
SILENT = b"\xf8\xff\xfe"


def pages(data: bytes) -> list[dict[str, Any]]:
    """An Ogg stream as its pages, each with its checksum verified."""
    out: list[dict[str, Any]] = []
    off = 0
    while off < len(data):
        assert data[off : off + 4] == b"OggS", f"no page starts at {off}"
        assert data[off + 4] == 0, "stream structure version"
        nseg = data[off + 26]
        segs = list(data[off + 27 : off + 27 + nseg])
        page = data[off : off + 27 + nseg + sum(segs)]
        assert struct.unpack("<I", page[22:26])[0] == ogg_crc(page[:22] + bytes(4) + page[26:]), f"page at {off}"
        out.append(
            {
                "flags": page[5],
                "granule": struct.unpack("<Q", page[6:14])[0],
                "serial": struct.unpack("<I", page[14:18])[0],
                "seq": struct.unpack("<I", page[18:22])[0],
                "segs": segs,
                "payload": page[27 + nseg :],
            }
        )
        off += len(page)
    return out


def real_opus(seconds: float, hz: int = 440) -> list[bytes]:
    """Opus of the controller's kind (CELT, 20 ms, mono, 64 kbit/s), of a sine, encoded by ffmpeg
    and taken out of its Ogg file again."""
    encoded = subprocess.run(
        [str(ffmpeg_path()), "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"sine=frequency={hz}:sample_rate=48000", "-t", str(seconds),
         "-ac", "1", "-c:a", "libopus", "-b:a", "64k", "-vbr", "off", "-frame_duration", "20",
         "-application", "lowdelay", "-f", "ogg", "pipe:1"],
        check=True, capture_output=True,
    ).stdout  # fmt: skip
    packets: list[bytes] = []
    held = b""
    for page in pages(encoded):  # the checksum this module computes is the one ffmpeg wrote
        at = 0
        for seg in page["segs"]:
            held += page["payload"][at : at + seg]
            at += seg
            if seg < 255:
                packets.append(held)
                held = b""
    return [p for p in packets if not p.startswith((b"OpusHead", b"OpusTags"))]


def decoded(path: Path) -> array.array[int]:
    """What ffmpeg makes of a recording: mono samples at 48 kHz. Anything it complains about fails the test."""
    done = subprocess.run(
        [str(ffmpeg_path()), "-hide_banner", "-loglevel", "warning", "-i", str(path),
         "-f", "s16le", "-ac", "1", "-ar", "48000", "pipe:1"],
        capture_output=True,
    )  # fmt: skip
    assert done.returncode == 0 and not done.stderr.strip(), done.stderr.decode(errors="replace")
    return array.array("h", done.stdout)


def frequency(samples: array.array[int]) -> float:
    """The pitch of a stretch of sound, by its rising zero crossings."""
    rising = sum(1 for a, b in itertools.pairwise(samples) if a < 0 <= b)
    return rising * 48000 / len(samples)


def mic_threads() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith("mic-pcm")]


def record(path: Path, packets: list[tuple[bytes, int, int]], **kw: Any) -> MicrophoneCapture:
    """A closed recording of ``(payload, RTP timestamp, sequence number)`` packets."""
    cap = MicrophoneCapture(str(path), **kw)
    for payload, timestamp, seq in packets:
        cap.write(payload, timestamp, seq)
    cap.close()
    return cap


# ---- the Ogg stream ---------------------------------------------------------------------------
def test_the_checksum_is_oggs() -> None:
    """Polynomial 0x04C11DB7, the most significant bit first, from a register of 0 and with no
    final XOR: not zlib's CRC-32. The value for ``123456789`` is the one the catalogues give for
    POSIX ``cksum`` (0x765E7680), which differs from Ogg's by its final XOR alone."""

    def bit_by_bit(data: bytes) -> int:
        crc = 0
        for byte in data:
            crc ^= byte << 24
            for _ in range(8):
                crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if crc & 0x80000000 else (crc << 1) & 0xFFFFFFFF
        return crc

    assert ogg_crc(b"") == 0 and ogg_crc(b"\x01") == 0x04C11DB7
    assert ogg_crc(b"123456789") == 0x765E7680 ^ 0xFFFFFFFF
    every_byte = bytes(range(256)) * 3 + b"OggS"
    assert ogg_crc(every_byte) == bit_by_bit(every_byte)


def test_a_recording_is_an_ogg_opus_stream_one_page_per_packet(tmp_path: Path) -> None:
    """RFC 7845: the identification header alone on the first page, flagged as the beginning; the
    comment header on the second; then the packets, each page's granule position the samples so
    far; a last page flagged as the end."""
    path = tmp_path / "mic.ogg"
    cap = MicrophoneCapture(str(path))
    for i in range(3):
        cap.write(MIC_FRAME, 1000 + i * 960, 7 + i)
    running = pages(path.read_bytes())  # no buffer: the file is a whole stream after every packet
    assert [p["flags"] for p in running] == [0x02, 0, 0, 0, 0]
    cap.close()
    cap.close()  # once is enough
    cap.write(MIC_FRAME, 3880, 10)  # and nothing is written behind the end
    got = pages(path.read_bytes())
    assert [p["flags"] for p in got] == [0x02, 0, 0, 0, 0, 0x04]
    assert [p["seq"] for p in got] == list(range(6)) and len({p["serial"] for p in got}) == 1
    assert [p["granule"] for p in got] == [0, 0, 960, 1920, 2880, 2880]
    head = got[0]["payload"]
    assert head[:8] == b"OpusHead" and len(head) == 19
    version, channels, pre_skip, rate, gain, family = struct.unpack("<BBHIhB", head[8:])
    assert (version, channels, pre_skip, rate, gain, family) == (1, 1, 0, 48000, 0, 0)
    tags = got[1]["payload"]
    assert tags[:8] == b"OpusTags" and tags[8:] == struct.pack("<I", 10) + b"stadiacast" + struct.pack("<I", 0)
    assert [p["payload"] for p in got[2:5]] == [MIC_FRAME] * 3 and got[2]["segs"] == [161]
    # The closing page has no lacing value. A single value of 0 would be a packet without a byte,
    # which no decoder takes for Opus.
    assert got[5]["segs"] == [] and got[5]["payload"] == b""
    assert cap.state() == {
        "file": str(path),
        "seconds": 0.06,
        "packets": 3,
        "bytes": 483,
        "concealed": 0,
        "late": 0,
        "ignored": 0,
        "pcm_file": None,
        "decoder": None,
    }
    # each recording is a stream of its own: two files put end to end still tell their pages apart
    record(tmp_path / "other.ogg", [(MIC_FRAME, 0, 0)])
    assert pages((tmp_path / "other.ogg").read_bytes())[0]["serial"] != got[0]["serial"]


def test_a_packet_says_itself_how_long_it_is() -> None:
    """RFC 6716: the first byte names the frame size and how many frames follow."""
    assert opus_samples(MIC_FRAME) == 960  # 0xf8: CELT, full band, 20 ms, one frame
    assert opus_samples(SILENT) == 960  # however little it carries
    assert [opus_samples(bytes([config << 3, 0])) for config in range(32)] == [
        *(480, 960, 1920, 2880) * 3,  # SILK: 10, 20, 40, 60 ms
        *(480, 960) * 2,  # hybrid: 10, 20 ms
        *(120, 240, 480, 960) * 4,  # CELT: 2.5, 5, 10, 20 ms
    ]
    assert opus_samples(b"\xf9\x00") == opus_samples(b"\xfa\x00") == 1920  # two frames, equal or not
    assert opus_samples(b"\xfb\x03") == 2880 and opus_samples(b"\xfb\x83") == 2880  # three; the flags above the count
    assert opus_samples(b"\xfb\x06") == 5760 and opus_samples(b"\xfb\x07") is None  # 120 ms is the most
    assert opus_samples(b"") is None and opus_samples(b"\xfb") is None and opus_samples(b"\xfb\x00") is None


def test_a_lost_packet_leaves_one_that_asks_to_be_concealed(tmp_path: Path) -> None:
    """The recording keeps the stream's time. RFC 7845 lets the granule position count nothing but
    the samples of the packets in the file, so each lost frame gets a packet in its place: the one
    byte that says which kind of frame is missing."""
    packets = [(MIC_FRAME, 5000 + i * 960, 100 + i) for i in range(8) if i not in (2, 5, 6)]
    cap = record(tmp_path / "mic.ogg", packets)
    got = pages((tmp_path / "mic.ogg").read_bytes())[2:-1]
    assert [p["payload"] for p in got] == [
        MIC_FRAME,
        MIC_FRAME,
        CONCEALED,
        MIC_FRAME,
        MIC_FRAME,
        CONCEALED,
        CONCEALED,
        MIC_FRAME,
    ]
    assert [p["granule"] for p in got] == [960 * n for n in range(1, 9)]
    assert (cap.packets, cap.concealed, cap.state()["seconds"]) == (5, 3, 0.16)


def test_a_long_silence_is_not_filled_and_the_clock_may_wrap(tmp_path: Path) -> None:
    cap = record(
        tmp_path / "mic.ogg",
        [
            (MIC_FRAME, 0xFFFFF880, 65534),
            (MIC_FRAME, 0xFFFFFC40, 65535),
            (MIC_FRAME, 0, 0),  # both counters wrap, and nothing is missing
            (MIC_FRAME, 960 + 48000, 1),  # one second is missing: fifty frames, still filled
            (MIC_FRAME, 49920 + 48960, 2),  # more than a second: a stream that began again
            (MIC_FRAME, 5000, 3),  # and a clock that jumps back fills nothing either
        ],
    )
    assert (cap.packets, cap.concealed, cap.late) == (6, 50, 0)
    assert pages((tmp_path / "mic.ogg").read_bytes())[-1]["granule"] == 56 * 960


def test_a_packet_that_comes_twice_or_late_is_left_out(tmp_path: Path) -> None:
    """Its place in the stream is taken: by itself the first time, by the packet that conceals it
    when its successor came first."""
    cap = record(
        tmp_path / "mic.ogg",
        [
            (MIC_FRAME, 0, 10),
            (MIC_FRAME, 960, 11),
            (MIC_FRAME, 960, 11),  # twice
            (MIC_FRAME, 2880, 13),  # 12 is missing here
            (b"\xf8late", 1920, 12),  # and comes behind 13
            (MIC_FRAME, 3840, 14),
        ],
    )
    got = pages((tmp_path / "mic.ogg").read_bytes())[2:-1]
    assert [p["payload"] for p in got] == [MIC_FRAME, MIC_FRAME, CONCEALED, MIC_FRAME, MIC_FRAME]
    assert (cap.packets, cap.concealed, cap.late) == (4, 1, 2)


def test_what_is_no_packet_of_the_microphone_is_left_out(tmp_path: Path) -> None:
    cap = record(
        tmp_path / "mic.ogg",
        [
            (MIC_FRAME, 0, 1),
            (b"", 960, 2),  # nothing
            (b"\xfb", 960, 3),  # a count of frames that is not there
            (b"\xf8" + bytes(1500), 960, 4),  # longer than a packet of the microphone gets
            (b"\xf8" + bytes(1499), 960, 5),  # the longest that is taken
        ],
    )
    assert (cap.packets, cap.ignored, cap.concealed) == (2, 3, 0)
    assert [len(p["payload"]) for p in pages((tmp_path / "mic.ogg").read_bytes())[2:-1]] == [161, 1500]


def test_a_stream_that_begins_again_is_taken_up(tmp_path: Path) -> None:
    """Sequence numbers that go on from somewhere behind the stream's last are late packets as far
    as anyone can tell, until five in a row say that the stream began again: the recording goes on
    with it, where it would have thrown away ten minutes of sound waiting for the numbers to pass."""
    said: list[tuple[str, dict[str, Any]]] = []
    path = tmp_path / "mic.ogg"
    cap = MicrophoneCapture(str(path), log=lambda event, **fields: said.append((event, fields)))
    for i in range(3):
        cap.write(MIC_FRAME, i * 960, 40000 + i)
    for i in range(4):  # four in a row behind it: late
        cap.write(b"\xf8old", 500000 + i * 960, 30000 + i)
    cap.write(MIC_FRAME, 3 * 960, 40003)  # and the stream goes on where it was
    assert (cap.packets, cap.late, cap.concealed) == (4, 4, 0)
    for i in range(8):  # eight in a row: at the fifth it is a stream
        cap.write(b"\xf8new%d" % i, 900000 + i * 960, 20000 + i)
    cap.close()
    assert (cap.packets, cap.late, cap.concealed) == (8, 8, 0)
    payloads = [p["payload"] for p in pages(path.read_bytes())[2:-1]]
    assert payloads == [MIC_FRAME] * 4 + [b"\xf8new%d" % i for i in range(4, 8)]
    assert [f for event, f in said if event == "mic_stream_restarted"] == [
        {"file": str(path), "seq": 20004, "was": 40003}
    ]


def test_a_packet_that_fills_its_last_segment_gets_the_closing_lacing_value() -> None:
    """A packet of 255 or 510 bytes ends on a segment boundary: a lacing value of 0 says it ended."""
    buf = io.BytesIO()
    writer = OggOpusWriter(buf, serial=7)
    for size in (254, 255, 256, 510):
        writer.write_packet(b"\xf8" + bytes(size - 1), 960)
    writer.write_packet(b"", 960)  # nothing: not a packet, no page
    with pytest.raises(ValueError, match="does not fit one Ogg page"):
        writer.write_packet(bytes(255 * 255), 960)
    writer.close()
    got = pages(buf.getvalue())
    assert [p["segs"] for p in got[2:6]] == [[254], [255, 0], [255, 1], [255, 255, 0]]
    assert [len(p["payload"]) for p in got[2:6]] == [254, 255, 256, 510]
    assert len(got) == 7 and {p["serial"] for p in got} == {7}
    assert writer.packets == 4 and writer.granule == got[-1]["granule"] == 4 * 960  # the refused one counts for nothing


def test_two_recordings_in_one_second_have_two_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(microphone, "time", SimpleNamespace(strftime=lambda _format: "184930"))
    first = capture_path(str(tmp_path), "c0")
    assert first == str(tmp_path / "mic_c0_184930.ogg")
    Path(first).touch()
    second = capture_path(str(tmp_path), "c0")
    assert second == str(tmp_path / "mic_c0_184930_2.ogg")
    Path(second).with_suffix(".pcm").touch()  # a PCM file alone holds its name as well
    assert capture_path(str(tmp_path), "c0") == str(tmp_path / "mic_c0_184930_3.ogg")
    assert capture_path(str(tmp_path), "c1") == str(tmp_path / "mic_c1_184930.ogg")


@needs_ffmpeg
def test_real_opus_plays_while_it_is_recorded_and_after(tmp_path: Path) -> None:
    """The stream as a player sees it: a tone goes in as Opus packets and comes out of ffmpeg whole,
    from the file that is still open and from the closed one."""
    packets = real_opus(2.0)
    assert len(packets) == 101 and {p[0] for p in packets} == {0xF8} and {len(p) for p in packets} == {160}
    path = tmp_path / "mic.ogg"
    cap = MicrophoneCapture(str(path))
    for i, packet in enumerate(packets):
        cap.write(packet, 5000 + i * 960, i)
    for moment in ("open", "closed"):
        samples = decoded(path)
        assert len(samples) == len(packets) * 960, moment
        assert abs(frequency(samples[24000:72000]) - 440) < 2, moment
        cap.close()


@needs_ffmpeg
def test_a_recording_with_lost_packets_plays_for_as_long_as_the_sound_lasted(tmp_path: Path) -> None:
    """A fifth of a second lost in the middle: the decoder conceals it, the tone is there on both
    sides of it, and the file is as long as the stream was."""
    packets = real_opus(2.0)
    kept = [(p, i * 960, i) for i, p in enumerate(packets) if not 40 <= i < 50]
    cap = record(tmp_path / "mic.ogg", kept)
    assert (cap.packets, cap.concealed) == (91, 10)
    samples = decoded(tmp_path / "mic.ogg")
    assert len(samples) == len(packets) * 960
    assert abs(frequency(samples[9600:33600]) - 440) < 2 and abs(frequency(samples[57600:86400]) - 440) < 2


@needs_ffmpeg
def test_a_silent_frame_is_three_bytes_and_as_long_as_any_other(tmp_path: Path) -> None:
    """A controller in a quiet room sends frames of digital silence among the others, three bytes
    each. They are frames like the rest: 20 ms in the file, and silence out of the decoder."""
    tone = real_opus(1.0)
    packets = tone[:20] + [SILENT] * 25 + tone[20:40]
    cap = record(tmp_path / "mic.ogg", [(p, i * 960, i) for i, p in enumerate(packets)])
    assert (cap.packets, cap.ignored, cap.concealed) == (65, 0, 0)
    samples = decoded(tmp_path / "mic.ogg")
    assert len(samples) == 65 * 960
    assert max(abs(s) for s in samples[25 * 960 : 44 * 960]) <= 1  # away from the tone on either side
    for before_or_after in (samples[5 * 960 : 15 * 960], samples[50 * 960 : 60 * 960]):
        assert abs(frequency(before_or_after) - 440) < 5


# ---- the PCM beside it --------------------------------------------------------------------------
def test_the_level_of_pcm_is_its_peak_and_its_rms() -> None:
    assert pcm_level(b"") is None and pcm_level(b"\x01") is None
    full = array.array("h", [32767, -32768] * 100).tobytes()
    peak, rms = pcm_level(full) or (1.0, 1.0)
    assert abs(peak) < 0.01 and abs(rms) < 0.01  # full scale is 0 dBFS
    half = array.array("h", [16384, -16384, 0, 0] * 100).tobytes()
    peak, rms = pcm_level(half + b"\x7f") or (1.0, 1.0)  # an odd byte at the end is not a sample
    assert abs(peak + 6.02) < 0.01 and abs(rms + 9.03) < 0.01
    peak, rms = pcm_level(bytes(200)) or (1.0, 1.0)  # silence has a floor, not minus infinity
    assert abs(peak + 90.31) < 0.01 and abs(rms + 90.31) < 0.01


@needs_ffmpeg
def test_the_pcm_file_gets_every_sample_and_nothing_is_left_running(tmp_path: Path) -> None:
    """Closing waits for nothing and loses nothing: ffmpeg is given the last page, decodes what it
    holds and ends, and the file is closed behind the last sample. That holds for a recording
    that ends before ffmpeg has started, too."""
    packets = real_opus(1.5)
    cap = MicrophoneCapture(str(tmp_path / "mic.ogg"), pcm=True)
    assert cap.decoder is not None and cap.pcm_path == str(tmp_path / "mic.pcm")
    for i, packet in enumerate(packets):
        cap.write(packet, i * 960, i)
    closed_at = time.monotonic()
    cap.close()
    assert time.monotonic() - closed_at < 0.5  # the event loop's part
    cap.join()
    assert time.monotonic() - closed_at < 2.5  # ffmpeg ends by itself, well before anyone would end it
    assert cap.ended and mic_threads() == []
    proc = cap.decoder._proc
    assert proc is not None and proc.stdout is not None and proc.stderr is not None
    assert proc.stdout.closed and proc.stderr.closed  # nothing of ffmpeg's is left open
    quick = MicrophoneCapture(str(tmp_path / "quick.ogg"), pcm=True)  # over before ffmpeg has started
    for i, packet in enumerate(packets[:5]):
        quick.write(packet, i * 960, i)
    quick.close()
    quick.join()
    assert quick.ended and mic_threads() == [] and len((tmp_path / "quick.pcm").read_bytes()) == 5 * 960 * 2
    pcm = (tmp_path / "mic.pcm").read_bytes()
    assert len(pcm) == len(packets) * 960 * 2  # the first frame and the last are there
    assert abs(frequency(array.array("h", pcm)[24000:48000]) - 440) < 2
    state = cap.state()["decoder"]
    assert state["running"] is False and state["reason"] is None and state["dropped"] == 0
    assert state["bytes"] == len(pcm) and state["peak_db"] is not None and state["rms_db"] < state["peak_db"]


@needs_ffmpeg
def test_the_pcm_file_follows_the_recording_while_it_is_open(tmp_path: Path) -> None:
    """A program reads the samples while they are made: what has arrived is in the PCM file a
    moment later, short of the packet ffmpeg still holds, and not when a buffer of its has filled."""
    packets = real_opus(0.6)
    pcm = tmp_path / "mic.pcm"
    cap = MicrophoneCapture(str(tmp_path / "mic.ogg"), pcm=True)
    try:
        for i, packet in enumerate(packets):
            cap.write(packet, i * 960, i)
        want = (len(packets) - 2) * 960 * 2
        until = time.monotonic() + 5
        while (not pcm.exists() or pcm.stat().st_size < want) and time.monotonic() < until:
            time.sleep(0.02)
        assert pcm.stat().st_size >= want
    finally:
        cap.close()
        cap.join()
    assert pcm.stat().st_size == len(packets) * 960 * 2


def test_without_ffmpeg_the_ogg_recording_is_all_there_is(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(microphone, "ffmpeg_path", lambda: None)
    said: list[tuple[str, dict[str, Any]]] = []
    cap = MicrophoneCapture(
        str(tmp_path / "mic.ogg"), pcm=True, log=lambda event, **fields: said.append((event, fields))
    )
    cap.write(MIC_FRAME, 1000, 1)
    cap.close()
    cap.join()
    assert cap.ended and not (tmp_path / "mic.pcm").exists()
    assert [p["flags"] for p in pages((tmp_path / "mic.ogg").read_bytes())] == [0x02, 0, 0, 0x04]
    assert cap.state()["pcm_file"] is None
    assert cap.state()["decoder"] == {
        "running": False,
        "reason": "ffmpeg is not on PATH",
        "bytes": 0,
        "dropped": 0,
        "peak_db": None,
        "rms_db": None,
    }
    assert ("mic_pcm_unavailable", {"reason": "ffmpeg is not on PATH"}) in said


def test_an_ffmpeg_that_does_not_start_leaves_no_pcm_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(microphone, "ffmpeg_path", lambda: str(tmp_path / "no-such-ffmpeg"))
    cap = MicrophoneCapture(str(tmp_path / "mic.ogg"), pcm=True)
    assert cap.decoder is not None
    cap.write(MIC_FRAME, 1000, 1)
    cap.decoder.join()  # the start is not the caller's to wait for; here it is waited for
    assert cap.decoder.ended and not cap.decoder.running and str(cap.decoder.reason).startswith("ffmpeg did not start")
    cap.close()
    cap.join()
    assert cap.ended and mic_threads() == [] and not (tmp_path / "mic.pcm").exists()
    assert cap.state()["pcm_file"] is None and cap.state()["packets"] == 1


def stand_in(tmp_path: Path, body: str) -> str:
    """A program in ffmpeg's place that does what ``body`` (Python) says."""
    script = tmp_path / "stand_in.py"
    script.write_text(f"import sys, time\n{body}\n")
    program = tmp_path / "ffmpeg"
    program.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}"\n')
    program.chmod(0o755)
    return str(program)


@posix_only
def test_a_decoder_that_does_not_read_loses_pages_and_holds_nobody_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recording is written on the event loop, so a page for the decoder never waits: past
    what its pipe and the queue hold, pages are dropped and counted. A decoder that does not end
    either is ended."""
    monkeypatch.setattr(microphone, "ffmpeg_path", lambda: stand_in(tmp_path, "time.sleep(600)"))
    monkeypatch.setattr(microphone, "DECODER_EXIT_TIMEOUT", 0.3)
    cap = MicrophoneCapture(str(tmp_path / "mic.ogg"), pcm=True)
    assert cap.decoder is not None
    packet = b"\xf8" + bytes(1199)
    sent = 3 * QUEUE_PAGES
    for i in range(sent):
        cap.write(packet, i * 960, i)
    assert 0 < cap.decoder.dropped < sent
    cap.close()
    cap.join()
    assert cap.ended and mic_threads() == []
    assert str(cap.state()["decoder"]["reason"]).startswith("ffmpeg ended with status")
    assert len(pages((tmp_path / "mic.ogg").read_bytes())) == sent + 3  # the Ogg file has them all


@posix_only
def test_a_recording_closed_before_a_stuck_ffmpeg_has_started_ends_all_the_same(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(microphone, "ffmpeg_path", lambda: stand_in(tmp_path, "time.sleep(600)"))
    monkeypatch.setattr(microphone, "DECODER_EXIT_TIMEOUT", 0.3)
    began = time.monotonic()
    cap = MicrophoneCapture(str(tmp_path / "mic.ogg"), pcm=True)
    cap.close()  # at once: there is no ffmpeg yet to give a deadline to
    cap.join()
    assert cap.ended and mic_threads() == [] and time.monotonic() - began < 3.0


@posix_only
def test_ffmpeg_runs_in_a_session_of_its_own(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ctrl-C in the receiver's terminal goes to every process of the terminal's foreground group.
    ffmpeg is not one of them: the receiver ends the recording, and ffmpeg ends behind its last
    page, where one that got the signal itself would end before it had it."""
    monkeypatch.setattr(microphone, "ffmpeg_path", lambda: stand_in(tmp_path, "sys.stdin.buffer.read()"))
    cap = MicrophoneCapture(str(tmp_path / "mic.ogg"), pcm=True)
    assert cap.decoder is not None
    try:
        until = time.monotonic() + 5
        while cap.decoder._proc is None and time.monotonic() < until:
            time.sleep(0.01)
        proc = cap.decoder._proc
        assert proc is not None
        assert os.getsid(proc.pid) != os.getsid(0) and os.getpgid(proc.pid) != os.getpgid(0)
    finally:
        cap.close()
        cap.join()
    assert cap.ended and mic_threads() == []


@posix_only
def test_a_decoder_that_ends_by_itself_says_why_and_the_recording_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = "sys.stdin.buffer.read(40)\nsys.stderr.write('first line\\nno decoder for this\\n')\nsys.exit(3)"
    monkeypatch.setattr(microphone, "ffmpeg_path", lambda: stand_in(tmp_path, body))
    said: list[tuple[str, dict[str, Any]]] = []
    cap = MicrophoneCapture(
        str(tmp_path / "mic.ogg"), pcm=True, log=lambda event, **fields: said.append((event, fields))
    )
    assert cap.decoder is not None
    cap.decoder.join()  # it ends on its own, while the recording is open
    assert cap.decoder.ended and not cap.decoder.running
    assert cap.decoder.reason == "ffmpeg ended with status 3: no decoder for this"
    for i in range(3):
        cap.write(MIC_FRAME, i * 960, i)
    cap.close()
    assert cap.ended and len(pages((tmp_path / "mic.ogg").read_bytes())) == 6
    assert [fields["reason"] for event, fields in said if event == "mic_pcm_ended"] == [cap.decoder.reason]


# ---- the answer and the flags -------------------------------------------------------------------
def test_the_answer_to_a_controllers_offer_names_no_audio_format() -> None:
    """A controller's offer has an audio SSRC and names no audio format, and the answer names none
    either: that is the answer a controller sends its microphone on."""
    p = Protos()
    offer = p.RtcSdpOffer()
    offer.audio_ssrc = 2365406975
    extension = offer.audio_extensions.add()
    extension.id, extension.uri = 6, "urn:ietf:params:rtp-hdrext:ssrc-audio-level"
    offer.raw_dtls = offer.delayed_acks = True
    ep = SimpleNamespace(local_params=lambda: {"ice_ufrag": "u", "ice_password": "pw", "fingerprint": "sha-256 AA"})
    assert not build_answer(p, offer, ep).sdp_answer.HasField("audio_format")  # type: ignore[arg-type]
    fmt = offer.audio_format.add()  # a format the offer does name is the answer's
    fmt.codec, fmt.channels, fmt.sample_rate = p.enum(f"{PKG_API}.RtcSdpAudioCodec", "AUDIO_CODEC_OPUS"), 1, 16000
    echoed = build_answer(p, offer, ep).sdp_answer.audio_format  # type: ignore[arg-type]
    assert (echoed.channels, echoed.sample_rate) == (1, 16000)


def test_the_flags_default_to_no_microphone() -> None:
    for args in (["--self-signed"], ["--preset", "live"]):
        cfg = parse_args(args)
        assert (cfg.mic_capture, cfg.mic_pcm) == ("off", False), args
    cfg = parse_args(["--self-signed", "--mic-capture", "ptt", "--mic-pcm"])
    assert (cfg.mic_capture, cfg.mic_pcm) == ("ptt", True)
    with pytest.raises(SystemExit):
        parse_args(["--self-signed", "--mic-capture", "always"])


def test_the_receiver_does_not_start_with_mic_pcm_and_no_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(receiver, "ffmpeg_path", lambda: None)

    async def run() -> None:
        rx = make_receiver(tmp_path, mic_capture="on", mic_pcm=True)
        try:
            with pytest.raises(SetupError, match="--mic-pcm decodes the recordings with ffmpeg"):
                await rx.start()
        finally:
            await rx.stop()

    asyncio.run(run())


# ---- through the display link ---------------------------------------------------------------------
async def call(fc: FakeController, rx: Any, method: str = "GET", body: dict[str, Any] | None = None) -> tuple[int, Any]:
    """One request to ``/_mic`` on the plain HTTP port, from this computer."""
    raw = json.dumps(body).encode() if body is not None else b""
    status, _, out = await fc.request(
        method, "x", "/_mic", raw, "application/json", port=rx.http.http_ports[0], tls=False
    )
    return status, json.loads(out) if out else None


def test_the_switch_starts_the_microphone_and_its_packets_are_a_recording_not_input(tmp_path: Path) -> None:
    """``set_microphone`` true goes out, the Opus that follows as SRTP is written to an Ogg file,
    none of it is taken for the input protocol, and false ends the recording."""

    async def run() -> None:
        rx = make_receiver(tmp_path)
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()  # the receiver has seen READY by its answer
            assert cs.mic_state() == {
                "on": False,
                "ptt": False,
                "pcm": False,
                "recording": None,
                "last": None,
                "stray": 0,
            }
            assert rx.live_state()["controllers"]["c0"]["microphone"] is False
            assert not fc.send_mic_frame()  # a microphone that is off sends nothing
            assert cs.mic_start() == {"ok": True, "on": True, "ticket": cs._rpc_ticket}
            await wait_for(lambda: fc.mic_requests == [True])
            assert fc.mic_on and rx.live_state()["controllers"]["c0"]["microphone"] is True
            for _ in range(3):
                assert fc.send_mic_frame()
            await wait_for(lambda: (cs.mic_state()["recording"] or {}).get("packets") == 3)
            state = cs.mic_state()
            recording = state["recording"]
            assert (recording["bytes"], recording["seconds"], recording["concealed"]) == (483, 0.06, 0)
            path = Path(str(recording["file"]))
            assert path.parent == tmp_path / "logs" and path.name.startswith("mic_c0_") and path.suffix == ".ogg"
            assert [p["flags"] for p in pages(path.read_bytes())] == [0x02, 0, 0, 0, 0]
            assert cs.status()["microphone"] == state and state["last"] is None
            # Opus is not the input protocol: nothing was decoded as input, or failed to be
            assert "rtp_payload_undecodable" not in rx.log.counts and "rtp_input" not in rx.log.counts
            assert cs.session["input_payload_type"] is None and cs.session["input_series"] == 0
            first = next(r for r in rx.log.records if r["event"] == "mic_rtp")
            assert (first["via"], first["pt"], first["ext"]) == ("srtp", MIC_PAYLOAD_TYPE, [(6, "ff")])
            assert first["ssrc"] == f"{MIC_SSRC:08x}"  # the one the offer names for audio

            assert cs.mic_stop()["on"] is False
            await wait_for(lambda: fc.mic_requests == [True, False])
            assert [p["flags"] for p in pages(path.read_bytes())][-1] == 0x04
            state = cs.mic_state()  # the recording that ended is still named
            assert (state["on"], state["recording"]) == (False, None)
            assert (state["last"]["file"], state["last"]["packets"]) == (str(path), 3)

            # a packet that was on its way when the switch went off is no recording of its own
            assert fc.send_mic_frame(even_if_off=True)
            await wait_for(lambda: cs.mic_state()["stray"] == 1)
            assert cs.mic_state()["recording"] is None and len(list((tmp_path / "logs").glob("mic_*"))) == 1
            assert rx.log.counts["mic_stray"] == 1
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


def test_before_the_link_is_ready_the_switch_stays_off_and_ready_still_asks(tmp_path: Path) -> None:
    """A request sent before the controller reads its requests is lost, and has stalled a real
    controller for good (the lab controller does the same here). Asked for a link that is not
    READY, nothing is sent and nothing is pretended: the endpoint answers 409, and
    ``--mic-capture on`` asks when the link is READY."""

    async def run() -> None:
        rx = make_receiver(tmp_path, mic_capture="on")
        await rx.start()
        fc = FakeController("127.0.0.1", rx.http.https_port, rx.protos, log=no_log, dtls_role="client",
                            controlling=True, raw_dtls=True, early_rpc="stall")  # fmt: skip
        try:
            await fc.boot()
            await fc.start_udp()
            await fc.discover()
            cs = single(rx)
            status, out = await call(fc, rx, "POST", {"do": "on"})
            assert status == 409 and out["ok"] is False and out["on"] is False
            assert out["error"] == "the display link is not READY: nothing was sent" and cs.mic_on is False
            status, out = await call(fc, rx, "POST", {"do": "off"})  # off is off, link or no link
            assert status == 200 and out["ok"] is True and out["on"] is False

            await fc.poll_gmsg()
            await fc.poll_gmsg()
            fc.connect()
            await fc.wait_connected(10)
            assert cs.mic_start() == {"ok": False, "on": False, "ticket": None}  # the link is up, and not READY
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            await wait_for(lambda: fc.mic_requests == [True])
            assert cs.mic_on is True and fc.early_rpc_records == 0
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


def test_a_controller_that_refuses_keeps_the_switch_off(tmp_path: Path) -> None:
    """An error in answer to true turns the switch off again, unless the switch has been thrown
    again since: then that request's answer is the one that counts."""

    async def run() -> None:
        rx = make_receiver(tmp_path)
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()  # the receiver has seen READY by its answer
            fc.mic_refuse = 13
            assert cs.mic_start()["ok"] is True  # it was sent
            await wait_for(lambda: rx.log.counts.get("mic_refused"))
            refused = next(r for r in rx.log.records if r["event"] == "mic_refused")
            assert (refused["code"], refused["error_message"], refused["stale"]) == (
                13,
                "refused by the lab controller",
                False,
            )
            assert cs.mic_on is False and rx.live_state()["controllers"]["c0"]["microphone"] is False

            fc.mic_refuse = 13  # refused again, and this time off and on follow before the answer is here
            cs.mic_start()
            cs.mic_stop()
            cs.mic_start()
            await wait_for(lambda: fc.mic_requests == [True, True, False, True])
            await wait_for(lambda: rx.log.counts["mic_refused"] == 2)
            assert [r["stale"] for r in rx.log.records if r["event"] == "mic_refused"] == [False, True]
            assert cs.mic_on is True and fc.mic_on is True
            assert fc.send_mic_frame()
            await wait_for(lambda: cs.mic_state()["recording"] is not None)
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


def test_the_endpoint_reads_for_anyone_and_switches_for_this_computer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``GET /_mic`` only reads. ``POST`` takes on, off, pcm and ptt, says what it does not know,
    and is refused to every computer but the receiver's own: the endpoints have no login, and
    this one makes a microphone record."""

    async def run() -> None:
        rx = make_receiver(tmp_path)
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)

        def from_the_network(method: str, body: bytes = b"") -> HttpRequest:
            return HttpRequest(
                method, "/_mic", "HTTP/1.1", [], body, ("192.168.2.9", 50000), "c9", None, None, path="/_mic"
            )

        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()  # the receiver has seen READY by its answer
            status, out = await call(fc, rx)
            assert status == 200
            assert (out["on"], out["ptt"], out["pcm"], out["controller"]) == (False, False, False, "c0")
            status, out = await call(fc, rx, "GET", {"do": "on"})  # a read with a body is still a read
            assert status == 200 and out["on"] is False and fc.mic_requests == []

            assert (await rx.handle(from_the_network("GET"))).status == 200
            answer = await rx.handle(from_the_network("POST", b'{"do": "on"}'))
            assert answer.status == 403 and b"only over the loopback address" in answer.body
            assert cs.mic_on is False and "mic_switch" not in rx.log.counts  # the switch was not touched

            status, out = await call(fc, rx, "POST", {"do": "on", "on": False})  # on, or not on?
            assert status == 400 and out == {"error": "do says on or off by itself; on goes with pcm and ptt"}
            assert cs.mic_on is False and "mic_switch" not in rx.log.counts
            status, out = await call(fc, rx, "POST", {"do": "on"})
            assert status == 200 and out == {"ok": True, "on": True, "ticket": cs._rpc_ticket, "controller": "c0"}
            await wait_for(lambda: fc.mic_requests == [True])
            assert fc.send_mic_frame()
            await wait_for(lambda: cs.mic_state()["recording"] is not None)
            status, out = await call(fc, rx, "POST", {"do": "off"})
            assert status == 200 and out["ok"] is True and out["on"] is False
            status, out = await call(fc, rx)
            assert (out["on"], out["recording"], out["last"]["packets"]) == (False, None, 1)

            monkeypatch.setattr("stadiacast.controller.ffmpeg_path", lambda: "/somewhere/ffmpeg")
            status, out = await call(fc, rx, "POST", {"do": "pcm", "on": True})
            assert status == 200 and out == {"ok": True, "pcm": True, "controller": "c0"} and cs.mic_pcm is True
            status, out = await call(fc, rx, "POST", {"do": "pcm", "on": False})
            assert status == 200 and cs.mic_state()["pcm"] is False
            monkeypatch.setattr("stadiacast.controller.ffmpeg_path", lambda: None)
            status, out = await call(fc, rx, "POST", {"do": "pcm"})  # on is what is meant without a word
            assert status == 400 and cs.mic_pcm is False
            assert out == {"error": "the PCM file is decoded by ffmpeg, and none was found on PATH"}
            status, out = await call(fc, rx, "POST", {"do": "ptt"})
            assert status == 200 and out == {"ok": True, "ptt": True, "controller": "c0"}
            assert cs.mic_state()["ptt"] is True
            status, out = await call(fc, rx, "POST", {"do": "ptt", "on": "false"})  # a word is not a yes or a no
            assert status == 400 and out == {"error": "on is true or false"} and cs.mic_ptt is True
            status, out = await call(fc, rx, "POST", {"do": "ptt", "on": False})
            assert status == 200 and cs.mic_ptt is False

            status, out = await call(fc, rx, "POST", {"do": "louder"})
            assert status == 400 and out == {"error": "unknown do='louder'", "actions": ["on", "off", "pcm", "ptt"]}
            status, _, raw = await fc.request("POST", "x", "/_mic", b"{not json", "application/json",
                                              port=rx.http.http_ports[0], tls=False)  # fmt: skip
            assert status == 400 and json.loads(raw) == {"error": "the body is not JSON"}
            status, _, raw = await fc.request("POST", "x", "/_mic", b"[" * 100000, "application/json",
                                              port=rx.http.http_ports[0], tls=False)  # fmt: skip
            assert status == 400 and json.loads(raw) == {"error": "the body is not JSON"}  # nested too deep to read
            assert (await call(fc, rx, "POST", {"do": "on", "controller": "c7"}))[0] == 404
            assert (await call(fc, rx, "DELETE"))[0] == 405
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


def test_changing_the_pcm_setting_ends_the_open_recording(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The two files of a recording cover the same sound: a recording that is open when the
    setting changes ends there, and the next packet starts one made the new way."""
    monkeypatch.setattr("stadiacast.controller.ffmpeg_path", lambda: "/somewhere/ffmpeg")  # the setting asks for one
    monkeypatch.setattr(microphone, "ffmpeg_path", lambda: None)  # and no decoder is started here

    async def run() -> None:
        rx = make_receiver(tmp_path)
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()  # the receiver has seen READY by its answer
            cs.mic_start()
            await wait_for(lambda: fc.mic_requests == [True])
            assert fc.send_mic_frame()
            await wait_for(lambda: cs.mic_state()["recording"] is not None)
            first = cs.mic_state()["recording"]
            assert first["decoder"] is None
            assert cs.set_mic_pcm(False) == {"ok": True, "pcm": False}  # as it was: nothing ends
            assert cs.mic_state()["recording"] is not None
            assert cs.set_mic_pcm(True) == {"ok": True, "pcm": True}
            state = cs.mic_state()
            assert (state["on"], state["recording"], state["last"]["file"]) == (True, None, first["file"])
            assert pages(Path(first["file"]).read_bytes())[-1]["flags"] == 0x04
            assert fc.send_mic_frame()
            await wait_for(lambda: cs.mic_state()["recording"] is not None)
            second = cs.mic_state()["recording"]
            assert second["file"] != first["file"] and second["decoder"]["reason"] == "ffmpeg is not on PATH"
            assert fc.mic_requests == [True]  # the controller was told nothing new
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


def test_push_to_talk_holds_the_switch_with_the_assistant_button(tmp_path: Path) -> None:
    """``--mic-capture ptt``: pressing the Assistant button turns the microphone on and letting go
    turns it off, each hold a recording of its own. A microphone that was on already is not the
    button's to turn off, one that was turned off during a hold stays off until the next press,
    and a hold ends when the button stops being the switch."""

    async def run() -> None:
        rx = make_receiver(tmp_path, mic_capture="ptt")
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        reports = 0

        async def report(buttons: int) -> None:
            nonlocal reports
            assert fc.send_input_rpc(buttons=buttons)
            reports += 1
            await wait_for(lambda: cs.session["display"]["input_rpc"] == reports)

        async def post(body: dict[str, Any]) -> None:
            assert (await call(fc, rx, "POST", body))[0] == 200

        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            assert cs.mic_state()["ptt"] is True and fc.mic_requests == []  # READY alone asks for nothing
            await report(int(Button.A))  # another button is not the switch
            assert fc.mic_requests == [] and cs.mic_on is False

            await report(ASSISTANT | int(Button.A))
            await wait_for(lambda: fc.mic_requests == [True])
            await report(ASSISTANT)  # still held: nothing new to say
            assert fc.send_mic_frame()
            await wait_for(lambda: cs.mic_state()["recording"] is not None)
            first = Path(str(cs.mic_state()["recording"]["file"]))
            await report(0)
            await wait_for(lambda: fc.mic_requests == [True, False])
            assert cs.mic_on is False and pages(first.read_bytes())[-1]["flags"] == 0x04
            await report(int(Button.A))  # let go is let go: no report after it says so again
            await report(0)
            assert fc.mic_requests == [True, False]

            await post({"do": "on"})  # asked for another way
            await wait_for(lambda: fc.mic_requests == [True, False, True])
            await report(ASSISTANT)
            await report(0)
            assert cs.mic_on is True and fc.mic_requests == [True, False, True]  # the tap ended nothing
            await post({"do": "off"})
            await wait_for(lambda: fc.mic_requests == [True, False, True, False])

            await report(ASSISTANT)
            await wait_for(lambda: fc.mic_requests == [True, False, True, False, True])
            await post({"do": "off"})  # turned off during the hold
            await wait_for(lambda: len(fc.mic_requests) == 6)
            await report(ASSISTANT | int(Button.A))  # still held: off stays off
            await report(0)
            assert cs.mic_on is False and len(fc.mic_requests) == 6

            await report(ASSISTANT)  # the next press
            await wait_for(lambda: len(fc.mic_requests) == 7 and fc.mic_requests[-1] is True)
            await post({"do": "ptt", "on": False})  # nothing would let go of it any more
            await wait_for(lambda: len(fc.mic_requests) == 8)
            assert fc.mic_requests[-1] is False and cs.mic_on is False
            await report(0)
            await report(ASSISTANT)  # the button is a button again
            assert len(fc.mic_requests) == 8
            await post({"do": "ptt", "on": True})
            await report(0)  # let go, seen by a button that is the switch
            await post({"do": "ptt", "on": False})
            await report(ASSISTANT)  # pressed while it is no switch
            await post({"do": "ptt", "on": True})  # and made the switch during that hold, which is no press
            await report(ASSISTANT | int(Button.A))
            await report(0)
            assert len(fc.mic_requests) == 8 and cs.mic_on is False
            await report(ASSISTANT)  # the next one is
            await wait_for(lambda: len(fc.mic_requests) == 9 and fc.mic_requests[-1] is True)
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


def test_push_to_talk_reads_the_display_links_reports_and_no_others(tmp_path: Path) -> None:
    """The display link's own report is the one whose buttons are mapped. A game session reports
    the controller as well, in a layout nobody has mapped: whatever its tenth bit is, it switches
    no microphone, and the display link's Assistant button goes on doing so during the game."""

    async def run() -> None:
        rx = make_receiver(tmp_path, mic_capture="ptt", display_rpc_delay=0.05, display_rpc_interval=0.1,
                           start_game="auto", start_game_delay=0.05)  # fmt: skip
        await rx.start()
        fc = await connect_firmware_flow(rx, game_mode=True, game_framing="raw")
        cs = single(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            await wait_for(lambda: cs.session["game"]["state"] == "game-started", 8)
            for buttons in (ASSISTANT, 0):
                seq = fc.send_game_input(buttons=buttons)
                assert await fc.wait_game_ack(5) == seq
            assert fc.mic_requests == [] and cs.mic_on is False
            assert fc.send_input_rpc(buttons=ASSISTANT)
            await wait_for(lambda: fc.mic_requests == [True])
            seq = fc.send_game_input(buttons=0)  # a report of the other kind ends no hold either
            assert await fc.wait_game_ack(5) == seq
            assert fc.mic_requests == [True] and cs.mic_on is True
            assert fc.send_input_rpc(buttons=0)
            await wait_for(lambda: fc.mic_requests == [True, False])
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


def test_a_boot_in_the_middle_of_a_hold_forgets_the_hold(tmp_path: Path) -> None:
    async def run() -> None:
        rx = make_receiver(tmp_path, mic_capture="ptt")
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            assert fc.send_input_rpc(buttons=ASSISTANT)
            await wait_for(lambda: fc.mic_requests == [True])
            fc.close()
            fc = await connect_firmware_flow(rx)  # the controller boots again, the button still down
            assert single(rx) is cs and cs.mic_on is False
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            assert fc.send_input_rpc(buttons=int(Button.A))  # nothing is held that could be let go
            await wait_for(lambda: cs.session["display"]["input_rpc"] == 1)
            assert fc.mic_requests == [] and cs.mic_on is False
            assert fc.send_input_rpc(buttons=ASSISTANT)  # and a press is a press
            await wait_for(lambda: fc.mic_requests == [True])
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


def test_mic_capture_on_asks_once_per_boot_and_a_new_boot_ends_the_recording(tmp_path: Path) -> None:
    async def run() -> None:
        rx = make_receiver(tmp_path, mic_capture="on")
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            await wait_for(lambda: fc.mic_requests == [True])
            cs.mic_stop()  # switched off by hand before the link's next milestone
            await wait_for(lambda: fc.mic_requests == [True, False])
            assert fc.send_input_rpc(buttons=1)
            await wait_for(lambda: cs.display_state == "input")
            assert fc.mic_requests == [True, False]  # which does not ask again: once per boot
            cs.mic_start()
            await wait_for(lambda: fc.mic_requests == [True, False, True])
            assert fc.send_mic_frame()
            await wait_for(lambda: cs.mic_state()["recording"] is not None)
            first = Path(str(cs.mic_state()["recording"]["file"]))

            fc.close()
            fc = await connect_firmware_flow(rx)  # the controller boots again
            assert single(rx) is cs and pages(first.read_bytes())[-1]["flags"] == 0x04
            assert cs.mic_state()["last"]["file"] == str(first)
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            await wait_for(lambda: fc.mic_requests == [True])  # asked again: this boot's microphone is off
            assert fc.send_mic_frame()
            await wait_for(lambda: cs.mic_state()["recording"] is not None)
            second = Path(str(cs.mic_state()["recording"]["file"]))
            assert second != first
        finally:
            fc.close()
            await rx.stop()
        assert pages(second.read_bytes())[-1]["flags"] == 0x04  # stopping the receiver ends it too

    asyncio.run(run())


def test_a_request_the_controller_does_not_answer_is_sent_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The request is one datagram. Lost on its way it is sent again after a moment, and so is one
    whose answer was lost, unless the sound has come meanwhile, which says the same. A controller
    that says nothing to four requests in a row is taken to be silent."""
    monkeypatch.setattr("stadiacast.controller.MIC_ANSWER_TIMEOUT", 0.1)

    async def run() -> None:
        rx = make_receiver(tmp_path)
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)

        def unanswered() -> list[tuple[bool, int, bool]]:
            return [
                (r["on"], r["tries"], r["again"] is not None) for r in rx.log.records if r["event"] == "mic_unanswered"
            ]

        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            fc.mic_lose_requests = 1  # true is lost on its way
            assert cs.mic_start()["ok"] is True
            await wait_for(lambda: fc.mic_requests == [True])
            assert unanswered() == [(True, 1, True)] and cs.mic_on is True and fc.mic_on is True
            await asyncio.sleep(0.3)
            assert fc.mic_requests == [True] and len(unanswered()) == 1  # answered: nothing more is sent

            fc.mic_lose_requests = 1  # false is lost on its way
            cs.mic_stop()
            await wait_for(lambda: fc.mic_requests == [True, False])
            assert unanswered()[1:] == [(False, 1, True)] and fc.mic_on is False

            monkeypatch.setattr("stadiacast.controller.MIC_ANSWER_TIMEOUT", 0.6)
            fc.mic_lose_answers = 1  # true arrives and its answer does not: the sound says the same
            cs.mic_start()
            await wait_for(lambda: fc.mic_requests == [True, False, True])
            assert fc.send_mic_frame()
            await asyncio.sleep(0.9)
            assert fc.mic_requests == [True, False, True] and len(unanswered()) == 2
            assert cs.mic_state()["recording"]["packets"] == 1
            cs.mic_stop()
            await wait_for(lambda: fc.mic_requests == [True, False, True, False])

            monkeypatch.setattr("stadiacast.controller.MIC_ANSWER_TIMEOUT", 0.1)
            fc.mic_lose_requests = 99  # a controller that hears nothing any more
            assert cs.mic_start()["on"] is True
            await wait_for(lambda: cs.mic_on is False)
            assert unanswered()[2:] == [(True, n, n < MIC_TRIES) for n in range(1, MIC_TRIES + 1)]
            assert rx.live_state()["controllers"]["c0"]["microphone"] is False
            assert fc.mic_requests == [True, False, True, False]
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


def test_sound_nobody_asked_for_gets_no_file_and_the_controller_is_told_to_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A controller that sends its microphone with the switch off did not hear it go off, or has
    kept it on from before. Nothing of it is written, and the controller is told to stop: a few
    times, should it go on regardless, and then no more."""
    monkeypatch.setattr("stadiacast.controller.MIC_ANSWER_TIMEOUT", 0.05)

    async def run() -> None:
        rx = make_receiver(tmp_path)
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)

        def told() -> list[tuple[bool, str]]:
            return [(r["on"], r["reason"]) for r in rx.log.records if r["event"] == "mic_switch"]

        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            fc.mic_on = True  # nobody asked
            assert fc.send_mic_frame()
            await wait_for(lambda: fc.mic_requests == [False])
            assert fc.mic_on is False and told() == [(False, "stray")]
            state = cs.mic_state()
            assert (state["on"], state["stray"], state["recording"], state["last"]) == (False, 1, None, None)
            assert not list((tmp_path / "logs").glob("mic_*"))

            fc.mic_on, fc.mic_lose_requests = True, 10**6  # one that goes on whatever it is told
            for _ in range(150):
                assert fc.send_mic_frame()
                await asyncio.sleep(0.01)
            assert told() == [(False, "stray")] * MIC_TRIES and fc.mic_requests == [False]
            strays = [r["n"] for r in rx.log.records if r["event"] == "mic_stray"]
            assert strays == [1] and cs.mic_state()["stray"] == 151 and cs.mic_state()["recording"] is None

            fc.mic_lose_requests = 0  # the switch thrown by hand counts afresh, and is heard
            cs.mic_stop()
            await wait_for(lambda: fc.mic_requests == [False, False])
            assert fc.mic_on is False
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


def test_stopping_the_receiver_tells_the_controller_to_stop(tmp_path: Path) -> None:
    """A controller left with its microphone on would go on sending, its light blinking, to a
    receiver that is gone."""

    async def run() -> None:
        rx = make_receiver(tmp_path, mic_capture="on")
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        stopped = False
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            await wait_for(lambda: fc.mic_requests == [True])
            assert fc.send_mic_frame()
            await wait_for(lambda: cs.mic_state()["recording"] is not None)
            stopped = True
            await rx.stop()
            await wait_for(lambda: fc.mic_requests == [True, False])
            assert fc.mic_on is False and cs.mic_on is False and cs.mic_state()["recording"] is None
        finally:
            fc.close()
            if not stopped:
                await rx.stop()

    asyncio.run(run())


def test_a_controller_switched_off_ends_its_recording(tmp_path: Path) -> None:
    async def run() -> None:
        rx = make_receiver(tmp_path)
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            cs.mic_start()
            await wait_for(lambda: fc.mic_requests == [True])
            assert fc.send_mic_frame()
            await wait_for(lambda: cs.mic_state()["recording"] is not None)
            path = Path(str(cs.mic_state()["recording"]["file"]))
            fc.power_off()
            await wait_for(lambda: cs.mic_state()["recording"] is None)
            assert cs.mic_on is False and pages(path.read_bytes())[-1]["flags"] == 0x04
            assert rx.live_state()["controllers"]["c0"]["microphone"] is False
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


def test_two_controllers_record_into_two_files(tmp_path: Path) -> None:
    async def run() -> None:
        rx = make_receiver(tmp_path)
        await rx.start()
        a = await connect_firmware_flow(rx, device_id="fake-a", cloud_device_id="cloud-fake-a")
        b = await connect_firmware_flow(rx, device_id="fake-b", cloud_device_id="cloud-fake-b")
        try:
            for fc in (a, b):
                await asyncio.wait_for(fc.display_ready.wait(), 5)
                await fc.keepalive()
            first, second = rx.find("fake-a"), rx.find("fake-b")
            assert first is not None and second is not None and (first.label, second.label) == ("c0", "c1")
            status, out = await call(a, rx, "POST", {"do": "on"})  # whose microphone?
            assert status == 400 and out["error"].startswith("several controllers") and not a.mic_requests
            for label in ("c0", "c1"):
                assert (await call(a, rx, "POST", {"do": "on", "controller": label}))[0] == 200
            await wait_for(lambda: a.mic_requests == [True] and b.mic_requests == [True])
            assert a.send_mic_frame() and a.send_mic_frame() and b.send_mic_frame()
            await wait_for(lambda: (first.mic_state()["recording"] or {}).get("packets") == 2)
            await wait_for(lambda: (second.mic_state()["recording"] or {}).get("packets") == 1)
            names = sorted(p.name.split("_")[1] for p in (tmp_path / "logs").glob("mic_*.ogg"))
            assert names == ["c0", "c1"]
            assert (await call(a, rx))[0] == 400  # a read wants to know as well
            status, _, raw = await a.request(
                "GET", "x", "/_mic?controller=c1", b"", "application/json", port=rx.http.http_ports[0], tls=False
            )
            out = json.loads(raw)
            assert status == 200 and out["controller"] == "c1" and out["recording"]["packets"] == 1
            assert (await call(a, rx, "POST", {"do": "off", "controller": "c0"}))[0] == 200
            await wait_for(lambda: a.mic_requests == [True, False])
            assert b.mic_requests == [True] and second.mic_state()["recording"] is not None  # the other goes on
        finally:
            a.close()
            b.close()
            await rx.stop()

    asyncio.run(run())


def test_a_run_directory_that_takes_no_file_switches_the_microphone_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("stadiacast.controller.capture_path", lambda _run_dir, _label: str(tmp_path / "gone" / "m.ogg"))

    async def run() -> None:
        rx = make_receiver(tmp_path)
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            cs.mic_start()
            await wait_for(lambda: fc.mic_requests == [True])
            assert fc.send_mic_frame()
            await wait_for(lambda: fc.mic_requests == [True, False])  # the controller is told to stop
            failed = [r for r in rx.log.records if r["event"] == "mic_recording_failed"]
            assert len(failed) == 1 and failed[0]["file"].endswith("m.ogg") and failed[0]["error"]
            state = cs.mic_state()
            assert (state["on"], state["recording"], state["last"]) == (False, None, None)
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


def test_a_disk_that_refuses_ends_the_recording_and_not_the_link(tmp_path: Path) -> None:
    class Full(io.RawIOBase):
        def write(self, _data: Any) -> int:
            raise OSError(28, "No space left on device")

    async def run() -> None:
        rx = make_receiver(tmp_path)
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()  # the receiver has seen READY by its answer
            cs.mic_start()
            await wait_for(lambda: fc.mic_requests == [True])
            assert fc.send_mic_frame()
            await wait_for(lambda: cs.mic_state()["recording"] is not None)
            assert cs._mic is not None
            cs._mic._file.close()
            cs._mic._file = cs._mic._writer.sink.file = Full()  # type: ignore[assignment,attr-defined]
            assert fc.send_mic_frame()
            await wait_for(lambda: fc.mic_requests == [True, False])  # the controller is told to stop
            failed = [r for r in rx.log.records if r["event"] == "mic_write_failed"]
            assert len(failed) == 1 and "No space left on device" in failed[0]["error"]  # said once
            assert cs.mic_on is False and cs.mic_state()["recording"] is None
            assert cs.mic_state()["last"]["packets"] == 1  # the packet the disk took
            assert (await fc.keepalive())["code"] == 0  # the link is as it was
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


@needs_ffmpeg
def test_stopping_the_receiver_sees_the_recording_and_its_decoder_out(tmp_path: Path) -> None:
    """``--mic-pcm`` through the whole receiver: the PCM file has every sample of what arrived, and
    no ffmpeg and no thread of the decoder outlives the receiver."""
    packets = real_opus(0.4)

    async def run() -> None:
        rx = make_receiver(tmp_path, mic_capture="on", mic_pcm=True)
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            await wait_for(lambda: fc.mic_requests == [True])
            for packet in packets:
                assert fc.send_mic_frame(packet)
            await wait_for(lambda: (cs.mic_state()["recording"] or {}).get("packets") == len(packets))
            state = cs.mic_state()
            assert state["pcm"] is True and state["recording"]["decoder"]["running"] is True
        finally:
            fc.close()
            await rx.stop()
        assert mic_threads() == []
        state = cs.mic_state()
        last = state["last"]
        assert state["recording"] is None and last["decoder"]["running"] is False and last["decoder"]["reason"] is None
        assert os.path.getsize(last["pcm_file"]) == len(packets) * 960 * 2 == last["decoder"]["bytes"]
        assert [r["event"] for r in rx.log.records if r["event"].startswith("mic_pcm")] == [
            "mic_pcm_started",
            "mic_pcm_ended",
        ]

    asyncio.run(run())
