"""The controller's microphone: the Ogg container, the #5 switch, PT 111, the endpoint and PTT.

The fake controller stands in for the pad's firmware: it answers ``DisplayToController`` #5 and
then sends Opus frames on the display link.  Everything here runs without a sound card, and the
one test that would need ffmpeg uses a deliberate stand-in instead.
"""

from __future__ import annotations

import asyncio
import io
import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from stadiacast import microphone
from stadiacast.microphone import (
    FRAME_SAMPLES,
    MIC_PAYLOAD_TYPE,
    MicrophoneCapture,
    OggOpusWriter,
    PcmDecoder,
    ogg_crc,
)
from stadiacast.protos import PKG_API, Protos
from stadiacast.signaling import build_answer
from tests.fake_controller import FakeController
from tests.helpers import connect_firmware_flow, make_receiver, single, wait_for

#: One 20 ms Opus packet: TOC 0xf8 (CELT, fullband, mono, one frame), then compressed silence.
OPUS = b"\xf8" + b"\x41" * 159
ASSISTANT = 1 << 10  # gamepad.Button.ASSISTANT


def pages(data: bytes) -> list[dict[str, Any]]:
    """Split an Ogg stream into pages, checking the CRC and the page sequence on the way."""
    out: list[dict[str, Any]] = []
    off = 0
    while off < len(data):
        assert data[off : off + 4] == b"OggS", f"no capture pattern at {off}"
        nseg = data[off + 26]
        segs = list(data[off + 27 : off + 27 + nseg])
        body = sum(segs)
        page = data[off : off + 27 + nseg + body]
        crc = struct.unpack("<I", page[22:26])[0]
        assert crc == ogg_crc(page[:22] + b"\x00\x00\x00\x00" + page[26:]), f"bad CRC at {off}"
        out.append(
            {
                "header": page[5],
                "granule": struct.unpack("<Q", page[6:14])[0],
                "seq": struct.unpack("<I", page[18:22])[0],
                "segs": segs,
                "payload": page[27 + nseg :],
            }
        )
        off += len(page)
    return out


# --------------------------------------------------------------- the container


def test_ogg_is_playable_and_ends_with_an_empty_eos_page(tmp_path: Path):
    """BOS + tags + one page per packet + EOS; every page's CRC is the Ogg one."""
    path = tmp_path / "mic.ogg"
    cap = MicrophoneCapture(str(path))
    for ts in (1000, 1960, 2920):
        cap.write(OPUS, ts)
    cap.close()
    got = pages(path.read_bytes())
    assert [p["header"] for p in got] == [0x02, 0x00, 0x00, 0x00, 0x00, 0x04]
    assert got[0]["payload"].startswith(b"OpusHead")
    assert got[0]["payload"][9] == 1  # mono
    assert struct.unpack("<I", got[0]["payload"][12:16])[0] == 48000
    assert got[1]["payload"].startswith(b"OpusTags")
    assert [p["granule"] for p in got] == [0, 0, 960, 1920, 2880, 2880]
    assert [p["seq"] for p in got] == list(range(len(got)))
    # An EOS page with a single 0 lacing value declares an *empty packet*: ffmpeg then ends an
    # otherwise playable file with "Packet processing failed: Invalid data found".
    assert got[-1]["segs"] == []
    assert got[2]["payload"] == OPUS
    assert cap.packets == 3 and cap.bytes == 3 * len(OPUS)


def test_granule_comes_from_the_rtp_clock(tmp_path: Path):
    """A 10 ms packet and a jump above 100 ms (a gap) must not distort the timeline."""
    cap = MicrophoneCapture(str(tmp_path / "mic.ogg"))
    cap.write(OPUS, 1000)
    cap.write(OPUS, 1480)  # 480 samples = 10 ms
    cap.write(OPUS, 1000000)  # a reboot / a lost second: fall back to the fixed 20 ms
    cap.close()
    assert [p["granule"] for p in pages((tmp_path / "mic.ogg").read_bytes())] == [0, 0, 960, 1440, 2400, 2400]


def test_an_empty_packet_is_not_a_packet():
    buf = io.BytesIO()
    w = OggOpusWriter(buf)
    w.write_packet(b"", FRAME_SAMPLES)
    w.close()
    assert w.packets == 0
    assert len(pages(buf.getvalue())) == 3  # BOS + tags + EOS


def test_pcm_without_ffmpeg_degrades_gracefully(tmp_path: Path, monkeypatch: Any):
    """The Ogg capture must survive a machine with no ffmpeg; the state says why."""
    monkeypatch.setattr(microphone.shutil, "which", lambda _name: None)
    dec = PcmDecoder(None, sink=io.BytesIO(), ffmpeg="no-such-ffmpeg-binary")
    assert dec.available is False and dec.reason
    cap = MicrophoneCapture(str(tmp_path / "mic.ogg"), pcm=True)
    cap.write(OPUS, 1000)
    cap.close()
    assert (tmp_path / "mic.pcm").exists()
    assert b"OpusHead" in (tmp_path / "mic.ogg").read_bytes()


def test_the_answer_picks_opus_when_the_offer_has_no_codec():
    """The pad offers an audio SSRC with audio_format=[]; the answerer has to name a codec."""
    p = Protos()
    offer = p.RtcSdpOffer()
    offer.audio_ssrc = 0x049A6E9C
    ep = SimpleNamespace(local_params=lambda: {"ice_ufrag": "u", "ice_password": "pw", "fingerprint": "sha-256 AA"})
    answer = build_answer(p, offer, ep).sdp_answer  # type: ignore[arg-type]
    assert answer.audio_format.codec == p.enum(f"{PKG_API}.RtcSdpAudioCodec", "AUDIO_CODEC_OPUS")
    assert (answer.audio_format.channels, answer.audio_format.sample_rate) == (2, 48000)
    # An offer that does name a codec still wins.
    fmt = offer.audio_format.add()
    fmt.codec = p.enum(f"{PKG_API}.RtcSdpAudioCodec", "AUDIO_CODEC_OPUS")
    fmt.channels, fmt.sample_rate = 1, 16000
    answer = build_answer(p, offer, ep).sdp_answer  # type: ignore[arg-type]
    assert answer.audio_format.channels == 1 and answer.audio_format.sample_rate == 16000


# --------------------------------------------------------------- through the display link


async def call(fc: FakeController, rx: Any, method: str = "GET", body: dict[str, Any] | None = None) -> tuple[int, Any]:
    """One request to /_mic on the plaintext port."""
    raw = json.dumps(body).encode() if body is not None else b""
    st, _, out = await fc.request(method, "x", "/_mic", raw, "application/json", port=rx.http.http_ports[0], tls=False)
    return st, json.loads(out) if out else None


def test_the_switch_opens_the_stream_and_frames_are_not_input(tmp_path: Path):
    """#5 goes out first, PT 111 reaches the Ogg writer, and nothing is decoded as input."""

    async def run() -> None:
        rx = make_receiver(tmp_path)
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            assert cs.mic_state()["on"] is False
            cs.mic_start()
            await wait_for(lambda: fc.mic_requests == [True])
            assert fc.mic_capture is True
            for _ in range(3):
                assert fc.send_mic_frame()
            await wait_for(lambda: cs.mic_state()["packets"] == 3)
            assert "rtp_payload_undecodable" not in rx.log.counts
            assert rx.log.counts.get("rtp_input", 0) == 0
            path = Path(str(cs.mic_state()["file"]))
            # Unbuffered: the running capture is already a valid Ogg stream, with no EOS yet.
            assert path.exists() and len(pages(path.read_bytes())) == 5  # BOS, tags, 3 frames
            assert cs.status()["session"]["mic_packets"] == 3

            cs.mic_stop()
            await wait_for(lambda: fc.mic_requests == [True, False])
            assert cs.mic_state()["on"] is False
            assert len(pages(path.read_bytes())) == 6  # the EOS page was written
            assert pages(path.read_bytes())[-1]["header"] & 0x04
            assert cs.mic_state()["file"] == str(path)  # the finished capture is still reported

            # A frame already in flight when the pad stopped must not leave a second, empty file.
            assert fc.send_mic_frame()
            await asyncio.sleep(0.05)
            assert cs.mic_state()["packets"] == 3
            assert len(list(tmp_path.glob("logs/mic_*.ogg"))) == 1
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


def test_mic_endpoint_and_assistant_push_to_talk(tmp_path: Path):
    """GET /_mic is read-only; POST on/off, pcm and ptt; the Assistant button drives #5."""

    async def run() -> None:
        rx = make_receiver(tmp_path, mic_ptt=True)
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            st, out = await call(fc, rx)
            assert st == 200 and out["on"] is False and out["ptt"] is True
            st, out = await call(fc, rx, "POST", {"do": "on"})
            assert st == 200 and out["on"] is True
            await wait_for(lambda: fc.mic_requests == [True])
            assert fc.send_mic_frame()
            await wait_for(lambda: cs.mic_state()["packets"] == 1)
            st, out = await call(fc, rx, "POST", {"do": "off"})
            assert st == 200 and out["on"] is False
            assert cs.mic_state()["packets"] == 1  # the finished capture is remembered

            st, out = await call(fc, rx, "POST", {"do": "pcm", "on": True})
            assert st == 200 and cs.mic_pcm is True
            st, out = await call(fc, rx, "POST", {"do": "ptt", "on": False})
            assert st == 200 and cs.mic_ptt is False
            assert (await call(fc, rx, "POST", {"do": "nonsense"}))[0] == 400

            # Re-arm PTT: holding Assistant opens the mic, releasing it closes the file.
            await call(fc, rx, "POST", {"do": "ptt", "on": True})
            fc.send_input_rpc(buttons=ASSISTANT)
            await wait_for(lambda: cs.mic_on is True)
            assert fc.mic_requests[-1] is True
            fc.send_input_rpc(buttons=0)
            await wait_for(lambda: cs.mic_on is False)
            assert fc.mic_requests[-1] is False
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


def test_mic_capture_starts_once_per_boot_at_ready(tmp_path: Path):
    """--mic-capture asks once, when the display link reaches READY (not per milestone)."""

    async def run() -> None:
        rx = make_receiver(tmp_path, mic_capture=True)
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            await wait_for(lambda: fc.mic_requests == [True])
            assert cs.mic_on is True
            fc.send_input_rpc(buttons=1)
            await wait_for(lambda: cs.display_state == "input")
            assert fc.mic_requests == [True]  # the input milestone does not ask again
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


def test_a_new_boot_closes_the_capture_and_arms_the_autostart(tmp_path: Path):
    """A second offer is a new boot: the Ogg file is finalised and #5 is sent again at READY."""

    async def run() -> None:
        rx = make_receiver(tmp_path, mic_capture=True)
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            await wait_for(lambda: fc.mic_requests == [True])
            assert fc.send_mic_frame()
            await wait_for(lambda: cs.mic_state()["packets"] == 1)
            path = Path(str(cs.mic_state()["file"]))
            cs.reset_boot_state("new offer")
            assert cs.mic_on is False and cs._mic is None
            assert pages(path.read_bytes())[-1]["header"] & 0x04
            assert cs._mic_autostarted is False
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


def test_the_payload_type_is_the_one_the_headset_uses():
    """Only the direction tells the mic uplink from the headset downlink."""
    assert MIC_PAYLOAD_TYPE == 111
    assert FRAME_SAMPLES == 960
