"""The controller's own microphone, as the display receives it.

The pad has a microphone (the one the Assistant button used).  It does not travel over party
chat, SCTP, or a second WebRTC session: it is an ordinary Opus RTP stream on the *display*
link's audio m-line, the same one we send the headset jack's audio down.  Two things are easy to
miss:

- The stream is switched on by ``DisplayToController`` #5 (a ``BoolValue``).  ``true`` starts
  one Opus packet every 20 ms within a frame or two; ``false`` stops it.  Nothing in the SDP
  says this, and the request is answered with an empty OK, so a silent microphone looks exactly
  like a broken decoder.
- The payload type is 111, which is also the type of the headset audio we send *to* the pad.
  Only the direction distinguishes them.  Handing those packets to the input protocol decoder
  (the receiver's old behaviour) drops every one as ``rtp_payload_undecodable``.

The encoder uses a fixed configuration: TOC ``0xf8`` (CELT, fullband, 20 ms, mono, one frame
per packet), about 64 kbit/s with no DTX, and RTP extension id 6 (``ssrc-audio-level``).  A
quiet room therefore produces full-rate packets of encoded silence, not a slow trickle -- do not
take that as a format error.

Opus cannot be decoded in pure Python, so this module only *packages* what arrives: a streaming
Ogg/Opus writer makes the capture playable on its own (``ffplay capture.ogg``), and an optional
long-lived ffmpeg turns the same Ogg pages into raw s16le PCM for a caller that wants to forward
the microphone somewhere.  With no ffmpeg on PATH the Ogg capture still works.
"""

from __future__ import annotations

import math
import os
import queue
import shutil
import struct
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any, BinaryIO, Protocol

#: Opus payload type of both the headset downlink and the microphone uplink.
MIC_PAYLOAD_TYPE = 111
#: What the pad actually encodes: 20 ms of mono audio at 48 kHz.
SAMPLE_RATE = 48000
CHANNELS = 1
FRAME_SAMPLES = 960
#: Ignore RTP timestamp jumps above 100 ms when deriving a frame's length from the clock.
MAX_TIMESTAMP_GAP = 4800
#: Ogg pages waiting for ffmpeg before new ones are dropped (~4 s of audio).
QUEUE_PAGES = 200

PCM_RATE = 48000
PCM_CHANNELS = 1
PCM_WIDTH = 2  # s16le

#: CRC-32 as used by Ogg: MSB-first, poly 0x04C11DB7, init 0, no reflection, no final XOR.
_CRC_TABLE: list[int] = []
for _i in range(256):
    _r = _i << 24
    for _ in range(8):
        _r = ((_r << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if _r & 0x80000000 else (_r << 1) & 0xFFFFFFFF
    _CRC_TABLE.append(_r)


class Sink(Protocol):
    """What :class:`OggOpusWriter` needs from its output."""

    def write(self, data: bytes) -> int: ...

    def flush(self) -> None: ...


def _no_log(*_args: Any, **_kwargs: Any) -> None:
    """A log sink for standalone use (the container and the decoder do not need one)."""


def ogg_crc(data: bytes) -> int:
    """Ogg's variant of CRC-32 (not the zlib one)."""
    crc = 0
    for b in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ _CRC_TABLE[((crc >> 24) & 0xFF) ^ b]
    return crc


def _lacing(length: int) -> list[int]:
    """Ogg lacing values for one packet (a trailing 0 when it is a multiple of 255)."""
    out = [255] * (length // 255)
    out.append(length % 255)
    return out


class OggOpusWriter:
    """Wrap arriving Opus packets in a playable Ogg stream.

    One page per packet: inefficient (a 27-byte header every 20 ms) but keeps packet boundaries
    exact, needs no buffering, and every player accepts it.
    """

    def __init__(
        self,
        sink: Sink,
        *,
        channels: int = CHANNELS,
        sample_rate: int = SAMPLE_RATE,
        pre_skip: int = 0,
        serial: int = 0x53544144,
    ) -> None:
        self.sink = sink
        self.serial = serial
        self.page_seq = 0
        self.granule = 0
        self.packets = 0
        self.bytes = 0
        self._write_page(b"OpusHead" + struct.pack("<BBHIhB", 1, channels, pre_skip, sample_rate, 0, 0), 0x02, 0)
        tags = b"OpusTags" + struct.pack("<I", 10) + b"stadiacast" + struct.pack("<I", 0)
        self._write_page(tags, 0x00, 0)

    def write_packet(self, payload: bytes, samples: int) -> None:
        """Append one Opus packet; ``samples`` is its length in 48 kHz samples (960 for 20 ms)."""
        if not payload:
            return
        self.granule += int(samples)
        self._write_page(payload, 0x00, self.granule)
        self.packets += 1
        self.bytes += len(payload)

    def close(self) -> None:
        """Terminate the stream with an empty EOS page."""
        self._write_page(b"", 0x04, self.granule)
        self.sink.flush()

    def _write_page(self, payload: bytes, header_type: int, granule: int) -> None:
        # An empty page (the EOS marker) must declare *no* lacing values.  A single 0 lacing
        # value declares an empty packet instead, and an empty Opus packet makes ffmpeg finish
        # an otherwise playable file with "Packet processing failed: Invalid data found".
        segments = _lacing(len(payload)) if payload else []
        if len(segments) > 255:
            raise ValueError(f"packet too large for one Ogg page: {len(payload)} bytes")
        header = bytearray(b"OggS")
        header += bytes([0, header_type])
        header += struct.pack("<QII", granule & 0xFFFFFFFFFFFFFFFF, self.serial, self.page_seq)
        header += b"\x00\x00\x00\x00"  # CRC placeholder
        header += bytes([len(segments)])
        header += bytes(segments)
        page = bytes(header) + payload
        crc = ogg_crc(page)
        self.sink.write(page[:22] + struct.pack("<I", crc) + page[26:])
        self.page_seq += 1


class PcmLevel:
    """Peak/RMS of the most recent PCM block, in dBFS."""

    __slots__ = ("blocks", "peak_db", "rms_db", "samples")

    def __init__(self) -> None:
        self.peak_db: float | None = None
        self.rms_db: float | None = None
        self.blocks = 0
        self.samples = 0

    def feed(self, pcm: bytes) -> None:
        n = len(pcm) // PCM_WIDTH
        if n <= 0:
            return
        peak = 0
        total = 0
        for i in range(0, n * PCM_WIDTH, PCM_WIDTH):
            v = int.from_bytes(pcm[i : i + PCM_WIDTH], "little", signed=True)
            a = -v if v < 0 else v
            if a > peak:
                peak = a
            total += v * v
        self.blocks += 1
        self.samples += n
        self.peak_db = 20.0 * math.log10(max(1, peak) / 32768.0)
        self.rms_db = 20.0 * math.log10(max(1.0, math.sqrt(total / n)) / 32768.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "peak_db": None if self.peak_db is None else round(self.peak_db, 1),
            "rms_db": None if self.rms_db is None else round(self.rms_db, 1),
            "blocks": self.blocks,
        }


class PcmDecoder:
    """One long-lived ffmpeg: Ogg/Opus in on stdin, s16le PCM out on stdout.

    Feeding the process goes through a bounded queue drained by a writer thread, so the RTP
    callback never blocks on a slow decoder; a full queue drops pages rather than stalling input.
    ``sink`` (a file, a socket wrapper, ...) receives the decoded bytes as they arrive.
    """

    def __init__(
        self,
        log: Callable[..., Any] | None = None,
        *,
        sink: BinaryIO,
        ffmpeg: str | None = None,
    ) -> None:
        self.log: Callable[..., Any] = log or _no_log
        self.sink: BinaryIO | None = sink
        self.level = PcmLevel()
        self.available = False
        self.reason: str | None = None
        self.read_bytes = 0
        self.dropped = 0
        self._proc: subprocess.Popen[bytes] | None = None
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=QUEUE_PAGES)
        self._writer: threading.Thread | None = None
        self._reader: threading.Thread | None = None
        exe = ffmpeg or shutil.which("ffmpeg")
        if not exe:
            self.reason = "ffmpeg not found on PATH"
            self.log("mic_pcm_unavailable", reason=self.reason)
            return
        try:
            self._proc = subprocess.Popen(
                [
                    exe,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-fflags",
                    "nobuffer",
                    "-i",
                    "pipe:0",
                    "-f",
                    "s16le",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    str(PCM_RATE),
                    "-ac",
                    str(PCM_CHANNELS),
                    "pipe:1",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except OSError as e:
            self.reason = str(e)
            self.log("mic_pcm_unavailable", reason=self.reason)
            return
        self.available = True
        self._writer = threading.Thread(target=self._drain, name="mic-pcm-w", daemon=True)
        self._reader = threading.Thread(target=self._read, name="mic-pcm-r", daemon=True)
        self._writer.start()
        self._reader.start()
        self.log("mic_pcm_started", ffmpeg=exe, rate=PCM_RATE, channels=PCM_CHANNELS)

    def write_page(self, page: bytes) -> None:
        """Hand one Ogg page to the decoder.  Never blocks."""
        if not self.available:
            return
        try:
            self._queue.put_nowait(page)
        except queue.Full:
            self.dropped += 1

    def _drain(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdin is not None
        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                proc.stdin.write(item)
                proc.stdin.flush()
            except (OSError, ValueError):
                self.available = False
                self.reason = "decoder closed its input"
                return

    def _read(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        while True:
            try:
                chunk = proc.stdout.read1(8192) if hasattr(proc.stdout, "read1") else proc.stdout.read(8192)
            except (OSError, ValueError):
                return
            if not chunk:
                return
            self.read_bytes += len(chunk)
            self.level.feed(chunk)
            if self.sink is not None:
                try:
                    self.sink.write(chunk)
                except (OSError, ValueError):
                    self.sink = None

    def close(self) -> None:
        """Close ffmpeg's input and let it flush the Ogg stream it has already been given."""
        if self._proc is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self.available = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "bytes": self.read_bytes,
            "dropped": self.dropped,
            **self.level.as_dict(),
        }


class PageSink:
    """File-like adapter that pushes Ogg pages into a :class:`PcmDecoder`."""

    def __init__(self, decoder: PcmDecoder) -> None:
        self.decoder = decoder

    def write(self, data: bytes) -> int:
        self.decoder.write_page(bytes(data))
        return len(data)

    def flush(self) -> None:
        pass  # ffmpeg is flushed by the decoder's writer thread


class TeeSink:
    """Mirror the Ogg bytes into the capture file and, when it is on, into the PCM decoder."""

    def __init__(self, *sinks: Sink | None) -> None:
        self.sinks = [s for s in sinks if s is not None]

    def write(self, data: bytes) -> int:
        for s in list(self.sinks):
            try:
                s.write(data)
            except (OSError, ValueError):
                self.sinks.remove(s)
        return len(data)

    def flush(self) -> None:
        for s in list(self.sinks):
            try:
                s.flush()
            except (OSError, ValueError):
                self.sinks.remove(s)


class MicrophoneCapture:
    """One recording of the pad's microphone, from the first packet until #5 goes false.

    ``path`` is an Ogg file.  With ``pcm=True`` a sibling ``.pcm`` file receives s16le mono
    48 kHz samples as ffmpeg decodes them live, so another process can forward the microphone
    while it is being captured.
    """

    def __init__(
        self,
        path: str,
        *,
        pcm: bool = False,
        log: Callable[..., Any] | None = None,
    ) -> None:
        self.log: Callable[..., Any] = log or _no_log
        self.path = path
        self.pcm_path: str | None = None
        self.packets = 0
        self.bytes = 0
        self.last_ts: int | None = None
        self._fh: BinaryIO | None = None
        self._pcm_fh: BinaryIO | None = None
        self._decoder: PcmDecoder | None = None
        self._writer: OggOpusWriter | None = None

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Unbuffered on purpose: a capture is useful while it is still running (ffplay, a tail,
        # another process forwarding the PCM), and a 600-byte buffer would keep the first pages
        # invisible for a long time.  The handles stay open until close().
        self._fh = open(path, "wb", buffering=0)  # noqa: SIM115
        sinks: list[Sink | None] = [self._fh]
        if pcm:
            self.pcm_path = os.path.splitext(path)[0] + ".pcm"
            self._pcm_fh = open(self.pcm_path, "wb", buffering=0)  # noqa: SIM115
            self._decoder = PcmDecoder(self.log, sink=self._pcm_fh)
            sinks.append(PageSink(self._decoder))
        self._writer = OggOpusWriter(TeeSink(*sinks), channels=CHANNELS)
        self.log("mic_open", path=path, pcm=self.pcm_path)

    def write(self, payload: bytes, timestamp: int) -> None:
        """One RTP payload: derive its length from the RTP clock and append it."""
        if self._writer is None:
            return
        samples = FRAME_SAMPLES
        if self.last_ts is not None:
            delta = (timestamp - self.last_ts) & 0xFFFFFFFF
            if 0 < delta <= MAX_TIMESTAMP_GAP:
                samples = delta
        self.last_ts = timestamp
        self._writer.write_packet(payload, samples)
        self.packets += 1
        self.bytes += len(payload)

    def close(self) -> None:
        """Write the EOS page and release both files and ffmpeg."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        for fh in (self._fh, self._pcm_fh):
            if fh is not None:
                try:
                    fh.close()
                except OSError:
                    pass
        self._fh = self._pcm_fh = None
        if self._decoder is not None:
            self._decoder.close()
        self.log("mic_close", path=self.path, packets=self.packets, bytes=self.bytes)

    def state(self) -> dict[str, Any]:
        """What ``/_mic`` and ``/_status`` report about this recording."""
        return {
            "packets": self.packets,
            "bytes": self.bytes,
            "file": self.path,
            "pcm_file": self.pcm_path,
            "pcm": None if self._decoder is None else self._decoder.as_dict(),
        }


def capture_path(log_dir: str | None, label: str) -> str:
    """A fresh ``mic_<label>_<HHMMSS>.ogg`` in the run directory; two quick taps never collide."""
    stamp = time.strftime("%H%M%S")
    path = os.path.join(log_dir or ".", f"mic_{label}_{stamp}.ogg")
    n = 1
    while os.path.exists(path) or os.path.exists(os.path.splitext(path)[0] + ".pcm"):
        n += 1
        path = os.path.join(log_dir or ".", f"mic_{label}_{stamp}_{n}.ogg")
    return path
