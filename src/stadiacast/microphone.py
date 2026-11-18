"""The controller's microphone, as a recording.

A controller that is asked for its microphone (``DisplayToController.set_microphone``, see
``docs/protocol.md``) sends it as Opus in RTP on the display link, one packet per 20 ms. Opus
cannot be decoded in plain Python, so this module packages what arrives: :class:`OggOpusWriter`
puts the packets into an Ogg stream as they come, which makes a file any player opens, also
while it is still being written. :class:`PcmDecoder` is optional: one ffmpeg that turns the same
Ogg stream into s16le PCM as it arrives, for a program that wants the samples. The Ogg
recording itself needs nothing.

:class:`MicrophoneCapture` is one recording: the Ogg file, and the PCM file beside it when asked
for. It keeps the stream's own time: a packet that was lost leaves a packet that asks the
decoder to conceal it, so that a second of sound is a second of file (a hole of more than a
second is a stream that stopped, and is left out). Its ``write`` and
``close`` are called on the event loop and never wait: the Ogg file takes a few hundred bytes
per packet, and everything around ffmpeg happens on threads of the decoder.
"""

from __future__ import annotations

import array
import contextlib
import math
import operator
import os
import queue
import secrets
import struct
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import IO, Any, BinaryIO, Protocol

from stadiacast.audio import ffmpeg_path

#: The RTP payload type of the microphone's Opus, which is also the one of the headset stream
#: the receiver sends: the two are told apart by their direction.
MIC_PAYLOAD_TYPE = 111
#: Opus counts its time at 48 kHz, and the controller encodes one channel.
SAMPLE_RATE = 48000
CHANNELS = 1
#: The samples of one frame by the configuration in a packet's first byte (RFC 6716, section
#: 3.1): SILK with 10, 20, 40 and 60 ms in three bandwidths, the hybrid with 10 and 20 ms in two,
#: CELT with 2.5, 5, 10 and 20 ms in four.
_FRAME_SAMPLES = (480, 960, 1920, 2880) * 3 + (480, 960) * 2 + (120, 240, 480, 960) * 4
#: The longest packet Opus allows: 120 ms.
MAX_PACKET_SAMPLES = 5760
#: A packet longer than this is not the microphone's (its packets have 161 bytes while the sound
#: is steady and under 300 while it changes), and what a page costs to write grows with its length.
MAX_PACKET_BYTES = 1500
#: The longest hole that is filled with concealment (one second). A longer one is a stream that
#: stopped and started again: the recording goes on behind it without the silence.
MAX_CONCEALED_SAMPLES = 48000
#: So many packets in a row, each the successor of the one before and all of them behind the
#: stream's last: those are no late packets, that is a stream which began again with numbers of
#: its own. No packet is overtaken by a tenth of a second of its successors.
RESTART_PACKETS = 5

#: The PCM ffmpeg is asked for.
PCM_RATE = 48000
PCM_CHANNELS = 1
PCM_WIDTH = 2  # s16le
#: Ogg pages that may wait for ffmpeg (4 s of audio). Past that the newest are dropped: a decoder
#: that has fallen behind loses audio, and the event loop that hands the pages over never waits.
QUEUE_PAGES = 200
#: How long ffmpeg has to end by itself once its recording is closed, before it is killed.
DECODER_EXIT_TIMEOUT = 3.0
#: Bytes kept of what ffmpeg wrote before it ended: the end of it goes into the log.
STDERR_KEPT = 2048

#: Ogg's CRC-32: most significant bit first, polynomial 0x04C11DB7, start 0, no final XOR. Not
#: the one of zlib.
_CRC_TABLE: list[int] = []
for _i in range(256):
    _r = _i << 24
    for _ in range(8):
        _r = ((_r << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if _r & 0x80000000 else (_r << 1) & 0xFFFFFFFF
    _CRC_TABLE.append(_r)


class Sink(Protocol):
    """Where :class:`OggOpusWriter` writes its pages."""

    def write(self, data: bytes, /) -> Any: ...


def _no_log(*_args: Any, **_kwargs: Any) -> None:
    """For use without a receiver: the container and the decoder need no log."""


def ogg_crc(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ _CRC_TABLE[((crc >> 24) & 0xFF) ^ b]
    return crc


def opus_samples(packet: bytes) -> int | None:
    """How long an Opus packet is, in samples at 48 kHz, read from its first byte and the frame
    count behind it (RFC 6716, section 3.2); ``None`` for what cannot be a packet."""
    if not packet:
        return None
    frame = _FRAME_SAMPLES[packet[0] >> 3]
    code = packet[0] & 3
    if code == 3:  # a number of frames, in the second byte
        if len(packet) < 2 or not packet[1] & 0x3F:
            return None
        frames = packet[1] & 0x3F
    else:
        frames = 1 if code == 0 else 2
    samples = frame * frames
    return samples if samples <= MAX_PACKET_SAMPLES else None


def _lacing(length: int) -> list[int]:
    """A packet's lacing values: 255 for every full segment, then what is left, which is 0 for a
    packet that fills its last segment."""
    return [255] * (length // 255) + [length % 255]


class OggOpusWriter:
    """An Ogg stream of the Opus packets it is given (RFC 7845), written as they come.

    Every packet is a page of its own. That spends 28 bytes per packet and buys a stream that is
    whole after each write: a reader of the file, or of a pipe, never waits for a page to fill.
    """

    def __init__(
        self, sink: Sink, *, channels: int = CHANNELS, sample_rate: int = SAMPLE_RATE, serial: int | None = None
    ) -> None:
        self.sink = sink
        #: A stream's own number, by chance: two files put end to end are then two streams.
        self.serial = secrets.randbits(32) if serial is None else serial
        self.page_seq = 0
        self.granule = 0  # samples so far, at 48 kHz
        self.packets = 0
        self.bytes = 0
        # The identification header: version 1, the channels, no pre-skip (every packet's samples
        # are the stream's: what the encoder looks ahead by is not known, and is a few
        # milliseconds), the rate the sound was taken at, no gain, mapping family 0 (mono or
        # stereo). Then the comment header: a vendor and no comments. Each has a page of its own,
        # the first one flagged as the beginning.
        self._page(b"OpusHead" + struct.pack("<BBHIhB", 1, channels, 0, sample_rate, 0, 0), 0x02, 0)
        vendor = b"stadiacast"
        self._page(b"OpusTags" + struct.pack("<I", len(vendor)) + vendor + struct.pack("<I", 0), 0x00, 0)

    def write_packet(self, payload: bytes, samples: int) -> None:
        """One Opus packet and the samples it holds (:func:`opus_samples`). The granule position
        counts those and nothing else, as RFC 7845 asks."""
        if not payload:
            return
        self._page(payload, 0x00, self.granule + samples)
        self.granule += samples  # behind the write: a page the sink refused is not in the stream
        self.packets += 1
        self.bytes += len(payload)

    def close(self) -> None:
        """End the stream with a page that carries the end flag and no packet."""
        self._page(b"", 0x04, self.granule)

    def _page(self, payload: bytes, flags: int, granule: int) -> None:
        # The closing page has no lacing value at all. One value of 0 would be a packet of no
        # bytes, which is not Opus: ffmpeg then ends a good file with "Invalid data found".
        segments = _lacing(len(payload)) if payload else []
        if len(segments) > 255:
            raise ValueError(f"a packet of {len(payload)} bytes does not fit one Ogg page")
        head = b"OggS" + struct.pack("<BBQIIIB", 0, flags, granule, self.serial, self.page_seq, 0, len(segments))
        page = head + bytes(segments) + payload
        self.sink.write(page[:22] + struct.pack("<I", ogg_crc(page)) + page[26:])
        self.page_seq += 1


def pcm_level(pcm: bytes) -> tuple[float, float] | None:
    """The peak and the RMS of s16le PCM in dBFS, or ``None`` for less than one sample."""
    samples = array.array("h", pcm[: len(pcm) - len(pcm) % PCM_WIDTH])
    if not samples:
        return None
    peak = max(max(samples), -min(samples), 1)
    # map keeps the squares out of Python bytecode, as audio.py does for its gain
    rms = math.sqrt(sum(map(operator.mul, samples, samples)) / len(samples))
    return 20.0 * math.log10(peak / 32768.0), 20.0 * math.log10(max(rms, 1.0) / 32768.0)


def _write_all(pipe: IO[bytes], data: bytes) -> None:
    """All of ``data`` into a pipe or a file opened without a buffer, where one write may take a part."""
    view = memoryview(data)
    while view:
        view = view[pipe.write(view) :]


class PcmDecoder:
    """One ffmpeg for one Ogg stream: the stream goes in, s16le PCM comes out, into the file
    ``path`` and to ``heard`` (either or both).

    None of it happens on the caller's thread, which is the event loop's. :meth:`write` puts a
    page into a queue with a limit (:data:`QUEUE_PAGES`), and :meth:`close` only says that the
    stream is over. One thread starts ffmpeg and hands it the pages. One reads the PCM, measures
    it, writes the file and calls ``heard``. One keeps the end of what ffmpeg says, for the log.
    After :meth:`close` the first closes ffmpeg's input behind the last page, ffmpeg decodes
    what it still holds and ends, and the second closes the file behind the last sample: closing
    loses nothing and waits for nothing. An ffmpeg that has not ended
    :data:`DECODER_EXIT_TIMEOUT` after :meth:`close` is killed. :meth:`join` waits for all of
    it, for a caller that is not the event loop.

    ``events`` names its three log records (``mic_pcm_started`` and so on), and ``rate`` is the
    rate of the samples, which ffmpeg converts to.
    """

    def __init__(
        self,
        path: str | None,
        log: Callable[..., Any] | None = None,
        *,
        rate: int = PCM_RATE,
        heard: Callable[[bytes], Any] | None = None,
        events: str = "mic_pcm",
    ) -> None:
        self.log: Callable[..., Any] = log or _no_log
        self.path: str | None = None  # the PCM file, when one was asked for and there is an ffmpeg to make it
        self.rate = rate
        self.reason: str | None = None  # why there is no PCM, or no more of it
        self.read_bytes = 0
        self.dropped = 0
        self.level: tuple[float, float] | None = None  # of the block read last
        self._heard = heard
        self._events = events
        self._names = events.replace("_", "-")  # of its threads
        self._exe = ffmpeg_path()
        self._proc: subprocess.Popen[bytes] | None = None
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._stderr: deque[bytes] = deque()
        self._lock = threading.Lock()  # for what close() and the threads decide between them
        self._closed = False
        self._empty = False  # closed with no packet in the stream
        self._over = self._exe is None  # ffmpeg has ended, or was never started
        self._deadline: threading.Timer | None = None
        if self._exe is None:
            self._unavailable("ffmpeg is not on PATH")
            return
        self.path = path
        try:
            self._thread(self._feed, f"{self._names}-in")
        except RuntimeError as e:  # a computer out of threads: the recording goes on without the samples
            self.path = None
            self._over = True
            self._unavailable(f"no thread for ffmpeg: {e}")

    def _thread(self, run: Callable[..., None], name: str, *args: Any) -> threading.Thread:
        thread = threading.Thread(target=run, name=name, args=args, daemon=True)
        thread.start()
        self._threads.append(thread)  # behind the start: join() never meets a thread that has not begun
        return thread

    def _unavailable(self, reason: str) -> None:
        self.reason = reason
        self.log(f"{self._events}_unavailable", reason=reason)

    @property
    def running(self) -> bool:
        """Pages are taken: ffmpeg runs, or is being started."""
        return self._exe is not None and self.reason is None and not self._closed

    @property
    def ended(self) -> bool:
        """ffmpeg has ended and the file is closed, or there never was an ffmpeg."""
        return not any(thread.is_alive() for thread in self._threads)

    def write(self, page: bytes) -> None:
        """One Ogg page for ffmpeg. Never waits: with the queue full the page is dropped."""
        if not self.running:
            return
        if self._queue.qsize() >= QUEUE_PAGES:
            self.dropped += 1
            return
        self._queue.put_nowait(page)

    def close(self, *, empty: bool = False) -> None:
        """The stream is over. Returns at once; the threads see the rest through. ``empty`` says
        that no packet was in it: ffmpeg calls that an error, and it is none."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._empty = empty
            self._queue.put_nowait(None)
            self._arm()

    def join(self) -> None:
        """Wait until ffmpeg has ended and the file is closed. Not for the event loop."""
        seen = 0
        while seen < len(self._threads):  # the threads start one another
            self._threads[seen].join(2 * DECODER_EXIT_TIMEOUT)
            seen += 1

    def _arm(self) -> None:
        """With the lock held, once the stream is over and ffmpeg runs: give ffmpeg its time to
        end. One that does not read keeps the first thread in its write, and one that does not
        end keeps the second in its read: both are let go by the end of the process."""
        if self._proc is None or self._over or self._deadline is not None:
            return
        self._deadline = threading.Timer(DECODER_EXIT_TIMEOUT, self._proc.kill)
        self._deadline.name = f"{self._names}-end"
        self._deadline.daemon = True
        try:
            self._deadline.start()
        except RuntimeError:  # no thread to wait on: ffmpeg ends now, with what it has
            self._proc.kill()
            return
        self._threads.append(self._deadline)

    def _finish(self) -> None:
        """ffmpeg has ended, or did not start: there is nothing left to end."""
        with self._lock:
            self._over = True
            if self._deadline is not None:
                self._deadline.cancel()

    def _feed(self) -> None:
        """Start ffmpeg, then hand it the pages until the stream is over."""
        path, exe = self.path, self._exe
        assert exe is not None
        command = [
            exe, "-hide_banner", "-loglevel", "error", "-nostats",
            # Told the format, ffmpeg decodes from the first page on, each packet as it comes.
            # (-fflags nobuffer would drop the packet it reads while it looks at the stream.)
            "-f", "ogg", "-i", "pipe:0",
            "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(self.rate), "-ac", str(PCM_CHANNELS),
            "-flush_packets", "1", "pipe:1",
        ]  # fmt: skip
        file: BinaryIO | None = None
        try:
            if path is not None:
                file = open(path, "wb", buffering=0)  # noqa: SIM115
        except OSError as e:
            self.path = None
            self._unavailable(f"no PCM file: {e}")
            self._finish()
            return
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                # A session of its own: Ctrl-C in the receiver's terminal is the receiver's to
                # act on. An ffmpeg that got it too would end before it had the last pages.
                start_new_session=True,
            )
        except OSError as e:
            if file is not None and path is not None:
                file.close()
                with contextlib.suppress(OSError):
                    os.unlink(path)
            self.path = None
            self._unavailable(f"ffmpeg did not start: {e}")
            self._finish()
            return
        assert proc.stdin is not None
        self.log(f"{self._events}_started", ffmpeg=exe, file=path, rate=self.rate, channels=PCM_CHANNELS)
        with self._lock:
            self._proc = proc
            if self._closed:  # the recording was shorter than ffmpeg's start
                self._arm()
        try:
            said = self._thread(self._keep_stderr, f"{self._names}-err", proc)
            self._thread(self._read, f"{self._names}-out", proc, file, said)
        except RuntimeError as e:  # no thread to read it with: an ffmpeg nobody reads is of no use
            proc.kill()
            proc.wait()
            for end in (proc.stdin, proc.stdout, proc.stderr, file):
                with contextlib.suppress(OSError):
                    if end is not None:
                        end.close()
            self.reason = f"ffmpeg could not be read: {e}"
            self._finish()
            self.log(f"{self._events}_ended", status=proc.returncode, reason=self.reason, bytes=0, dropped=self.dropped)
            return
        try:
            while (page := self._queue.get()) is not None:
                _write_all(proc.stdin, page)
        except OSError:  # ffmpeg has ended by itself: the thread that reads it says why
            pass
        finally:
            with contextlib.suppress(OSError):
                proc.stdin.close()

    def _read(self, proc: subprocess.Popen[bytes], file: BinaryIO | None, said: threading.Thread) -> None:
        assert proc.stdout is not None
        odd = b""  # half a sample, when a read ended inside one
        try:
            while chunk := proc.stdout.read(8192):
                self.read_bytes += len(chunk)
                if file is not None:
                    _write_all(file, chunk)
                if self._heard is not None:
                    self._heard(chunk)
                block = odd + chunk
                whole = len(block) - len(block) % PCM_WIDTH
                self.level = pcm_level(block[:whole]) or self.level
                odd = block[whole:]
        except OSError as e:  # the disk is full, or the file was taken away
            self.reason = f"the PCM file could not be written: {e}"
            proc.kill()
        finally:
            for end in (file, proc.stdout):
                with contextlib.suppress(OSError):
                    if end is not None:
                        end.close()
        code = proc.wait()
        self._finish()
        self._queue.put_nowait(None)  # an ffmpeg that ended by itself has the first thread waiting for pages
        said.join(1.0)  # what ffmpeg said last is read by then
        with self._lock:
            last = b"".join(self._stderr).decode(errors="replace").strip()
        if self.reason is None and not self._empty and (code != 0 or not self._closed):
            self.reason = f"ffmpeg ended with status {code}" + (f": {last.splitlines()[-1]}" if last else "")
        self.log(f"{self._events}_ended", status=code, reason=self.reason, bytes=self.read_bytes, dropped=self.dropped)

    def _keep_stderr(self, proc: subprocess.Popen[bytes]) -> None:
        assert proc.stderr is not None
        kept = 0
        with contextlib.suppress(OSError):
            while chunk := proc.stderr.read(4096):
                with self._lock:
                    self._stderr.append(chunk)
                    kept += len(chunk)
                    while kept - len(self._stderr[0]) >= STDERR_KEPT:
                        kept -= len(self._stderr.popleft())
        with contextlib.suppress(OSError):
            proc.stderr.close()

    def state(self) -> dict[str, Any]:
        peak, rms = self.level or (None, None)
        return {
            "running": self.running,
            "reason": self.reason,
            "bytes": self.read_bytes,
            "dropped": self.dropped,
            "peak_db": None if peak is None else round(peak, 1),
            "rms_db": None if rms is None else round(rms, 1),
        }


class PacketOrder:
    """A stream's packets, placed as they arrive: what goes into the stream for each of them.

    The sequence number says whether a packet is news: one that comes twice, or after its
    successors, is left out, since its place in the stream is taken. The timestamp says what is
    missing in front of it: every lost frame gets a packet of the one byte that asks the decoder
    to conceal a frame (RFC 7845, section 4.1), up to :data:`MAX_CONCEALED_SAMPLES`.
    :data:`RESTART_PACKETS` in a row behind the stream are a stream that began again, and the
    stream goes on with it: ``restarted`` is told its number and the one it had.
    """

    def __init__(self, restarted: Callable[[int, int], Any] | None = None) -> None:
        self.concealed = 0  # packets that were lost, each with a packet in its place that says so
        self.late = 0  # packets that came twice, or after the ones behind them: left out
        self.ignored = 0  # what was not an Opus packet, or too long for the microphone's
        self._restarted = restarted
        self._last_seq: int | None = None
        self._next_ts = 0  # the RTP timestamp the next packet has when nothing is lost
        self._behind: tuple[int, int] | None = None  # the last packet behind the stream, and how many in a row

    def place(self, payload: bytes, timestamp: int, seq: int) -> list[tuple[bytes, int]]:
        """The packets this arrival puts into the stream, each with its samples: none, or the
        ones that stand for what was lost and then the packet itself."""
        samples = opus_samples(payload) if len(payload) <= MAX_PACKET_BYTES else None
        if samples is None:
            self.ignored += 1
            return []
        placed: list[tuple[bytes, int]] = []
        if self._last_seq is not None:
            if not 0 < (seq - self._last_seq) & 0xFFFF < 0x8000:
                follows = self._behind is not None and (seq - self._behind[0]) & 0xFFFF == 1
                self._behind = (seq, self._behind[1] + 1 if self._behind is not None and follows else 1)
                if self._behind[1] < RESTART_PACKETS:
                    self.late += 1
                    return []
                if self._restarted is not None:
                    self._restarted(seq, self._last_seq)
            else:
                missing = (timestamp - self._next_ts) & 0xFFFFFFFF
                if 0 < missing <= MAX_CONCEALED_SAMPLES:
                    frame = _FRAME_SAMPLES[payload[0] >> 3]
                    lost = missing // frame
                    placed.extend([(bytes([payload[0] & 0xFC]), frame)] * lost)  # this configuration, no data
                    self.concealed += lost
        self._behind = None
        self._last_seq = seq
        self._next_ts = (timestamp + samples) & 0xFFFFFFFF
        placed.append((payload, samples))
        return placed


class _Pages:
    """The Ogg writer's sink: every page goes into the file, and to the decoder when there is one."""

    def __init__(self, file: BinaryIO, decoder: PcmDecoder | None) -> None:
        self.file = file
        self.decoder = decoder

    def write(self, data: bytes) -> None:
        _write_all(self.file, data)
        if self.decoder is not None:
            self.decoder.write(data)


class MicrophoneCapture:
    """One recording of a controller's microphone: an Ogg file, written as the packets arrive.

    With ``pcm`` a file of the same name ending in ``.pcm`` gets the samples as ffmpeg decodes
    them (s16le, 48 kHz, mono), a moment behind the Ogg file. A write the disk refuses is an
    ``OSError`` to the caller, who then closes the recording.
    """

    def __init__(self, path: str, *, pcm: bool = False, log: Callable[..., Any] | None = None) -> None:
        self.log: Callable[..., Any] = log or _no_log
        self.path = path
        self.packets = 0  # written as they came
        self.payload_bytes = 0
        self._order = PacketOrder(lambda seq, was: self.log("mic_stream_restarted", file=path, seq=seq, was=was))
        self._closed = False
        self._refused = False  # the disk refused a write: the last page will be refused too
        # Without a buffer: each page is one write, and the file is a whole Ogg stream after it,
        # for whoever reads it while the recording runs.
        self._file: BinaryIO = open(path, "wb", buffering=0)  # noqa: SIM115
        self.decoder = PcmDecoder(os.path.splitext(path)[0] + ".pcm", self.log) if pcm else None
        try:
            self._writer = OggOpusWriter(_Pages(self._file, self.decoder))
        except OSError:
            self._file.close()
            if self.decoder is not None:
                self.decoder.close()
            raise
        self.log("mic_recording", file=path, pcm_file=self.pcm_path)

    @property
    def pcm_path(self) -> str | None:
        return None if self.decoder is None else self.decoder.path

    @property
    def concealed(self) -> int:
        """Packets that were lost, each with a packet in its place that says so."""
        return self._order.concealed

    @property
    def late(self) -> int:
        """Packets that came twice, or after the ones behind them: not written."""
        return self._order.late

    @property
    def ignored(self) -> int:
        """What was not an Opus packet, or too long for the microphone's."""
        return self._order.ignored

    def write(self, payload: bytes, timestamp: int, seq: int) -> None:
        """One RTP payload with its RTP timestamp and sequence number, placed by
        :class:`PacketOrder`: left out when it is late or no packet, written behind what stands
        for the packets lost in front of it otherwise."""
        if self._closed:
            return
        placed = self._order.place(payload, timestamp, seq)
        if not placed:
            return
        try:
            for packet, samples in placed:
                self._writer.write_packet(packet, samples)
        except OSError:
            self._refused = True
            raise
        self.packets += 1
        self.payload_bytes += len(payload)

    def close(self) -> None:
        """End the Ogg stream and close its file; the decoder ends by itself behind it."""
        if self._closed:
            return
        self._closed = True
        try:
            self._writer.close()
        except OSError as e:
            if not self._refused:  # the disk that refused a packet refuses the last page too: said once
                self.log("mic_write_failed", file=self.path, error=str(e))
        finally:
            with contextlib.suppress(OSError):
                self._file.close()
            if self.decoder is not None:
                self.decoder.close()
        self.log("mic_recorded", **self.state())

    @property
    def ended(self) -> bool:
        """Closed, and nothing of it runs any more."""
        return self._closed and (self.decoder is None or self.decoder.ended)

    def join(self) -> None:
        """Wait for the decoder to end and its PCM file to be closed. Not for the event loop."""
        if self.decoder is not None:
            self.decoder.join()

    def state(self) -> dict[str, Any]:
        """What ``/_mic`` and ``/_status`` say about this recording."""
        return {
            "file": self.path,
            "seconds": round(self._writer.granule / SAMPLE_RATE, 2),
            "packets": self.packets,
            "bytes": self.payload_bytes,
            "concealed": self.concealed,
            "late": self.late,
            "ignored": self.ignored,
            "pcm_file": self.pcm_path,
            "decoder": None if self.decoder is None else self.decoder.state(),
        }


def capture_path(run_dir: str, label: str) -> str:
    """A name no recording has yet in the run directory: ``mic_<label>_<HHMMSS>.ogg``, with a
    number behind it when that second already has one."""
    stamp = time.strftime("%H%M%S")
    path = os.path.join(run_dir, f"mic_{label}_{stamp}.ogg")
    n = 1
    while os.path.exists(path) or os.path.exists(os.path.splitext(path)[0] + ".pcm"):
        n += 1
        path = os.path.join(run_dir, f"mic_{label}_{stamp}_{n}.ogg")
    return path


__all__ = [
    "MIC_PAYLOAD_TYPE",
    "MicrophoneCapture",
    "OggOpusWriter",
    "PacketOrder",
    "PcmDecoder",
    "capture_path",
    "ogg_crc",
    "opus_samples",
    "pcm_level",
]
