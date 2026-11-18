"""The controller's microphone as a microphone of this computer.

A program picks its microphone among the computer's input devices, and nothing but a device is
one to it. Where the sound server lets a program make a device, the receiver makes it: on Linux
a source of PulseAudio or PipeWire named "Stadia Controller microphone". Windows and macOS have
no input device without a driver, so there the sound is played into the output side of a
loopback device that is installed (VB-CABLE, BlackHole), and programs record its input side.

:class:`MicrophoneDevice` is the receiver's one device, with a :class:`Voice` for each
controller that is connected: the controller's Opus packets go through an ffmpeg that stays for
as long as the controller does, and come back as samples in a small buffer. The :class:`Output`
asks for 20 ms of sound at a time, at the pace of its own clock, and gets the voices added up,
or silence. A program that records therefore gets silence between two holds of a button, as
from any microphone, and the device never runs dry.

The microphone is not meant to be heard on this computer. Without a name, an output is a device
the receiver made or one whose name says it is a loopback; a device that is named is the
choice of whoever named it, refused only where it is plainly a mistake (the computer's
speakers, the device the headset's sound is captured from). A sound server that takes the
receiver's device away would play its stream on the computer's own output instead: the stream
is ended the moment the server says so.
"""

from __future__ import annotations

import abc
import array
import contextlib
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from stadiacast.audio import ffmpeg_path
from stadiacast.capture import LOOPBACK_HINTS, MIC_FEED_SINK, sound_server_user
from stadiacast.errors import SetupError
from stadiacast.microphone import OggOpusWriter, PacketOrder, PcmDecoder

#: What programs see, where the receiver makes the device itself.
DEVICE_NAME = "Stadia Controller microphone"
#: The device's rate where the receiver chooses it, and the length of what an output asks for.
RATE = 48000
TICK = 0.02
#: A voice plays again once it holds this much, after it ran dry: what evens out the network.
#: A controller's packets, sent every 20 ms, come up to 80 ms apart over Wi-Fi now and then.
PRIME_SECONDS = 0.08
#: A voice that holds more than this lets go of its oldest sound, down to the amount above: a
#: controller whose clock runs ahead of the computer's must not push the device ever further
#: behind what is said.
MAX_SECONDS = 0.16
#: A voice that holds less than it waits for, and got nothing more for this long, says it: it is
#: the end of what was said, and must not wait for the beginning of the next thing said.
FLUSH_SECONDS = 0.06
#: The first PipeWire whose PulseAudio server links a remapped source to a sink's monitor.
PIPEWIRE_MIN = (0, 3, 64)
#: How long a sound server gets to answer ``pactl``.
PACTL_TIMEOUT = 5.0
#: An output that has not asked for this long has lost its device.
STALL_SECONDS = 1.0
#: How often the device looks at its output, and opens one that lost its device anew. A voice
#: whose ffmpeg ended gets a new one as often at most.
REOPEN_SECONDS = 5.0
#: Fragments that mark the playback side of a virtual cable on Windows.
CABLE_HINTS = ("vb-audio", "virtual cable", "virtual audio cable", "steam streaming microphone", "voicemeeter")
#: The loopback device a Mac is given for the microphone, beside the headset's BlackHole 2ch.
SECOND_LOOPBACK = "blackhole 16ch"

Read = Callable[[int], bytes]
T = TypeVar("T")


def _no_log(*_args: Any, **_kwargs: Any) -> None:
    """For use without a receiver."""


def mix(parts: list[bytes]) -> bytes:
    """Voices of the same length added up, sample by sample, and kept within 16 bits."""
    total = array.array("h", parts[0])
    for part in parts[1:]:
        for i, value in enumerate(array.array("h", part)):
            both = total[i] + value
            total[i] = min(32767, max(-32768, both))
    return total.tobytes()


def spread(mono: bytes, channels: int) -> bytes:
    """One channel as the first two of ``channels``, the others silent."""
    if channels == 1:
        return mono
    samples = array.array("h", mono)
    out = array.array("h", bytes(2 * channels * len(samples)))
    out[0::channels] = samples
    out[1::channels] = samples
    return out.tobytes()


class Voice:
    """One controller's microphone as samples at the device's rate.

    The packets are put in order as for a recording (:class:`~stadiacast.microphone.PacketOrder`),
    go to ffmpeg as one Ogg stream for as long as the controller stays, and what ffmpeg decodes,
    each packet as it comes, waits here for the output. :meth:`take` is the output's side: it
    gets nothing while the voice has nothing to say, and after a silence the voice holds
    :data:`PRIME_SECONDS` before it speaks again, so that packets which arrive unevenly leave no
    holes. It begins with that much and no more: sound that came in one piece behind a hole (the
    network held it back, or it stands for packets that were lost) has the silence the hole
    already was taken off its oldest end, and the device stays as close behind what is said as
    it was. The stream goes on over the times the microphone is off, as the controller's
    numbering of its packets does: a decoder that kept its place needs no time to find it again.
    """

    def __init__(self, label: str, rate: int = RATE, log: Callable[..., Any] | None = None) -> None:
        self.label = label
        self.rate = rate
        self.packets = 0
        self.gaps = 0  # times it ran dry while it played: the end of what was said, or a hole in it
        self.dropped_ms = 0  # sound let go of: it had run ahead of the output, or came behind a hole
        self._lock = threading.Lock()  # the buffer, between ffmpeg's reader and the output
        self._buffer = bytearray()
        self._playing = False
        self._heard_at = 0.0  # when ffmpeg last gave sound
        self._order = PacketOrder()
        said = log or _no_log
        self._decoder = PcmDecoder(
            None,
            lambda event, **fields: said(event, controller=label, **fields),
            rate=rate,
            heard=self._heard,
            events="mic_voice",
        )
        self._stream = OggOpusWriter(self._decoder)

    def feed(self, payload: bytes, timestamp: int, seq: int) -> None:
        """One RTP payload of the controller's microphone. Never waits."""
        placed = self._order.place(payload, timestamp, seq)
        for packet, samples in placed:
            self._stream.write_packet(packet, samples)
        self.packets += bool(placed)

    def skip(self, payload: bytes, timestamp: int, seq: int) -> None:
        """A packet that came while the microphone was off, still on its way when the switch
        went: nobody is to hear it, and the packet behind it is not to take it for lost."""
        self._order.place(payload, timestamp, seq)

    def _heard(self, pcm: bytes) -> None:
        """What ffmpeg decoded, on its reader's thread."""
        with self._lock:
            self._heard_at = time.monotonic()
            self._buffer += pcm
            if len(self._buffer) > int(self.rate * MAX_SECONDS) * 2:
                gone = len(self._buffer) - int(self.rate * PRIME_SECONDS) * 2
                gone -= gone % 2
                del self._buffer[:gone]
                self.dropped_ms += gone * 500 // self.rate

    def take(self, nbytes: int) -> bytes | None:
        """The next ``nbytes`` of this voice, or ``None`` while it has nothing to say."""
        with self._lock:
            if not self._playing:
                held = max(nbytes, int(self.rate * PRIME_SECONDS) * 2)
                if len(self._buffer) < held:
                    if len(self._buffer) < 2 or time.monotonic() - self._heard_at < FLUSH_SECONDS:
                        return None  # nothing to say, or the beginning of it
                else:
                    late = len(self._buffer) - held
                    late -= late % 2
                    del self._buffer[:late]
                    self.dropped_ms += late * 500 // self.rate
                self._playing = True
            if len(self._buffer) >= nbytes:
                out = bytes(self._buffer[:nbytes])
                del self._buffer[:nbytes]
                return out
            have = len(self._buffer) - len(self._buffer) % 2  # it ran dry: what is left, then silence
            out = bytes(self._buffer[:have]) + bytes(nbytes - have)
            del self._buffer[:have]
            self._playing = False
            self.gaps += 1
            return out

    def close(self) -> None:
        """The controller is gone: ffmpeg ends by itself behind the last packet. Never waits."""
        with contextlib.suppress(OSError):
            self._stream.close()
        self._decoder.close(empty=self.packets == 0)

    def join(self) -> None:
        """Wait until ffmpeg has ended. Not for the event loop."""
        self._decoder.join()

    @property
    def ended(self) -> bool:
        return self._decoder.ended

    @property
    def failed(self) -> bool:
        """Its ffmpeg ended by itself, or never started: nothing it is fed is heard."""
        return self._decoder.reason is not None

    def state(self) -> dict[str, Any]:
        with self._lock:
            buffered = len(self._buffer)
        return {
            "packets": self.packets,
            "concealed": self._order.concealed,
            "late": self._order.late,
            "ignored": self._order.ignored,
            "gaps": self.gaps,
            "dropped_ms": self.dropped_ms,
            "buffered_ms": buffered * 500 // self.rate,
            "decoder": self._decoder.state(),
        }


class Output(abc.ABC):
    """Where the device's sound goes: something programs can record from.

    :meth:`start` finds or makes it and begins to ask ``read`` for mono 16-bit samples at
    :attr:`rate`, on a thread of its own and at the pace of its own clock; a computer where it
    cannot be had is a :class:`~stadiacast.errors.SetupError` that says what to do. An output
    that was stopped can be started again. Neither :meth:`start` nor :meth:`stop` is for the
    event loop: both talk to the sound system.
    """

    #: The device as the output knows it, and the name a program picks to record it (nothing
    #: where the output cannot tell).
    name: str = ""
    microphone: str = ""
    rate: int = RATE
    #: Why it stopped by itself, while that is the case.
    error: str | None = None
    #: What the person at the computer should know about the device, if anything.
    note: str | None = None
    #: While it is open, this process sees the sound devices as they were when it opened.
    pins_devices: bool = False

    @abc.abstractmethod
    def start(self, read: Read) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    def check(self) -> bool:
        """Whether the device is still there, for an output that would go on asking without
        it. Asked every :data:`REOPEN_SECONDS`, never on the event loop."""
        return True

    def abort(self) -> None:  # noqa: B027 (most outputs have nothing to give up)
        """A :meth:`start` that is under way on another thread is to give up soon."""

    @property
    def loop(self) -> str:
        """The capture device that hears what is played here, as ``--audio-source`` names it:
        the headset's sound must not be captured from it, or the headset would hear itself."""
        return self.name

    def heard_as(self) -> str:
        """What a program is pointed at to hear it, as part of a sentence."""
        return f'the microphone "{self.microphone}"' if self.microphone else f'the recording side of "{self.name}"'

    def describe(self) -> dict[str, Any]:
        return {
            "kind": type(self).__name__,
            "name": self.name,
            "microphone": self.microphone,
            "rate": self.rate,
            "error": self.error,
            "note": self.note,
        }


def _tool(name: str) -> str:
    exe = shutil.which(name)
    if exe is None:
        raise SetupError(
            f"{name} not found: on Linux the microphone is made a device of PulseAudio or PipeWire with pactl and"
            " pacat (the pulseaudio-utils package)"
        )
    return exe


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:  # somebody else's process: it is there
        return True
    return True


def _receiver_alive(pid: int) -> bool:
    """Whether the process ``pid`` is a receiver that runs: the number a receiver that is gone left
    on its modules can belong to another program by now."""
    if not _alive(pid):
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:  # nothing more to know of it: it runs
        return True
    return b"stadiacast" in command.lower()


def _also_the_output(name: str) -> str:
    return (
        f'"{name}" is also the computer\'s sound output: what the computer plays is in the microphone as well,'
        " until another output is chosen in the sound settings"
    )


class PulseOutput(Output):
    """Linux: a source of the sound server, made by the receiver and removed with it.

    A null sink takes the sound, which ``pacat`` plays into it, and a source remapped over the
    sink's monitor is what programs record: a monitor itself is hidden by browsers, by Discord
    and by games. Both are modules of the server and outlive whoever loaded them. Each carries
    the number of the receiver's process: what a receiver that is gone left behind is unloaded
    first, and a receiver that still runs keeps the server's one device to itself.

    The sink is listed among the computer's outputs, under a name that says what it is: a
    computer whose sound is sent there has it in the microphone. A computer without a sound card
    sends it there by itself, since the silent output its sound server keeps for it goes when
    another appears: the output then says so (:attr:`note`).

    A sink that is unloaded under ``pacat`` does not end its stream: both servers move the
    stream to the computer's own output, where the microphone would be heard. PipeWire is told
    not to (``node.dont-reconnect``). PulseAudio cannot be told, so ``pacat`` runs with ``-v``,
    says where its stream is, and is ended the moment that is anywhere but the sink.

    ``pacat`` reads its pipe when the server asks it for sound, and what waits in the pipe is
    that much later in the microphone. PulseAudio's null sink asks for nothing in its first two
    seconds, and a pipe holds two thirds of a second, which would stay in it for good. So the
    pump looks at what is unread before it writes: with :attr:`AHEAD` blocks waiting it writes
    no more, and blocks that waited all through a second are one too many each, and are left
    out once.
    """

    FEED = MIC_FEED_SINK
    SOURCE = "stadiacast_mic"
    #: Blocks of 20 ms that may wait unread in ``pacat``'s pipe: more than a sound server takes at once.
    AHEAD = 8
    #: Ticks after which the pump looks at what waited all along: a second.
    WINDOW = 50

    def __init__(self, log: Callable[..., Any] | None = None) -> None:
        self.log: Callable[..., Any] = log or _no_log
        self.name = self.microphone = DEVICE_NAME
        self.server = ""
        self.flowing = False  # pacat took all it was given, through the last second
        self.let_go_ms = 0  # sound that was not written, because pacat was not taking it
        self._proc: subprocess.Popen[bytes] | None = None
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._aborted = threading.Event()  # the receiver stops: nothing is opened again
        self._stalled = False  # the sound server did not answer: asked nothing more in this start, stop or check
        self._said = ""  # the last thing pacat said

    def _pactl(self, *args: str, cleaning: bool = False) -> str:
        """``pactl`` with these arguments, as the user whose sound server it is. Once the server
        has not answered, the start, stop or check under way asks it nothing more; once the
        receiver stops, nothing but its clean-up (``cleaning``) is asked."""
        if self._stalled:
            raise SetupError("the sound server did not answer")
        if self._aborted.is_set() and not cleaning:
            raise SetupError("the receiver is stopping")
        user = sound_server_user()
        # in English, whatever the session speaks: the server's name is read from the answer
        env = {**user.get("env", os.environ), "LC_ALL": "C"}
        try:
            done: subprocess.CompletedProcess[str] = subprocess.run(
                [_tool("pactl"), *args],
                capture_output=True,
                text=True,
                timeout=PACTL_TIMEOUT,
                check=False,
                **{**user, "env": env},
            )
        except subprocess.TimeoutExpired as e:
            self._stalled = True
            raise SetupError(f"pactl {args[0]} got no answer from the sound server in {PACTL_TIMEOUT:g} s") from e
        except (OSError, subprocess.SubprocessError) as e:
            raise SetupError(f"pactl {args[0]} did not run ({e})") from e
        if done.returncode != 0:
            said = (done.stderr or done.stdout).strip().splitlines()
            raise SetupError(
                f"pactl {args[0]} failed ({said[-1] if said else 'no message'}): the sound server belongs to the"
                " user who is logged in, so run the receiver as that user, or with sudo from that user's session"
            )
        return done.stdout.strip()

    def _modules(self, *, cleaning: bool = False) -> list[tuple[int, int | None]]:
        """The modules in the sound server that are a receiver's microphone, found by their
        names: each one's number, and the process it says it belongs to."""
        found = []
        for line in self._pactl("list", "short", "modules", cleaning=cleaning).splitlines():
            if f"source_name={self.SOURCE}" in line or f"sink_name={self.FEED}" in line:
                owner = re.search(r'stadiacast\.pid="?(\d+)', line)
                found.append((int(line.split()[0]), int(owner.group(1)) if owner else None))
        return found

    def _unload(self, indices: list[int] | None = None) -> None:
        """Unload these modules, or this receiver's own. Its own are looked up each time: a
        number kept from before would be another module's once the sound server has restarted.
        The source goes first: it was loaded behind its sink."""
        if indices is None:
            try:
                indices = [index for index, owner in self._modules(cleaning=True) if owner == os.getpid()]
            except SetupError:
                return
        for index in sorted(indices, reverse=True):
            with contextlib.suppress(SetupError):
                self._pactl("unload-module", str(index), cleaning=True)

    def _open(self) -> subprocess.Popen[bytes]:
        named = [line.partition(":")[2].strip() for line in self._pactl("info").splitlines() if "Server Name" in line]
        self.server = named[0] if named else ""
        pipewire = re.search(r"PipeWire (\d+)\.(\d+)\.(\d+)", self.server)
        if pipewire and tuple(int(n) for n in pipewire.groups()) < PIPEWIRE_MIN:
            raise SetupError(
                f"this PipeWire ({'.'.join(pipewire.groups())}) cannot make the microphone a device: it takes"
                f" {'.'.join(str(n) for n in PIPEWIRE_MIN)} or later"
            )
        me = os.getpid()
        modules = self._modules()
        theirs = sorted({owner for _index, owner in modules if owner not in (None, me) and _receiver_alive(owner)})
        if theirs:
            raise SetupError(
                f"another receiver (process {theirs[0]}) has its microphone device in this sound server, which"
                " holds one: stop that receiver, or start this one without --mic-device"
            )
        if modules:  # of a receiver that is gone
            self._unload([index for index, _owner in modules])
            self.log("mic_device_left_behind", modules=len(modules))
        sinks = [
            line.split()[1] for line in self._pactl("list", "short", "sinks").splitlines() if len(line.split()) > 1
        ]
        self.note = _also_the_output(f"{DEVICE_NAME} feed") if all(sink == "auto_null" for sink in sinks) else None
        # The whole property list in quotes, and each value in the other kind: both servers read
        # a name with spaces that way, and neither reads one whose spaces are escaped.
        self._pactl(
            "load-module", "module-null-sink", f"sink_name={self.FEED}", "format=s16le", f"rate={self.rate}",
            "channels=1", "channel_map=mono",
            f"sink_properties='device.description=\"{DEVICE_NAME} feed\" stadiacast.pid=\"{me}\"'",
        )  # fmt: skip
        self._pactl(
            "load-module", "module-remap-source", f"master={self.FEED}.monitor", f"source_name={self.SOURCE}",
            "channels=1", "channel_map=mono", "master_channel_map=mono",
            f"source_properties='device.description=\"{DEVICE_NAME}\" device.icon_name=\"audio-input-microphone\""
            f" stadiacast.pid=\"{me}\"'",
        )  # fmt: skip
        user = sound_server_user()
        try:
            return subprocess.Popen(
                [
                    _tool("pacat"), "--playback", "--raw", "--format=s16le", f"--rate={self.rate}", "--channels=1",
                    "--channel-map=mono", f"--device={self.FEED}", "--latency-msec=20", "--client-name=Stadiacast",
                    f"--stream-name={DEVICE_NAME}", "--property=node.dont-reconnect=true", "-v",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
                **{**user, "env": {**user.get("env", os.environ), "LC_ALL": "C"}},  # what it says is read
            )  # fmt: skip
        except OSError as e:
            raise SetupError(f"pacat did not start ({e})") from e

    def start(self, read: Read) -> None:
        if self._aborted.is_set():
            raise SetupError("the receiver is stopping")
        self._stalled = False
        self.error = None
        self._said = ""
        try:
            proc = self._open()
        except SetupError:
            self._unload()
            raise
        self._proc = proc
        self._stop = stop = threading.Event()
        says = threading.Thread(target=self._listen, args=(proc,), name="mic-device-says", daemon=True)
        pump = threading.Thread(target=self._pump, args=(read, proc, stop, says), name="mic-device-out", daemon=True)
        self._threads = [pump, says]
        says.start()
        pump.start()

    def _listen(self, proc: subprocess.Popen[bytes]) -> None:
        """What ``pacat`` says: where its stream is, and in the end why it ended. A stream that
        is anywhere but in the sink is ended at once."""
        assert proc.stderr is not None
        held = b""
        with contextlib.suppress(OSError, ValueError):
            while chunk := proc.stderr.read(1024):
                *lines, held = (held + chunk).replace(b"\r", b"\n").split(b"\n")
                for raw in lines:
                    line = raw.replace(b"\x1b[K", b"").decode("utf-8", "replace").strip()
                    if not line or line.startswith("Time:"):
                        continue
                    self._said = line
                    at = re.match(r"(?:Connected to|Stream moved to) device (\S+) \(", line)
                    if at and at.group(1) != self.FEED:
                        self.error = f"the sound server put the microphone's stream on {at.group(1)!r}: it was ended"
                        proc.kill()

    def _pump(self, read: Read, proc: subprocess.Popen[bytes], stop: threading.Event, says: threading.Thread) -> None:
        """Twenty milliseconds of sound into ``pacat`` every twenty milliseconds, until the
        output is stopped or ``pacat`` ends: the sound server went away, or the sink was taken
        out of it."""
        import fcntl
        import termios

        assert proc.stdin is not None
        frames = self.rate // 50
        size = 2 * frames
        unread = array.array("i", [0])
        least: int | None = None  # the least that waited unread at a tick of this window
        ticks = skip = 0
        steady = True
        due = time.monotonic()
        try:
            while not stop.is_set():
                block = read(frames)  # asked for even when it is not written: the voices keep their pace
                fcntl.ioctl(proc.stdin.fileno(), termios.FIONREAD, unread)
                least = unread[0] if least is None else min(least, unread[0])
                if skip or unread[0] >= self.AHEAD * size:
                    skip = max(0, skip - 1)
                    steady = False
                    self.let_go_ms += int(1000 * TICK)
                else:
                    proc.stdin.write(block)
                ticks += 1
                if ticks == self.WINDOW:
                    skip = least // size
                    self.flowing = steady and not skip
                    least, ticks, steady = None, 0, True
                due += TICK
                wait = due - time.monotonic()
                if wait > 0:
                    stop.wait(wait)
                elif wait < -0.2:  # the computer stood still: no catching up with a burst
                    due = time.monotonic()
        except (OSError, ValueError):  # pacat is gone
            pass
        self.flowing = False
        if stop.is_set():
            return
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            proc.wait(timeout=2)
        says.join(1)  # what pacat said last is read by then
        if self.error is None:
            self.error = f"pacat ended: {self._said or 'no message'}"

    def stop(self) -> None:
        self._stalled = False
        self._stop.set()
        proc, self._proc = self._proc, None
        threads, self._threads = self._threads, []
        mine = [thread for thread in threads if thread is not threading.current_thread()]
        for thread in mine[:1]:
            thread.join(1)  # the pump: between two writes it sees the flag at once
        if proc is not None:
            with contextlib.suppress(OSError, ValueError):
                if proc.stdin is not None:
                    proc.stdin.close()  # pacat plays what it has and ends
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:  # a sound server that stands still holds it, and the pump with it
                proc.kill()
                proc.wait()
            for thread in mine:
                thread.join(1)
            with contextlib.suppress(OSError, ValueError):
                if proc.stderr is not None:
                    proc.stderr.close()
        self._unload()

    def abort(self) -> None:
        """The receiver stops: a start under way gives up at its next question to the sound
        server, and none is begun again. :meth:`stop` still takes the device away."""
        self._aborted.set()

    def check(self) -> bool:
        """Both modules are still in the sound server. A source that was unloaded leaves
        ``pacat`` playing into a sink nobody records. Also looks whether the sink has become the
        computer's own output since the start, and says so."""
        self._stalled = False
        try:
            there = sum(owner == os.getpid() for _index, owner in self._modules()) == 2
            info = self._pactl("info")
        except SetupError:
            return True  # a server that is gone has ended pacat: the pump has said so
        if not there:
            self.error = self.error or "the microphone's device was taken out of the sound server"
            return False
        note = _also_the_output(f"{DEVICE_NAME} feed") if f"Default Sink: {self.FEED}\n" in info + "\n" else None
        if note != self.note:
            self.note = note
            self.log("mic_device_note", note=note)
        return True

    @property
    def loop(self) -> str:
        return f"{self.FEED}.monitor"

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "server": self.server, "flowing": self.flowing, "let_go_ms": self.let_go_ms}


def _loopback_named(name: str) -> bool:
    return any(hint in name.lower() for hint in LOOPBACK_HINTS)


def _named(devices: list[dict[str, Any]], wanted: str) -> dict[str, Any] | None:
    """A device by its name, or by a fragment of it when no name is the whole of it."""
    key = wanted.lower()
    return next((d for d in devices if key == str(d["name"]).lower()), None) or next(
        (d for d in devices if key in str(d["name"]).lower()), None
    )


class PortAudioOutput(Output):
    """macOS: the output side of a loopback device that is installed, found by its name.

    Programs record the same device's input side. Without a name it is the first loopback
    device besides the one the headset's sound is captured from (``avoid``): played into that
    one, the microphone would go back into the headset. One that is not the computer's sound
    output comes first, since what the computer plays would be in the microphone with it. A
    device without an input side is no loopback, and what is played into it would be heard, not
    recorded: it is refused.
    """

    def __init__(self, name: str | None = None, *, avoid: str | None = None) -> None:
        self._wanted = name
        self._avoid = avoid
        self._stream: Any = None
        self._read: Read | None = None
        self._channels = 2

    def _choose(self, devices: list[dict[str, Any]], default: str = "") -> dict[str, Any]:
        outputs = [d for d in devices if int(d["max_output_channels"]) > 0]
        loops = [d for d in outputs if int(d["max_input_channels"]) > 0 and _loopback_named(str(d["name"]))]
        avoid = (self._avoid or "").lower()
        install = (
            "install a second one beside it (brew install blackhole-16ch, then restart the Mac)"
            if avoid
            else "install one (brew install blackhole-16ch, then restart the Mac)"
        ) + ", or name another loopback device with --mic-device NAME"
        if self._wanted:
            found = _named(outputs, self._wanted)
            if found is None:
                raise SetupError(
                    f"no output device matches {self._wanted!r}; loopback devices here:"
                    f" {[str(d['name']) for d in loops] or 'none'}"
                )
        else:
            free = [d for d in loops if str(d["name"]).lower() != avoid]
            # BlackHole 16ch first, the one installed for the microphone, whatever order the Mac
            # lists its devices in; then one that is not where the Mac plays its own sound
            found = min(
                free,
                key=lambda d: (str(d["name"]).lower() != SECOND_LOOPBACK, str(d["name"]) == default),
                default=None,
            )
            if found is None:
                besides = f" besides {self._avoid!r}, which the headset's sound is captured from" if avoid else ""
                raise SetupError(f"no loopback device to play the microphone into{besides}: {install}")
        chosen = str(found["name"])
        if int(found["max_input_channels"]) < 1:
            raise SetupError(
                f"{chosen!r} has no input side: the microphone played into it would be heard, and no program"
                f" could record it. It takes a loopback device: {install}"
            )
        if chosen.lower() == avoid:
            raise SetupError(
                f"{chosen!r} is the device the headset's sound is captured from: the microphone played into it"
                f" would go back into the headset. It takes another loopback device: {install}"
            )
        return found

    def start(self, read: Read) -> None:
        from stadiacast.capture import _sounddevice

        self.error = None
        sd = _sounddevice("the microphone's device")
        try:
            devices = [dict(d) for d in sd.query_devices()]
        except Exception as e:  # sounddevice.PortAudioError, which cannot be named before the import
            raise SetupError(f"PortAudio cannot list the sound devices ({e})") from e
        try:
            default = str(sd.query_devices(kind="output")["name"])
        except Exception:  # a computer without an output
            default = ""
        found = self._choose(devices, default)
        self.name = self.microphone = str(found["name"])
        if self.name == default:
            self.note = _also_the_output(self.name)
        elif not _loopback_named(self.name):  # somebody named it: a loopback device under a name of its own, or not
            self.note = (
                f'"{self.name}" does not say that it is a loopback device: on a real device the microphone is heard,'
                " and a program that records it gets that device's own microphone"
            )
        else:
            self.note = None
        self._channels = min(2, int(found["max_output_channels"]))
        self._read = read
        try:
            self._stream = sd.RawOutputStream(
                samplerate=self.rate,
                blocksize=self.rate // 50,
                device=int(found["index"]),
                channels=self._channels,
                dtype="int16",
                latency="low",  # 20 ms at a time, as the capture of the headset's sound asks for
                callback=self._callback,
            )
            self._stream.start()
        except Exception as e:
            self._stream = None
            raise SetupError(f"{self.name!r} could not be opened for the microphone ({e})") from e

    def _callback(self, outdata: Any, frames: int, _time: Any, _status: Any) -> None:
        assert self._read is not None
        outdata[:] = spread(self._read(frames), self._channels)

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as e:  # a failing close has nothing left to do
                self.error = repr(e)


class WasapiOutput(Output):
    """Windows: the playback side of a virtual cable that is installed.

    Without a name it is VB-CABLE's ``CABLE Input``, and programs record ``CABLE Output``: the
    recording side of a cable carries the name of the same driver, which is how it is found.
    The stream is opened in the format the endpoint has, which shared mode asks for: the voices
    are decoded at its rate and go to its first two channels. The computer's default output is
    refused unless its name says it is a cable: the microphone would be heard on it. So is the
    device whose loopback the headset's sound is captured from (``avoid``, as that capture names
    it): the microphone would go back into the headset. A device that was named and does not
    look like a cable is taken, with a word about it (:attr:`note`) and no guess at where it
    could be recorded.

    PortAudio lists the sound devices when its first session opens and keeps that list until
    the last one closes. This output's session stays open, so :attr:`pins_devices` says that
    whoever needs the list as it is now has to have the output closed for a moment.
    """

    pins_devices = True

    def __init__(self, name: str | None = None, *, avoid: str | None = None) -> None:
        self._wanted = name
        self._avoid = avoid
        self._p: Any = None
        self._stream: Any = None
        self.channels = 2

    def _choose(self, outputs: list[dict[str, Any]], default: str) -> dict[str, Any]:
        install = (
            "install VB-CABLE (https://vb-audio.com/Cable/, run its setup as administrator, then restart), or"
            " name another cable's playback side with --mic-device NAME"
        )
        if self._wanted:
            found = _named(outputs, self._wanted)
            if found is None:
                raise SetupError(
                    f"no playback device matches {self._wanted!r}; there are: {[str(d['name']) for d in outputs]}"
                )
        else:
            cables = [d for d in outputs if "vb-audio" in str(d["name"]).lower() and "cable" in str(d["name"]).lower()]
            found = (
                next((d for d in cables if str(d["name"]).lower().startswith("cable input")), None)
                or next((d for d in cables if "16ch" not in str(d["name"]).lower()), None)
                or next(iter(cables), None)
            )
            if found is None:
                raise SetupError(f"no virtual cable to play the microphone into: {install}")
        chosen = str(found["name"])
        if chosen == default and not self._cable(chosen):
            raise SetupError(
                f"{chosen!r} is the computer's default output: the microphone would be heard on it. It takes a"
                f" virtual cable: {install}"
            )
        if self._avoid == f"{chosen} [Loopback]":
            raise SetupError(
                f"{chosen!r} is the device the headset's sound is captured from: the microphone played into it"
                " would go back into the headset. Choose the speakers as the computer's output in the sound"
                " settings, or name another cable's playback side with --mic-device NAME"
            )
        return found

    @staticmethod
    def _cable(name: str) -> bool:
        return any(hint in name.lower() for hint in CABLE_HINTS)

    @staticmethod
    def _other_side(playback: str, inputs: list[dict[str, Any]]) -> str:
        """The recording device of the cable whose playback side is ``playback``: Windows names
        both sides after their driver, in brackets at the end. Nothing when that finds no one
        device."""
        driver = playback[playback.rfind("(") :] if playback.endswith(")") and "(" in playback else ""
        sides = [str(d["name"]) for d in inputs if driver and str(d["name"]).endswith(driver)]
        return sides[0] if len(sides) == 1 else ""

    def start(self, read: Read) -> None:
        from stadiacast.loopback import _pyaudio

        self.error = None
        pa = _pyaudio("the microphone's device")
        try:
            self._p = pa.PyAudio()
            wasapi = self._p.get_host_api_info_by_type(pa.paWASAPI)
            every = [dict(self._p.get_device_info_by_index(i)) for i in range(self._p.get_device_count())]
            ours = [d for d in every if d.get("hostApi") == wasapi["index"] and not d.get("isLoopbackDevice")]
            outputs = [d for d in ours if int(d.get("maxOutputChannels", 0)) > 0]
            inputs = [d for d in ours if int(d.get("maxInputChannels", 0)) > 0]
            default = next((str(d["name"]) for d in outputs if d["index"] == wasapi.get("defaultOutputDevice")), "")
            found = self._choose(outputs, default)
            name = str(found["name"])
            rate, channels = int(found["defaultSampleRate"]), int(found["maxOutputChannels"])

            def give(_in: Any, frames: int, _time: Any, _status: Any) -> tuple[bytes, Any]:
                return spread(read(frames), channels), pa.paContinue

            self._stream = self._p.open(
                format=pa.paInt16,
                channels=channels,
                rate=rate,
                output=True,
                output_device_index=int(found["index"]),
                frames_per_buffer=rate // 50,
                stream_callback=give,
            )
        except SetupError:
            self.stop()
            raise
        except Exception as e:
            self.stop()
            raise SetupError(f"the microphone's playback device could not be opened ({e})") from e
        self.name, self.rate, self.channels = name, rate, channels
        if self._cable(name):
            self.microphone = self._other_side(name, inputs)
            self.note = _also_the_output(name) if name == default else None
        else:  # somebody named it: it may be a cable under a name of its own, or a pair of speakers
            self.microphone = ""
            self.note = f'"{name}" does not say that it is a virtual cable: on a real output the microphone is heard'

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.stop_stream()
                stream.close()
        p, self._p = self._p, None
        if p is not None:
            with contextlib.suppress(Exception):
                p.terminate()

    @property
    def loop(self) -> str:
        return f"{self.name} [Loopback]"

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "channels": self.channels}


def open_output(name: str | None, *, avoid: str | None = None, log: Callable[..., Any] | None = None) -> Output:
    """The output for this system: ``name`` is a device to look for, or nothing for the system's
    own way (a device the receiver makes on Linux, VB-CABLE on Windows, a loopback device on
    macOS). ``avoid`` is the device the headset's sound is captured from, as that capture names
    it: never the one the microphone is played into."""
    if sys.platform.startswith("linux"):
        if name:
            raise SetupError("on Linux the receiver makes the microphone's device itself: --mic-device takes no name")
        return PulseOutput(log)
    if sys.platform == "win32":
        return WasapiOutput(name, avoid=avoid)
    if sys.platform == "darwin":
        return PortAudioOutput(name, avoid=avoid)
    raise SetupError("--mic-device is for Linux, Windows and macOS")


class MicrophoneDevice:
    """The receiver's microphone device: the controllers' voices, added up for one output.

    A thread looks at the output every :data:`REOPEN_SECONDS`. One that stopped asking, or
    whose device is gone (the sound server restarted, the cable was taken out), is opened anew
    until it asks again.
    """

    def __init__(self, output: Output, log: Callable[..., Any] | None = None) -> None:
        self.output = output
        self.log: Callable[..., Any] = log or _no_log
        self._lock = threading.Lock()  # the voices, between the receiver and the output
        self._voices: dict[str, Voice] = {}
        self._gone: list[Voice] = []  # closed, their ffmpeg seeing the rest through
        self._renewed: dict[str, float] = {}  # when a voice whose ffmpeg had ended last got a new one
        self._started = False
        self._asked = 0.0  # when the output last asked
        self._switch = threading.Lock()  # one thread at a time starts or stops the output
        self._stopping = threading.Event()
        self._watcher: threading.Thread | None = None

    def start(self) -> None:
        """Find or make the device and begin to feed it. Not for the event loop."""
        if ffmpeg_path() is None:
            raise SetupError("--mic-device decodes the microphone with ffmpeg, and none was found on PATH")
        with self._switch:
            self._asked = time.monotonic()
            self.output.start(self.read)
        self._started = True
        self._stopping.clear()
        self._watcher = threading.Thread(target=self._watch, name="mic-device-watch", daemon=True)
        self._watcher.start()
        self.log("mic_device", **self.output.describe())

    def _silent(self) -> bool:
        return time.monotonic() - self._asked >= STALL_SECONDS

    def _lost(self) -> bool:
        # A second look a moment later: a computer that slept wakes with an old time.
        if self._silent() and not self._stopping.wait(STALL_SECONDS) and self._silent():
            return True
        if self._stopping.is_set():
            return False
        with self._switch:
            return not self.output.check()

    def _watch(self) -> None:
        said: str | None = None  # while the device is lost: the reason last written to the log
        while not self._stopping.wait(REOPEN_SECONDS):
            try:
                lost = self._lost()
            except Exception:  # a look that failed shows nothing: the next one is in a moment
                lost = False
            if not lost or self._stopping.is_set():
                continue
            if said is None:
                said = self.output.error or "it takes no more sound"
                self.log("mic_device_lost", name=self.output.name, error=said)
            with self._switch:
                if self._stopping.is_set():
                    return
                with contextlib.suppress(Exception):  # what is left of it: it is opened anew all the same
                    self.output.stop()
                try:
                    self.output.start(self.read)
                except Exception as e:  # any failure is a try that failed: there is another in a moment
                    if self._stopping.is_set():
                        return
                    reason = str(e) if isinstance(e, SetupError) else f"{type(e).__name__}: {e}"
                    self.output.error = reason
                    if reason != said:  # why it does not come back, said once
                        said = reason
                        self.log("mic_device_lost", name=self.output.name, error=said)
                    continue
                self._asked = time.monotonic()
            said = None
            self._renew()
            self.log("mic_device_back", **self.output.describe())

    def _renew(self) -> None:
        """The voices are decoded at the device's rate: those of another get a new ffmpeg. A
        controller that went meanwhile gets none."""
        with self._lock:
            stale = [label for label, voice in self._voices.items() if voice.rate != self.output.rate]
        for label in stale:
            made = Voice(label, self.output.rate, self.log)  # starts ffmpeg: not under the lock
            unused = made
            with self._lock:
                old = self._voices.get(label)
                if self._started and old is not None and old.rate != made.rate:
                    self._voices[label] = made
                    self._gone = [v for v in self._gone if not v.ended] + [old]
                    unused = old
            unused.close()  # the voice it replaces, or the new one when its controller went meanwhile

    def fresh(self, resolve: Callable[[], T]) -> T:
        """What ``resolve`` returns, which lists the sound devices: with the output closed
        meanwhile where an open output keeps that list as it was (:attr:`Output.pins_devices`).
        Not for the event loop."""
        if not self.output.pins_devices:
            return resolve()
        with self._switch:
            if not self._started or self._stopping.is_set():
                return resolve()
            self.output.stop()
            try:
                return resolve()
            finally:
                try:
                    self.output.start(self.read)
                except Exception as e:  # the thread that looks at the output opens it again
                    self.output.error = str(e)
                else:
                    self._asked = time.monotonic()
                    self._renew()  # it may have come back at another rate

    def voice(self, label: str) -> Voice | None:
        """The voice of the controller ``label``, made when there is none: its ffmpeg then runs
        before the first packet needs it. Nothing once the device is stopped."""
        with self._lock:
            if not self._started:
                return None
            voice = self._voices.get(label)
        if voice is not None:
            return voice
        made = Voice(label, self.output.rate, self.log)
        with self._lock:
            if self._started and label not in self._voices:
                self._voices[label] = made
                return made
            voice = self._voices.get(label)
        made.close()  # the device was stopped meanwhile
        return voice

    def feed(self, label: str, payload: bytes, timestamp: int, seq: int) -> None:
        """One RTP payload of a controller's microphone. Never waits."""
        voice = self.voice(label)
        if voice is not None and voice.failed:  # its ffmpeg ended by itself: another one, now and then
            now = time.monotonic()
            if now - self._renewed.get(label, float("-inf")) < REOPEN_SECONDS:
                return
            self._renewed[label] = now
            self.drop(label)
            voice = self.voice(label)
        if voice is not None:
            voice.feed(payload, timestamp, seq)

    def skip(self, label: str, payload: bytes, timestamp: int, seq: int) -> None:
        """A packet of a controller whose microphone is off: see :meth:`Voice.skip`."""
        with self._lock:
            voice = self._voices.get(label)
        if voice is not None:
            voice.skip(payload, timestamp, seq)

    def drop(self, label: str) -> None:
        """The controller is gone, or boots anew: its voice ends. Never waits."""
        with self._lock:
            voice = self._voices.pop(label, None)
            if voice is not None:
                self._gone = [v for v in self._gone if not v.ended] + [voice]
        if voice is not None:
            voice.close()

    def read(self, frames: int) -> bytes:
        """``frames`` samples of what the voices say, or of silence: the output's question."""
        self._asked = time.monotonic()
        nbytes = 2 * frames
        with self._lock:
            voices = list(self._voices.values())
        parts = [part for voice in voices if (part := voice.take(nbytes)) is not None]
        if not parts:
            return bytes(nbytes)
        return parts[0] if len(parts) == 1 else mix(parts)

    def stop(self) -> None:
        """Remove the device and end the voices. Not for the event loop."""
        with self._lock:
            started, self._started = self._started, False
            voices, self._voices = list(self._voices.values()), {}
            gone, self._gone = self._gone + voices, []
        self._stopping.set()
        self.output.abort()  # a start that the watching thread is in gives up
        watcher, self._watcher = self._watcher, None
        if watcher is not None:
            watcher.join()
        if started:
            with self._switch:
                self.output.stop()
        for voice in voices:
            voice.close()
        for voice in gone:
            voice.join()
        if started:
            self.log("mic_device_removed", name=self.output.name)

    def state(self) -> dict[str, Any]:
        with self._lock:
            voices = dict(self._voices)
        return {**self.output.describe(), "voices": {label: voice.state() for label, voice in voices.items()}}


__all__ = [
    "DEVICE_NAME",
    "MicrophoneDevice",
    "Output",
    "PortAudioOutput",
    "PulseOutput",
    "Voice",
    "WasapiOutput",
    "mix",
    "open_output",
    "spread",
]
