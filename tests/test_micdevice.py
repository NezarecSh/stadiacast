"""The controller's microphone as a device of the computer: the voices and what an output gets
from them, the three outputs' choice of a device, the sound server's tools as they are called,
and the receiver with a device in place of a recording. The device itself is a stand-in here.
The real ones are in the lab tests of Linux and Windows (``tests/test_linux_lab.py``,
``tests/test_windows_lab.py``) and, for a Mac, in the last test of this file, which runs when
STADIACAST_MAC_LAB is set and plays two seconds of a tone into the Mac's loopback device. The
tests that decode sound need ffmpeg and skip without it."""

from __future__ import annotations

import array
import asyncio
import json
import math
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from stadiacast import controller, micdevice, microphone, receiver
from stadiacast.audio import Source, ffmpeg_path
from stadiacast.capture import pick_loopback
from stadiacast.errors import SetupError
from stadiacast.gamepad import Button
from stadiacast.micdevice import (
    DEVICE_NAME,
    MicrophoneDevice,
    Output,
    PortAudioOutput,
    PulseOutput,
    Voice,
    WasapiOutput,
    mix,
    open_output,
    spread,
)
from tests.fake_controller import FakeController
from tests.helpers import connect_firmware_flow, make_receiver, single, wait_for
from tests.test_microphone import real_opus

needs_ffmpeg = pytest.mark.skipif(ffmpeg_path() is None, reason="needs ffmpeg (with libopus) on PATH")
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="the sound server's tools are POSIX programs")

ASSISTANT = int(Button.ASSISTANT)
FRAME = 960


def samples(*values: int) -> bytes:
    return array.array("h", values).tobytes()


def strength(pcm: bytes, hz: float, rate: int = 48000) -> float:
    """How much of a pitch a stretch of sound holds (Goertzel), as the amplitude of that sine."""
    data = array.array("h", pcm[: len(pcm) // 2 * 2])
    coefficient = 2 * math.cos(2 * math.pi * hz / rate)
    before = last = 0.0
    for value in data:
        before, last = last, value + coefficient * last - before
    return math.sqrt(max(0.0, before * before + last * last - coefficient * before * last)) * 2 / max(1, len(data))


def loud(pieces: list[bytes], hz: float, above: float = 1000.0, rate: int = 48000) -> int:
    """How many pieces of sound hold a pitch. The pieces are what an output asked for, 20 ms each:
    a count of them is not upset by a hole in the sound, which the strength of a whole second is
    (the tone goes on behind the hole with another phase)."""
    return sum(strength(piece, hz, rate) > above for piece in pieces)


class Asked(Output):
    """An output that asks for 20 ms every 20 ms on a thread of its own, as a sound device does,
    and keeps what it was given."""

    def __init__(self, rate: int = 48000) -> None:
        self.name = self.microphone = "the lab's device"
        self.rate = rate
        self.given: list[bytes] = []
        self.started = self.stopped = 0
        self.refuses: list[str | Exception] = []  # what the next starts fail with, one each
        self.next_rate: int | None = None  # the device's rate from the next start on
        self.there = True  # what check() answers
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, read: micdevice.Read) -> None:
        self.error = None
        self.rate = self.next_rate or self.rate  # known before the device is open, as an endpoint's format is
        if self.refuses:
            refusal = self.refuses.pop(0)
            raise refusal if isinstance(refusal, Exception) else SetupError(refusal)
        self.started += 1
        self.there = True
        self._stop = stop = threading.Event()

        def ask() -> None:
            due = time.monotonic()
            while not stop.is_set():
                self.given.append(read(self.rate // 50))
                due += 0.02
                stop.wait(max(0.0, due - time.monotonic()))

        self._thread = threading.Thread(target=ask, name="lab-device", daemon=True)
        self._thread.start()

    def lose(self, error: str) -> None:
        """The device goes away under the output: it asks no more, and says why."""
        self._stop.set()
        self.error = error

    def check(self) -> bool:
        return self.there

    def stop(self) -> None:
        self.stopped += 1
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2)

    def sound(self) -> bytes:
        return b"".join(self.given)


def voice_threads() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith(("mic-voice", "mic-device", "lab-device"))]


def wait_until(reached: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not reached():
        assert time.monotonic() < deadline, "not reached in time"
        time.sleep(0.01)


# ---- adding up ----------------------------------------------------------------------------------
def test_voices_are_added_up_and_kept_within_sixteen_bits() -> None:
    assert mix([samples(100, -200, 30000), samples(5, 5, 5)]) == samples(105, -195, 30005)
    assert mix([samples(30000, -30000), samples(10000, -10000)]) == samples(32767, -32768)
    assert mix([samples(1, 2), samples(10, 20), samples(100, 200)]) == samples(111, 222)
    assert spread(samples(1, -2, 3), 1) == samples(1, -2, 3)
    assert spread(samples(1, -2, 3), 2) == samples(1, 1, -2, -2, 3, 3) and spread(b"", 2) == b""
    assert spread(samples(1, -2), 4) == samples(1, 1, 0, 0, -2, -2, 0, 0)  # the first two of a wider device


# ---- a voice's buffer ---------------------------------------------------------------------------
def test_a_voice_waits_for_eighty_milliseconds_and_starts_again_after_it_ran_dry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What ffmpeg decodes waits for the output. A voice says nothing until it holds 80 ms, so that
    packets which come unevenly leave no holes; one that ran dry (what was said is over, or a
    packet is late) gives what it has, then silence, and waits for 80 ms again."""
    monkeypatch.setattr(microphone, "ffmpeg_path", lambda: None)  # the buffer alone: no decoder is started
    voice = Voice("c0")
    frame = samples(*[7] * FRAME)
    assert voice.take(len(frame)) is None
    for _ in range(3):
        voice._heard(frame)
        assert voice.take(len(frame)) is None  # 20, 40, 60 ms: not yet
    voice._heard(frame)
    assert [voice.take(len(frame)) for _ in range(4)] == [frame] * 4  # 80 ms: it plays
    voice._heard(frame[:1000] + b"\x07")  # a part of a frame, and half a sample
    assert voice.take(len(frame)) == frame[:1000] + bytes(len(frame) - 1000)  # what is left, then silence
    assert (voice.gaps, voice.state()["buffered_ms"]) == (1, 0)
    assert voice.take(len(frame)) is None  # dry: nothing, not even the half sample
    voice._heard(b"\x00" + frame * 3)  # the other half, and 60 ms
    assert voice.take(len(frame)) is None
    voice._heard(frame)
    assert voice.take(len(frame)) == frame and voice.state()["buffered_ms"] == 60
    voice.close()


def test_the_end_of_what_was_said_does_not_wait_for_the_next_thing_said(monkeypatch: pytest.MonkeyPatch) -> None:
    """A packet that comes just as the voice ran dry is less than the voice waits for. When
    nothing follows it, it was the end: it is said a moment later, so that the next thing said
    begins with itself and not with the end of the last."""
    monkeypatch.setattr(microphone, "ffmpeg_path", lambda: None)
    voice = Voice("c0")
    frame = samples(*[7] * FRAME)
    voice._heard(frame)
    assert voice.take(len(frame)) is None  # it may be the beginning of something
    voice._heard_at -= micdevice.FLUSH_SECONDS  # and then nothing more came
    assert voice.take(len(frame)) == frame
    assert voice.take(len(frame)) == bytes(len(frame)) and voice.gaps == 1  # and with that it has run dry
    assert voice.take(len(frame)) is None
    voice._heard(frame[:1001])  # with half a sample whose other half never comes
    voice._heard_at -= 1.0
    assert voice.take(len(frame)) == frame[:1000] + bytes(len(frame) - 1000)
    assert voice.take(len(frame)) is None and voice.take(len(frame)) is None and voice.gaps == 2
    voice.close()


def test_a_voice_that_runs_ahead_lets_go_of_its_oldest_sound(monkeypatch: pytest.MonkeyPatch) -> None:
    """A controller's clock is not the computer's. Sound that piles up past 160 ms is dropped
    down to 80 ms, the oldest first: the device says what is being said, not what was."""
    monkeypatch.setattr(microphone, "ffmpeg_path", lambda: None)
    voice = Voice("c0")
    for n in range(8):
        voice._heard(samples(*[n] * FRAME))
    assert (voice.dropped_ms, voice.state()["buffered_ms"]) == (0, 160)
    voice._heard(samples(*[8] * FRAME))  # 180 ms: over
    assert (voice.dropped_ms, voice.state()["buffered_ms"]) == (100, 80)
    assert [voice.take(2 * FRAME) for _ in range(4)] == [samples(*[n] * FRAME) for n in (5, 6, 7, 8)]
    voice.close()


# ---- through ffmpeg -----------------------------------------------------------------------------
@needs_ffmpeg
def test_a_device_gets_what_the_controllers_say_and_silence_between() -> None:
    """One controller's tone comes out of the device as that tone, a second one's is added to
    it, and an output that asks while nobody speaks is given silence, without end."""
    said: list[str] = []
    out = Asked()
    device = MicrophoneDevice(out, log=lambda event, **_fields: said.append(event))
    device.start()
    try:
        device.voice("c0")
        low, high = real_opus(1.0, 440), real_opus(1.0, 1500)
        time.sleep(0.3)
        quiet = len(out.given)
        assert quiet >= 10 and not any(out.sound())  # asked all the while, and given silence
        began = time.monotonic()
        for i, packet in enumerate(low):
            device.feed("c0", packet, 7000 + i * 960, 100 + i)
            if i >= 25:  # the second controller joins half way
                device.feed("c1", high[i - 25], i * 960, i)
            time.sleep(max(0.0, began + (i + 1) * 0.02 - time.monotonic()))
        time.sleep(0.3)
        state = device.state()
    finally:
        device.stop()
    said_then = out.given[quiet:]
    first, second = said_then[:24], said_then[30:50]  # the first half second, and the second one
    assert loud(first, 440) >= 18 and loud(first, 1500) == 0  # one voice
    assert loud(second, 440) >= 16 and loud(second, 1500) >= 16  # both
    assert not any(said_then[-1])  # and silence again behind them
    assert state["name"] == "the lab's device" and sorted(state["voices"]) == ["c0", "c1"]
    voice = state["voices"]["c0"]
    assert (voice["packets"], voice["concealed"], voice["late"]) == (len(low), 0, 0)
    assert voice["decoder"]["running"] is True and voice["decoder"]["reason"] is None
    assert (out.started, out.stopped) == (1, 1) and voice_threads() == []
    assert said == ["mic_device", "mic_voice_started", "mic_voice_started", "mic_voice_ended", "mic_voice_ended",
                    "mic_device_removed"]  # fmt: skip


@needs_ffmpeg
def test_a_voice_goes_on_over_a_pause_and_over_lost_packets() -> None:
    """Two holds of a button are one stream to the decoder, as they are to the controller's
    encoder: the second is heard from its first packet. Packets lost on the way are concealed
    once the next one shows that they are missing; by then the device has gone silent for
    them, and that silence is taken off the sound that stands for them, so that the voice is
    no further behind after the hole than it was before."""
    out = Asked()
    device = MicrophoneDevice(out)
    device.start()
    try:
        device.voice("c0")
        tone = real_opus(2.0)
        time.sleep(0.2)
        heard: list[float] = []
        n = 0
        for _hold in range(2):
            began = time.monotonic()
            asked = len(out.given)
            for i in range(40):
                if not 20 <= i < 26:  # six packets lost in the middle of each
                    device.feed("c0", tone[n], n * 960, n)
                n += 1
                time.sleep(max(0.0, began + (i + 1) * 0.02 - time.monotonic()))
            time.sleep(0.3)
            sounding = [k for k, piece in enumerate(out.given[asked:]) if strength(piece, 440) > 1000]
            heard.append((sounding[-1] - sounding[0] + 1) * 0.02)
            assert sounding[0] <= 7, sounding[0]  # within a seventh of a second of its first packet
        state = device.state()["voices"]["c0"]
    finally:
        device.stop()
    assert all(0.7 <= seconds <= 0.86 for seconds in heard), heard  # 0.8 s each, from its first sound to its last
    assert (state["packets"], state["concealed"]) == (68, 12)
    # each hold ran dry twice, in its hole and at its end; about 60 ms a hole of what came behind it were let go of
    assert state["gaps"] >= 4 and 60 <= state["dropped_ms"] <= 200, state


@needs_ffmpeg
def test_a_voice_that_is_dropped_ends_its_ffmpeg_and_a_new_one_begins_anew() -> None:
    out = Asked()
    device = MicrophoneDevice(out)
    device.start()
    try:
        first = device.voice("c0")
        assert device.voice("c0") is first
        tone = real_opus(0.4)
        for i, packet in enumerate(tone):
            device.feed("c0", packet, i * 960, 60000 + i)
        device.drop("c0")
        device.drop("c0")  # once is enough
        first.join()
        assert first.ended and device.state()["voices"] == {}
        second = device.voice("c0")  # the controller booted: its numbers begin anywhere
        assert second is not first
        for i, packet in enumerate(tone):
            device.feed("c0", packet, i * 960, 5 + i)
        assert second.state()["packets"] == len(tone) and second.state()["late"] == 0
    finally:
        device.stop()
    assert voice_threads() == []


def test_a_device_needs_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(micdevice, "ffmpeg_path", lambda: None)
    out = Asked()
    with pytest.raises(SetupError, match="--mic-device decodes the microphone with ffmpeg"):
        MicrophoneDevice(out).start()
    assert out.started == 0


# ---- a device that goes away ----------------------------------------------------------------------
@needs_ffmpeg
def test_an_output_that_stops_asking_is_opened_anew(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sound server that restarts, a cable that is taken out: the output asks no more. The
    device then opens it again and again until it asks, and says in the log that it was lost,
    why it does not come back, and that it is back."""
    monkeypatch.setattr(micdevice, "REOPEN_SECONDS", 0.1)
    monkeypatch.setattr(micdevice, "STALL_SECONDS", 0.15)
    said: list[tuple[str, dict[str, Any]]] = []
    out = Asked()
    device = MicrophoneDevice(out, log=lambda event, **fields: said.append((event, fields)))
    device.start()
    try:
        first = device.voice("c0")
        time.sleep(0.5)
        assert (out.started, out.stopped) == (1, 0) and [e for e, _ in said] == ["mic_device", "mic_voice_started"]
        assert said[1][1]["controller"] == "c0"  # whose voice it is

        out.refuses = ["the sound server does not answer", "the sound server does not answer"]
        out.lose("pacat ended: Connection terminated")
        asked = len(out.given)
        wait_until(lambda: out.started == 2)
        assert out.stopped == 3  # stopped before each of the three tries
        wait_until(lambda: len(out.given) > asked)
        wait_until(lambda: said[-1][0] == "mic_device_back")
        assert [e for e, _ in said[2:]] == ["mic_device_lost", "mic_device_lost", "mic_device_back"]
        assert said[2][1] == {"name": "the lab's device", "error": "pacat ended: Connection terminated"}
        assert said[3][1] == {"name": "the lab's device", "error": "the sound server does not answer"}  # said once
        assert said[4][1]["error"] is None and device.state()["error"] is None
        assert device.voice("c0") is first  # the same device as before: the voice goes on

        # It comes back at another rate, which shows before a first try fails: the voices are
        # decoded anew for the rate it has when it is open.
        out.next_rate = 44100
        out.refuses = ["the endpoint is being set up"]
        out.lose("the stream ended")
        wait_until(lambda: out.started == 3)
        wait_until(lambda: [e for e, _ in said].count("mic_device_back") == 2)
        again = device.voice("c0")
        assert again is not first and again is not None and again.rate == 44100
        assert len(out.given[-1]) == 2 * 882
    finally:
        device.stop()
    assert voice_threads() == []


@needs_ffmpeg
def test_an_output_whose_device_is_gone_is_opened_anew_though_it_asks(monkeypatch: pytest.MonkeyPatch) -> None:
    """An output can go on asking for sound that reaches nobody: a source that was taken out of
    the sound server leaves the player playing. The device asks the output now and then
    whether what it plays into is still there."""
    monkeypatch.setattr(micdevice, "REOPEN_SECONDS", 0.1)
    said: list[tuple[str, dict[str, Any]]] = []
    out = Asked()
    device = MicrophoneDevice(out, log=lambda event, **fields: said.append((event, fields)))
    device.start()
    try:
        time.sleep(0.4)
        assert (out.started, out.stopped) == (1, 0)
        out.there = False
        wait_until(lambda: out.started == 2)
        wait_until(lambda: said[-1][0] == "mic_device_back")
        assert said[1] == ("mic_device_lost", {"name": "the lab's device", "error": "it takes no more sound"})
        time.sleep(0.4)
        assert (out.started, out.stopped) == (2, 1)  # and it stays
    finally:
        device.stop()


@needs_ffmpeg
def test_whoever_lists_the_sound_devices_gets_them_as_they_are() -> None:
    """On Windows an open output keeps PortAudio's list of the devices as it was when it opened.
    The headset's capture is looked up with the output closed for that moment."""
    out = Asked()
    device = MicrophoneDevice(out)
    device.start()
    try:
        assert device.fresh(lambda: (out.started, out.stopped)) == (1, 0)  # nothing to close: the list is not kept
        out.pins_devices = True
        assert device.fresh(lambda: (out.started, out.stopped)) == (1, 1)  # closed while it is looked up
        assert (out.started, out.stopped) == (2, 1)  # and open again
        with pytest.raises(SetupError, match="no loopback device"):
            device.fresh(lambda: (_ for _ in ()).throw(SetupError("no loopback device")))
        assert (out.started, out.stopped) == (3, 2)  # whatever the lookup said
        out.refuses = ["the cable is busy"]
        assert device.fresh(lambda: "looked up") == "looked up"
        assert device.state()["error"] == "the cable is busy"  # left to the thread that opens it again
    finally:
        device.stop()
    assert device.fresh(lambda: "after the stop") == "after the stop" and out.started == 3


@needs_ffmpeg
def test_a_voice_whose_ffmpeg_ended_gets_another() -> None:
    """An ffmpeg that ends while its controller is connected takes that controller's voice with
    it. The next packet makes a new one, and at most one every few seconds."""
    said: list[tuple[str, dict[str, Any]]] = []
    out = Asked()
    device = MicrophoneDevice(out, log=lambda event, **fields: said.append((event, fields)))
    device.start()
    try:
        tone = real_opus(0.4)
        for i, packet in enumerate(tone[:10]):
            device.feed("c0", packet, i * 960, i)
        first = device.voice("c0")
        assert first is not None
        wait_until(lambda: first._decoder._proc is not None)
        first._decoder._proc.kill()
        wait_until(lambda: first.failed)
        device.feed("c0", tone[10], 10 * 960, 10)
        second = device.voice("c0")
        assert second is not first and second is not None and second.packets == 1
        wait_until(lambda: second._decoder._proc is not None)
        second._decoder._proc.kill()
        wait_until(lambda: second.failed)
        device.feed("c0", tone[11], 11 * 960, 11)
        assert device.voice("c0") is second  # not again so soon
        ended = [fields for event, fields in said if event == "mic_voice_ended"]
        assert len(ended) == 2 and all(e["controller"] == "c0" and e["reason"] for e in ended)
    finally:
        device.stop()
    assert voice_threads() == []


@needs_ffmpeg
def test_a_voice_that_never_spoke_ends_without_a_complaint() -> None:
    """A controller that never switched its microphone on leaves a stream with no packet in
    it, which ffmpeg ends on with an error. That is no failure of the voice."""
    said: list[tuple[str, dict[str, Any]]] = []
    out = Asked()
    device = MicrophoneDevice(out, log=lambda event, **fields: said.append((event, fields)))
    device.start()
    try:
        voice = device.voice("c0")
        assert voice is not None
        wait_until(lambda: voice._decoder._proc is not None)
        device.drop("c0")
        voice.join()
    finally:
        device.stop()
    ended = [fields for event, fields in said if event == "mic_voice_ended"]
    assert len(ended) == 1 and ended[0]["reason"] is None and ended[0]["bytes"] == 0, ended


@needs_ffmpeg
def test_a_device_that_is_stopped_makes_no_voice() -> None:
    """A packet that is still on its way when the receiver stops starts no ffmpeg."""
    said: list[str] = []
    out = Asked()
    device = MicrophoneDevice(out, log=lambda event, **_fields: said.append(event))
    assert device.voice("c0") is None  # not started yet
    device.start()
    device.stop()
    device.feed("c0", b"\xf8\xff\xfe", 0, 0)
    device.skip("c0", b"\xf8\xff\xfe", 960, 1)
    assert device.voice("c0") is None and device.state()["voices"] == {}
    assert said == ["mic_device", "mic_device_removed"] and voice_threads() == []


@needs_ffmpeg
def test_a_try_that_fails_in_any_way_is_followed_by_another(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever a try to open the device again fails with (PortAudio raises errors of its own while
    Windows' audio service restarts), the thread that opens it lives on and tries again."""
    monkeypatch.setattr(micdevice, "REOPEN_SECONDS", 0.1)
    monkeypatch.setattr(micdevice, "STALL_SECONDS", 0.15)
    said: list[tuple[str, dict[str, Any]]] = []
    out = Asked()
    device = MicrophoneDevice(out, log=lambda event, **fields: said.append((event, fields)))
    device.start()
    try:
        out.refuses = [OSError(-9999, "Unanticipated host error")]
        out.lose("the stream ended")
        wait_until(lambda: said[-1][0] == "mic_device_back")
        lost = [fields["error"] for event, fields in said if event == "mic_device_lost"]
        assert lost == ["the stream ended", "OSError: [Errno -9999] Unanticipated host error"]
        assert out.started == 2
    finally:
        device.stop()
    assert voice_threads() == []


@needs_ffmpeg
def test_a_device_that_comes_back_at_another_rate_while_it_was_closed_gets_voices_for_it() -> None:
    """The output closed for a moment, while the devices were listed, can come back at another
    rate (its format was changed meanwhile): the voices are decoded anew for it."""
    out = Asked()
    out.pins_devices = True
    device = MicrophoneDevice(out)
    device.start()
    try:
        first = device.voice("c0")
        out.next_rate = 44100
        assert device.fresh(lambda: "listed") == "listed"
        again = device.voice("c0")
        assert again is not first and again is not None and again.rate == 44100
    finally:
        device.stop()
    assert voice_threads() == []


@needs_ffmpeg
def test_a_controller_that_goes_while_its_voice_is_made_anew_gets_none(monkeypatch: pytest.MonkeyPatch) -> None:
    out = Asked()
    device = MicrophoneDevice(out)
    device.start()
    made: list[Voice] = []
    try:
        assert device.voice("c0") is not None
        out.rate = 44100  # the device came back at another rate
        plain = micdevice.Voice

        def voice_while_the_controller_goes(label: str, rate: int, log: Any) -> Voice:
            device.drop(label)  # the controller goes while its new ffmpeg starts
            made.append(plain(label, rate, log))
            return made[-1]

        monkeypatch.setattr(micdevice, "Voice", voice_while_the_controller_goes)
        device._renew()
        assert device.state()["voices"] == {} and len(made) == 1
        made[0].join()
        assert made[0].ended
    finally:
        device.stop()
    assert voice_threads() == []


# ---- which device -------------------------------------------------------------------------------
def mac(name: str, inputs: int, outputs: int, index: int = 0) -> dict[str, Any]:
    return {
        "name": name,
        "max_input_channels": inputs,
        "max_output_channels": outputs,
        "index": index,
        "default_samplerate": 48000.0,
    }


MAC = [
    mac("SAMSUNG", 0, 6, 0),
    mac("USB Microphone", 2, 0, 1),
    mac("BlackHole 2ch", 2, 2, 2),
    mac("BlackHole 16ch", 16, 16, 3),
    mac("Mac mini Speakers", 0, 2, 4),
    mac("Multi-Output Device", 0, 2, 5),
]


def test_on_a_mac_it_is_a_loopback_device_and_speakers_are_refused() -> None:
    """The microphone is played into something programs can record: a device with an input side,
    by its name or, without one, BlackHole 16ch, or else the first loopback device besides the
    one the headset's sound is captured from, and besides the computer's own output while there
    is another. Speakers have no input side and are refused, whoever names them."""
    assert PortAudioOutput()._choose(MAC)["name"] == "BlackHole 16ch"
    assert PortAudioOutput()._choose(list(reversed(MAC)))["name"] == "BlackHole 16ch"  # whatever the order
    assert PortAudioOutput(avoid="BlackHole 2ch")._choose(MAC)["name"] == "BlackHole 16ch"
    assert PortAudioOutput()._choose(MAC, "BlackHole 2ch")["name"] == "BlackHole 16ch"  # the computer plays into 2ch
    assert PortAudioOutput("blackhole 16")._choose(MAC)["name"] == "BlackHole 16ch"
    assert PortAudioOutput("BlackHole 2ch")._choose(MAC)["name"] == "BlackHole 2ch"  # the whole name first
    with pytest.raises(SetupError, match=r"'BlackHole 2ch' is the device the headset's sound is captured from"):
        PortAudioOutput("BlackHole 2ch", avoid="BlackHole 2ch")._choose(MAC)
    one = [d for d in MAC if d["name"] != "BlackHole 16ch"]
    assert PortAudioOutput()._choose(one, "BlackHole 2ch")["name"] == "BlackHole 2ch"  # there is no other
    loopback = [mac("Loopback Audio", 2, 2, 7), *one]
    assert PortAudioOutput()._choose(loopback, "BlackHole 2ch")["name"] == "Loopback Audio"  # not where the Mac plays
    # and the headset's sound, for its part, is BlackHole 2ch whatever order the Mac lists them in
    assert pick_loopback(["BlackHole 16ch", "BlackHole 2ch"]) == "BlackHole 2ch"
    assert pick_loopback(["Loopback Audio", "BlackHole 16ch"]) == "Loopback Audio"
    with pytest.raises(
        SetupError,
        match=r"no loopback device to play the microphone into besides 'BlackHole 2ch', which the headset's sound"
        r" is captured from: install a second one beside it \(brew install blackhole-16ch",
    ):
        PortAudioOutput(avoid="BlackHole 2ch")._choose(one)
    with pytest.raises(
        SetupError, match=r"no loopback device to play the microphone into: install one \(brew install blackhole-16ch"
    ):
        PortAudioOutput()._choose([d for d in MAC if "BlackHole" not in d["name"]])
    for speakers in ("SAMSUNG", "Mac mini Speakers", "Multi-Output Device"):
        with pytest.raises(SetupError, match="has no input side: the microphone played into it would be heard"):
            PortAudioOutput(speakers)._choose(MAC)
    with pytest.raises(
        SetupError, match=r"no output device matches 'USB Microphone'; loopback devices here: \['BlackHole"
    ):
        PortAudioOutput("USB Microphone")._choose(MAC)


class LabSounddevice:
    """Just enough of ``sounddevice`` for an output: the devices, the computer's own output, and
    a stream that asks its callback when the test does, into a buffer of exactly the size
    PortAudio hands over."""

    def __init__(self, devices: list[dict[str, Any]], default: str | None) -> None:
        self.devices, self.default = devices, default
        self.streams: list[LabSounddevice] = []
        self.opened: dict[str, Any] = {}
        self.state = "new"

    def query_devices(self, kind: str | None = None) -> Any:
        if kind is None:
            return self.devices
        assert kind == "output"
        if self.default is None:
            raise OSError("no default output device")
        return next(d for d in self.devices if d["name"] == self.default)

    def RawOutputStream(self, **kwargs: Any) -> LabSounddevice:  # noqa: N802 (sounddevice's name)
        if "fails" in str(self.devices[kwargs["device"]]["name"]):
            raise OSError("Internal PortAudio error [PaErrorCode -9986]")
        self.opened = kwargs
        return self

    def start(self) -> None:
        self.state = "started"

    def ask(self, frames: int) -> bytes:
        outdata = memoryview(bytearray(2 * frames * self.opened["channels"]))  # a buffer takes no other length
        self.opened["callback"](outdata, frames, None, None)
        return bytes(outdata)

    def stop(self) -> None:
        self.state = "stopped"

    def close(self) -> None:
        self.state = "closed"


def test_a_macs_loopback_device_is_opened_for_twenty_milliseconds_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    sd = LabSounddevice(MAC, "Mac mini Speakers")
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    out = PortAudioOutput(avoid="BlackHole 2ch")
    out.start(lambda frames: samples(*range(frames)))
    assert (out.name, out.microphone, out.rate, out.note) == ("BlackHole 16ch", "BlackHole 16ch", 48000, None)
    assert out.loop == "BlackHole 16ch"  # what a capture of it is called
    assert out.heard_as() == 'the microphone "BlackHole 16ch"'
    callback = sd.opened.pop("callback")
    assert sd.opened == {
        "samplerate": 48000, "blocksize": 960, "device": 3, "channels": 2, "dtype": "int16", "latency": "low",
    }  # fmt: skip
    sd.opened["callback"] = callback
    assert sd.state == "started" and sd.ask(3) == samples(0, 0, 1, 1, 2, 2)  # the one channel on both
    out.stop()
    assert sd.state == "closed"

    # the only loopback device is where the computer's own sound goes: taken, and said
    sd = LabSounddevice([d for d in MAC if d["name"] != "BlackHole 16ch"], "BlackHole 2ch")
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    out = PortAudioOutput()
    out.start(lambda frames: bytes(2 * frames))
    assert out.name == "BlackHole 2ch" and out.note is not None
    assert out.note.startswith('"BlackHole 2ch" is also the computer\'s sound output: what the computer plays is in')
    out.stop()

    # A device that was named and does not say it is a loopback device (a USB headset has an input
    # side too): taken, with a word about it.
    monkeypatch.setitem(sys.modules, "sounddevice", LabSounddevice([*MAC, mac("USB Headset", 1, 2, 6)], "SAMSUNG"))
    out = PortAudioOutput("usb headset")
    out.start(silence)
    assert out.name == "USB Headset" and out.note == (
        '"USB Headset" does not say that it is a loopback device: on a real device the microphone is heard, and a'
        " program that records it gets that device's own microphone"
    )
    out.stop()

    sd = LabSounddevice([mac("Loopback that fails", 2, 2, 0)], None)  # and a computer without an output
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    with pytest.raises(SetupError, match=r"'Loopback that fails' could not be opened for the microphone \(Internal"):
        PortAudioOutput().start(lambda frames: bytes(2 * frames))

    monkeypatch.setitem(sys.modules, "sounddevice", None)  # the audio extra is not installed
    with pytest.raises(SetupError, match=r"the microphone's device needs the audio extra: pip install"):
        PortAudioOutput().start(lambda frames: bytes(2 * frames))


def win(name: str, index: int, rate: int = 48000, outputs: int = 2, inputs: int = 0, api: int = 2) -> dict[str, Any]:
    return {
        "name": name,
        "index": index,
        "hostApi": api,
        "defaultSampleRate": float(rate),
        "maxOutputChannels": outputs,
        "maxInputChannels": inputs,
    }


WINDOWS = [
    win("Speakers (Steam Streaming Speakers)", 2),
    win("Speakers (Realtek(R) Audio)", 3),
    win("CABLE In 16ch (VB-Audio Virtual Cable)", 4, outputs=16),
    win("CABLE Input (VB-Audio Virtual Cable)", 5, 44100, outputs=8),
    win("Speakers (Steam Streaming Microphone)", 6),
]
RECORDING = [
    win("Microphone (Realtek(R) Audio)", 7, outputs=0, inputs=2),
    win("CABLE Output (VB-Audio Virtual Cable)", 8, 44100, outputs=0, inputs=8),
    win("Microphone (Steam Streaming Microphone)", 9, outputs=0, inputs=2),
]


def test_on_windows_it_is_the_cable_and_speakers_that_are_the_default_are_refused() -> None:
    """Without a name it is VB-CABLE. A device that is named is taken, unless it is the output
    the computer plays on and no cable by its name, or the one whose loopback goes to the
    headset."""
    default = "Speakers (Realtek(R) Audio)"
    assert WasapiOutput()._choose(WINDOWS, default)["name"] == "CABLE Input (VB-Audio Virtual Cable)"
    only_wide = [d for d in WINDOWS if "CABLE Input" not in d["name"]]
    assert WasapiOutput()._choose(only_wide, default)["name"] == "CABLE In 16ch (VB-Audio Virtual Cable)"
    others = [
        win("Hi-Fi Cable Input (VB-Audio Hi-Fi Cable)", 1),
        win("VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)", 2),
    ]
    assert WasapiOutput()._choose(others, default)["name"] == "Hi-Fi Cable Input (VB-Audio Hi-Fi Cable)"
    steam = WasapiOutput("steam streaming microphone")._choose(WINDOWS, default)["name"]
    assert steam == "Speakers (Steam Streaming Microphone)"
    # Steam's other device is where a game's sound goes to another computer: no cable
    assert WasapiOutput._cable(steam) and not WasapiOutput._cable("Speakers (Steam Streaming Speakers)")
    assert not WasapiOutput._cable(default) and WasapiOutput._cable("Line 1 (Virtual Audio Cable)")
    # a sound card's own mixer is no cable, whatever it calls itself
    assert not WasapiOutput._cable("Speakers (Sonic Studio Virtual Mixer)")
    mixer = "Speakers (Sonic Studio Virtual Mixer)"
    with pytest.raises(SetupError, match="is the computer's default output"):
        WasapiOutput("sonic")._choose([*WINDOWS, win(mixer, 11)], mixer)
    with pytest.raises(SetupError, match=r"no virtual cable to play the microphone into: install VB-CABLE"):
        WasapiOutput()._choose([d for d in WINDOWS if "VB-Audio" not in d["name"]], default)
    with pytest.raises(SetupError, match="is the computer's default output: the microphone would be heard on it"):
        WasapiOutput("realtek")._choose(WINDOWS, default)
    with pytest.raises(SetupError, match=r"no playback device matches 'headphones'; there are: \['Speakers"):
        WasapiOutput("headphones")._choose(WINDOWS, default)
    # a cable that Windows made the default output after its installation is still the cable
    cable = "CABLE Input (VB-Audio Virtual Cable)"
    assert WasapiOutput()._choose(WINDOWS, cable)["name"] == cable
    # unless the headset's sound is that output's loopback: the microphone would be in the headset
    with pytest.raises(SetupError, match=r"is the device the headset's sound is captured from.*Choose the speakers"):
        WasapiOutput(avoid=f"{cable} [Loopback]")._choose(WINDOWS, cable)
    assert WasapiOutput(avoid=f"{default} [Loopback]")._choose(WINDOWS, default)["name"] == cable

    side = WasapiOutput._other_side
    assert side("CABLE Input (VB-Audio Virtual Cable)", RECORDING) == "CABLE Output (VB-Audio Virtual Cable)"
    assert side("CABLE In 16ch (VB-Audio Virtual Cable)", RECORDING) == "CABLE Output (VB-Audio Virtual Cable)"
    assert side("Speakers (Steam Streaming Microphone)", RECORDING) == "Microphone (Steam Streaming Microphone)"
    assert side("Line 1 (Virtual Audio Cable)", RECORDING) == ""  # no recording device of that driver
    two = [*RECORDING, win("Line In (Realtek(R) Audio)", 10, outputs=0, inputs=2)]
    assert side("Speakers (Realtek(R) Audio)", two) == "" and side("Speakers", RECORDING) == ""


class LabPyAudio:
    """Just enough of ``pyaudiowpatch`` for an output: one object plays the module, the PortAudio
    session and the stream it opens. The devices are listed once by each host API, as Windows
    has them, the older one with its names cut short."""

    paInt16 = 8  # noqa: N815 (pyaudiowpatch's name)
    paContinue = 0  # noqa: N815 (pyaudiowpatch's name)
    paWASAPI = 13  # noqa: N815 (pyaudiowpatch's name)

    def __init__(self, default: str) -> None:
        loopbacks = [{**win(f"{d['name']} [Loopback]", 20 + i, inputs=2, outputs=0), "isLoopbackDevice": True}
                     for i, d in enumerate(WINDOWS)]  # fmt: skip
        older = [win(str(d["name"])[:31], 30 + i, api=0) for i, d in enumerate(WINDOWS)]
        self.infos = {int(d["index"]): d for d in [*older, *WINDOWS, *RECORDING, *loopbacks]}
        self.default = next(int(d["index"]) for d in WINDOWS if d["name"] == default)
        self.opened: dict[str, Any] = {}
        self.state = "new"

    def PyAudio(self) -> LabPyAudio:  # noqa: N802 (pyaudiowpatch's name)
        return self

    def get_host_api_info_by_type(self, kind: int) -> dict[str, Any]:
        assert kind == self.paWASAPI
        return {"index": 2, "defaultOutputDevice": self.default}

    def get_device_count(self) -> int:
        return len(self.infos)

    def get_device_info_by_index(self, i: int) -> dict[str, Any]:
        return list(self.infos.values())[i]

    def open(self, **kwargs: Any) -> LabPyAudio:
        if kwargs["rate"] != int(self.infos[kwargs["output_device_index"]]["defaultSampleRate"]):
            raise OSError("[Errno -9997] Invalid sample rate")  # shared mode takes the endpoint's own
        self.opened = kwargs
        self.state = "open"
        return self

    def stop_stream(self) -> None:
        self.state = "stopped"

    def close(self) -> None:
        self.state = "closed"

    def terminate(self) -> None:
        self.state = "terminated"


def test_a_cable_is_opened_in_its_own_format_and_its_recording_side_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    pa = LabPyAudio("Speakers (Realtek(R) Audio)")
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", pa)
    out = WasapiOutput()
    out.start(lambda frames: samples(*range(frames)))
    assert (out.name, out.rate, out.channels) == ("CABLE Input (VB-Audio Virtual Cable)", 44100, 8)
    assert (out.microphone, out.note) == ("CABLE Output (VB-Audio Virtual Cable)", None)
    assert out.loop == "CABLE Input (VB-Audio Virtual Cable) [Loopback]"  # what a capture of it is called
    assert out.heard_as() == 'the microphone "CABLE Output (VB-Audio Virtual Cable)"'
    give = pa.opened.pop("stream_callback")
    assert pa.opened == {
        "format": 8, "channels": 8, "rate": 44100, "output": True, "output_device_index": 5, "frames_per_buffer": 882,
    }  # fmt: skip
    assert give(None, 2, None, None) == (samples(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0), 0)
    assert out.describe() == {
        "kind": "WasapiOutput",
        "name": "CABLE Input (VB-Audio Virtual Cable)",
        "microphone": "CABLE Output (VB-Audio Virtual Cable)",
        "rate": 44100,
        "error": None,
        "note": None,
        "channels": 8,
    }
    out.stop()
    assert pa.state == "terminated"

    # the cable has become the computer's own output, as its installation can leave it: said
    pa = LabPyAudio("CABLE Input (VB-Audio Virtual Cable)")
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", pa)
    out = WasapiOutput()
    out.start(lambda frames: bytes(2 * frames))
    assert out.note is not None and out.note.startswith('"CABLE Input (VB-Audio Virtual Cable)" is also the computer')
    out.stop()

    pa = LabPyAudio("Speakers (Realtek(R) Audio)")
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", pa)
    with pytest.raises(SetupError, match="'Speakers \\(Realtek\\(R\\) Audio\\)' is the computer's default output"):
        WasapiOutput("speakers (realtek").start(lambda frames: bytes(2 * frames))
    assert pa.state == "terminated"  # nothing is left open behind a refusal
    pa.infos[9]["name"] = "Microphone (Steam Streaming Microphone) 2"  # a recording side no rule finds
    out = WasapiOutput("steam streaming microphone")
    out.start(lambda frames: bytes(2 * frames))
    assert out.microphone == "" and out.heard_as() == 'the recording side of "Speakers (Steam Streaming Microphone)"'
    assert out.note is None
    out.stop()

    # A device that was named and is no cable by its name, and not the computer's output: taken,
    # with a word about it, and without a guess at the microphone of the same sound card.
    pa = LabPyAudio("CABLE Input (VB-Audio Virtual Cable)")
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", pa)
    out = WasapiOutput("realtek")
    out.start(lambda frames: bytes(2 * frames))
    assert (out.name, out.microphone) == ("Speakers (Realtek(R) Audio)", "")
    assert out.note == (
        '"Speakers (Realtek(R) Audio)" does not say that it is a virtual cable: on a real output the microphone is'
        " heard"
    )
    out.stop()

    monkeypatch.setitem(sys.modules, "pyaudiowpatch", None)  # the audio extra is not installed
    with pytest.raises(SetupError, match=r"the microphone's device needs the audio extra: pip install"):
        WasapiOutput().start(lambda frames: bytes(2 * frames))


@posix_only
def test_a_capture_of_the_microphones_device_goes_by_the_name_the_output_expects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The headset's sound is refused a capture of the device that carries the microphone, by the
    name the capture has. Each output says that name as the capture of its system makes it."""
    from stadiacast import capture, loopback

    pa = LabPyAudio("Speakers (Realtek(R) Audio)")
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", pa)
    cable = WasapiOutput()
    cable.start(silence)
    cable.stop()
    assert loopback.describe("cable input").name == cable.loop

    monkeypatch.setitem(sys.modules, "sounddevice", LabSounddevice(MAC, "Mac mini Speakers"))
    hole = PortAudioOutput(avoid="BlackHole 2ch")
    hole.start(silence)
    hole.stop()
    assert capture.open_portaudio("BlackHole 16ch").name == hole.loop

    tools(tmp_path)
    (tmp_path / "parec").write_text("#!/bin/sh\n")
    (tmp_path / "parec").chmod(0o755)
    (tmp_path / "default").write_text("stadiacast_mic_feed")  # a computer whose one output is the feed
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    assert capture.linux_monitor().name == PulseOutput().loop


def test_each_system_has_its_own_way(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert isinstance(open_output(None), PulseOutput)
    with pytest.raises(SetupError, match="on Linux the receiver makes the microphone's device itself"):
        open_output("BlackHole 16ch")
    monkeypatch.setattr(sys, "platform", "win32")
    assert isinstance(open_output("CABLE"), WasapiOutput)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert isinstance(open_output(None, avoid="BlackHole 2ch"), PortAudioOutput)
    monkeypatch.setattr(sys, "platform", "freebsd14")
    with pytest.raises(SetupError, match="--mic-device is for Linux, Windows and macOS"):
        open_output(None)


# ---- the sound server's tools, as they are called -----------------------------------------------
def tools(
    directory: Path,
    *,
    server: str = "PulseAudio (on PipeWire 1.0.5)",
    left: str = "",
    sinks: str = "57\\talsa_output.usb-headphones.analog-stereo\\tPipeWire\\ts16le 2ch 48000Hz\\tSUSPENDED\\n",
) -> Path:
    """Stand-ins for ``pactl`` and ``pacat``, first on PATH. ``pactl`` notes how it was called and
    answers as a sound server would: it keeps the modules it was told to load in the file
    ``modules`` (``left`` are those it holds at the start) and numbers them from 41 on, ``sinks``
    are its outputs and the file ``default`` names the one the computer plays on. ``pacat`` notes
    its arguments and counts what it is given."""
    calls = directory / "calls"
    (directory / "modules").write_text(left)
    (directory / "next").write_text("41")
    (directory / "default").write_text("alsa_output.usb-headphones.analog-stereo")
    pactl = directory / "pactl"
    pactl.write_text(
        "#!/bin/sh\n"
        f'cd "{directory}"\n'
        'printf "%s\\n" "$*" >> calls\n'
        'case "$1" in\n'
        f'  info) echo "Server Name: {server}"; echo "Default Sink: $(cat default)";;\n'
        f'  list) case "$3" in modules) cat modules;; sinks) printf "{sinks}";; esac;;\n'
        "  load-module) n=$(cat next); echo $((n + 1)) > next; shift; name=$1; shift\n"
        '    printf "%s\\t%s\\t%s\\n" "$n" "$name" "$*" >> modules; echo "$n";;\n'
        '  unload-module) grep -v "^$2\t" modules > modules.new; mv modules.new modules;;\n'
        "esac\n"
    )
    pacat = directory / "pacat"
    pacat.write_text(f'#!/bin/sh\nprintf "pacat %s\\n" "$*" >> "{calls}"\nexec wc -c > "{directory}/played"\n')
    for tool in (pactl, pacat):
        tool.chmod(0o755)
    return calls


def standing_still(directory: Path) -> None:
    """The stand-in ``pactl`` notes each call and then, while the file ``hang`` is there, stands
    still as a sound server that does not answer."""
    plain = (directory / "pactl").read_text()
    recorded = 'printf "%s\\n" "$*" >> calls\n'
    assert recorded in plain
    (directory / "pactl").write_text(plain.replace(recorded, recorded + "[ -f hang ] && exec sleep 30\n"))


ME = os.getpid()
NULL_SINK = (
    "load-module module-null-sink sink_name=stadiacast_mic_feed format=s16le rate=48000 channels=1"
    ' channel_map=mono sink_properties=\'device.description="Stadia Controller microphone feed"'
    f' stadiacast.pid="{ME}"\''
)
REMAP_SOURCE = (
    "load-module module-remap-source master=stadiacast_mic_feed.monitor source_name=stadiacast_mic channels=1"
    " channel_map=mono master_channel_map=mono source_properties='device.description=\"Stadia Controller"
    f' microphone" device.icon_name="audio-input-microphone" stadiacast.pid="{ME}"\''
)
PACAT = (
    "pacat --playback --raw --format=s16le --rate=48000 --channels=1 --channel-map=mono"
    " --device=stadiacast_mic_feed --latency-msec=20 --client-name=Stadiacast"
    " --stream-name=Stadia Controller microphone --property=node.dont-reconnect=true -v"
)


def silence(frames: int) -> bytes:
    return bytes(2 * frames)


@posix_only
def test_on_linux_the_receiver_makes_the_device_and_takes_it_away(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A null sink and a source remapped over its monitor, each named in words and marked with
    the receiver's process, with what a receiver that is gone left behind unloaded first;
    ``pacat`` plays into that sink and is told to stay there; and at the end the source goes
    before the sink it stands on."""
    left = "17\tmodule-null-sink\tsink_name=stadiacast_mic_feed rate=48000\n9\tmodule-alsa-card\tdevice_id=0\n"
    left += "18\tmodule-remap-source\tmaster=stadiacast_mic_feed.monitor source_name=stadiacast_mic\n"
    calls = tools(tmp_path, left=left)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    said: list[tuple[str, dict[str, Any]]] = []
    out = PulseOutput(lambda event, **fields: said.append((event, fields)))
    out.start(silence)
    try:
        assert (out.name, out.microphone, out.rate, out.server) == (
            DEVICE_NAME,
            DEVICE_NAME,
            48000,
            "PulseAudio (on PipeWire 1.0.5)",
        )
        time.sleep(0.5)
    finally:
        out.stop()
    made = calls.read_text().splitlines()
    assert made == [
        "info",
        "list short modules",
        "unload-module 18",  # the source that was left behind, before its sink
        "unload-module 17",
        "list short sinks",
        NULL_SINK,
        REMAP_SOURCE,
        PACAT,
        "list short modules",
        "unload-module 42",  # the source
        "unload-module 41",
    ]
    assert (tmp_path / "modules").read_text() == "9\tmodule-alsa-card\tdevice_id=0\n"  # the sound card's stays
    played = int((tmp_path / "played").read_text())
    assert 15 * 1920 <= played <= 35 * 1920 and played % 1920 == 0  # 20 ms at a time, at the clock's pace
    assert said == [("mic_device_left_behind", {"modules": 2})] and voice_threads() == []
    assert out.note is None  # the computer has an output of its own
    assert out.loop == "stadiacast_mic_feed.monitor"  # what a capture of it is called


@posix_only
def test_a_sound_server_holds_one_receivers_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The modules say which process loaded them. Those of a receiver that still runs are left
    alone, and a second receiver is told; those of one that is gone are unloaded."""
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "stadiacast"])
    try:
        mark = f'device.description="x" stadiacast.pid="{other.pid}"'
        left = f"61\tmodule-null-sink\tsink_name=stadiacast_mic_feed sink_properties='{mark}'\n"
        left += f"62\tmodule-remap-source\tsource_name=stadiacast_mic source_properties='{mark}'\n"
        calls = tools(tmp_path, left=left)
        monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
        with pytest.raises(
            SetupError, match=rf"another receiver \(process {other.pid}\) has its microphone device in this sound"
        ):
            PulseOutput().start(silence)
        assert calls.read_text().splitlines() == ["info", "list short modules", "list short modules"]
        assert (tmp_path / "modules").read_text() == left  # nothing of the other receiver's was touched
    finally:
        other.kill()
        other.wait()
    said: list[str] = []
    out = PulseOutput(lambda event, **_fields: said.append(event))  # that receiver is gone now
    out.start(silence)
    out.stop()
    assert said == ["mic_device_left_behind"] and (tmp_path / "modules").read_text() == ""
    assert [line for line in calls.read_text().splitlines() if line.startswith("unload")] == [
        "unload-module 62",
        "unload-module 61",
        "unload-module 42",
        "unload-module 41",
    ]


@posix_only
def test_modules_are_unloaded_by_what_they_are_and_not_by_a_number_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sound server that restarted numbers its modules anew: the numbers the receiver's had
    are other modules' now, and are left alone."""
    calls = tools(tmp_path)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    out = PulseOutput()
    out.start(silence)
    wait_until(lambda: calls.read_text().splitlines()[-1].startswith("pacat"))  # the player has begun
    restarted = "41\tmodule-alsa-card\tdevice_id=0\n42\tmodule-bluetooth-discover\t\n"
    (tmp_path / "modules").write_text(restarted)
    before = len(calls.read_text().splitlines())
    out.stop()
    assert calls.read_text().splitlines()[before:] == ["list short modules"]
    assert (tmp_path / "modules").read_text() == restarted


@posix_only
def test_a_stream_that_the_sound_server_moves_is_ended_at_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sink that is unloaded under ``pacat`` does not end its stream: the server moves it to
    the computer's own output, where the microphone would be heard. ``pacat`` says where its
    stream is, and is ended the moment that is not the sink."""
    tools(tmp_path)
    (tmp_path / "pacat").write_text(
        "#!/bin/sh\n"
        f': > "{tmp_path}/heard"\n'
        'echo "Connected to device stadiacast_mic_feed (index: 41, suspended: no)." >&2\n'
        "head -c 9600 > /dev/null\n"
        'printf "Time: 0.100 sec; Latency: 20000 usec.\\r" >&2\n'
        'echo "Stream moved to device alsa_output.usb-headphones.analog-stereo (57, not suspended)." >&2\n'
        f'exec cat > "{tmp_path}/heard"\n'
    )
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    out = PulseOutput()
    out.start(silence)
    try:
        wait_until(lambda: out.error is not None)
        assert out.error == (
            "the sound server put the microphone's stream on 'alsa_output.usb-headphones.analog-stereo': it was ended"
        )
        assert (tmp_path / "heard").stat().st_size <= 4 * 1920  # a few blocks at most got there
        wait_until(lambda: voice_threads() == [])  # it asks no more: the device opens it anew
    finally:
        out.stop()

    (tmp_path / "pacat").write_text(
        '#!/bin/sh\necho "Connected to device auto_null (index: 0, suspended: no)." >&2\nexec cat > /dev/null\n'
    )
    out = PulseOutput()  # a server that put it elsewhere from the start
    out.start(silence)
    try:
        wait_until(lambda: out.error is not None)
        assert out.error == "the sound server put the microphone's stream on 'auto_null': it was ended"
    finally:
        out.stop()


@posix_only
def test_the_output_looks_whether_its_device_is_still_there(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A source that is unloaded leaves ``pacat`` playing into a sink nobody records, and a
    computer whose output goes away plays into the sink: both show when the output is asked."""
    tools(tmp_path)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    said: list[tuple[str, dict[str, Any]]] = []
    out = PulseOutput(lambda event, **fields: said.append((event, fields)))
    out.start(silence)
    try:
        assert out.check() and out.note is None and said == []
        (tmp_path / "default").write_text("stadiacast_mic_feed")  # the headphones were switched off
        assert out.check() and out.note is not None and out.note.startswith('"Stadia Controller microphone feed" is')
        assert out.check() and said == [("mic_device_note", {"note": out.note})]  # said once
        (tmp_path / "default").write_text("alsa_output.usb-headphones.analog-stereo")
        assert out.check() and out.note is None and said[-1] == ("mic_device_note", {"note": None})
        modules = (tmp_path / "modules").read_text().splitlines(keepends=True)
        (tmp_path / "modules").write_text(modules[0])  # the source was unloaded by somebody
        assert out.check() is False and out.error == "the microphone's device was taken out of the sound server"
        (tmp_path / "pactl").write_text('#!/bin/sh\necho "Connection refused" >&2\nexit 1\n')
        assert out.check()  # a server that is gone ends pacat, which is how that is seen
    finally:
        out.stop()


@posix_only
def test_a_computer_without_a_sound_card_plays_into_the_microphone_and_is_told(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The silent output a sound server keeps for a computer without a sound card goes when
    another output appears: the microphone's feed is then where the computer plays."""
    tools(tmp_path, sinks="36\\tauto_null\\tPipeWire\\tfloat32le 2ch 48000Hz\\tSUSPENDED\\n")
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    out = PulseOutput()
    out.start(silence)
    out.stop()
    assert out.note == (
        '"Stadia Controller microphone feed" is also the computer\'s sound output: what the computer plays is in the'
        " microphone as well, until another output is chosen in the sound settings"
    )


@posix_only
def test_a_sound_server_that_cannot_do_it_says_so_and_nothing_is_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    calls = tools(tmp_path, server="PulseAudio (on PipeWire 0.3.48)")
    with pytest.raises(
        SetupError, match=r"this PipeWire \(0\.3\.48\) cannot make the microphone a device: it takes 0\.3\.64"
    ):
        PulseOutput().start(silence)
    assert calls.read_text().splitlines() == ["info", "list short modules"]  # nothing was loaded or played

    (tmp_path / "pactl").write_text('#!/bin/sh\necho "Connection failure: Connection refused" >&2\nexit 1\n')
    with pytest.raises(SetupError, match=r"pactl info failed \(Connection failure: Connection refused\).*logged in"):
        PulseOutput().start(silence)

    (tmp_path / "pactl").unlink()
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(SetupError, match=r"pactl not found.*pulseaudio-utils"):
        PulseOutput().start(silence)
    assert voice_threads() == []


@posix_only
def test_a_device_that_cannot_be_finished_is_taken_away_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A source that does not load, a ``pacat`` that does not start: what was loaded until then
    is unloaded."""
    calls = tools(tmp_path)
    plain = (tmp_path / "pactl").read_text()
    refuses = 'test "$2" = module-remap-source && { echo "Failure: Module initialization failed" >&2; exit 1; }'
    (tmp_path / "pactl").write_text(plain.replace("  load-module)", f"  load-module) {refuses};"))
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    with pytest.raises(SetupError, match=r"pactl load-module failed \(Failure: Module initialization failed\)"):
        PulseOutput().start(silence)
    assert calls.read_text().splitlines()[-2:] == ["list short modules", "unload-module 41"]
    assert not any(line.startswith("pacat") for line in calls.read_text().splitlines())

    (tmp_path / "pactl").write_text(plain)
    (tmp_path / "pacat").write_text("#!/nowhere/to/be/found\n")
    with pytest.raises(SetupError, match=r"pacat did not start \("):
        PulseOutput().start(silence)
    assert calls.read_text().splitlines()[-3:] == ["list short modules", "unload-module 43", "unload-module 42"]
    assert (tmp_path / "modules").read_text() == ""


@posix_only
def test_a_player_that_ends_is_said_and_the_device_is_made_anew(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``pacat`` ends when the sound server goes away. The output then asks no more and says why,
    which is what the device waits for to open it again: the modules are loaded anew."""
    calls = tools(tmp_path)
    (tmp_path / "pacat").write_text('#!/bin/sh\nhead -c 9600 > /dev/null\necho "Connection terminated" >&2\nexit 1\n')
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    asked: list[int] = []

    def read(frames: int) -> bytes:
        asked.append(frames)
        return bytes(2 * frames)

    out = PulseOutput()
    out.start(read)
    try:
        wait_until(lambda: out.error is not None)
        assert out.error == "pacat ended: Connection terminated" and 5 <= len(asked) <= 60, (out.error, len(asked))
        so_far = len(asked)
        time.sleep(0.1)
        assert len(asked) == so_far and voice_threads() == []  # it asks no more
        out.stop()
        out.start(read)
        assert out.error is None
        wait_until(lambda: len(asked) > so_far)
    finally:
        out.stop()
    loaded = [line.split()[1] for line in calls.read_text().splitlines() if line.startswith("load-module")]
    assert loaded == ["module-null-sink", "module-remap-source"] * 2
    unloaded = [line for line in calls.read_text().splitlines() if line.startswith("unload-module")]
    assert unloaded == ["unload-module 42", "unload-module 41", "unload-module 44", "unload-module 43"]


@posix_only
def test_a_sound_server_that_does_not_answer_is_given_up_on_quickly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sound server that stands still is asked once and given up on: a start says so after one
    wait, and a receiver that stops while its device is being opened again waits for one call
    at most, and for its own clean-up, not for every call that was still to come."""
    calls = tools(tmp_path)
    standing_still(tmp_path)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    monkeypatch.setattr(micdevice, "PACTL_TIMEOUT", 0.3)
    (tmp_path / "hang").touch()
    began = time.monotonic()
    with pytest.raises(SetupError, match=r"pactl info got no answer from the sound server in 0\.3 s"):
        PulseOutput().start(silence)
    assert time.monotonic() - began < 1.5 and calls.read_text().splitlines() == ["info"]

    # Now the device runs, its player ends, and the server stands still while it is made anew.
    (tmp_path / "hang").unlink()
    (tmp_path / "pacat").write_text("#!/bin/sh\nhead -c 3840 > /dev/null\n")
    monkeypatch.setattr(micdevice, "REOPEN_SECONDS", 0.1)
    monkeypatch.setattr(micdevice, "STALL_SECONDS", 0.15)
    monkeypatch.setattr(micdevice, "ffmpeg_path", lambda: "ffmpeg")  # no voice is made: nothing is decoded
    device = MicrophoneDevice(PulseOutput())
    device.start()
    (tmp_path / "hang").touch()
    wait_until(lambda: device.output.error is not None and "no answer" in device.output.error, timeout=10)
    began = time.monotonic()
    device.stop()  # the reopen under way gives up at its question, and the clean-up asks once
    assert time.monotonic() - began < 1.5, time.monotonic() - began
    assert voice_threads() == []
    asked = len(calls.read_text().splitlines())
    with pytest.raises(SetupError, match="the receiver is stopping"):
        device.output.start(silence)  # nothing is opened again, and the sound server is not asked
    assert len(calls.read_text().splitlines()) == asked


@pytest.mark.skipif(not Path("/proc/self/cmdline").exists(), reason="a process's command line is read in /proc")
def test_modules_whose_process_is_another_program_by_now_are_left_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The process number a receiver that is gone left on its modules can be another program's by
    now: that is no receiver, and the modules are taken as left behind."""
    other = subprocess.Popen(["sleep", "30"])
    try:
        mark = f'device.description="x" stadiacast.pid="{other.pid}"'
        left = f"61\tmodule-null-sink\tsink_name=stadiacast_mic_feed sink_properties='{mark}'\n"
        left += f"62\tmodule-remap-source\tsource_name=stadiacast_mic source_properties='{mark}'\n"
        tools(tmp_path, left=left)
        monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
        said: list[str] = []
        out = PulseOutput(lambda event, **_fields: said.append(event))
        out.start(silence)
        out.stop()
        assert said == ["mic_device_left_behind"] and (tmp_path / "modules").read_text() == ""
    finally:
        other.kill()
        other.wait()


@posix_only
def test_one_answer_that_does_not_come_ends_no_later_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sound server that stood still once (a Bluetooth profile switch, a busy computer) is asked
    again at the next check, and what changed meanwhile is seen."""
    tools(tmp_path)
    standing_still(tmp_path)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    monkeypatch.setattr(micdevice, "PACTL_TIMEOUT", 0.3)
    out = PulseOutput()
    out.start(silence)
    try:
        (tmp_path / "hang").touch()
        assert out.check()  # no answer: nothing is known, and the player says when the server is gone
        (tmp_path / "hang").unlink()
        modules = (tmp_path / "modules").read_text().splitlines(keepends=True)
        (tmp_path / "modules").write_text(modules[0])  # and meanwhile the source was unloaded
        assert out.check() is False
    finally:
        out.stop()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="what waits unread in a pipe is asked of Linux")
def test_sound_that_the_player_does_not_take_is_not_kept_for_later(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PulseAudio's null sink asks ``pacat`` for nothing in its first seconds, and a pipe would
    hold two thirds of a second for it, which the microphone would then be late by for good. The
    pump writes no more once eight blocks wait, and leaves out as many blocks as waited all
    through a window: when the player takes sound again, it is the sound of that moment."""
    tools(tmp_path)
    # a pacat that takes nothing for 0.6 s and then one block every 20 ms, noting how old each one is
    (tmp_path / "pacat").write_text(
        f"#!{sys.executable}\n"
        "import struct, sys, time\n"
        "time.sleep(0.6)\n"
        "ages = []\n"
        "due = time.monotonic()\n"
        "while len(block := sys.stdin.buffer.read(1920)) == 1920:\n"
        "    ages.append(time.monotonic() - struct.unpack('<d', block[:8])[0])\n"
        "    due += 0.02\n"
        "    time.sleep(max(0.0, due - time.monotonic()))\n"
        f"open({str(tmp_path / 'ages')!r}, 'w').write(' '.join(f'{{age:.3f}}' for age in ages))\n"
    )
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    monkeypatch.setattr(PulseOutput, "WINDOW", 10)  # a fifth of a second
    out = PulseOutput()
    out.start(lambda frames: struct.pack("<d", time.monotonic()) * (frames // 4))
    try:
        time.sleep(0.5)
        assert out.flowing is False and 200 <= out.let_go_ms <= 400, out.describe()  # eight blocks wait, no more
        wait_until(lambda: out.flowing)
        time.sleep(0.4)
        assert out.flowing and 500 <= out.let_go_ms <= 760, out.describe()  # the wait, and the eight that waited
        assert out.describe()["let_go_ms"] == out.let_go_ms
    finally:
        out.stop()
    ages = [float(age) for age in (tmp_path / "ages").read_text().split()]
    assert max(ages[:5]) > 0.15, ages  # what had gathered while it took nothing
    assert len(ages) > 30 and max(ages[-15:]) < 0.05, ages  # and then what is said now


# ---- the receiver with a device -----------------------------------------------------------------
async def say(fc: FakeController, packets: list[bytes]) -> None:
    """The lab controller's microphone sends these, one every 20 ms by the clock as a controller's does."""
    began = time.monotonic()
    for i, packet in enumerate(packets):
        assert fc.send_mic_frame(packet)
        await asyncio.sleep(max(0.0, began + (i + 1) * 0.02 - time.monotonic()))


async def call(fc: FakeController, rx: Any, method: str = "GET", body: dict[str, Any] | None = None) -> tuple[int, Any]:
    raw = json.dumps(body).encode() if body is not None else b""
    status, _, out = await fc.request(
        method, "x", "/_mic", raw, "application/json", port=rx.http.http_ports[0], tls=False
    )
    return status, json.loads(out) if out else None


@needs_ffmpeg
def test_the_receivers_device_carries_a_controllers_microphone_and_no_file_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--mic ptt --mic-device``: the controller is asked while its Assistant button is held, what
    it sends is heard in the device, and the run directory gets no recording. The device is
    there from the start and gone at the end, whoever speaks."""
    out = Asked()
    asked_for: list[tuple[str | None, str | None]] = []

    def lab_output(name: str | None, *, avoid: str | None = None, **_kw: Any) -> Output:
        asked_for.append((name, avoid))
        return out

    monkeypatch.setattr(receiver, "open_output", lab_output)
    tone = real_opus(1.0)

    async def run() -> None:
        rx = make_receiver(tmp_path, mic="ptt", mic_device="")
        await rx.start()
        assert out.started == 1 and asked_for == [(None, None)] and rx.mic_device is not None
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            await wait_for(lambda: "c0" in rx.mic_device.state()["voices"])  # its decoder waits from READY on
            status, state = await call(fc, rx)
            assert status == 200 and (state["on"], state["record"], state["recording"]) == (False, False, None)
            assert state["device"]["name"] == "the lab's device" and state["device"]["voices"]["c0"]["packets"] == 0
            assert rx.status()["mic_device"]["microphone"] == "the lab's device"

            assert fc.send_input_rpc(buttons=ASSISTANT)
            await wait_for(lambda: fc.mic_requests == [True])
            began = len(out.given)
            await say(fc, tone)
            assert fc.send_input_rpc(buttons=0)
            await wait_for(lambda: fc.mic_requests == [True, False])
            await asyncio.sleep(0.2)
            assert loud(out.given[began:], 440) >= len(tone) - 8  # a second of it, but for its two ends
            assert not any(out.given[-1])  # silence again
            voice = rx.mic_device.state()["voices"]["c0"]
            assert (voice["packets"], voice["concealed"], voice["late"]) == (len(tone), 0, 0)
            assert cs.mic_state()["recording"] is None and cs.mic_state()["last"] is None
            assert not list((tmp_path / "logs").glob("mic_*"))
            assert [r["n"] for r in rx.log.records if r["event"] == "mic_rtp"] == [1, 2, 3]

            fc.power_off()  # the controller is switched off: its voice ends, the device stays
            await wait_for(lambda: rx.mic_device.state()["voices"] == {})
            assert out.stopped == 0
        finally:
            fc.close()
            await rx.stop()
        assert (out.started, out.stopped) == (1, 1) and voice_threads() == []
        assert [r["event"] for r in rx.log.records if r["event"].startswith("mic_device")] == [
            "mic_device",
            "mic_device_removed",
        ]

    asyncio.run(run())


@needs_ffmpeg
def test_packets_that_come_with_the_switch_off_are_neither_heard_nor_missed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A controller numbers its packets on from one hold of the button to the next. Those still
    on their way when the switch went off are not for the device, and the first packet of the
    next hold is not to conceal them as if they had been lost."""
    out = Asked()
    monkeypatch.setattr(receiver, "open_output", lambda _name, **_kw: out)
    tone = real_opus(1.0)

    async def run() -> None:
        rx = make_receiver(tmp_path, mic="ptt", mic_device="")
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            assert fc.send_input_rpc(buttons=ASSISTANT)
            await wait_for(lambda: fc.mic_requests == [True])
            await say(fc, tone[:20])
            assert fc.send_input_rpc(buttons=0)
            await wait_for(lambda: fc.mic_requests == [True, False])
            assert rx.mic_device is not None
            await wait_for(lambda: rx.mic_device.state()["voices"]["c0"]["packets"] == 20)
            await asyncio.sleep(0.3)  # what was said in that hold is out
            asked = len(out.given)
            for packet in tone[20:23]:
                assert fc.send_mic_frame(packet, even_if_off=True)
            await wait_for(lambda: cs.mic_state()["stray"] == 3)
            await asyncio.sleep(0.2)
            assert not any(b"".join(out.given[asked:]))  # nobody heard them
            assert fc.send_input_rpc(buttons=ASSISTANT)
            await wait_for(lambda: fc.mic_requests.count(True) == 2)
            await say(fc, tone[23:43])
            await wait_for(lambda: rx.mic_device.state()["voices"]["c0"]["packets"] == 40)
            voice = rx.mic_device.state()["voices"]["c0"]
            assert (voice["concealed"], voice["late"]) == (0, 0), voice
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


@needs_ffmpeg
def test_a_device_and_a_recording_take_the_same_sound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = Asked()
    monkeypatch.setattr(receiver, "open_output", lambda _name, **_kw: out)
    tone = real_opus(0.6)

    async def run() -> None:
        rx = make_receiver(tmp_path, mic="on", mic_device="", mic_record=True)
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            await wait_for(lambda: fc.mic_requests == [True])
            await say(fc, tone)
            await wait_for(lambda: (cs.mic_state()["recording"] or {}).get("packets") == len(tone))
            assert rx.mic_device is not None and rx.mic_device.state()["voices"]["c0"]["packets"] == len(tone)
            status, switched = await call(fc, rx, "POST", {"do": "record", "on": False})
            assert status == 200 and switched == {"ok": True, "record": False, "controller": "c0"}
            state = cs.mic_state()  # the recording ended, the microphone is on for the device
            assert (state["on"], state["record"], state["recording"]) == (True, False, None)
            assert state["last"]["packets"] == len(tone) and fc.mic_requests == [True]
        finally:
            fc.close()
            await rx.stop()
        assert loud(out.given, 440) >= len(tone) - 8

    asyncio.run(run())


@needs_ffmpeg
def test_a_recording_that_fails_leaves_the_device_its_microphone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run directory that takes no file ends the recording. The microphone stays on for the
    device, which a recording that was all there was would not."""
    out = Asked()
    monkeypatch.setattr(receiver, "open_output", lambda _name, **_kw: out)
    monkeypatch.setattr(controller, "capture_path", lambda _run_dir, _label: str(tmp_path / "gone" / "m.ogg"))

    async def run() -> None:
        rx = make_receiver(tmp_path, mic="on", mic_device="", mic_record=True)
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            await wait_for(lambda: fc.mic_requests == [True])
            for _ in range(3):
                assert fc.send_mic_frame()
            assert rx.mic_device is not None
            await wait_for(lambda: rx.mic_device.state()["voices"]["c0"]["packets"] == 3)
            state = cs.mic_state()
            assert (state["on"], state["record"], state["recording"]) == (True, False, None)
            assert fc.mic_requests == [True] and rx.log.counts.get("mic_recording_failed") == 1  # said once
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


@needs_ffmpeg
def test_the_headsets_sound_is_never_captured_from_the_microphones_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A capture of the device the microphone is played into would put the microphone into the
    headset: ``--audio-source`` is refused it at READY, and ``/_audio`` whenever it asks. The
    capture is looked up with the output closed where an open one keeps the list of devices."""
    out = Asked()
    out.pins_devices = True
    monkeypatch.setattr(receiver, "open_output", lambda _name, **_kw: out)
    pipe = ["-f", "s16le", "-ar", "48000", "-ac", "2", "-i", "pipe:0"]
    monkeypatch.setattr(
        controller, "source_from_spec", lambda _spec: Source(pipe, "capture the lab's device", pcm=True)
    )

    async def run() -> None:
        rx = make_receiver(tmp_path, mic_device="", audio_source="device:lab")
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            await wait_for(lambda: rx.log.counts.get("audio_source_invalid") == 1)
            refused = next(r for r in rx.log.records if r["event"] == "audio_source_invalid")
            assert refused["error"].startswith(
                "\"the lab's device\" carries the controllers' microphones (--mic-device): captured for the headset"
            )
            assert (out.started, out.stopped) == (2, 1)  # closed while the capture was looked up
            with pytest.raises(SetupError, match=r"carries the controllers' microphones \(--mic-device\)"):
                await cs.audio_start_spec("system")
            assert cs.audio_state() == {"playing": False} and (out.started, out.stopped) == (3, 2)
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


def test_the_device_is_told_where_the_headsets_sound_comes_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Windows and on a Mac the microphone is played into a device of the kind the headset's
    sound is captured from. The output is told which one that is, so as to take another."""
    asked: list[str] = []

    def source(spec: str) -> Source:
        asked.append(spec)
        return Source([], "capture BlackHole 2ch", pcm=True)

    monkeypatch.setattr(receiver, "source_from_spec", source)
    for n, (platform, spec, avoid) in enumerate(
        [
            ("darwin", "system", "BlackHole 2ch"),
            ("win32", "device:cable", "BlackHole 2ch"),
            ("linux", "system", None),  # there the receiver makes a device of its own
            ("darwin", "tone", None),
            ("darwin", None, None),
        ]
    ):
        monkeypatch.setattr(sys, "platform", platform)
        assert make_receiver(tmp_path / str(n), mic_device="", audio_source=spec)._headset_capture() == avoid
    assert asked == ["system", "device:cable"]

    def missing(_spec: str) -> Source:
        raise SetupError("no loopback device among the audio inputs")

    monkeypatch.setattr(receiver, "source_from_spec", missing)  # the headset's stream says that itself
    monkeypatch.setattr(sys, "platform", "darwin")
    assert make_receiver(tmp_path / "none", mic_device="", audio_source="system")._headset_capture() is None


def test_a_microphone_is_not_switched_on_for_nothing(tmp_path: Path) -> None:
    """With neither a device nor a recording to take it, the controller is not asked at all: its
    light would blink for nobody."""

    async def run() -> None:
        rx = make_receiver(tmp_path)
        await rx.start()
        fc = await connect_firmware_flow(rx)
        cs = single(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            status, out = await call(fc, rx, "POST", {"do": "on"})
            assert status == 409 and out["ok"] is False and out["on"] is False and "nowhere" not in out
            assert out["error"].startswith("nothing takes the microphone: start the receiver with --mic-device or")
            assert fc.mic_requests == [] and cs.mic_on is False
            assert (await call(fc, rx, "POST", {"do": "ptt", "on": True}))[0] == 200
            assert fc.send_input_rpc(buttons=ASSISTANT)  # the button is the switch, and has nothing to switch
            await wait_for(lambda: cs.session["display"]["input_rpc"] == 1)
            assert fc.mic_requests == [] and cs.mic_on is False

            assert (await call(fc, rx, "POST", {"do": "record", "on": True}))[1]["record"] is True
            assert fc.send_input_rpc(buttons=0) and fc.send_input_rpc(buttons=ASSISTANT)
            await wait_for(lambda: fc.mic_requests == [True])  # now there is somewhere
            assert fc.send_mic_frame()
            await wait_for(lambda: cs.mic_state()["recording"] is not None)
            status, out = await call(fc, rx, "POST", {"do": "record", "on": False})  # and nowhere again
            assert status == 200 and out["record"] is False
            await wait_for(lambda: fc.mic_requests == [True, False])
            assert cs.mic_on is False and cs.mic_state()["recording"] is None
            assert [r["reason"] for r in rx.log.records if r["event"] == "mic_switch"][-1] == "nowhere"
        finally:
            fc.close()
            await rx.stop()

    asyncio.run(run())


@needs_ffmpeg
def test_a_receiver_started_twice_ends_on_its_ports_before_it_touches_the_sound_system(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second receiver by mistake must not get as far as the sound server, where the first
    one's device is: the ports are tried first."""
    opened: list[Output] = []

    def lab_output(_name: str | None, **_kw: Any) -> Output:
        opened.append(Asked())
        return opened[-1]

    monkeypatch.setattr(receiver, "open_output", lab_output)

    async def run() -> None:
        with socket.socket() as first:  # the port the first receiver listens on
            first.bind(("127.0.0.1", 0))
            first.listen(1)
            rx = make_receiver(tmp_path, mic="ptt", mic_device="", https_port=first.getsockname()[1])
            try:
                with pytest.raises(OSError, match="cannot listen on"):
                    await rx.start()
            finally:
                await rx.stop()
        assert opened == [] and rx.mic_device is None

    asyncio.run(run())


def test_a_device_that_cannot_be_had_stops_the_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def none(_name: str | None, **_kw: Any) -> Output:
        raise SetupError("no virtual cable to play the microphone into: install VB-CABLE")

    monkeypatch.setattr(receiver, "open_output", none)

    async def run() -> None:
        rx = make_receiver(tmp_path, mic="on", mic_device="")
        try:
            with pytest.raises(SetupError, match="no virtual cable to play the microphone into"):
                await rx.start()
        finally:
            await rx.stop()
        assert rx.mic_device is None and "ready" not in rx.log.counts

    asyncio.run(run())


#: A program that records a device for some seconds into a file, as raw samples: started as a
#: process of its own, since that is what a program which listens to the microphone is.
RECORDER = """
import sys, time
import sounddevice as sd
with open(sys.argv[2], "wb") as out:
    with sd.RawInputStream(samplerate=48000, blocksize=960, device=sys.argv[1], channels=1, dtype="int16",
                           callback=lambda indata, frames, at, status: out.write(bytes(indata))):
        time.sleep(float(sys.argv[3]))
"""


@pytest.mark.skipif(
    sys.platform != "darwin" or not os.environ.get("STADIACAST_MAC_LAB") or ffmpeg_path() is None,
    reason="needs a Mac with a loopback device (BlackHole), ffmpeg and STADIACAST_MAC_LAB=1",
)
def test_a_macs_loopback_device_carries_the_microphone(tmp_path: Path) -> None:
    """The whole chain on a Mac: a controller's packets, decoded, played into the loopback device
    the output picks, and recorded from that device's input side by another program."""
    out = PortAudioOutput()
    device = MicrophoneDevice(out)
    device.start()
    heard = tmp_path / "heard.pcm"
    try:
        assert out.note is None, out.note  # the computer's own sound goes elsewhere
        device.voice("c0")
        recorder = subprocess.Popen([sys.executable, "-c", RECORDER, out.name, str(heard), "4"])
        time.sleep(1.5)  # it has begun
        assert recorder.poll() is None, "the recorder could not open the device"
        tone = real_opus(2.0)
        began = time.monotonic()
        for i, packet in enumerate(tone):
            device.feed("c0", packet, i * 960, i)
            time.sleep(max(0.0, began + (i + 1) * 0.02 - time.monotonic()))
        assert recorder.wait(timeout=10) == 0
        state = device.state()
    finally:
        device.stop()
    recorded = heard.read_bytes()
    pieces = [recorded[at : at + 1920] for at in range(0, len(recorded) - 1919, 1920)]
    sounding = [n for n, piece in enumerate(pieces) if loud([piece], 440)]
    holes = sounding[-1] - sounding[0] + 1 - len(sounding)
    print(f"the microphone through {out.name}: {len(sounding)} of {len(tone)} pieces, {holes} holes")
    assert len(sounding) >= len(tone) - 6 and holes <= 2, (len(pieces), len(sounding), holes)
    assert state["error"] is None and state["voices"]["c0"]["packets"] == len(tone)
