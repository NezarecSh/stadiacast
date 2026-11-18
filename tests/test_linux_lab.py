"""The Linux-only paths against the real thing: the pad uinput creates, read back through its
event node the way a game reads it, a game's force feedback requests, and what the computer
plays, recorded from a real sound server. Runs on Linux when STADIACAST_LINUX_LAB is set (the
Linux lab workflow); skipped elsewhere. The pad tests need ``/dev/uinput`` and the event nodes
to open (as root, or with the udev rule of the README), and a pad appears on the system while
one runs: A, the left stick and the left trigger are what it presses. The sound tests play a
tone to the default output for a few seconds and make a microphone named "Stadia Controller
microphone" for as long, and skip without a sound server or ffmpeg."""

from __future__ import annotations

import asyncio
import ctypes
import errno
import os
import shutil
import struct
import subprocess
import sys
import time
from typing import Any

import pytest

from stadiacast import uinput_gamepad as uinput
from stadiacast.audio import ffmpeg_path
from stadiacast.uinput_gamepad import (
    ABS_HAT0Y,
    ABS_X,
    ABS_Y,
    ABS_Z,
    AXES,
    BTN_A,
    EV_ABS,
    EV_FF,
    EV_KEY,
    EV_SYN,
    EVENT,
    FF_BITS,
    FF_GAIN,
    FF_PERIODIC,
    FF_RUMBLE,
    FF_SINE,
    KEYS,
    NEUTRAL,
    UinputPad,
    ioc,
)
from tests.helpers import connect_firmware_flow, make_receiver, single, wait_for
from tests.test_micdevice import loud, say
from tests.test_microphone import real_opus

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or not os.environ.get("STADIACAST_LINUX_LAB"),
    reason="needs Linux, the right to open /dev/uinput and STADIACAST_LINUX_LAB=1",
)

EVIOCGID = ioc("r", "E", 0x02, 8)
EVIOCGEFFECTS = ioc("r", "E", 0x84, 4)
EVIOCSFF = ioc("w", "E", 0x80, ctypes.sizeof(uinput._Effect))
EVIOCRMFF = ioc("w", "E", 0x81, 4)


class Game:
    """A pad's event node as a game holds it: it reads the state and the events, and uploads,
    plays and erases force feedback effects."""

    def __init__(self, path: str) -> None:
        import fcntl

        self.ioctl: Any = fcntl.ioctl
        deadline = time.monotonic() + 3
        while True:  # udev sets the node's permissions a moment after the kernel made it
            try:
                self.fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
                break
            except PermissionError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)

    def identity(self) -> tuple[int, ...]:
        """Bus, vendor, product and version."""
        raw = bytearray(8)
        self.ioctl(self.fd, EVIOCGID, raw)
        return struct.unpack("HHHH", raw)

    def name(self) -> str:
        raw = bytearray(80)
        self.ioctl(self.fd, ioc("r", "E", 0x06, len(raw)), raw)
        return raw.split(b"\0")[0].decode()

    def codes(self, kind: int) -> set[int]:
        """The codes of an event type that the device has; type 0 lists the types themselves."""
        raw = bytearray(96)
        self.ioctl(self.fd, ioc("r", "E", 0x20 + kind, len(raw)), raw)
        return {bit for bit in range(len(raw) * 8) if raw[bit // 8] & (1 << (bit % 8))}

    def axis(self, code: int) -> tuple[int, ...]:
        """Value, minimum, maximum, fuzz, flat and resolution."""
        raw = bytearray(24)
        self.ioctl(self.fd, ioc("r", "E", 0x40 + code, len(raw)), raw)
        return struct.unpack("6i", raw)

    def held(self) -> set[int]:
        raw = bytearray(96)
        self.ioctl(self.fd, ioc("r", "E", 0x18, len(raw)), raw)
        return {bit for bit in range(len(raw) * 8) if raw[bit // 8] & (1 << (bit % 8))}

    def effects(self) -> int:
        raw = bytearray(4)
        self.ioctl(self.fd, EVIOCGEFFECTS, raw)
        return int(struct.unpack("i", raw)[0])

    def events(self) -> list[tuple[int, int, int]]:
        """What has arrived since the last call, as (type, code, value)."""
        out: list[tuple[int, int, int]] = []
        while True:
            try:
                data = os.read(self.fd, EVENT.size * 64)
            except BlockingIOError:
                return out
            out += [EVENT.unpack_from(data, at)[2:] for at in range(0, len(data), EVENT.size)]

    def upload(self, strong: int, weak: int, length_ms: int, delay_ms: int = 0, slot: int = -1) -> int:
        """Store a rumble effect on the pad (``slot`` -1: a new one) and return its slot. The call
        returns once the pad has answered."""
        effect = uinput._Effect(type=FF_RUMBLE, id=slot, replay_length=length_ms, replay_delay=delay_ms)
        effect.u.rumble.strong_magnitude = strong
        effect.u.rumble.weak_magnitude = weak
        self.ioctl(self.fd, EVIOCSFF, effect)
        return int(effect.id)

    def upload_sine(self, magnitude: int, length_ms: int) -> int:
        """Store a periodic effect, a sine of that magnitude, and return its slot."""
        effect = uinput._Effect(type=FF_PERIODIC, id=-1, replay_length=length_ms)
        effect.u.periodic.waveform = FF_SINE
        effect.u.periodic.period = 100
        effect.u.periodic.magnitude = magnitude
        self.ioctl(self.fd, EVIOCSFF, effect)
        return int(effect.id)

    def play(self, slot: int, count: int = 1) -> None:
        os.write(self.fd, EVENT.pack(0, 0, EV_FF, slot, count))

    def gain(self, gain: int) -> None:
        os.write(self.fd, EVENT.pack(0, 0, EV_FF, FF_GAIN, gain))

    def erase(self, slot: int) -> None:
        self.ioctl(self.fd, EVIOCRMFF, slot)

    def close(self) -> None:
        os.close(self.fd)


def wait_until(cond: Any, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.01)


def test_the_kernel_takes_the_pad_for_an_xbox_360_controller() -> None:
    pad = UinputPad("lab")
    try:
        assert pad.device is not None and pad.device.startswith("/dev/input/event")
        game = Game(pad.device)
        try:
            assert game.name() == "Microsoft X-Box 360 pad"
            assert game.identity() == (3, 0x045E, 0x028E, 0x0114)  # USB, Microsoft, Xbox 360 controller
            assert game.codes(0) == {EV_SYN, EV_KEY, EV_ABS, EV_FF}
            assert game.codes(EV_KEY) == set(KEYS.values())
            assert game.codes(EV_ABS) == set(AXES)
            assert game.codes(EV_FF) == set(FF_BITS) and FF_PERIODIC in FF_BITS
            assert game.effects() == 16
            for code, (minimum, maximum, fuzz, flat) in AXES.items():
                assert game.axis(code)[1:] == (minimum, maximum, fuzz, flat, 0), code
        finally:
            game.close()
        device = pad.device
    finally:
        pad.close()
    wait_until(lambda: not os.path.exists(device))  # closing the pad unplugs it
    pad.close()  # a second close is nothing


def test_reports_reach_a_game_as_state_and_as_events() -> None:
    pad = UinputPad("lab")
    try:
        assert pad.device is not None
        game = Game(pad.device)
        try:
            pad.apply(NEUTRAL)
            game.events()
            pad.apply(
                {**NEUTRAL, "wButtons": (1 << 12) | 1, "sThumbLX": 20000, "sThumbLY": -20000, "bLeftTrigger": 255}
            )
            assert game.events() == [
                (EV_KEY, BTN_A, 1),
                (EV_ABS, ABS_X, 20000),
                (EV_ABS, ABS_Y, 19999),
                (EV_ABS, ABS_Z, 255),
                (EV_ABS, ABS_HAT0Y, -1),
                (EV_SYN, 0, 0),
            ]
            assert game.held() == {BTN_A}
            assert (game.axis(ABS_X)[0], game.axis(ABS_Y)[0], game.axis(ABS_Z)[0]) == (20000, 19999, 255)
            pad.apply(
                {**NEUTRAL, "wButtons": (1 << 12) | 1, "sThumbLX": 20000, "sThumbLY": -20000, "bLeftTrigger": 255}
            )
            assert game.events() == []  # the same report again is no news
            pad.reset()
            assert game.events()[-1] == (EV_SYN, 0, 0) and game.held() == set()
            assert (game.axis(ABS_X)[0], game.axis(ABS_Z)[0], game.axis(ABS_HAT0Y)[0]) == (0, 0, 0)
        finally:
            game.close()
    finally:
        pad.close()


def test_two_pads_are_two_devices() -> None:
    first, second = UinputPad("c0"), UinputPad("c1")
    try:
        assert first.device is not None and second.device is not None and first.device != second.device
        games = [Game(first.device), Game(second.device)]
        try:
            second.apply({**NEUTRAL, "wButtons": 1 << 12})
            assert [game.held() for game in games] == [set(), {BTN_A}]
            first.apply({**NEUTRAL, "bLeftTrigger": 9})
            assert [game.axis(ABS_Z)[0] for game in games] == [9, 0]
        finally:
            for game in games:
                game.close()
    finally:
        first.close()
        second.close()


def test_a_games_rumble_comes_out_as_motor_levels_over_time() -> None:
    pad = UinputPad("lab")
    levels: list[tuple[int, int]] = []
    stamps: list[float] = []  # when the pad's own thread reported each level

    def report(strong: int, weak: int) -> None:
        levels.append((strong, weak))
        stamps.append(time.monotonic())

    try:
        assert pad.device is not None
        pad.on_rumble(report)
        game = Game(pad.device)
        try:
            pulse = game.upload(0xC000, 0x4000, length_ms=150)
            assert pulse == 0 and levels == []  # stored on the pad, not played
            game.play(pulse)
            wait_until(lambda: len(levels) >= 2)  # it ends by itself
            assert levels == [(0xC0, 0x40), (0, 0)] and 0.1 < stamps[1] - stamps[0] < 2.0

            hold = game.upload(0xFFFF, 0x8000, length_ms=0)
            assert hold == 1
            game.play(hold)
            wait_until(lambda: levels[-1] == (0xFF, 0x80))
            time.sleep(0.3)
            assert levels[-1] == (0xFF, 0x80)  # no length: until it is stopped
            game.gain(0x7FFF)
            wait_until(lambda: levels[-1] == (0x7F, 0x3F))
            game.gain(0xFFFF)
            assert game.upload(0x2000, 0x1000, length_ms=0, slot=hold) == hold  # changed while it plays
            wait_until(lambda: levels[-1] == (0x20, 0x10))
            game.play(pulse)  # effects that play together add up
            wait_until(lambda: (0xE0, 0x50) in levels)
            wait_until(lambda: levels[-1] == (0x20, 0x10))
            game.play(hold, 0)
            wait_until(lambda: levels[-1] == (0, 0))
            game.play(hold)
            wait_until(lambda: levels[-1] == (0x20, 0x10))
            game.erase(hold)  # an effect that is taken off the pad stops
            wait_until(lambda: levels[-1] == (0, 0))
        finally:
            game.close()
    finally:
        pad.close()


def test_an_effect_sent_again_while_it_plays_starts_its_run_again() -> None:
    """As on a real pad: a game that keeps a short effect alive by updating it goes on rumbling."""
    pad = UinputPad("lab")
    levels: list[tuple[int, int]] = []
    stamps: list[float] = []

    def report(strong: int, weak: int) -> None:
        levels.append((strong, weak))
        stamps.append(time.monotonic())

    try:
        assert pad.device is not None
        pad.on_rumble(report)
        game = Game(pad.device)
        try:
            slot = game.upload(0xFFFF, 0xFFFF, length_ms=300)
            game.play(slot)
            wait_until(lambda: levels == [(0xFF, 0xFF)])
            time.sleep(0.1)
            assert game.upload(0x8000, 0x8000, length_ms=300, slot=slot) == slot
            wait_until(lambda: len(levels) >= 3)
            assert levels == [(0xFF, 0xFF), (0x80, 0x80), (0, 0)]
            assert 0.25 < stamps[2] - stamps[1] < 2.0  # 300 ms from the update, not the 200 left of the first run
        finally:
            game.close()
    finally:
        pad.close()


def test_a_periodic_effect_rumbles_both_motors() -> None:
    pad = UinputPad("lab")
    levels: list[tuple[int, int]] = []
    try:
        assert pad.device is not None
        pad.on_rumble(lambda strong, weak: levels.append((strong, weak)))
        game = Game(pad.device)
        try:
            slot = game.upload_sine(0x4000, length_ms=0)
            game.play(slot)
            wait_until(lambda: levels == [(0x80, 0x80)])
            game.play(slot, 0)
            wait_until(lambda: levels == [(0x80, 0x80), (0, 0)])
        finally:
            game.close()
    finally:
        pad.close()


def test_a_game_and_a_pad_can_leave_in_either_order() -> None:
    """A game that closes the node with effects on the pad has them erased by the kernel, which
    asks the pad about each: a pad that did not answer would hold the game for half a minute. A
    pad that goes first leaves the game with a device that is gone."""
    pad = UinputPad("lab")
    try:
        assert pad.device is not None
        game = Game(pad.device)
        for _ in range(4):
            game.play(game.upload(0x1000, 0x1000, length_ms=0))
        began = time.monotonic()
        game.close()
        assert time.monotonic() - began < 2
    finally:
        pad.close()

    pad = UinputPad("lab")
    assert pad.device is not None
    game = Game(pad.device)
    try:
        game.play(game.upload(0x1000, 0x1000, length_ms=0))
        began = time.monotonic()
        pad.close()
        assert time.monotonic() - began < 2
        with pytest.raises(OSError) as gone:
            game.upload(0x1000, 0x1000, length_ms=0)
        assert gone.value.errno == errno.ENODEV
    finally:
        game.close()


def test_a_controller_is_a_pad_for_this_computers_games_and_feels_their_rumble(tmp_path) -> None:
    """The whole path with the lab controller: its input, on the display link and then on the
    game session ``--gamepad`` starts, reaches the kernel's device, and a force feedback effect a
    game plays there moves the controller's motors."""

    async def run() -> None:
        rx = make_receiver(tmp_path, gamepad="uinput", start_game_delay=0.5, display_rpc_delay=0.05)
        await rx.start()
        fc = await connect_firmware_flow(rx, game_mode=True, game_framing="raw")
        game = None
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            assert fc.send_input_rpc(buttons=1 << 12, lx=20000, ly=-20000, lt=255)  # A, left stick, left trigger
            await wait_for(lambda: rx.status()["gamepads"].get("c0", {}).get("frames", 0) >= 1)
            status = rx.status()["gamepads"]["c0"]
            assert status["kind"] == "uinput" and status["slot"] is None
            game = Game(status["device"])
            assert game.name() == "Microsoft X-Box 360 pad" and game.held() == {BTN_A}
            # past XInput's deadzone, which the receiver applies: right and down, and the trigger
            assert game.axis(ABS_X)[0] > 8000 and game.axis(ABS_Y)[0] > 8000 and game.axis(ABS_Z)[0] == 255

            await wait_for(lambda: single(rx).session["game"]["state"] == "game-started", 8)
            seq = fc.send_game_input(buttons=0)  # in a game the reports come on the game session
            assert await fc.wait_game_ack(5) == seq
            await wait_for(lambda: rx.status()["gamepads"]["c0"]["frames"] >= 2)
            assert game.held() == set() and game.axis(ABS_Z)[0] == 0

            # what a game does through SDL: one effect, uploaded and played
            slot = await asyncio.to_thread(game.upload, 0xFFFF, 0x8000, 0)
            game.play(slot)
            await wait_for(lambda: fc.motors == (65535, 32896))
            assert rx.status()["gamepads"]["c0"]["rumbles"] >= 1
            game.play(slot, 0)
            await wait_for(lambda: fc.motors == (0, 0))

            fc.power_off()  # the controller goes, and its pad with it
            await wait_for(lambda: single(rx).session["state"] == "dtls-closed")
            await wait_for(lambda: not os.path.exists(status["device"]))
            assert rx.status()["gamepads"] == {}
        finally:
            if game is not None:
                game.close()
            fc.close()
            await rx.stop()

    asyncio.run(run())


def test_what_the_computer_plays_reaches_the_headset(tmp_path) -> None:
    """``--audio-source system`` with a real sound server: a tone played to the default output is
    what the lab controller's headset gets, recorded from the output's monitor."""
    ffmpeg = ffmpeg_path()
    if ffmpeg is None:
        pytest.skip("needs ffmpeg (with libopus) on PATH")
    tools = [shutil.which(tool) for tool in ("pactl", "parec", "paplay")]
    if None in tools or subprocess.run(["pactl", "info"], capture_output=True, timeout=10, check=False).returncode:
        pytest.skip("needs a PulseAudio or PipeWire server that pactl reaches, with parec and paplay")
    tone = tmp_path / "tone.wav"
    made = [ffmpeg, "-v", "error", "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=40"]
    subprocess.run([*made, "-ac", "2", str(tone)], check=True, timeout=60)

    async def run() -> None:
        rx = make_receiver(tmp_path, audio_source="system")
        await rx.start()
        fc = await connect_firmware_flow(rx)
        try:
            await asyncio.wait_for(fc.display_ready.wait(), 5)
            await fc.keepalive()
            await wait_for(lambda: rx.log.counts.get("audio_capture", 0) >= 1, timeout=15)
            started = next(r for r in rx.log.records if r["event"] == "audio_capture")
            assert str(started["device"]).endswith(".monitor") and (started["rate"], started["channels"]) == (48000, 2)
            await wait_for(lambda: len(fc.audio_frames) >= 150, timeout=20)  # three seconds of it
            state = single(rx).audio_state()
            assert state["playing"] and state["source"] == f"capture {started['device']}"
            assert state["stats"]["send_failed"] == 0 and state["stats"]["level"] > 0.01, state
            sizes = [len(frame.payload) for frame in fc.audio_frames[-50:]]
            assert sum(size > 40 for size in sizes) >= 40, sizes  # a tone; silence is three bytes a frame
            stopped = await single(rx).audio_stop()
            assert stopped["playing"] is False
        finally:
            fc.close()
            await rx.stop()

    player = subprocess.Popen(["paplay", str(tone)])
    try:
        time.sleep(0.5)
        assert player.poll() is None, "paplay could not play to the default output"
        asyncio.run(run())
    finally:
        player.kill()
        player.wait()


def test_a_controllers_microphone_is_a_microphone_of_this_computer(tmp_path) -> None:
    """``--mic on --mic-device`` with a real sound server: "Stadia Controller microphone" is a source
    of its own and no monitor, what the lab controller's microphone sends is recorded from it, the
    computer's output stays the one it was, what a receiver that was killed left behind is
    replaced, and nothing is left when the receiver stops."""
    if ffmpeg_path() is None:
        pytest.skip("needs ffmpeg (with libopus) on PATH")
    tools = [shutil.which(tool) for tool in ("pactl", "pacat", "parec")]
    if None in tools or subprocess.run(["pactl", "info"], capture_output=True, timeout=10, check=False).returncode:
        pytest.skip("needs a PulseAudio or PipeWire server that pactl reaches, with pacat and parec")

    def pactl(*args: str) -> str:
        said = subprocess.run(
            ["pactl", *args], capture_output=True, text=True, timeout=10, check=True, env={**os.environ, "LC_ALL": "C"}
        )
        return said.stdout.strip()

    def ours(kind: str) -> list[str]:
        return [line for line in pactl("list", "short", kind).splitlines() if "stadiacast_mic" in line]

    def default_sink() -> str:
        return next(line.partition(": ")[2] for line in pactl("info").splitlines() if line.startswith("Default Sink"))

    assert ours("modules") == [], "a receiver's microphone is in the sound server already"
    output = default_sink()
    pactl("load-module", "module-null-sink", "sink_name=stadiacast_mic_feed")  # as a receiver that was killed leaves it
    tone = real_opus(2.0)
    heard = tmp_path / "heard.pcm"

    async def run() -> None:
        rx = make_receiver(tmp_path, mic="on", mic_device="")
        await rx.start()
        recorder = fc = None
        try:
            assert rx.mic_device is not None and rx.log.counts.get("mic_device_left_behind") == 1
            assert len(ours("modules")) == 2 and len(ours("sinks")) == 1
            (source,) = (block for block in pactl("list", "sources").split("\n\n") if "Name: stadiacast_mic\n" in block)
            assert "Description: Stadia Controller microphone\n" in source and "Monitor of Sink: n/a" in source
            state = rx.mic_device.state()
            assert state["error"] is None and state["server"], state
            await wait_for(lambda: rx.mic_device.state()["flowing"], timeout=10)  # PulseAudio's sink takes its time
            if output == "auto_null":  # a computer without a sound card: the feed is the one output it has
                assert "is also the computer's sound output" in state["note"], state
            else:
                assert default_sink() == output and state["note"] is None, state
            recording = ["parec", "--device=stadiacast_mic", "--format=s16le", "--rate=48000", "--channels=1", "--raw"]
            with heard.open("wb") as fh:
                recorder = subprocess.Popen([*recording, "--latency-msec=20"], stdout=fh)
                fc = await connect_firmware_flow(rx)
                await asyncio.wait_for(fc.display_ready.wait(), 5)
                await fc.keepalive()
                await wait_for(lambda: fc.mic_requests == [True])
                await asyncio.sleep(0.5)  # the recorder has begun
                assert recorder.poll() is None, "parec could not record the microphone"
                await say(fc, tone)
                await asyncio.sleep(0.5)
            voice = rx.mic_device.state()["voices"]["c0"]
            assert (voice["packets"], voice["late"]) == (len(tone), 0), voice
        finally:
            if recorder is not None:
                recorder.terminate()
                recorder.wait()
            if fc is not None:
                fc.close()
            await rx.stop()

    try:
        asyncio.run(run())
    finally:
        left = ours("modules")
        for line in left:
            subprocess.run(["pactl", "unload-module", line.split()[0]], check=False, timeout=10)
    assert left == [] and ours("sources") == [] and default_sink() == output
    recorded = heard.read_bytes()
    pieces = [recorded[at : at + 1920] for at in range(0, len(recorded) - 1919, 1920)]
    sounding = [n for n, piece in enumerate(pieces) if loud([piece], 440)]
    holes = sounding[-1] - sounding[0] + 1 - len(sounding)
    print(f"the microphone through the sound server: {len(sounding)} of {len(tone)} pieces, {holes} holes")
    assert len(sounding) >= len(tone) - 6 and holes <= 2, (len(pieces), len(sounding), holes)  # two seconds of it


def test_a_sink_that_is_taken_from_the_receiver_is_never_played_elsewhere(tmp_path) -> None:
    """A sound server moves a stream whose sink is unloaded to the computer's own output, where
    the microphone would be heard. The receiver's stream is ended instead, and the device is
    made anew."""
    if ffmpeg_path() is None:
        pytest.skip("needs ffmpeg (with libopus) on PATH")
    tools = [shutil.which(tool) for tool in ("pactl", "pacat", "parec")]
    if None in tools or subprocess.run(["pactl", "info"], capture_output=True, timeout=10, check=False).returncode:
        pytest.skip("needs a PulseAudio or PipeWire server that pactl reaches, with pacat and parec")

    def pactl(*args: str) -> str:
        said = subprocess.run(
            ["pactl", *args], capture_output=True, text=True, timeout=10, check=True, env={**os.environ, "LC_ALL": "C"}
        )
        return said.stdout.strip()

    def ours() -> list[str]:
        return [line for line in pactl("list", "short", "modules").splitlines() if "stadiacast_mic" in line]

    def elsewhere() -> list[str]:
        """The sinks other than the microphone's that a stream of the receiver plays on."""
        names = {line.split()[0]: line.split()[1] for line in pactl("list", "short", "sinks").splitlines()}
        found = []
        for stream in pactl("list", "sink-inputs").split("\n\n"):
            if 'application.name = "Stadiacast"' in stream:
                sink = next(line.split()[1] for line in stream.splitlines() if line.strip().startswith("Sink:"))
                if sink in names and names[sink] != "stadiacast_mic_feed":  # not on a sink at all is nowhere
                    found.append(names[sink])
        return found

    assert ours() == [], "a receiver's microphone is in the sound server already"

    async def run() -> None:
        rx = make_receiver(tmp_path, mic="on", mic_device="")
        await rx.start()
        try:
            assert rx.mic_device is not None
            await wait_for(lambda: rx.mic_device.state()["flowing"], timeout=10)
            assert elsewhere() == []
            sink = next(line.split()[0] for line in ours() if "module-null-sink" in line)
            pactl("unload-module", sink)
            seen: list[str] = []
            deadline = time.monotonic() + 20
            while rx.log.counts.get("mic_device_back", 0) == 0:
                assert time.monotonic() < deadline, f"the device did not come back: {rx.mic_device.state()}"
                seen += elsewhere()
                await asyncio.sleep(0.05)
            assert seen == [], f"the microphone's stream was on {sorted(set(seen))}"
            lost = next(r for r in rx.log.records if r["event"] == "mic_device_lost")
            print(f"the sink was taken away: {lost['error']}")
            assert len(ours()) == 2 and elsewhere() == []
            await wait_for(lambda: rx.mic_device.state()["flowing"], timeout=10)
        finally:
            await rx.stop()

    try:
        asyncio.run(run())
    finally:
        left = ours()
        for line in left:
            subprocess.run(["pactl", "unload-module", line.split()[0]], check=False, timeout=10)
    assert left == []
