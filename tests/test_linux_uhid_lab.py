"""The Linux Stadia pad against the real kernel: the device uhid creates from the report
descriptor, read back through its hidraw node the way SDL reads it and through its event node the
way a game reads it, rumble written to the hidraw node the way SDL writes it, and a game's force
feedback through the kernel's hid-google-stadiaff driver. Runs on Linux when STADIACAST_LINUX_LAB
is set, with the right to open ``/dev/uhid`` and the pad's nodes (as root, or with the udev rule
of the README); a Stadia Controller appears on the system while one runs."""

from __future__ import annotations

import glob
import os
import sys
import time
from collections.abc import Iterator

import pytest

from stadiacast.gamepad import Button, GamepadState
from stadiacast.uhid_gamepad import PAD_NAME, UhidPad, input_report
from tests.test_linux_lab import Game, wait_until


def _uhid_opens() -> bool:
    try:
        os.close(os.open("/dev/uhid", os.O_RDWR))
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or not os.environ.get("STADIACAST_LINUX_LAB") or not _uhid_opens(),
    reason="needs Linux, the right to open /dev/uhid and STADIACAST_LINUX_LAB=1",
)

EV_KEY, EV_ABS, EV_FF = 0x01, 0x03, 0x15
BTN_A, BTN_B, BTN_X, BTN_Y, BTN_TL, BTN_TR = 0x130, 0x131, 0x133, 0x134, 0x136, 0x137
BTN_SELECT, BTN_START, BTN_MODE, BTN_THUMBL, BTN_THUMBR = 0x13A, 0x13B, 0x13C, 0x13D, 0x13E
BTN_TRIGGER_HAPPY1, BTN_TRIGGER_HAPPY2, BTN_TRIGGER_HAPPY3, BTN_TRIGGER_HAPPY4 = 0x2C0, 0x2C1, 0x2C2, 0x2C3
ABS_X, ABS_Y, ABS_Z, ABS_RZ, ABS_GAS, ABS_BRAKE, ABS_HAT0X, ABS_HAT0Y = 0, 1, 2, 5, 9, 10, 0x10, 0x11
#: What the kernel makes of a wired Stadia Controller, less the three media keys of its headset
#: socket: the keys in the order programs number them, and the axes.
KEYS = [BTN_A, BTN_B, BTN_X, BTN_Y, BTN_TL, BTN_TR, BTN_SELECT, BTN_START, BTN_MODE, BTN_THUMBL, BTN_THUMBR]
KEYS += [BTN_TRIGGER_HAPPY1, BTN_TRIGGER_HAPPY2, BTN_TRIGGER_HAPPY3, BTN_TRIGGER_HAPPY4]
AXES = {ABS_X, ABS_Y, ABS_Z, ABS_RZ, ABS_GAS, ABS_BRAKE, ABS_HAT0X, ABS_HAT0Y}


@pytest.fixture
def pad() -> Iterator[UhidPad]:
    made = UhidPad("lab")
    try:
        yield made
    finally:
        made.close()


def _event_node(hidraw: str) -> str:
    """The event node of the HID device behind a hidraw node."""
    name = os.path.basename(hidraw)
    found: list[str] = []
    wait_until(lambda: bool(found.extend(glob.glob(f"/sys/class/hidraw/{name}/device/input/input*/event*")) or found))
    return f"/dev/input/{os.path.basename(min(found))}"


def _open_hidraw(path: str) -> int:
    deadline = time.monotonic() + 3
    while True:  # udev sets the node's permissions a moment after the kernel made it
        try:
            return os.open(path, os.O_RDWR | os.O_NONBLOCK)
        except PermissionError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)


def _driver(hidraw: str) -> str:
    """The HID driver bound to the pad: ``stadia`` (hid-google-stadiaff) where the kernel has it."""
    return os.path.basename(os.path.realpath(f"/sys/class/hidraw/{os.path.basename(hidraw)}/device/driver"))


def test_the_kernel_takes_the_pad_for_a_stadia_controller(pad: UhidPad) -> None:
    assert pad.device is not None and os.path.exists(pad.device)
    game = Game(_event_node(pad.device))
    try:
        assert game.identity() == (0x03, 0x18D1, 0x9400, 0x0100)
        assert game.name() == PAD_NAME
        assert sorted(game.codes(EV_KEY)) == KEYS  # one more or one less would renumber the buttons
        assert game.codes(EV_ABS) == AXES
        for stick in (ABS_X, ABS_Y, ABS_Z, ABS_RZ):
            assert game.axis(stick)[1:3] == (1, 255)
        for trigger in (ABS_GAS, ABS_BRAKE):
            assert game.axis(trigger)[1:3] == (0, 255)
        assert game.axis(ABS_HAT0X)[1:3] == game.axis(ABS_HAT0Y)[1:3] == (-1, 1)
        driver = _driver(pad.device)
        assert driver in ("stadia", "hid-generic")
        assert (EV_FF in game.codes(0)) == (driver == "stadia")  # force feedback comes with the kernel's driver
    finally:
        game.close()


def test_input_reaches_the_hidraw_node_as_the_report_and_the_event_node_as_keys(pad: UhidPad) -> None:
    assert pad.device is not None
    raw = _open_hidraw(pad.device)
    game = Game(_event_node(pad.device))
    try:
        game.events()
        state = GamepadState(buttons=int(Button.ASSISTANT | Button.CAPTURE | Button.A | Button.X), lx=32767, l2=255)
        pad.apply_state(state)
        reports: list[bytes] = []

        def read() -> bool:
            try:
                reports.append(os.read(raw, 64))
            except BlockingIOError:
                pass
            return bool(reports)

        wait_until(read)
        assert reports[0] == input_report(state)  # what SDL's Stadia driver reads
        events: list[tuple[int, int, int]] = []
        wait_until(lambda: bool(events.extend(game.events()) or (EV_KEY, BTN_TRIGGER_HAPPY1, 1) in events))
        pressed = {code for kind, code, value in events if kind == EV_KEY and value == 1}
        assert pressed == {BTN_TRIGGER_HAPPY1, BTN_TRIGGER_HAPPY2, BTN_A, BTN_X, BTN_TRIGGER_HAPPY4}  # 4: L2 held
        assert (EV_ABS, ABS_X, 255) in events and (EV_ABS, ABS_BRAKE, 255) in events
    finally:
        game.close()
        os.close(raw)


def test_a_hidraw_write_of_report_5_is_rumble(pad: UhidPad) -> None:
    """What SDL does to rumble a Stadia Controller."""
    assert pad.device is not None
    levels: list[tuple[int, int]] = []
    pad.on_rumble(lambda strong, weak: levels.append((strong, weak)))
    raw = _open_hidraw(pad.device)
    try:
        assert os.write(raw, b"\x05\x00\x80\x00\x40") == 5
        wait_until(lambda: (0x80, 0x40) in levels)
        os.write(raw, b"\x05\x00\x00\x00\x00")
        wait_until(lambda: levels[-1] == (0, 0))
    finally:
        os.close(raw)


def test_a_games_force_feedback_reaches_the_motors_through_the_kernels_driver(pad: UhidPad) -> None:
    """A game that rumbles through the event node: hid-google-stadiaff turns the effect into a
    SET_REPORT of report 5, which the pad answers."""
    assert pad.device is not None
    if _driver(pad.device) != "stadia":
        pytest.skip("hid-google-stadiaff did not take the pad (not built, or not loaded)")
    levels: list[tuple[int, int]] = []
    pad.on_rumble(lambda strong, weak: levels.append((strong, weak)))
    game = Game(_event_node(pad.device))
    try:
        slot = game.upload(0x8000, 0x4000, 0)
        game.play(slot)
        wait_until(lambda: (0x80, 0x40) in levels)
        game.play(slot, 0)
        wait_until(lambda: levels[-1] == (0, 0))
    finally:
        game.close()
