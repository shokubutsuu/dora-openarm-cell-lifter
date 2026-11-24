# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import types

import pytest


import dora_openarm_cell_lifter.main as main_module  # noqa: E402


@pytest.mark.parametrize(
    ("joystick_y", "expected_elevation", "expected_velocity"),
    [
        (1.0, 28.8732414637843, 30.0),
        (-1.0, -18.8732414637843, 30.0),
    ],
)
def test_joystick_y_produces_expected_elevation_action(
    joystick_y, expected_elevation, expected_velocity
):
    action_elevation, applied_vel = main_module._calc_elevation_action_from_joystick(
        current_elevation=5.0,
        joystick_y=joystick_y,
        dt=1.0,
        lead_length=5.0,
    )

    assert action_elevation == pytest.approx(expected_elevation)
    assert applied_vel == pytest.approx(expected_velocity)


@pytest.mark.parametrize("joystick_y", [-0.15, 0.0, 0.15])
def test_deadzone_joystick_y_does_not_produce_elevation_action(joystick_y):
    action_elevation, applied_vel = main_module._calc_elevation_action_from_joystick(
        current_elevation=5.0,
        joystick_y=joystick_y,
        dt=1.0,
        lead_length=5.0,
    )

    assert action_elevation is None
    assert applied_vel == 0.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("off", None),
        ("", None),
        (None, None),
        ("0", 0.0),
        ("120.5", 120.5),
        ("-10", 0.0),
        ("999", 300.0),
    ],
)
def test_start_elevation_is_off_or_clamped_to_the_stroke(value, expected):
    assert main_module._start_elevation(value, screw_length=300.0) == expected


def test_start_elevation_rejects_other_text():
    with pytest.raises(ValueError):
        main_module._start_elevation("bottom", screw_length=300.0)


def test_bottom_ignores_the_reversal_spike_after_the_runway():
    # The two runs that calibrated at the runway position: one ~1 Nm sample
    # 8 ms into the descent.
    bottom = main_module.BottomDetector(started=0.0)
    assert not bottom.update(0.004, -0.24)
    assert not bottom.update(0.008, -1.02)


def test_bottom_needs_the_torque_to_stay_over_the_limit():
    bottom = main_module.BottomDetector(started=0.0)
    t = main_module.CALIB_SETTLE_S
    samples = [1.0] * (main_module.CALIB_CONFIRM_SAMPLES - 1) + [0.1] + [1.0] * main_module.CALIB_CONFIRM_SAMPLES
    results = [bottom.update(t + i * 0.004, tau) for i, tau in enumerate(samples)]
    assert results.index(True) == len(samples) - 1


def test_bottom_reached_during_the_settle_time_is_taken_after_it():
    # Starting at the bottom, the short descent can hit it within the settle
    # time; it keeps pushing, so it is taken once the settle time is over.
    bottom = main_module.BottomDetector(started=0.0)
    times = [i * 0.004 for i in range(200)]
    first = next(t for t in times if bottom.update(t, -1.2))
    assert main_module.CALIB_SETTLE_S <= first < main_module.CALIB_SETTLE_S + 0.05


def test_unwrapper_follows_a_wrap_of_the_feedback():
    unwrapper = main_module.PositionUnwrapper()
    unwrapper.update(-12.4)
    assert unwrapper.update(12.4) == pytest.approx(-12.6)
    assert unwrapper.update(-12.4) == pytest.approx(-12.4)


class _StillLifter:
    """A lifter whose motor reports the same position and no torque."""

    def __init__(self):
        self.motor = types.SimpleNamespace(
            get_position=lambda: 1.0, get_torque=lambda: 0.0
        )

    def get_arm(self):
        return types.SimpleNamespace(
            posvel_control_all=lambda params: None, get_motors=lambda: [self.motor]
        )

    def recv_all(self):
        pass


def test_homing_gives_up_when_the_lifter_does_not_move(monkeypatch, capsys):
    clock = [0.0]
    monkeypatch.setattr(main_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        main_module.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s)
    )
    args = types.SimpleNamespace(screw_length=300.0, lead_length=5.0, debug=False)
    assert main_module._home(_StillLifter(), args) is False
    assert clock[0] == pytest.approx(main_module.HOMING_STALL_S, abs=0.01)
    assert "not moving" in capsys.readouterr().out
