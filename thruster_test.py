"""
thruster_test.py - measure the acceleration the shuttle ACTUALLY achieves when you command it.

For each body axis (X, Y, Z) and each test magnitude it commands +a for a few seconds, then -a
for the same time (so the shuttle ends roughly where it started), and reports the acceleration
it measured from the change in velocity.

How to read the report:
  * achieved ~ commanded at small magnitudes, then stops growing  -> that axis is capped at that value
  * achieved far below commanded even for small commands          -> something scales your commands
  * "off-axis" is large                                           -> commanded direction isn't what you get
  * very different caps on different axes                         -> thrust depends on direction; you may
                                                                     need to rotate the shuttle to aim its
                                                                     strong axis where you want to go

Use it INSTEAD of your normal control code for one run. In Controller.py, before the loop:

    from thruster_test import ThrusterTest
    test = ThrusterTest()
    def to_body(v):      # Phobos frame -> shuttle body frame
        return np.asarray(st.frames.FramedVector(np.asarray(v, float), phobosFrame)
                            .ExprIn(shuttleFrame), dtype=float)

and in the loop, in place of the avoider / governor / SetCommand lines:

    accel_body, report = test.command(t_from_start, vel, to_body)
    if report:
        status(report)
    SetCommand(accel_body, np.zeros(3))      # note: already in the body frame, no conversion

Also prints how fast the sim clock runs compared with the wall clock, in case the sim itself is slow.
"""

import time
import numpy as np

AXIS_NAMES = ["X", "Y", "Z"]


class ThrusterTest:
    def __init__(self, magnitudes=(1.0, 5.0, 20.0, 100.0), seconds=2.0):
        # Plan: for every axis and magnitude, push one way then the other.
        self.plan = [(axis, sign * a) for axis in range(3) for a in magnitudes for sign in (+1, -1)]
        self.seconds = seconds
        self.step = 0
        self.step_start_time = None
        self.step_start_velocity = None
        self.wall_start = None
        self.sim_start = None
        self.finished = False

    def command(self, sim_time, velocity, to_body):
        """Returns (acceleration command in the BODY frame, report text or None)."""
        if self.finished:
            return np.zeros(3), None
        velocity = np.asarray(velocity, float)
        report = None

        if self.step_start_time is None:                       # first call: start the first step
            self.step_start_time, self.step_start_velocity = sim_time, velocity.copy()
            self.wall_start, self.sim_start = time.time(), sim_time

        elapsed = sim_time - self.step_start_time
        if elapsed >= self.seconds:                            # step finished: measure and report
            axis, commanded = self.plan[self.step]
            achieved = to_body(velocity - self.step_start_velocity) / elapsed   # m/s^2 per body axis
            off_axis = np.linalg.norm(np.delete(achieved, axis))
            report = (f"cmd {AXIS_NAMES[axis]}{commanded:+8.1f} m/s^2 -> achieved "
                      f"{achieved[axis]:+8.3f} m/s^2  (off-axis {off_axis:6.3f})")
            self.step += 1
            if self.step >= len(self.plan):
                self.finished = True
                sim_elapsed = sim_time - self.sim_start
                wall_elapsed = max(time.time() - self.wall_start, 1e-9)
                report += f"\nTEST DONE. Sim clock ran at {sim_elapsed / wall_elapsed:.2f}x real time."
                return np.zeros(3), report
            self.step_start_time, self.step_start_velocity = sim_time, velocity.copy()

        axis, commanded = self.plan[self.step]
        accel = np.zeros(3)
        accel[axis] = commanded
        return accel, report
