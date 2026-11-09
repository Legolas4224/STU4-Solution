import random
import numpy as np
import rock_tracker as rt
from dataclasses import dataclass

<<<<<<< Updated upstream
r = lambda: random.uniform(1.0, 10.0)

radar_items = [
    np.array([[r(), r(), r()], [r(), r(), r()], [r(), r(), r()]]) for _ in range(500)
]
tracker = rt.Tracker()
loc = np.array([0.0, 0.0, 0.0])
tracker.update(radar_items, loc, 0.0)
=======
r = lambda : random.uniform(1.0, 10.0)
matx = lambda n=10: [np.array([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]) for _ in range(n)]

@dataclass 
class RadarItems:
    items = [np.array([[r(), r(), r()],[r(), r(), r()],[r(), 0.0, 0.0]]) for _ in range(50)]
    vel = np.array([r(), r(), r()])
>>>>>>> Stashed changes

    def step(self, dt: float):
        self.items[0] += (self.vel * dt)


def check(name, ok):
    print(("PASS" if ok else "FAIL"), "-", name)

# --- Tracker: a rock moving at (-2, 0, 0) m/s should be tracked as ONE rock with that velocity
tracker, shuttle = rt.Tracker(), np.zeros(3)
radar_items = RadarItems()

for i in range(11):
    t = i * 0.5
    world = np.array([100.0, 0, 0]) + np.array([-2.0, 0, 0]) * t
    tracker.update(radar_items.items, shuttle, t)
    radar_items.step(t)
rock = tracker.rocks[0]
check("tracker keeps one rock", len(tracker.rocks) == 1)
check("tracker estimates velocity", rock.velocity is not None and np.allclose(rock.velocity, [-2, 0, 0], atol=0.1))

# --- Tracker: rocks not seen for max_age seconds are forgotten
tracker.update(radar_items.items, shuttle, 5.0 + 20.0)
check("tracker forgets old rocks", len(tracker.rocks) == 0)

# --- Avoider: a rock dead ahead forces a dodge; a rock far off to the side does not
avoider, wanted = rt.Avoider(), np.array([8.0, 0, 0])
blocker = rt.Rock(np.array([50.0, 0, 0]), 10.0, 0.0)
check("avoider dodges blocker", not np.allclose(avoider.choose(wanted, [blocker], shuttle, 0.0), wanted))
far_rock = rt.Rock(np.array([50.0, 500.0, 0]), 10.0, 0.0)
check("avoider ignores far rock", np.allclose(rt.Avoider().choose(wanted, [far_rock], shuttle, 0.0), wanted))

# --- BurnGovernor: tiny error = no burn, big error = burn
gov = rt.BurnGovernor(gain=5.0)
check("governor coasts on small error", np.allclose(gov.acceleration([1.05, 0, 0], [1, 0, 0]), 0))
check("governor burns on big error", np.linalg.norm(gov.acceleration([5, 0, 0], [0, 0, 0])) > 0)

# --- GoalPlanner: from rest it should head at the target at cruise speed
planner = rt.GoalPlanner(cruise_speed=8.0)
v = planner.wanted_velocity(np.zeros(3), np.zeros(3), np.array([1000.0, 0, 0]))
check("planner heads for target at cruise speed", np.allclose(v, [8, 0, 0]))