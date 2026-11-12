import random
import numpy as np
import rock_tracker as rt
from dataclasses import dataclass
import matplotlib.pyplot as plt
import random

r = lambda: random.uniform(1.0, 10.0)

matx = lambda n=10: [
    np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]) for _ in range(n)
]


@dataclass
class RadarItems:
    items = [
        np.array([[r(), r(), r()], [r(), r(), r()], [r(), 0.0, 0.0]]) for _ in range(50)
    ]
    vel = np.array([r(), r(), r()])

    def step(self, dt: float):
        self.items[0] += self.vel * dt


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


cone = rt.Cone()
cone.reset(np.zeros(3), np.array([1000.0, 1000.0, 0.0]))
cone.set_angle_degrees(20.0)

sum = 0
for i, r in enumerate(tracker.rocks):
    in_cone = cone.check_collision(r.position)
    if in_cone:
        sum += 1
    print(f"[{i}]{in_cone}")
print(f"sum in cone: {sum}")
r = tracker.rocks[0]
print(type(r.position), type(r.velocity), np.asarray(r.position, float))


# PLOT
fig = plt.figure(figsize=(12, 12))
ax = plt.axes(projection="3d")

xs = [r.position[0] for r in tracker.rocks]
ys = [r.position[1] for r in tracker.rocks]
zs = [r.position[2] for r in tracker.rocks]

ax.scatter(xs, ys, zs)
plt.show()
