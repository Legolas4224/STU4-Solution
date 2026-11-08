import numpy as np
import rock_tracker as rt
import random

r = lambda: random.uniform(1.0, 10.0)

radar_items = [
    np.array([[r(), r(), r()], [r(), r(), r()], [r(), r(), r()]]) for _ in range(500)
]
tracker = rt.Tracker()
loc = np.array([0.0, 0.0, 0.0])
tracker.update(radar_items, loc, 0.0)

for r in tracker.rocks:
    print(f"{r.position}")
r = tracker.rocks[0]
print(type(r.position), type(r.velocity), np.asarray(r.position, float))
