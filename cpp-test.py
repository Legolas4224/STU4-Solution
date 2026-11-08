import numpy as np
import rock_tracker as rt

radar_items = []
radar_items.append(np.array(
    [[1.5, 2.6, 3.7], [4.0, 5.0, 6.0], [7.0, 0.0, 0.0]]
))
tracker = rt.Tracker()
loc = np.array([0.0, 0.0, 0.0])
tracker.update(radar_items, loc, 0.0)

r = tracker.rocks[0]
print(type(r.position), type(r.velocity), np.asarray(r.position, float))
