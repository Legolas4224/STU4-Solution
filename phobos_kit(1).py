"""
phobos_kit.py - the simplest version of everything we built, in one readable file.

Read it top to bottom. Each part uses only the parts above it.

    1. TRACKING     Rock, Tracker       remember each rock, estimate its velocity
    2. AVOIDANCE    clearance, Avoider  pick a velocity that stays clear of the rocks
    3. FUEL         BurnGovernor        don't fire thrusters for tiny velocity errors
    4. GOAL         GoalPlanner         any path that ends inside the goal sphere is good enough
    5. LIVE MAP     LiveMap             top-down matplotlib view that updates live
    6. CAMERA       project_to_image... draw radar contacts onto a camera image (optional)

Conventions used everywhere:
  * Positions and velocities are in the Phobos frame, in metres and metres/second.
  * "Radar item" = one 3x3 array from GetNearestDebrisLocations():
        item[0, :]  rock centre RELATIVE to the shuttle (m)
        item[1, 0]  distance to its surface     item[1, 1]  range rate (m/s)
        item[1, 2]  diameter (m)                item[2, 0]  time of measurement (s since sim start)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque


# =====================================================================================
# 1. TRACKING - the radar forgets everything between readings, so we keep a list of rocks
# =====================================================================================

class Rock:
    def __init__(self, position, diameter, time):
        self.position = position          # latest ABSOLUTE position
        self.diameter = diameter
        self.velocity = None              # unknown until we have watched it for a while
        self.last_seen = time
        self.history = deque(maxlen=30)   # recent (time, position) samples; capped so memory can't grow
        self.history.append((time, position))

    def add_sighting(self, position, diameter, time):
        if time <= self.last_seen:        # the radar repeated an old reading: ignore it
            return
        self.position = position
        self.diameter = 0.8 * self.diameter + 0.2 * diameter     # smooth the noisy diameter
        self.last_seen = time
        self.history.append((time, position))
        self.update_velocity()

    def update_velocity(self):
        times = np.array([t for t, p in self.history])
        points = np.array([p for t, p in self.history])
        if len(times) < 3 or times[-1] - times[0] < 1.0:
            return                        # too little data: a velocity now would be mostly noise
        # Fit a straight line through the positions. Its slope (per axis) is the velocity.
        self.velocity = np.polyfit(times - times[0], points, 1)[0]

    def predicted_position(self, time):
        if self.velocity is None:
            return self.position
        return self.position + self.velocity * (time - self.last_seen)


class Tracker:
    def __init__(self, gate=15.0, max_age=5.0):
        self.rocks = []
        self.gate = gate          # sightings closer than this (m) count as the same rock
        self.max_age = max_age    # forget rocks not seen for this long (s)

    def update(self, radar_items, shuttle_position, now):
        for item in radar_items:
            # The radar gives positions relative to the shuttle. Add the shuttle's own
            # position so that all sightings share one fixed frame.
            position = shuttle_position + item[0, :]
            diameter, time = item[1, 2], item[2, 0]
            rock = self.find_match(position, diameter, time)
            if rock is None:
                self.rocks.append(Rock(position, diameter, time))    # new rock
            else:
                rock.add_sighting(position, diameter, time)
        self.rocks = [r for r in self.rocks if now - r.last_seen < self.max_age]

    def find_match(self, position, diameter, time):
        """The closest existing rock that is a similar size and within the gate, else None."""
        best, best_distance = None, self.gate
        for rock in self.rocks:
            if abs(rock.diameter - diameter) > 0.25 * rock.diameter:
                continue                                              # sizes too different
            distance = np.linalg.norm(rock.predicted_position(time) - position)
            if distance < best_distance:
                best, best_distance = rock, distance
        return best


# =====================================================================================
# 2. AVOIDANCE - choose the velocity to fly
# =====================================================================================

def clearance(my_velocity, rel_pos, rock_vel, keep_out, horizon=15.0):
    """
    If we fly at my_velocity, how close do we get to the worst rock in the next `horizon` seconds,
    after subtracting its keep-out radius? Positive = safe, negative = collision course.
      rel_pos  (n,3)  where each rock is, relative to us
      rock_vel (n,3)  how fast each rock is moving
      keep_out (n,)   radius we want to stay outside (rock radius + safety margin)
    """
    rel_vel = rock_vel - my_velocity                       # how each rock moves as seen from us
    speed_squared = (rel_vel ** 2).sum(axis=1) + 1e-9
    # time when the rock is closest to us (from minimising |rel_pos + t * rel_vel|)
    t = -(rel_pos * rel_vel).sum(axis=1) / speed_squared
    t = np.clip(t, 0.0, horizon)                           # only look from now until the horizon
    closest = np.linalg.norm(rel_pos + t[:, None] * rel_vel, axis=1)
    return (closest - keep_out).min()


def two_sideways_directions(direction):
    """Two unit vectors that are perpendicular to `direction` and to each other."""
    helper = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    side1 = np.cross(direction, helper)
    side1 /= np.linalg.norm(side1)
    side2 = np.cross(direction, side1)
    return side1, side2


class Avoider:
    def __init__(self, margin=6.0, unknown_extra=4.0, horizon=15.0):
        self.margin = margin                # metres of clearance beyond each rock's surface
        self.unknown_extra = unknown_extra  # extra margin for rocks whose velocity we don't know yet
        self.horizon = horizon
        self.last_dodge = None              # remembered so we don't flip-flop between dodges

    def choose(self, wanted_velocity, rocks, shuttle_position, now):
        """Return the velocity to fly: the wanted one if it is safe, else the smallest safe change."""
        wanted_velocity = np.asarray(wanted_velocity, float)
        speed = np.linalg.norm(wanted_velocity)
        if speed < 1e-3 or len(rocks) == 0:
            return wanted_velocity

        # Describe every rock as plain arrays.
        rel_pos = np.array([r.predicted_position(now) - shuttle_position for r in rocks])
        rock_vel = np.array([r.velocity if r.velocity is not None else np.zeros(3) for r in rocks])
        keep_out = np.array([r.diameter / 2 + self.margin
                             + (self.unknown_extra if r.velocity is None else 0.0) for r in rocks])

        def score(velocity):
            return clearance(velocity, rel_pos, rock_vel, keep_out, self.horizon)

        # 1. Nothing in the way? Fly exactly as wanted (no thruster use at all).
        if score(wanted_velocity) > 0:
            self.last_dodge = None
            return wanted_velocity

        # 2. Still dodging? Keep the same dodge while it still works (stops flip-flopping).
        if self.last_dodge is not None and score(self.last_dodge) > 0:
            return self.last_dodge

        # 3. Try sideways dodges, SMALLEST first: a small velocity change costs the least fuel.
        side1, side2 = two_sideways_directions(wanted_velocity / speed)
        tried = [wanted_velocity]
        for size in (0.05, 0.1, 0.2, 0.35, 0.5, 0.75):         # as a fraction of our speed
            for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                sideways = np.cos(angle) * side1 + np.sin(angle) * side2
                candidate = wanted_velocity + size * speed * sideways
                if score(candidate) > 0:
                    self.last_dodge = candidate
                    return candidate
                tried.append(candidate)

        # 4. No sideways dodge works: try slowing down.
        for fraction in (0.5, 0.25, 0.0):
            candidate = wanted_velocity * fraction
            if score(candidate) > 0:
                self.last_dodge = candidate
                return candidate
            tried.append(candidate)

        # 5. Nothing is fully safe: take the least-bad option we looked at.
        self.last_dodge = max(tried, key=score)
        return self.last_dodge


# =====================================================================================
# 3. FUEL - only fire the thrusters when the velocity error is worth a burn
# =====================================================================================

class BurnGovernor:
    def __init__(self, gain=5.0, start=0.3, stop=0.1):
        self.gain = gain        # acceleration (m/s^2) per m/s of velocity error
        self.start = start      # start burning when the error is bigger than this (m/s)
        self.stop = stop        # stop burning once the error is smaller than this (m/s)
        self.burning = False

    def acceleration(self, wanted_velocity, current_velocity):
        error = np.asarray(wanted_velocity, float) - np.asarray(current_velocity, float)
        size = np.linalg.norm(error)
        if self.burning and size < self.stop:
            self.burning = False
        elif not self.burning and size > self.start:
            self.burning = True
        return self.gain * error if self.burning else np.zeros(3)


# =====================================================================================
# 4. GOAL - the target is a SPHERE, so we only steer when our path would miss it
# =====================================================================================

class GoalPlanner:
    """
    Decides the velocity we WANT (before avoidance). Instead of constantly steering at the exact
    centre of the target, it lets us coast while our current path already passes well inside the
    goal sphere, and only steers when the path drifts towards the edge. Less steering = less fuel.
    """

    def __init__(self, cruise_speed, goal_radius=50.0, start_fraction=0.7, stop_fraction=0.3):
        self.cruise_speed = cruise_speed
        self.goal_radius = goal_radius
        self.start = start_fraction * goal_radius   # start steering if we would miss the centre by more than this
        self.stop = stop_fraction * goal_radius     # ...and keep steering until the miss is under this
        self.correcting = True

    def miss_distance(self, shuttle_position, shuttle_velocity, target_position):
        """How far from the target centre we would pass if we kept our current velocity."""
        speed = np.linalg.norm(shuttle_velocity)
        if speed < 0.98 * self.cruise_speed:
            return np.inf                                   # not at cruise speed yet: keep accelerating
        heading = shuttle_velocity / speed
        to_target = target_position - shuttle_position
        along = to_target @ heading                         # how far ahead the target is along our heading
        if along <= 0:
            return np.inf                                   # we are moving away from it
        return np.linalg.norm(to_target - along * heading)  # sideways distance at closest approach

    def wanted_velocity(self, shuttle_position, shuttle_velocity, target_position):
        miss = self.miss_distance(shuttle_position, shuttle_velocity, target_position)
        if self.correcting and miss < self.stop:
            self.correcting = False                         # on a good path: coast
        elif not self.correcting and miss > self.start:
            self.correcting = True                          # drifting off: steer again
        if not self.correcting:
            return np.asarray(shuttle_velocity, float)      # wanting what we have = no thruster use
        to_target = target_position - shuttle_position
        return self.cruise_speed * to_target / np.linalg.norm(to_target)


# =====================================================================================
# 5. LIVE MAP - top-down view (X right, Y up) centred on the shuttle
# =====================================================================================

class LiveMap:
    def __init__(self, view=100.0, every=5):
        self.view = view            # half-width of the map (m)
        self.every = every          # redraw every N calls (drawing is slow)
        self.calls = 0
        plt.ion()
        self.fig = plt.figure("Live rock map")      # same name every time = same window
        self.ax = self.fig.add_subplot(111)

    def draw(self, rocks, shuttle_position, target_position=None, goal_radius=None):
        self.calls += 1
        if self.calls % self.every:
            return
        ax = self.ax
        ax.clear()
        for rock in rocks:
            colour = "tab:gray" if rock.velocity is None else "tab:red"   # gray = velocity not known yet
            ax.add_patch(plt.Circle(rock.position[:2], rock.diameter / 2, color=colour, alpha=0.5))
            if rock.velocity is not None:                                 # arrow = 5 seconds of motion
                ax.arrow(rock.position[0], rock.position[1],
                         5 * rock.velocity[0], 5 * rock.velocity[1], head_width=2, color="black")
        ax.plot(shuttle_position[0], shuttle_position[1], "k^", markersize=10)
        if target_position is not None:
            ax.plot(target_position[0], target_position[1], "g*", markersize=14)
            if goal_radius is not None:                                   # the goal sphere, seen from above
                ax.add_patch(plt.Circle(target_position[:2], goal_radius, fill=False, color="green"))
        ax.set_xlim(shuttle_position[0] - self.view, shuttle_position[0] + self.view)
        ax.set_ylim(shuttle_position[1] - self.view, shuttle_position[1] + self.view)
        ax.set_aspect("equal")
        ax.set_title(f"{len(rocks)} tracked rocks")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# =====================================================================================
# 6. CAMERA (optional) - where would a rock appear in the camera image?
#    Uses the wiki's camera model: yaw turns the camera LEFT, pitch turns it DOWN.
#    Assumes image-right = shuttle -Y and FOV is the horizontal field of view; check
#    this against a real image (the wiki doesn't say).
# =====================================================================================

CAMERA_POSITION = {"top": np.array([1.27, 0.0, 0.9]), "bottom": np.array([1.675, 0.0, -1.025])}


def focal_length_in_pixels(fov_deg, image_size):
    return (image_size / 2) / np.tan(np.radians(fov_deg) / 2)


def project_to_image(point_in_body, camera, yaw_deg, pitch_deg, fov_deg, image_size):
    """Pixel (x, y) and depth of a point given relative to the shuttle in the BODY frame, or None."""
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    # the camera's three axes, written in body coordinates
    forward = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)])
    left = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    up = np.cross(forward, left)

    offset = np.asarray(point_in_body, float) - CAMERA_POSITION[camera]
    depth = offset @ forward
    if depth <= 0.1:
        return None                                   # behind the camera
    f = focal_length_in_pixels(fov_deg, image_size)
    x = image_size / 2 - f * (offset @ left) / depth  # left of centre -> smaller x
    y = image_size / 2 - f * (offset @ up) / depth    # above centre   -> smaller y
    return x, y, depth


def draw_rocks_on_image(image, rocks_in_body, camera, yaw_deg, pitch_deg, fov_deg):
    """rocks_in_body: list of (position relative to shuttle in BODY frame, diameter). image: square PIL image."""
    from PIL import ImageDraw
    marked = image.copy()
    pen = ImageDraw.Draw(marked)
    size = marked.size[0]
    f = focal_length_in_pixels(fov_deg, size)
    for position, diameter in rocks_in_body:
        hit = project_to_image(position, camera, yaw_deg, pitch_deg, fov_deg, size)
        if hit is None:
            continue
        x, y, depth = hit
        radius = max(f * (diameter / 2) / depth, 2.0)
        pen.ellipse([x - radius, y - radius, x + radius, y + radius], outline=(255, 60, 60), width=2)
        pen.text((x + radius + 2, y - radius), f"{depth:.0f}m", fill=(255, 220, 0))
    return marked


# =====================================================================================
# HOW TO USE IN Controller.py  (this is the whole control loop of the "smart" part)
# =====================================================================================
#
#   from phobos_kit import Tracker, Avoider, BurnGovernor, GoalPlanner, LiveMap
#
#   GOAL_RADIUS = 50.0                  # metres - CHECK this against the real safe-area size
#   tracker  = Tracker()                # before the while loop
#   avoider  = Avoider()
#   governor = BurnGovernor(gain=5.0)
#   planner  = GoalPlanner(cruise_speed=20.0, goal_radius=GOAL_RADIUS)
#   live_map = LiveMap(view=100.0, every=5)
#
#   # inside the loop, after you have loc, vel and t_from_start:
#   tracker.update(GetNearestDebrisLocations(50), loc, t_from_start)
#
#   wanted_velocity = planner.wanted_velocity(loc, vel, target_loc)   # steer only if we'd miss the sphere
#   safe_velocity = avoider.choose(wanted_velocity, tracker.rocks, loc, t_from_start)
#   accel = governor.acceleration(safe_velocity, vel)                # Phobos frame
#   accel = st.frames.FramedVector(accel, phobosFrame).ExprIn(shuttleFrame)
#   SetCommand(accel, angularAcceleration_rad_s2)
#
#   live_map.draw(tracker.rocks, loc, target_loc, goal_radius=GOAL_RADIUS)
#
# Logging helper and version stamp (put near the top of the user-code section):
#
#   LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "controller_log.txt")
#   log_file = open(LOG_PATH, "w", buffering=1)
#   def status(message, key="Status"):
#       print(message, flush=True)
#       st.OnScreenLogMessage(message, key, st.Severity.Info)
#       log_file.write(message + "\n")
#
#   import subprocess
#   try:
#       GIT_REV = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
#                     cwd=os.path.dirname(os.path.abspath(__file__)), text=True).strip()
#   except Exception:
#       GIT_REV = "unknown"
#   status(f"git rev {GIT_REV}")
#
# Camera overlay (inside your existing "if image is not None:" block):
#
#   def to_body(v):      # Phobos frame -> shuttle body frame
#       return np.asarray(st.frames.FramedVector(np.asarray(v, float), phobosFrame)
#                           .ExprIn(shuttleFrame), dtype=float)
#   rocks_in_body = [(to_body(r.position - loc), r.diameter) for r in tracker.rocks]
#   draw_rocks_on_image(imageRGB, rocks_in_body, "top", 30.0, -10.0, cameraParams.FOV).save("cam.png")
