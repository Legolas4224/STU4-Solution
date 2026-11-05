"""
Kalman filter / tracking playground  (pygame-ce + numpy)

Install:  pip install pygame-ce numpy

What this gives you
-------------------
* A hidden "true" object that wanders around the screen (random acceleration, bounces off walls).
* A noisy sensor that measures its (x, y) position at a low rate, with optional dropouts and outliers.
* A Tracker class that is a deliberately dumb stub. YOUR JOB: replace its guts with a Kalman filter.

The tracker only ever sees the measurements (never the truth), exactly like real life.
The truth is drawn so you can see how well you're doing, and RMSE is shown live.

Units: pixels and seconds.

Controls
--------
SPACE  pause           R  reset
UP/DOWN     sensor noise sigma +/-
LEFT/RIGHT  sensor rate (Hz) -/+
A / Z       object manoeuvrability (random accel sigma) +/-
D  toggle dropouts     O  toggle outliers (false readings)
T  toggle trails       ESC quit
"""

from collections import deque

import numpy as np
import pygame

W, H = 960, 680
FPS = 60
DT = 1.0 / FPS
rng = np.random.default_rng()


# --------------------------------------------------------------------------
# The world (don't peek at this from the tracker!)
# --------------------------------------------------------------------------
class Target:
    def __init__(self, accel_sigma=120.0):
        self.accel_sigma = accel_sigma      # px/s^2, std dev of random acceleration
        self.min_speed, self.max_speed = 40.0, 260.0
        self.reset()

    def reset(self):
        self.pos = np.array([W / 2, H / 2], dtype=float)
        ang = rng.uniform(0, 2 * np.pi)
        self.vel = 120.0 * np.array([np.cos(ang), np.sin(ang)])

    def step(self, dt):
        # white-noise acceleration: this is the process noise your Q should model
        a = rng.normal(0.0, self.accel_sigma, 2)
        self.vel += a * dt

        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel *= self.max_speed / speed
        elif speed < self.min_speed:
            self.vel = self.vel / (speed + 1e-9) * self.min_speed

        self.pos += self.vel * dt

        # bounce off walls (with a margin)
        m = 20
        for i, lim in enumerate((W, H)):
            if self.pos[i] < m:
                self.pos[i] = m
                self.vel[i] = abs(self.vel[i])
                
            elif self.pos[i] > lim - m:
                self.pos[i] = lim - m
                self.vel[i] = -abs(self.vel[i])
                


class Sensor:
    """Returns a noisy (x, y) position now and then, or None when nothing is detected."""

    def __init__(self):
        self.sigma = 15.0          # px, measurement noise std dev (your R = sigma^2 * I)
        self.rate = 5.0            # Hz
        self.dropouts = False
        self.p_dropout = 0.25
        self.outliers = False
        self.p_outlier = 0.08
        self.timer = 0.0

    def step(self, dt, true_pos):
        self.timer += dt
        period = 1.0 / self.rate
        if self.timer < period:
            return None
        self.timer -= period

        if self.dropouts and rng.random() < self.p_dropout:
            return None

        z = true_pos + rng.normal(0.0, self.sigma, 2)
        if self.outliers and rng.random() < self.p_outlier:
            z = true_pos + rng.normal(0.0, self.sigma * 10, 2)   # wild reading
        return z


# --------------------------------------------------------------------------
# YOUR CODE GOES HERE
# --------------------------------------------------------------------------
class Tracker:
    def __init__(self, sensor_sigma):
        self.sensor_sigma = sensor_sigma
        self.sigma_a = 1000.0              # tuning knob for Q
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        self.reset()

    def reset(self):
        self.x = None
        self.P = None

    def predict(self, dt):
        if self.x is None:
            return
        F = np.eye(4)
        F[0,2] = dt
        F[1,3] = dt

        G = np.array([[dt**2 / 2, 0],
              [0,         dt**2 / 2],
              [dt,        0],
              [0,         dt]])

        Q = self.sigma_a**2 * (G @ G.T)

        self.x = F @ self.x
        print(self.x)
        self.P = F @ self.P @ F.T + Q

        # TODO: build F and Q from dt, then x = F x ; P = F P F^T + Q

    def update(self, z):
        z = np.asarray(z, dtype=float)
        R = np.eye(2) * self.sensor_sigma**2
        if self.x is None:
            self.x = np.array([z[0], z[1], 0, 0])
            self.P = np.diag([self.sensor_sigma**2, self.sensor_sigma**2, 100**2, 100**2])
            # TODO: initialise x and P as in step 2
            return

        y = z - self.H @ self.x                  # innovation (2,)
        S = self.H @ self.P @ self.H.T + R       # innovation covariance (2x2)
        K = self.P @ self.H.T @ np.linalg.inv(S) # gain (4x2)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        # TODO: the five update lines above

    @property
    def position(self): return None if self.x is None else self.x[:2]
    @property
    def velocity(self): return None if self.x is None else self.x[2:]
    @property
    def cov(self): return None if self.P is None else self.P[:2, :2]




class DumbTracker:
    """
    Stub tracker: just believes the latest measurement. Replace with a Kalman filter.

    The sim calls:
        predict(dt)   every frame (1/60 s), whether or not a measurement arrived
        update(z)     only on frames where the sensor produced a reading (z = np.array([x, y]))

    The display reads:
        position   -> np.array([x, y]) current best estimate
        cov        -> 2x2 position covariance (optional). Return None to hide the ellipse.
                      If you fill it in you get a 2-sigma uncertainty ellipse drawn for free.
        velocity   -> np.array([vx, vy]) (optional, drawn as a little arrow)

    Things to try, roughly in order:
        1. Constant-velocity model, state [x, y, vx, vy], measure position only.
        2. Tune Q and R. Watch what happens when Q is too small / too big.
        3. Turn on dropouts (D) and see the filter coast on prediction alone.
        4. Turn on outliers (O) and add gating (Mahalanobis distance test on the innovation).
        5. Start the filter uninitialised: first measurement sets position, large P for velocity.
    """

    def __init__(self, sensor_sigma):
        self.sensor_sigma = sensor_sigma
        self.reset()
        self.past_positions = deque([], 10)
        self.prediction = None

    def reset(self):
        self.x = None                        # state estimate (replace with your state vector)
        self.P = None                        # state covariance
        self.past_positions = deque([], 10)
        self.prediction = None
        self.last_z = None
        self.t_since_update = 0.0

    def predict(self, dt):
        self.t_since_update += dt 
        if self.x is not None :
            x = self.x[:2]
            v = self.x[2:]
            #velocity = [1]
            #print(f"This is the position extracted in the predict method: {x}")
            #print(f"This is the velocity extracted in the predict method: {v}")
            if len(self.past_positions) > 0 :
                #print(f"Past States: \n{np.array(self.past_positions)}")
                state_arr = np.array(self.past_positions)
                mean_vx, mean_vy = state_arr[:, 2:].mean(axis=0)
                #print(f"Mean: {mean_vx}x {mean_vy}y")
                self.v_avg = np.array([mean_vx, mean_vy])
                predicted_pos = x + self.v_avg * dt *50
                print(f"Predicted Position: {predicted_pos}")
                self.prediction = predicted_pos
                #self.x[:2] = predicted_pos
                return predicted_pos

            #print(average_v)

            #return predicted_position
        pass                                 # TODO: x = F x ; P = F P F^T + Q

    def update(self, z):
        z = np.array(z, dtype=float)
        if self.last_z is None:
            # first measurement: position known, velocity unknown
            self.x = np.array([z[0], z[1], 0.0, 0.0])
        else:
            v = (z - self.last_z) / self.t_since_update   # px/s, handles dropouts too
            self.x = np.array([z[0], z[1], v[0], v[1]])
            self.past_positions.append(self.x.copy())
        self.last_z = z
        self.t_since_update = 0.0
        
        #x_prev = self.x
        #self.x = np.array(z, dtype=float)    # TODO: replace with the real Kalman update
        ##print(f"Received updated z value: {z}")
        #try :
        #    pos_change = self.x-x_prev
        #    velocity = pos_change * 5 # This is the sampling rate
        #    #print(f"Velocity: {velocity}")
        #    self.v = velocity
        #    self.x = np.array([z[0], z[1], velocity[0], velocity[1]])
        #    self.past_positions.append(self.x)
        #except Exception as e :
        #    print(f"+++++++++++++++++++++++++++++++++++++++++++\nException in update loop.... might just be a None as self.x. Also might not be!\nHave a look:\n{e}\n+++++++++++++++++++++++++++++++++++++++++++")
        #    pass

    @property
    def position(self):
        return None if self.x is None else self.x[:2]

    @property
    def velocity(self):
        return None if self.x is None else self.x[2:]

    @property
    def cov(self):
        return None                          # TODO: return self.P[:2, :2] once you have one


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------
BG = (14, 16, 22)
C_TRUE = (80, 220, 120)
C_MEAS = (240, 90, 90)
C_EST = (90, 160, 255)
C_TEXT = (200, 205, 215)


def draw_cov_ellipse(surf, center, P, n_sigma=2.0, color=C_EST):
    P = np.asarray(P, dtype=float)
    if not np.all(np.isfinite(P)):
        return
    vals, vecs = np.linalg.eigh(P)
    radii = n_sigma * np.sqrt(np.clip(vals, 0, None))
    t = np.linspace(0, 2 * np.pi, 48)
    pts = center[:, None] + vecs @ (radii[:, None] * np.vstack((np.cos(t), np.sin(t))))
    if np.max(radii) > 1e-3:
        pygame.draw.lines(surf, color, True, pts.T.tolist(), 1)


def rmse(buf):
    return float(np.sqrt(np.mean(buf))) if buf else float("nan")


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Kalman playground")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)

    target = Target()
    sensor = Sensor()
    tracker = Tracker(sensor.sigma)

    true_trail = deque(maxlen=250)
    est_trail = deque(maxlen=250)
    meas_hist = deque(maxlen=40)          # (z, age_seconds)
    err_est = deque(maxlen=600)           # squared error, estimate vs truth
    err_meas = deque(maxlen=600)          # squared error, raw measurements vs truth
    paused = False
    trails = True
    running = True

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                k = e.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k == pygame.K_r:
                    target.reset()
                    tracker.reset()
                    sensor.timer = 0.0
                    for d in (true_trail, est_trail, meas_hist, err_est, err_meas):
                        d.clear()
                elif k == pygame.K_UP:
                    sensor.sigma = min(80.0, sensor.sigma + 2.5)
                elif k == pygame.K_DOWN:
                    sensor.sigma = max(0.5, sensor.sigma - 2.5)
                elif k == pygame.K_RIGHT:
                    sensor.rate = min(60.0, sensor.rate + 1.0)
                elif k == pygame.K_LEFT:
                    sensor.rate = max(0.5, sensor.rate - 1.0)
                elif k == pygame.K_a:
                    target.accel_sigma = min(600.0, target.accel_sigma + 20.0)
                elif k == pygame.K_z:
                    target.accel_sigma = max(0.0, target.accel_sigma - 20.0)
                elif k == pygame.K_d:
                    sensor.dropouts = not sensor.dropouts
                elif k == pygame.K_o:
                    sensor.outliers = not sensor.outliers
                elif k == pygame.K_t:
                    trails = not trails

        # ---------------- simulate ----------------
        if not paused:
            target.step(DT)
            tracker.sensor_sigma = sensor.sigma   # lets you read R from here if you want

            tracker.predict(DT)
            z = sensor.step(DT, target.pos)
            #print(f"Z VALUE IS: {z}")
            if z is not None:
                tracker.update(z)
                meas_hist.append([z, 0.0])
                err_meas.append(float(np.sum((z - target.pos) ** 2)))

            for m in meas_hist:
                m[1] += DT

            est = tracker.position
            true_trail.append(target.pos.copy())
            if est is not None:
                est_trail.append(np.array(est, dtype=float))
                err_est.append(float(np.sum((np.asarray(est) - target.pos) ** 2)))

        # ---------------- draw ----------------
        screen.fill(BG)

        if trails:
            if len(true_trail) > 1:
                pygame.draw.lines(screen, (40, 90, 55), False, [p.tolist() for p in true_trail], 1)
            if len(est_trail) > 1:
                pygame.draw.lines(screen, (40, 65, 110), False, [p.tolist() for p in est_trail], 1)

        for z, age in meas_hist:                       # measurements fade out over 3 s
            f = max(0.0, 1.0 - age / 3.0)
            col = tuple(int(BG[i] + (C_MEAS[i] - BG[i]) * f) for i in range(3))
            pygame.draw.circle(screen, col, z.astype(int).tolist(), 3)

        pygame.draw.circle(screen, C_TRUE, target.pos.astype(int).tolist(), 7, 2)
        try :
            if tracker.prediction is not None:
                pygame.draw.circle(screen, "red", (tracker.prediction.astype(int)).tolist(), 3, 2)
                pygame.draw.line(screen, "red", tracker.position, tracker.prediction.astype(int).tolist(), 1)
                #print(f"Tracker prediction: {tracker.prediction}")
        except :
            pass

        est = tracker.position
        if est is not None:
            est = np.asarray(est, dtype=float)
            if tracker.cov is not None:
                draw_cov_ellipse(screen, est, tracker.cov)
            pygame.draw.circle(screen, C_EST, est.astype(int).tolist(), 5)

            v = tracker.velocity
            if v is not None:
                pygame.draw.line(screen, C_EST, est.tolist(), (est + 0.5 * np.asarray(v)).tolist(), 1)
            #if tracker.v_avg is not None:
            #    v_avg = tracker.v_avg
            #    pygame.draw.line(screen, "yellow", est.tolist(), (est + 0.5 * np.asarray(v_avg)).tolist(), 2)

        lines = [
            f"sensor sigma {sensor.sigma:.1f} px  (UP/DOWN)     rate {sensor.rate:.0f} Hz  (LEFT/RIGHT)",
            f"object accel sigma {target.accel_sigma:.0f} px/s^2  (A/Z)",
            f"dropouts {'ON' if sensor.dropouts else 'off'} (D)    outliers {'ON' if sensor.outliers else 'off'} (O)"
            f"    trails (T)    pause (SPACE)    reset (R)",
            f"RMSE  raw measurements: {rmse(err_meas):6.1f} px    tracker: {rmse(err_est):6.1f} px",
        ]
        for i, s in enumerate(lines):
            screen.blit(font.render(s, True, C_TEXT), (10, 8 + i * 20))

        # legend
        y = H - 24
        for col, label, x in ((C_TRUE, "truth", 10), (C_MEAS, "measurement", 90), (C_EST, "estimate", 220)):
            pygame.draw.circle(screen, col, (x, y + 7), 5)
            screen.blit(font.render(label, True, C_TEXT), (x + 12, y))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
