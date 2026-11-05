"""
avoidance.py - pick a collision-free velocity, using tracked debris motion.

Input is the desired shuttle velocity (Phobos frame) and a snapshot of the debris
tracks from DebrisTracker.snapshot(). We test many candidate shuttle velocities
V (different directions, at full / half / quarter speed, plus stopping) and keep
the cheapest one that stays clear of every debris object over the next
`horizon` seconds.

Relative position of a debris object at time t, if we fly at constant V:
        p(t) = p0 + t * (v_debris - V)
Its closest approach has a closed form, so this is cheap to evaluate for
hundreds of candidates x dozens of contacts.

Keep-out radius per object = diameter/2 + margin + k*sigma_p + k*sigma_v*t
(capped), so uncertain tracks get a wider berth that grows with look-ahead time.
If nothing is fully safe, we take the candidate with the best worst-case
clearance instead of blindly stopping.
"""

import numpy as np


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


class Avoider:
    def __init__(self, margin=6.0, horizon=15.0, k_sigma=2.0, max_inflate=25.0,
                 n_dirs=400, speed_scales=(1.0, 0.5, 0.25), w_prev=0.3):
        self.margin = margin            # extra clearance beyond debris radius (m); covers shuttle size
        self.horizon = horizon          # look-ahead time (s); should cover your braking time
        self.k_sigma = k_sigma          # how many sigmas of track uncertainty to keep clear
        self.max_inflate = max_inflate  # cap on uncertainty inflation (m)
        self.dirs = fib_sphere(n_dirs)
        self.scales = np.asarray(speed_scales, float)
        self.w_prev = w_prev            # bias towards last choice (reduces flicker)
        self.v_prev = None
        self.last_info = {}             # diagnostics for logging

    def pick(self, v_des, snap):
        """Return a safe velocity (Phobos frame) as close as possible to v_des."""
        v_des = np.asarray(v_des, float)
        speed = np.linalg.norm(v_des)
        if speed < 1e-3 or len(snap["id"]) == 0:
            self.v_prev = v_des
            self.last_info = {"blocked": False, "n_tracks": len(snap["id"])}
            return v_des

        u = v_des / speed
        nd = len(self.dirs)
        # candidates: every direction at every speed scale, plus "stop"
        V = (self.dirs[None] * (speed * self.scales[:, None, None])).reshape(-1, 3)
        V = np.vstack([V, np.zeros((1, 3))])
        scale_pen = np.concatenate([np.repeat(1.0 - self.scales, nd), [3.0]])
        dir_cost = np.concatenate([np.tile(1.0 - self.dirs @ u, len(self.scales)), [0.0]])
        cost = 2.0 * scale_pen + dir_cost
        if self.v_prev is not None and np.linalg.norm(self.v_prev) > 1e-3:
            up = self.v_prev / np.linalg.norm(self.v_prev)
            prev_cost = np.concatenate([np.tile(1.0 - self.dirs @ up, len(self.scales)), [0.0]])
            cost = cost + self.w_prev * prev_cost

        # --- closest approach of every (debris, candidate) pair over [0, horizon] ---
        P = snap["p_rel"]                                  # (m,3) relative to shuttle
        Vd = snap["v_abs"]                                 # (m,3) debris velocity
        W = Vd[:, None, :] - V[None, :, :]                 # (m,n,3) relative velocity
        WW = np.einsum("mnk,mnk->mn", W, W)
        PW = np.einsum("mnk,mk->mn", W, P)
        t = np.clip(-PW / (WW + 1e-9), 0.0, self.horizon)
        d = np.linalg.norm(P[:, None, :] + t[..., None] * W, axis=2)
        inflate = self.k_sigma * (snap["sigma_p"][:, None] + snap["sigma_v"][:, None] * t)
        R = (snap["diameter"] / 2 + self.margin)[:, None] + np.minimum(inflate, self.max_inflate)
        clearance = (d - R).min(axis=0)                    # worst-case margin per candidate
        safe = clearance > 0

        j0 = int(np.argmax(self.dirs @ u))                 # sampled direction closest to v_des
        if safe.any():
            k = int(np.argmin(np.where(safe, cost, np.inf)))
        else:
            k = int(np.argmax(clearance))                  # least-bad option
        self.v_prev = V[k]
        self.last_info = {"blocked": not bool(safe[j0]),   # was flying straight at the goal unsafe?
                          "any_safe": bool(safe.any()), "clearance": float(clearance[k]),
                          "n_tracks": len(snap["id"])}
        return V[k]
