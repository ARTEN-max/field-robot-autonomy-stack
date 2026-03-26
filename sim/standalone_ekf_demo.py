"""
Standalone EKF Simulation
--------------------------
Demonstrates the EKF localizer running on a simulated corn-field traverse.

No ROS2 required — just numpy + matplotlib.

Run:
    python sim/standalone_ekf_demo.py

What it shows:
    - Ground truth path (sinusoidal heading variation, like a robot
      navigating uneven rows)
    - Noisy dead-reckoning  (odometry only — drifts badly)
    - EKF estimate          (fusing odometry + sparse GPS)
    - 2-σ uncertainty ellipses at GPS update instants
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse


# ── Simulation parameters ─────────────────────────────────────────────
DT          = 0.05          # control loop period [s]
SIM_TIME    = 60.0          # total simulation duration [s]
GPS_RATE    = 0.1           # GPS update probability per step (~2 Hz)

# Noise levels
ODOM_V_NOISE   = 0.03       # [m/s]
ODOM_W_NOISE   = 0.015      # [rad/s]
GPS_NOISE_STD  = 1.5        # [m]

# EKF process noise
Q_DIAG  = (0.04, 0.04, 0.008)
R_DIAG  = (GPS_NOISE_STD**2, GPS_NOISE_STD**2)

# Field layout — corn rows are ~75 cm apart; robot drives ~5 rows in
FIELD_WIDTH  = 20.0         # [m]
FIELD_LENGTH = 80.0         # [m]
ROW_SPACING  = 0.76         # [m]


# ── Helper functions ───────────────────────────────────────────────────

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def motion_model(x, v, w, dt):
    xp, yp, th = x
    if abs(w) < 1e-6:
        return np.array([xp + v*dt*math.cos(th),
                         yp + v*dt*math.sin(th),
                         wrap(th)])
    r   = v / w
    dth = w * dt
    return np.array([xp + r*(math.sin(th+dth) - math.sin(th)),
                     yp + r*(-math.cos(th+dth) + math.cos(th)),
                     wrap(th + dth)])


def jacobian_G(x, v, w, dt):
    th = x[2]
    if abs(w) < 1e-6:
        return np.array([[1, 0, -v*dt*math.sin(th)],
                         [0, 1,  v*dt*math.cos(th)],
                         [0, 0,  1]])
    r   = v / w
    dth = w * dt
    return np.array([[1, 0,  r*(math.cos(th+dth) - math.cos(th))],
                     [0, 1,  r*(math.sin(th+dth) - math.sin(th))],
                     [0, 0,  1]])


def cov_ellipse(ax, mean, cov2x2, n_std=2.0, **kwargs):
    """Draw a 2-D covariance ellipse."""
    vals, vecs = np.linalg.eigh(cov2x2)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = math.degrees(math.atan2(*vecs[:, 0][::-1]))
    w, h  = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ell   = Ellipse(xy=mean, width=w, height=h, angle=angle, **kwargs)
    ax.add_patch(ell)


# ── Simulation loop ────────────────────────────────────────────────────

def run():
    rng = np.random.default_rng(42)

    steps = int(SIM_TIME / DT)

    # ground truth
    gt   = np.zeros((steps, 3))
    odom = np.zeros((steps, 3))
    ekf  = np.zeros((steps, 3))

    # EKF state & covariance
    x_hat = np.zeros(3)
    P     = np.diag([0.5, 0.5, 0.05])
    Q     = np.diag(Q_DIAG)
    R     = np.diag(R_DIAG)
    H     = np.array([[1,0,0],[0,1,0]])

    # Dead-reckoning state (no corrections)
    x_dr  = np.zeros(3)

    # Ground-truth initial heading: pointing "north" through the rows
    x_true = np.array([1.5, 0.0, math.pi/2])   # start between rows 1 & 2

    gps_points = []

    for i in range(steps):
        t = i * DT

        # ── Generate ground-truth control inputs ──────────────────────
        # Robot drives up rows with gentle sinusoidal lateral oscillation
        # simulating the real-world heading chatter
        v_true = 0.5                                  # [m/s]
        w_true = 0.12 * math.sin(2 * math.pi * t / 8) # gentle weave

        # ── Propagate ground truth ────────────────────────────────────
        x_true = motion_model(x_true, v_true, w_true, DT)
        gt[i]  = x_true

        # ── Simulate noisy odometry ───────────────────────────────────
        v_meas = v_true + rng.normal(0, ODOM_V_NOISE)
        w_meas = w_true + rng.normal(0, ODOM_W_NOISE)

        # Dead-reckoning (no EKF)
        x_dr   = motion_model(x_dr, v_meas, w_meas, DT)
        odom[i] = x_dr

        # ── EKF Predict ───────────────────────────────────────────────
        x_hat = motion_model(x_hat, v_meas, w_meas, DT)
        G     = jacobian_G(x_hat, v_meas, w_meas, DT)
        P     = G @ P @ G.T + Q

        # ── EKF Update (sparse GPS) ───────────────────────────────────
        if rng.random() < GPS_RATE:
            z = x_true[:2] + rng.normal(0, GPS_NOISE_STD, 2)
            y_res = z - H @ x_hat
            S     = H @ P @ H.T + R
            K     = P @ H.T @ np.linalg.inv(S)
            x_hat = x_hat + K @ y_res
            x_hat[2] = wrap(x_hat[2])
            P     = (np.eye(3) - K @ H) @ P
            gps_points.append((z, P[:2, :2].copy()))

        ekf[i] = x_hat

    return gt, odom, ekf, gps_points


# ── Plotting ───────────────────────────────────────────────────────────

def plot(gt, odom, ekf, gps_points):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes:
        ax.set_facecolor("#0f1117")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    # ── Left: trajectory ──────────────────────────────────────────────
    ax = axes[0]

    # Draw corn rows
    for row_x in np.arange(0, FIELD_WIDTH, ROW_SPACING):
        ax.axvline(row_x, color="#2d5a1b", linewidth=0.6, alpha=0.4)

    ax.plot(gt[:,0],   gt[:,1],   color="#00ff88", lw=1.5, label="Ground truth",    zorder=3)
    ax.plot(odom[:,0], odom[:,1], color="#ff6b35", lw=1.0, linestyle="--",
            label="Dead-reckoning (drift)", alpha=0.7, zorder=2)
    ax.plot(ekf[:,0],  ekf[:,1],  color="#4fc3f7", lw=1.5, label="EKF estimate",    zorder=4)

    # GPS measurements + covariance ellipses
    if gps_points:
        gx = [p[0][0] for p in gps_points]
        gy = [p[0][1] for p in gps_points]
        ax.scatter(gx, gy, s=12, color="#ffeb3b", zorder=5, label="GPS meas.", alpha=0.7)
        for z, cov in gps_points[::3]:
            cov_ellipse(ax, z, cov, n_std=2, color="#ffeb3b", alpha=0.08, zorder=1)

    ax.set_xlim(-1, FIELD_WIDTH + 1)
    ax.set_xlabel("East  [m]", color="white")
    ax.set_ylabel("North [m]", color="white")
    ax.set_title("Corn Field Traverse — EKF vs Dead-Reckoning", color="white", pad=10)
    leg = ax.legend(facecolor="#1e1e2e", edgecolor="#333", labelcolor="white", fontsize=8)
    ax.set_aspect("equal")

    # ── Right: error over time ────────────────────────────────────────
    ax2 = axes[1]
    steps = len(gt)
    t = np.arange(steps) * 0.05

    odom_err = np.linalg.norm(odom[:,:2] - gt[:,:2], axis=1)
    ekf_err  = np.linalg.norm(ekf[:,:2]  - gt[:,:2], axis=1)

    ax2.plot(t, odom_err, color="#ff6b35", lw=1.2, label="Dead-reckoning error", alpha=0.8)
    ax2.plot(t, ekf_err,  color="#4fc3f7", lw=1.5, label="EKF error")
    ax2.set_xlabel("Time [s]", color="white")
    ax2.set_ylabel("Position error [m]", color="white")
    ax2.set_title("Localisation Error Over Time", color="white", pad=10)
    ax2.legend(facecolor="#1e1e2e", edgecolor="#333", labelcolor="white", fontsize=8)
    ax2.set_ylim(bottom=0)
    ax2.grid(color="#333", linewidth=0.5)

    final_odom = odom_err[-1]
    final_ekf  = ekf_err[-1]
    ax2.text(0.98, 0.95,
             f"Final drift: {final_odom:.2f} m (DR)  /  {final_ekf:.2f} m (EKF)",
             transform=ax2.transAxes, ha="right", va="top",
             color="white", fontsize=8,
             bbox=dict(boxstyle="round", facecolor="#1e1e2e", edgecolor="#555"))

    plt.tight_layout(pad=2.0)
    out = "sim/ekf_demo.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    print("Running EKF simulation…")
    gt, odom, ekf, gps = run()
    plot(gt, odom, ekf, gps)
    print("Done.")
