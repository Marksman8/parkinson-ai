"""Dark-themed scientific visualizer for ODE simulation results."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Color palette ────────────────────────────────────────────────────
BG    = "#06080f"
SURF  = "#0d1117"
BLUE  = "#4f8ef7"
PURP  = "#a78bfa"
CYAN  = "#06b6d4"
GREEN = "#10b981"
AMBER = "#f59e0b"
RED   = "#ef4444"
MUTED = "#546e7a"


def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(SURF)
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#1e2535")
    if xlabel: ax.set_xlabel(xlabel, color="white", fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, color="white", fontsize=10)
    if title:  ax.set_title(title,   color="white", fontsize=11, pad=10)


def _save(fig, path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path


def plot_hodgkin_huxley(result, save_path: str = "data/outputs/hh_simulation.png") -> str:
    t, y   = result.t, result.y
    V, m, h, n = y[0], y[1], y[2], y[3]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), facecolor=BG)
    fig.subplots_adjust(hspace=0.4)

    # Panel 1 — membrane potential
    axes[0].plot(t, V, color=BLUE, lw=1.8, label="Membrane Potential (mV)")
    axes[0].axhline(-65, color=MUTED, lw=0.8, ls="--", label="Resting (−65 mV)")
    axes[0].fill_between(t, V, -65, where=(V > -65), alpha=0.08, color=BLUE)
    axes[0].legend(facecolor=SURF, labelcolor="white", framealpha=0.9, fontsize=9)
    _style_ax(axes[0],
              title="Hodgkin-Huxley Dopaminergic Neuron — Action Potential",
              xlabel="Time (ms)", ylabel="V (mV)")

    # Panel 2 — gating variables
    axes[1].plot(t, m, color=PURP,  lw=1.4, label="m  Na⁺ activation")
    axes[1].plot(t, h, color=CYAN,  lw=1.4, label="h  Na⁺ inactivation")
    axes[1].plot(t, n, color=GREEN, lw=1.4, label="n  K⁺ activation")
    axes[1].legend(facecolor=SURF, labelcolor="white", framealpha=0.9, fontsize=9)
    _style_ax(axes[1],
              title="Ion Channel Gating Variables",
              xlabel="Time (ms)", ylabel="Gate probability")

    return _save(fig, save_path)


def plot_dopamine_kinetics(result, save_path: str = "data/outputs/dopamine_kinetics.png") -> str:
    pd_loss  = result.params.get("pd_loss", 0.0)
    severity = (
        "Healthy Neuron"   if pd_loss < 0.2 else
        "Mild PD"          if pd_loss < 0.5 else
        "Moderate PD"      if pd_loss < 0.8 else
        "Severe PD"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    fig.subplots_adjust(wspace=0.35)

    labels = ["Vesicular Store", "Synaptic Cleft [DA]", "Extracellular [DA]"]
    colors = [BLUE, AMBER, RED]

    # Panel 1 — time series
    for i, (label, color) in enumerate(zip(labels, colors)):
        axes[0].plot(result.t, result.y[i], color=color, lw=2.0, label=label)
    axes[0].legend(facecolor=SURF, labelcolor="white", framealpha=0.9, fontsize=9)
    _style_ax(axes[0],
              title=f"Dopamine Kinetics — {severity} ({pd_loss*100:.0f}% neurodegeneration)",
              xlabel="Time (s)", ylabel="Dopamine (a.u.)")

    # Panel 2 — steady-state bar comparison
    healthy_sim = result.y[:, -1]
    ax2 = axes[1]
    x = np.arange(len(labels))
    bars = ax2.bar(x, healthy_sim, color=[BLUE, AMBER, RED], alpha=0.85,
                   edgecolor="#1e2535", linewidth=1.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Vesicular", "Synaptic", "Extracellular"],
                        color="white", fontsize=9)
    _style_ax(axes[1],
              title="Steady-State Dopamine Levels",
              xlabel="Compartment", ylabel="Concentration (a.u.)")

    return _save(fig, save_path)
