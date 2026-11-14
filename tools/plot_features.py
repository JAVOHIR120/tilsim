"""Draw the .npz files from tools/features.py as images: spectrogram with pitch curve on top, loudness below.

    python tools/plot_features.py out/first-uz/_feat --per-image 4
"""
import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("folder", type=Path)
ap.add_argument("--per-image", type=int, default=4)
args = ap.parse_args()
files = sorted(args.folder.glob("*.npz"))
for page in range(math.ceil(len(files) / args.per_image)):
    chunk = files[page * args.per_image:(page + 1) * args.per_image]
    fig, axes = plt.subplots(len(chunk) * 2, 1, figsize=(13, 3.3 * len(chunk)),
                             gridspec_kw={"height_ratios": [3, 1] * len(chunk)})
    for i, path in enumerate(chunk):
        d = np.load(path)
        sr, hop = int(d["sr"]), int(d["hop"])
        t_end = d["mel"].shape[1] * hop / sr
        ax, ax2 = axes[2 * i], axes[2 * i + 1]
        ax.imshow(d["mel"], origin="lower", aspect="auto", extent=[0, t_end, 0, 100], cmap="magma", vmin=-80, vmax=0)
        f0 = d["f0"]
        t = np.arange(len(f0)) * hop / sr
        twin = ax.twinx()
        twin.plot(t, f0, color="cyan", lw=1.6)
        twin.set_ylim(60, 300)
        twin.set_ylabel("F0 Hz", color="cyan")
        ax.set_title(f"{path.stem}: {str(d['text'])}", fontsize=10, loc="left")
        ax.set_ylabel("mel")
        ax.set_xlim(0, t_end)
        rms = d["rms_db"]
        ax2.plot(np.arange(len(rms)) * hop / sr, rms, color="black", lw=1)
        ax2.set_xlim(0, t_end)
        ax2.set_ylim(-80, 0)
        ax2.set_ylabel("dB")
        ax2.grid(alpha=.3)
    axes[-1].set_xlabel("seconds")
    fig.tight_layout()
    out = args.folder / f"page{page + 1}.png"
    fig.savefig(out, dpi=72)
    plt.close(fig)
    print("saved", out)
