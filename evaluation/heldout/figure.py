"""Hình kết quả held-out: recall theo detector trên 2 video (kèm khoảng Wilson 95%).
Dùng: python -m evaluation.heldout.figure <out.pdf> [<out2.pdf> ...]"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DETS = [("plain", "Plain"), ("balanced", "Balanced"), ("coco", "COCO-pretrained")]
COLORS = {"plain": "#2a78d6", "balanced": "#eb6834", "coco": "#1baf7a"}
VIDEOS = [("hem", "Alley camera (held out)"), ("junction", "Junction, rider height (held out)")]
INK, MUTED = "#0b0b0b", "#52514e"


def main(outs):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.4), sharey=True)
    for ax, (v, title) in zip(axes, VIDEOS):
        d = json.loads((ROOT / "evaluation" / "results" / f"heldout_{v}.json").read_text(encoding="utf-8"))
        n = sum(d["n_gt"].values())
        for i, (k, label) in enumerate(DETS):
            r = d["detectors"][k]
            lo, hi = r["recall_ci"]
            ax.bar(i, 100 * r["recall"], width=0.62, color=COLORS[k], edgecolor="white", linewidth=2)
            ax.errorbar(i, 100 * r["recall"], yerr=[[100 * (r["recall"] - lo)], [100 * (hi - r["recall"])]],
                        color=INK, capsize=3, linewidth=1)
            ax.text(i, 100 * hi + 3, f"{100 * r['recall']:.0f}%", ha="center", fontsize=8.5)
        ax.set_xticks(range(3), [x[1] for x in DETS], fontsize=8)
        ax.set_title(f"{title}\n{n} labelled vehicle boxes in 40 frames", loc="left", fontsize=9)
        ax.set_ylim(0, 110)
        ax.tick_params(colors=MUTED, length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Recall at conf 0.25, IoU 0.5 (%)", color=MUTED)
    fig.tight_layout()
    for o in outs:
        fig.savefig(o, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1:])
