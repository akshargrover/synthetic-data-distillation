"""
Build every figure and the summary table used in the paper from results/*.

Figures are 300-dpi PNGs sized for an IEEE single column (3.5 in wide).
Colours: validated categorical slots (blue / orange / aqua) with marker or hatch
as secondary encoding; sequential blue ramp for confusion matrices.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import confusion_matrix

import config as C

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"
SEQ = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
R = C.RESULTS_DIR
SHORT = {
    "card_arrival": "card_arrival", "card_delivery_estimate": "card_deliv_est",
    "lost_or_stolen_card": "lost_stolen_card", "compromised_card": "compromised_card",
    "exchange_rate": "exchange_rate", "card_payment_wrong_exchange_rate": "wrong_exch_rate",
    "top_up_failed": "top_up_failed", "top_up_reverted": "top_up_reverted",
    "transfer_not_received_by_recipient": "transfer_not_recv", "cash_withdrawal_charge": "atm_charge",
}


def agg_runs():
    runs = pd.read_csv(R / "student_runs.csv")
    g = runs.groupby(["group", "config"]).agg(
        f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
        n_train=("n_train", "first"), train_s=("train_seconds", "mean"),
        runs=("seed", "count")).reset_index()
    g.to_csv(R / "student_summary.csv", index=False)
    return runs, g


def fig_pipeline():
    """Two-row 'snake' layout so the diagram fits one IEEE column."""
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 58)
    w, h = 29, 21
    boxes = {  # name: (x, y, text, colour)
        "seed": (1, 34, "Human seeds\n80 examples\n(8 per class)", "#f1f0ec"),
        "p1": (35.5, 34, "Phase 1\nTeacher LLM\npersona x style", "#cde2fb"),
        "p2": (70, 34, "Phase 2\nheuristic + MiniLM\ndedup + centroid", "#cde2fb"),
        "p3": (70, 2, "Phase 3\nDistilBERT\nfine-tuning", "#cde2fb"),
        "p4": (35.5, 2, "Phase 4\nhidden human test\nF1 / latency", "#f1f0ec"),
        "dep": (1, 2, "CPU deployment\n67M params\nINT8 option", "#f1f0ec"),
    }
    for x, y, t, c in boxes.values():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                    fc=c, ec=INK2, lw=0.7))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=6.3, color=INK)
    arrow = dict(arrowstyle="-|>", color=INK2, lw=0.9)
    ax.annotate("", xy=(35, 44.5), xytext=(30.6, 44.5), arrowprops=arrow)
    ax.annotate("", xy=(69.5, 44.5), xytext=(65, 44.5), arrowprops=arrow)
    ax.annotate("", xy=(84.5, 23.6), xytext=(84.5, 33.4), arrowprops=arrow)
    ax.annotate("", xy=(65, 12.5), xytext=(69.5, 12.5), arrowprops=arrow)
    ax.annotate("", xy=(30.6, 12.5), xytext=(35, 12.5), arrowprops=arrow)
    fig.savefig(C.FIG_DIR / "fig1_pipeline.png"); plt.close(fig)


def fig_funnel(fs):
    s = fs["synthetic"]
    stages = ["Raw", "Heuristic", "Semantic\ndedup", "Centroid"]
    vals = [s["raw"], s["after_heuristic"], s["after_semantic"], s["after_centroid"]]
    p = fs.get("synthetic_plain")
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    x = np.arange(4)
    if p:
        pv = [p["raw"], p["after_heuristic"], p["after_semantic"], p["after_centroid"]]
        # express as % of raw so the two prompting modes are comparable
        ax.bar(x - 0.19, np.array(vals) / vals[0] * 100, 0.36, color=BLUE, label=f"Persona+style (raw={vals[0]})")
        ax.bar(x + 0.19, np.array(pv) / pv[0] * 100, 0.36, color=ORANGE, hatch="////", edgecolor="white",
               lw=0, label=f"Plain (raw={pv[0]})")
        for xi, v in zip(x, vals):
            ax.text(xi - 0.19, v / vals[0] * 100 + 1.5, str(v), ha="center", fontsize=6, color=INK2)
        for xi, v in zip(x, pv):
            ax.text(xi + 0.19, v / pv[0] * 100 + 1.5, str(v), ha="center", fontsize=6, color=INK2)
        ax.set_ylabel("Samples retained (% of raw)")
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                  handlelength=1.2, columnspacing=1.0)
        ax.set_ylim(0, 108)
    ax.set_xticks(x, stages)
    fig.savefig(C.FIG_DIR / "fig2_funnel.png"); plt.close(fig)


def fig_main(g, teacher):
    order = [("Teacher zero-shot", teacher["zero_shot"]["macro_f1"], 0, ORANGE),
             ("Teacher few-shot", teacher["few_shot"]["macro_f1"], 0, ORANGE)]
    lookup = g[g.group == "main"].set_index("config")
    for name, lab in [("Seeds only", "Student: seeds only"),
                      ("Seeds + raw synthetic", "Student: + raw synthetic"),
                      ("Synthetic only (full filter)", "Student: synthetic only"),
                      ("Seeds + full filter (proposed)", "Student: proposed"),
                      ("Oracle: all real training data", "Student: all real data")]:
        r = lookup.loc[name]
        order.append((lab, r.f1_mean, r.f1_std, AQUA if "real" in lab else BLUE))
    cl = pd.read_csv(R / "classical_baselines.csv")
    mlr = cl[(cl.method == "MiniLM + LR") & (cl.data == "Seeds only")].macro_f1.iloc[0]
    order.insert(-1, ("MiniLM + LR: seeds only", mlr, 0, GREY))
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    y = np.arange(len(order))[::-1]
    for yi, (lab, m, s, c) in zip(y, order):
        ax.barh(yi, m * 100, xerr=s * 100 if s else None, color=c, height=0.62,
                error_kw=dict(lw=0.7, capsize=2, ecolor=INK))
        ax.text(m * 100 + (s or 0) * 100 + 1, yi, f"{m*100:.1f}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y, [o[0] for o in order])
    ax.set_xlabel("Macro-F1 on hidden human test set (%)")
    ax.set_xlim(0, 105); ax.grid(axis="y", visible=False)
    fig.savefig(C.FIG_DIR / "fig3_main_results.png"); plt.close(fig)


def fig_scaling(g, teacher):
    sc = g[g.group == "scaling"].copy()
    sc["n"] = sc.config.str.extract(r"n=(\d+)").astype(int)
    sc["kind"] = sc.config.str.split().str[0]
    main = g[g.group == "main"].set_index("config")
    seeds_only = main.loc["Seeds only"]
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    for kind, col, mk, lab in [("synthetic", BLUE, "o", "Seeds + synthetic"),
                               ("real", AQUA, "s", "Seeds + real")]:
        d = sc[sc.kind == kind].sort_values("n")
        xs = np.r_[0, d.n.values]
        ms = np.r_[seeds_only.f1_mean, d.f1_mean.values] * 100
        ss = np.r_[seeds_only.f1_std, d.f1_std.values] * 100
        ax.plot(xs, ms, color=col, marker=mk, ms=4, lw=1.5, label=lab)
        ax.fill_between(xs, ms - ss, ms + ss, color=col, alpha=0.15, lw=0)
    full = main.loc["Seeds + full filter (proposed)"]
    nfull = (full.n_train - 80) / 10
    ax.plot([nfull], [full.f1_mean * 100], marker="D", ms=5, color=BLUE, mec="white", ls="none")
    ax.annotate(f"all filtered (~{nfull:.0f}/class)", (nfull, full.f1_mean * 100),
                textcoords="offset points", xytext=(0, -6), fontsize=6, color=INK2, ha="center",
                va="top")
    tz = teacher["zero_shot"]["macro_f1"] * 100
    ax.axhline(tz, color=ORANGE, ls="--", lw=1)
    ax.text(198, tz - 0.9, "teacher zero-shot", color=INK2, fontsize=6, ha="right", va="top")
    ax.set_xlabel("Additional training samples per class")
    ax.set_ylabel("Macro-F1 (%)")
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.62))
    fig.savefig(C.FIG_DIR / "fig5_scaling.png"); plt.close(fig)


def fig_threshold(g, fs):
    th = g[g.group == "threshold"].copy()
    th["tau"] = th.config.str.extract(r"tau=([\d.]+)").astype(float)
    m90 = g[(g.group == "main") & (g.config == "Seeds + dedup (tau=0.90)")].iloc[0]
    th = pd.concat([th, pd.DataFrame([{**m90.to_dict(), "tau": 0.90}])]).sort_values("tau")
    sweep = fs["threshold_sweep"]
    kept = [sweep[str(t) if str(t) in sweep else f"{t}"]["after_semantic"] for t in th.tau]
    labels = ["none" if t > 1 else f"{t:.2f}" for t in th.tau]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.7))
    x = np.arange(len(th))
    axes[0].bar(x, kept, color=BLUE, width=0.6)
    axes[0].set_xticks(x, labels); axes[0].set_title("Samples kept", color=INK)
    axes[0].set_xlabel(r"Dedup threshold $\tau$")
    axes[1].errorbar(x, th.f1_mean * 100, yerr=th.f1_std * 100, color=BLUE, marker="o", ms=4,
                     lw=1.5, capsize=2)
    axes[1].set_xticks(x, labels); axes[1].set_title("Student macro-F1 (%)", color=INK)
    axes[1].set_xlabel(r"Dedup threshold $\tau$")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(C.FIG_DIR / "fig4_threshold.png"); plt.close(fig)


def fig_confusion():
    files = [("Teacher zero-shot", R / "preds_teacher_zero_shot.csv"),
             ("Proposed student", R / "preds_student_Seeds__full_filter_proposed.csv")]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.95))
    ids = [f"C{i+1}" for i in range(len(C.LABELS))]       # class ids defined in Table I
    for k, (ax, (title, f)) in enumerate(zip(axes, files)):
        d = pd.read_csv(f)
        cm = confusion_matrix(d.label, d.pred, labels=C.LABELS)
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=40)
        for i in range(len(C.LABELS)):
            for j in range(len(C.LABELS)):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=4.6,
                            color="white" if cm[i, j] > 22 else INK)
        ax.set_xticks(range(10), ids, rotation=90, fontsize=5)
        ax.set_yticks(range(10), ids if k == 0 else [""] * 10, fontsize=5)
        ax.tick_params(length=0, pad=1.5)
        ax.grid(False); ax.set_title(title, color=INK, fontsize=7)
        ax.set_xlabel("Predicted", fontsize=6)
        if k == 0:
            ax.set_ylabel("True", fontsize=6)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(C.FIG_DIR / "fig6_confusion.png"); plt.close(fig)


def fig_latency(teacher, bench, g):
    main = g[g.group == "main"].set_index("config")
    pts = [("Teacher zero-shot\n(GPU)", teacher["zero_shot"]["latency_ms_mean"], teacher["zero_shot"]["macro_f1"], ORANGE, "o"),
           ("Teacher few-shot\n(GPU)", teacher["few_shot"]["latency_ms_mean"], teacher["few_shot"]["macro_f1"], ORANGE, "s"),
           ("Student FP32\n(CPU)", bench["student_cpu_fp32"]["latency_ms_mean"], bench["student_cpu_fp32"]["macro_f1"], BLUE, "o"),
           ("Student INT8\n(CPU)", bench["student_cpu_int8"]["latency_ms_mean"], bench["student_cpu_int8"]["macro_f1"], BLUE, "s")]
    if "minilm_lr_cpu" in bench:
        pts.append(("MiniLM+LR\n(CPU)", bench["minilm_lr_cpu"]["latency_ms_mean"],
                    bench["minilm_lr_cpu"]["macro_f1"], GREY, "D"))
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    offsets = {"Student INT8\n(CPU)": (0, -9, "center"), "Student FP32\n(CPU)": (6, -10, "left")}
    for lab, l, f, c, mk in pts:
        ax.plot(l, f * 100, marker=mk, ms=6, color=c, mec="white", mew=0.8, ls="none")
        dx, dy, ha = offsets.get(lab, (6, -3, "left"))
        ax.annotate(lab, (l, f * 100), textcoords="offset points", xytext=(dx, dy), fontsize=6,
                    color=INK2, ha=ha, va="top" if ha == "center" else "baseline")
    ax.set_xscale("log")
    ax.set_xlabel("Mean latency per query, batch = 1 (ms, log scale)")
    ax.set_ylabel("Macro-F1 (%)")
    lo = min(p[2] for p in pts) * 100
    ax.set_ylim(lo - 6, 100)
    ax.set_xlim(min(p[1] for p in pts) / 3, max(p[1] for p in pts) * 6)
    fig.savefig(C.FIG_DIR / "fig7_latency.png"); plt.close(fig)


def per_class_table(teacher):
    t = teacher["zero_shot"]["per_class"]
    d = pd.read_csv(R / "preds_student_Seeds__full_filter_proposed.csv")
    from sklearn.metrics import classification_report
    s = classification_report(d.label, d.pred, labels=C.LABELS, output_dict=True, zero_division=0)
    rows = [dict(label=l, teacher_f1=t[l]["f1-score"], student_f1=s[l]["f1-score"]) for l in C.LABELS]
    pd.DataFrame(rows).to_csv(R / "per_class_f1.csv", index=False)


def main():
    runs, g = agg_runs()
    teacher = json.loads((R / "teacher_eval.json").read_text())
    fs = json.loads((R / "filter_stats.json").read_text())
    fig_pipeline()
    fig_funnel(fs)
    fig_main(g, teacher)
    fig_scaling(g, teacher)
    fig_threshold(g, fs)
    fig_confusion()
    per_class_table(teacher)
    bp = R / "benchmark.json"
    if bp.exists():
        fig_latency(teacher, json.loads(bp.read_text()), g)
    print(g.to_string())


if __name__ == "__main__":
    main()
