"""
ARMA Time Series Analysis — Bachelor / Bachelorette
====================================================
Treats each season as a weekly time series.  Three signals are modeled:
  1. elimination_rate  — fraction of *alive* contestants eliminated each week
  2. date_rose_rate    — fraction of alive contestants who received a date rose
  3. alive_pct         — fraction of original cast still alive (normalised 0-1)

The script fits ARMA(p, q) to each signal for every season available and
produces four publication-quality figures:
  Fig 1  — ACF / PACF across all seasons (averaged) → order selection guide
  Fig 2  — Fitted vs Actual elimination rate for every season
  Fig 3  — ARMA forecast (one-step-ahead) + residual diagnostics heat-map
  Fig 4  — Model fit quality dashboard (AIC, RMSE, order p/q) across seasons
"""

import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

# ── colour palette ──────────────────────────────────────────────────────────
BACH_PINK  = "#C9184A"
BACH_ROSE  = "#FF4D6D"
BACH_LIGHT = "#FFB3C6"
BACH_GOLD  = "#F4A261"
BACH_TEAL  = "#2A9D8F"
BACH_DARK  = "#1D1B1E"
BG         = "#FFF8F9"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.edgecolor":    "#DDCCCC",
    "axes.labelcolor":   BACH_DARK,
    "xtick.color":       BACH_DARK,
    "ytick.color":       BACH_DARK,
    "text.color":        BACH_DARK,
    "font.family":       "DejaVu Sans",
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.color":        "#DDCCCC",
})


# ── 1.  LOAD & AGGREGATE ─────────────────────────────────────────────────────

def load_and_aggregate(path: str) -> pd.DataFrame:
    """Return one row per (SHOW, SEASON, WEEK) with computed rates."""
    raw = pd.read_csv(path)

    agg = (
        raw.groupby(["SHOW", "SEASON", "WEEK"])
        .agg(
            n_rows          = ("CONTESTANT", "count"),
            n_alive_start   = ("ALIVE",               "sum"),   # alive at start of week
            n_eliminated    = ("ELIMINATED_THIS_WEEK", "sum"),
            n_date_rose     = ("DATE_ROSE",            "sum"),
            n_weekly_rose   = ("WEEKLY_ROSE",          "sum"),
            n_has_date      = ("HAS_DATE",             "sum"),
        )
        .reset_index()
    )

    # Alive at START of the week = alive_now + eliminated_this_week
    agg["alive_start"] = agg["n_alive_start"] + agg["n_eliminated"]

    # Elimination rate: fraction of still-alive contestants sent home this week
    agg["elim_rate"] = agg["n_eliminated"] / agg["alive_start"].replace(0, np.nan)

    # Date-rose rate among those who got a date
    agg["date_rose_rate"] = agg["n_date_rose"] / agg["n_has_date"].replace(0, np.nan)

    # Normalised alive percentage (relative to Week-1 pool)
    # Normalise alive percentage
    first_alive_map = (
        agg.groupby(["SHOW", "SEASON"])["alive_start"].first()
    )
    agg["alive_pct"] = agg.apply(
        lambda r: r["alive_start"] / first_alive_map.loc[(r["SHOW"], r["SEASON"])]
        if first_alive_map.loc[(r["SHOW"], r["SEASON"])] > 0 else np.nan,
        axis=1
    )
    agg = agg.sort_values(["SHOW", "SEASON", "WEEK"]).reset_index(drop=True)
    return agg


# ── 2.  AUTO ORDER SELECTION ─────────────────────────────────────────────────

def select_arma_order(series: pd.Series,
                      p_max: int = 4,
                      q_max: int = 2) -> tuple[int, int]:
    """Grid search over (p, q) in [0..p_max] × [0..q_max], pick by AIC."""
    series = series.dropna()
    if len(series) < 6:
        return 1, 0          # too short to fit meaningfully

    best_aic, best_order = np.inf, (1, 0)
    for p, q in itertools.product(range(p_max + 1), range(q_max + 1)):
        if p == 0 and q == 0:
            continue
        try:
            res = ARIMA(series, order=(p, 0, q)).fit()
            if res.aic < best_aic:
                best_aic, best_order = res.aic, (p, q)
        except Exception:
            pass
    return best_order


# ── 3.  FIT ARMA PER SEASON ──────────────────────────────────────────────────

def fit_all_seasons(agg: pd.DataFrame,
                    signal: str = "elim_rate") -> list[dict]:
    """Fit ARMA to `signal` for every (SHOW, SEASON). Return list of result dicts."""
    results = []
    for (show, season), grp in agg.groupby(["SHOW", "SEASON"]):
        ts = grp.set_index("WEEK")[signal].dropna()
        if len(ts) < 5:
            continue

        p, q = select_arma_order(ts)
        try:
            model = ARIMA(ts, order=(p, 0, q)).fit()
        except Exception:
            continue

        fitted    = model.fittedvalues
        residuals = model.resid
        rmse      = np.sqrt(np.mean(residuals ** 2))

        # one-step-ahead forecast
        fc_obj    = model.get_forecast(steps=1)
        forecast  = fc_obj.predicted_mean.values[0]
        fc_ci     = fc_obj.conf_int().values[0]           # (lower, upper)

        # Ljung-Box test on residuals (last lag tested)
        lb = acorr_ljungbox(residuals, lags=[min(4, len(ts) - 2)], return_df=True)
        lb_pval = lb["lb_pvalue"].values[-1]

        results.append(dict(
            show=show, season=season,
            ts=ts, fitted=fitted, residuals=residuals,
            p=p, q=q, aic=model.aic, rmse=rmse,
            forecast=forecast, fc_ci=fc_ci,
            lb_pval=lb_pval,
            n_weeks=len(ts),
        ))
    return results


# ── 4.  FIGURE 1 — ACF / PACF ────────────────────────────────────────────────

def plot_acf_pacf(agg: pd.DataFrame,
                  signal: str = "elim_rate",
                  out: str = "fig1_acf_pacf.png"):
    """Average ACF/PACF across all seasons to suggest ARMA order."""
    max_lags = 6
    all_series = []
    for _, grp in agg.groupby(["SHOW", "SEASON"]):
        ts = grp.set_index("WEEK")[signal].dropna()
        if len(ts) >= max_lags + 2:
            ts_z = (ts - ts.mean()) / (ts.std() or 1)
            all_series.append(ts_z.values)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "ACF & PACF of Elimination Rate — Pooled Across All Seasons\n"
        "(Use to guide ARMA order selection: AR order ≈ last significant PACF lag, "
        "MA order ≈ last significant ACF lag)",
        fontsize=11, color=BACH_DARK, y=1.01
    )

    # Plot ACF/PACF on a single representative stacked series
    stacked = np.concatenate(all_series)
    plot_acf(stacked,  lags=max_lags, ax=axes[0], color=BACH_ROSE,
             alpha=0.05, vlines_kwargs={"colors": BACH_ROSE})
    plot_pacf(stacked, lags=max_lags, ax=axes[1], color=BACH_TEAL,
              alpha=0.05, vlines_kwargs={"colors": BACH_TEAL}, method="ywm")

    for ax, title in zip(axes, ["Autocorrelation Function (ACF)",
                                 "Partial Autocorrelation Function (PACF)"]):
        ax.set_title(title, fontsize=12, color=BACH_DARK)
        ax.set_xlabel("Lag (weeks)", fontsize=10)
        ax.axhline(0, color=BACH_DARK, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#DDCCCC")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✓ {out}")


# ── 5.  FIGURE 2 — FITTED VS ACTUAL (grid) ───────────────────────────────────

def plot_fitted_vs_actual(results: list[dict],
                          out: str = "fig2_fitted_actual.png"):
    """One subplot per season: actual elimination rate + ARMA fitted values."""
    n      = len(results)
    ncols  = 4
    nrows  = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 3.2),
                             squeeze=False)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "ARMA Fitted vs Actual — Weekly Elimination Rate per Season",
        fontsize=14, color=BACH_DARK, fontweight="bold", y=1.01
    )

    for idx, res in enumerate(results):
        ax   = axes[idx // ncols][idx % ncols]
        ts   = res["ts"]
        fit  = res["fitted"].reindex(ts.index)

        ax.plot(ts.index, ts.values,  color=BACH_ROSE,  lw=2,
                marker="o", markersize=4, label="Actual", zorder=3)
        ax.plot(fit.index, fit.values, color=BACH_TEAL, lw=1.8,
                linestyle="--", marker="s", markersize=3,
                label=f"ARMA({res['p']},{res['q']})", zorder=2)

        # one-step forecast marker
        next_week = ts.index.max() + 1
        ax.errorbar(next_week, res["forecast"],
                    yerr=[[res["forecast"] - res["fc_ci"][0]],
                          [res["fc_ci"][1]  - res["forecast"]]],
                    fmt="D", color=BACH_GOLD, markersize=6,
                    capsize=4, zorder=4, label="Forecast")

        title_color = BACH_DARK
        lb_str = f"LB p={res['lb_pval']:.2f}" if not np.isnan(res['lb_pval']) else ""
        ax.set_title(
            f"{res['show']} S{res['season']}  |  ARMA({res['p']},{res['q']})\n"
            f"AIC={res['aic']:.1f}  RMSE={res['rmse']:.3f}  {lb_str}",
            fontsize=7.5, color=title_color
        )
        ax.set_xlabel("Week", fontsize=8)
        ax.set_ylabel("Elim. Rate", fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=6, loc="upper right")

    # hide unused panels
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✓ {out}")


# ── 6.  FIGURE 3 — RESIDUAL HEAT-MAP ─────────────────────────────────────────

def plot_residual_heatmap(results: list[dict],
                          out: str = "fig3_residual_heatmap.png"):
    """Heat-map of ARMA residuals: rows = seasons, columns = weeks."""
    max_weeks = max(res["n_weeks"] for res in results)
    labels    = [f"{res['show'][:4]} S{res['season']}" for res in results]
    matrix    = np.full((len(results), max_weeks), np.nan)

    for i, res in enumerate(results):
        r = res["residuals"].values
        matrix[i, :len(r)] = r

    fig, ax = plt.subplots(figsize=(max(12, max_weeks * 0.9), max(6, len(results) * 0.45)))
    fig.patch.set_facecolor(BG)

    vmax = np.nanpercentile(np.abs(matrix), 95)
    cmap = plt.get_cmap("RdBu_r")
    im   = ax.imshow(matrix, aspect="auto", cmap=cmap,
                     vmin=-vmax, vmax=vmax, interpolation="nearest")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xticks(range(max_weeks))
    ax.set_xticklabels([f"W{w+1}" for w in range(max_weeks)], fontsize=8)
    ax.set_xlabel("Week", fontsize=10)
    ax.set_title(
        "ARMA Residual Heat-Map — All Seasons\n"
        "Red = model under-predicts eliminations  |  Blue = over-predicts",
        fontsize=11, color=BACH_DARK
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Residual (actual − fitted)", fontsize=9)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✓ {out}")


# ── 7.  FIGURE 4 — MODEL QUALITY DASHBOARD ───────────────────────────────────

def plot_model_dashboard(results: list[dict],
                         out: str = "fig4_model_dashboard.png"):
    """Summary dashboard: AIC, RMSE, selected (p,q) across all seasons."""
    df = pd.DataFrame([{
        "label":  f"{r['show'][:4]} S{r['season']:02d}",
        "show":   r["show"],
        "season": r["season"],
        "p":      r["p"],
        "q":      r["q"],
        "aic":    r["aic"],
        "rmse":   r["rmse"],
        "n":      r["n_weeks"],
        "lb":     r["lb_pval"],
    } for r in results]).sort_values(["show", "season"])

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "ARMA Model Quality Dashboard — All Bachelor / Bachelorette Seasons",
        fontsize=14, color=BACH_DARK, fontweight="bold"
    )

    gs   = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax1  = fig.add_subplot(gs[0, 0])
    ax2  = fig.add_subplot(gs[0, 1])
    ax3  = fig.add_subplot(gs[1, 0])
    ax4  = fig.add_subplot(gs[1, 1])

    colors = [BACH_ROSE if s == "Bachelor" else BACH_TEAL for s in df["show"]]

    # — AIC by season —
    ax1.bar(df["label"], df["aic"], color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_title("AIC by Season  (lower = better fit)", fontsize=10)
    ax1.set_ylabel("AIC")
    ax1.tick_params(axis="x", rotation=75, labelsize=6.5)
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color=BACH_ROSE, label="Bachelor"),
                         Patch(color=BACH_TEAL, label="Bachelorette")],
               fontsize=8)

    # — RMSE by season —
    ax2.bar(df["label"], df["rmse"], color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_title("RMSE by Season  (lower = tighter fit)", fontsize=10)
    ax2.set_ylabel("RMSE (elimination rate)")
    ax2.tick_params(axis="x", rotation=75, labelsize=6.5)

    # — Selected (p, q) scatter —
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(df))
    sc = ax3.scatter(df["p"] + jitter, df["q"] + jitter,
                     c=[BACH_ROSE if s == "Bachelor" else BACH_TEAL for s in df["show"]],
                     s=70, edgecolors="white", linewidths=0.7, alpha=0.85)
    ax3.set_xlabel("AR order (p)", fontsize=10)
    ax3.set_ylabel("MA order (q)", fontsize=10)
    ax3.set_title("Selected ARMA(p, q) Orders", fontsize=10)
    ax3.set_xticks(range(int(df["p"].max()) + 1))
    ax3.set_yticks(range(int(df["q"].max()) + 1))

    # annotate counts
    for (p, q), sub in df.groupby(["p", "q"]):
        ax3.text(p, q + 0.22, f"n={len(sub)}", ha="center", fontsize=7.5,
                 color=BACH_DARK)

    # — Ljung-Box p-values —
    bar_colors = [BACH_GOLD if v >= 0.05 else "#E63946" for v in df["lb"]]
    ax4.bar(df["label"], df["lb"], color=bar_colors, edgecolor="white", linewidth=0.5)
    ax4.axhline(0.05, color=BACH_DARK, linestyle="--", lw=1.2,
                label="α = 0.05  (residuals should be ≥ this)")
    ax4.set_title("Ljung-Box p-value  (residual white-noise test)", fontsize=10)
    ax4.set_ylabel("p-value")
    ax4.tick_params(axis="x", rotation=75, labelsize=6.5)
    ax4.legend(fontsize=8)
    ax4.set_ylim(0, 1)
    from matplotlib.patches import Patch
    ax4.legend(handles=[
        Patch(color=BACH_GOLD,  label="Residuals ≈ white noise (good)"),
        Patch(color="#E63946",  label="Residual autocorrelation (re-specify)"),
    ] + [plt.Line2D([0],[0], color=BACH_DARK, linestyle="--", label="α = 0.05")],
               fontsize=7.5)

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✓ {out}")


# ── 8.  TEXT SUMMARY ─────────────────────────────────────────────────────────

def print_summary(results: list[dict]):
    df = pd.DataFrame([{
        "Show":   r["show"],
        "Season": r["season"],
        "Weeks":  r["n_weeks"],
        "p":      r["p"],
        "q":      r["q"],
        "AIC":    round(r["aic"],  2),
        "RMSE":   round(r["rmse"], 4),
        "LB_p":   round(r["lb_pval"], 3),
        "Forecast (next-wk elim rate)": round(r["forecast"], 3),
    } for r in results]).sort_values(["Show", "Season"])

    print("\n" + "=" * 90)
    print("  ARMA RESULTS — ELIMINATION RATE TIME SERIES")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)

    good = (df["LB_p"] >= 0.05).sum()
    print(f"\n  Seasons with white-noise residuals (LB p ≥ 0.05): {good}/{len(df)}")
    best = df.sort_values("RMSE").iloc[0]
    print(f"  Best RMSE: {best['Show']} Season {best['Season']}  RMSE={best['RMSE']}")
    most_common = df.groupby(["p","q"]).size().idxmax()
    print(f"  Most-selected order: ARMA{most_common}\n")


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA = "/mnt/user-data/uploads/contestant_weekly.csv"
    OUT  = "/mnt/user-data/outputs"

    import os
    os.makedirs(OUT, exist_ok=True)

    print("\n[1/5]  Loading and aggregating data …")
    agg = load_and_aggregate(DATA)
    print(f"       {agg['SEASON'].nunique()} seasons × {agg['WEEK'].max()} max weeks")

    print("\n[2/5]  Fitting ARMA models …")
    results = fit_all_seasons(agg, signal="elim_rate")
    print(f"       Fitted {len(results)} season models")

    print("\n[3/5]  Generating Figure 1 — ACF / PACF …")
    plot_acf_pacf(agg, signal="elim_rate",
                  out=f"{OUT}/fig1_acf_pacf.png")

    print("\n[4/5]  Generating Figure 2 — Fitted vs Actual …")
    plot_fitted_vs_actual(results,
                          out=f"{OUT}/fig2_fitted_actual.png")

    print("\n[4b]   Generating Figure 3 — Residual Heat-Map …")
    plot_residual_heatmap(results,
                          out=f"{OUT}/fig3_residual_heatmap.png")

    print("\n[5/5]  Generating Figure 4 — Model Quality Dashboard …")
    plot_model_dashboard(results,
                         out=f"{OUT}/fig4_model_dashboard.png")

    print_summary(results)
    print("\n  All outputs saved to", OUT)
