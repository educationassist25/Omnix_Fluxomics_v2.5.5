"""
core.barplots — dynamic bar plots (v2.2, replaces the box plots).

One plot per metabolite: x = isotopologue (M+0 … M+n); within each isotopologue one bar per group × time point
(group hue, lighter = earlier time). Bar height = mean of the biological replicates, error bar = SD or SEM,
dots = individual replicates. Isotopologues are never summed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .style import INK, INK2, GRID, group_colors

FONT = dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=12, color=INK)
ERRORS = {"SD": "SD", "SEM": "SEM", "none": "No error bar"}


def condition_colors(arms: list, times: list) -> dict:
    """Group hue, lighter → darker with time."""
    import matplotlib.colors as mc
    gc = group_colors(arms)
    out = {}
    for g in arms:
        base = np.array(mc.to_rgb(gc[g]))
        for k, h in enumerate(times):
            f = 0.55 - 0.55 * (k / max(len(times) - 1, 1))          # 0.55 (light) … 0 (full colour)
            out[(g, h)] = mc.to_hex(base + (1 - base) * f)
    return out


def bar_summary(X: pd.DataFrame, arms: list, times: list, isos: list) -> pd.DataFrame:
    """Mean / SD / SEM / n per isotopologue × group × time (X: Isotopologue, Arm, Hours, TimeLabel, Value)."""
    rows = []
    for g in arms:
        for h in times:
            s = X[(X.Arm == g) & np.isclose(X.Hours, h)]
            if s.empty:
                continue
            for iso in isos:
                v = s[s.Isotopologue == iso].Value.dropna().to_numpy(float)
                n = len(v)
                rows.append(dict(Isotopologue=iso, Group=g, Hours=h, Time=s.TimeLabel.iloc[0], n=n,
                                 Mean=float(v.mean()) if n else np.nan,
                                 SD=float(v.std(ddof=1)) if n > 1 else np.nan,
                                 SEM=float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan))
    return pd.DataFrame(rows)


def _conds(X, arms, times):
    return [(g, h) for g in arms for h in times if len(X[(X.Arm == g) & np.isclose(X.Hours, h)])]


def iso_bar_plotly(X: pd.DataFrame, arms: list, times: list, isos: list, ylab: str, title: str = "",
                   points: bool = True, error: str = "SD", height: int = 500, xlab: str = "Isotopologue"):
    import plotly.graph_objects as go
    cc = condition_colors(arms, times)
    conds = _conds(X, arms, times)
    T = bar_summary(X, arms, times, isos)
    k_n = max(len(conds), 1)
    w = 0.84 / k_n
    xi = {iso: j for j, iso in enumerate(isos)}
    rng = np.random.default_rng(0)
    fig = go.Figure()
    for k, (g, h) in enumerate(conds):
        off = (k - (k_n - 1) / 2) * w
        d = T[(T.Group == g) & np.isclose(T.Hours, h)]
        lab = f"{g} · {d.Time.iloc[0]}"
        err = d[error].to_numpy(float) if error in ("SD", "SEM") else None
        fig.add_trace(go.Bar(x=[xi[i] + off for i in d.Isotopologue], y=d.Mean, width=w * 0.92, name=lab,
                             legendgroup=lab, marker=dict(color=cc[(g, h)], line=dict(width=0)),
                             error_y=dict(type="data", array=np.nan_to_num(err), visible=True, thickness=1.1,
                                          width=2.5, color="#3a3940") if err is not None else None,
                             customdata=np.column_stack([
                                 d.Isotopologue.astype(str).to_numpy(), d.n.to_numpy(),
                                 np.nan_to_num(d[error].to_numpy(float)) if err is not None else np.zeros(len(d))]),
                             hovertemplate="%{customdata[0]}<br>mean %{y:.3g}"
                                           + (f" ± %{{customdata[2]:.2g}} ({error})" if err is not None else "")
                                           + "<br>n = %{customdata[1]}<extra>" + lab + "</extra>"))
        if points:
            s = X[(X.Arm == g) & np.isclose(X.Hours, h) & X.Isotopologue.isin(isos)].dropna(subset=["Value"])
            xs = np.array([xi[i] + off for i in s.Isotopologue]) + rng.uniform(-w * 0.22, w * 0.22, len(s))
            fig.add_trace(go.Scatter(x=xs, y=s.Value, mode="markers", legendgroup=lab, showlegend=False,
                                     marker=dict(size=5, color="white", line=dict(color="#2b2a30", width=0.9)),
                                     customdata=np.c_[s.Sample, s.Isotopologue],
                                     hovertemplate="%{customdata[0]}<br>%{customdata[1]}: %{y:.3g}<extra></extra>"))
    fig.update_layout(template="plotly_white", font=FONT, height=height, barmode="overlay", bargap=0,
                      title=dict(text=title, x=0, font=dict(size=15)), margin=dict(l=60, r=20, t=60, b=50),
                      legend=dict(orientation="h", y=-0.14, x=0, font=dict(size=11)),
                      yaxis_title=ylab, xaxis_title=xlab)
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(isos))), ticktext=isos,
                     range=[-0.6, len(isos) - 0.4], showgrid=False)
    fig.update_yaxes(gridcolor=GRID, rangemode="tozero")
    return fig


def iso_bar_mpl(X: pd.DataFrame, arms: list, times: list, isos: list, ylab: str, title: str = "", points=True,
                error: str = "SD", width=10.0, height=4.2, xlab: str = ""):
    import matplotlib.pyplot as plt
    cc = condition_colors(arms, times)
    conds = _conds(X, arms, times)
    T = bar_summary(X, arms, times, isos)
    k_n = max(len(conds), 1)
    w = 0.84 / k_n
    xi = {iso: j for j, iso in enumerate(isos)}
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(width, height))
    for k, (g, h) in enumerate(conds):
        off = (k - (k_n - 1) / 2) * w
        d = T[(T.Group == g) & np.isclose(T.Hours, h)]
        pos = np.array([xi[i] + off for i in d.Isotopologue])
        err = d[error].to_numpy(float) if error in ("SD", "SEM") else None
        ax.bar(pos, d.Mean, width=w * 0.92, color=cc[(g, h)], label=f"{g} · {d.Time.iloc[0]}", zorder=2,
               yerr=None if err is None else np.nan_to_num(err), error_kw=dict(lw=0.8, capsize=1.8, capthick=0.8,
                                                                              ecolor="#3a3940"))
        if points:
            s = X[(X.Arm == g) & np.isclose(X.Hours, h) & X.Isotopologue.isin(isos)].dropna(subset=["Value"])
            xs = np.array([xi[i] + off for i in s.Isotopologue]) + rng.uniform(-w * 0.2, w * 0.2, len(s))
            ax.scatter(xs, s.Value, s=8, facecolor="white", edgecolor="#2b2a30", lw=0.6, zorder=3)
    ax.set_xticks(range(len(isos))); ax.set_xticklabels(isos, fontsize=8)
    ax.set_xlim(-0.6, len(isos) - 0.4)
    ax.set_ylabel(ylab, fontsize=9); ax.tick_params(labelsize=8)
    if xlab:
        ax.set_xlabel(xlab, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, ncol=min(len(conds), 6), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    cap = {"SD": "mean ± SD", "SEM": "mean ± SEM"}.get(error, "mean")
    ax.set_title(f"{title}  ({cap}; dots = replicates)" if points else f"{title}  ({cap})", loc="left", fontsize=11)
    fig.tight_layout()
    return fig
