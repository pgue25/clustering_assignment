"""
Gym member segments: who should get the monthly $10 extras credit?

Reproduces every number in the slides:
  python gym_segments.py
Outputs go to ./outputs (elbow plot, centers, segment table, member assignments).
"""
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
DATA = Path(__file__).parent / "data" / "gym_members.csv"
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

# Usage + who the member is. Left out: member_id (an identifier) and
# monthly_extras_spend (spending, not usage; it is the outcome the credit targets).
FEATURES = ["age", "visits_per_month", "avg_visit_minutes",
            "classes_per_month", "months_member"]
SPEND = "monthly_extras_spend"
K_RANGE = range(1, 9)
CHOSEN_K = 4

# Finance assumptions (per member per month)
CREDIT_COST = 10
SPEND_THRESHOLD = 20      # members spending >= $20 would spend the same anyway
LIFT_MARGIN = 25 * 0.60   # $25 extra spend at 60% margin = $15


def credit_net(spend):
    """Gym's monthly net from giving one member the credit (finance's rules)."""
    spend = np.asarray(spend)
    return np.where(spend >= SPEND_THRESHOLD, -CREDIT_COST, LIFT_MARGIN - CREDIT_COST)


# ---------------------------------------------------------------- 1. Split
df = pd.read_csv(DATA)
assert df.shape == (330, 7)
print(f"Rows: {len(df)}, avg extras spend: ${df[SPEND].mean():.2f}")

train, test = train_test_split(df, test_size=0.30, random_state=SEED)
print(f"Train: {len(train)}  Test: {len(test)}")

scaler = StandardScaler().fit(train[FEATURES])          # scaling fit on train only
X_train = scaler.transform(train[FEATURES])
X_test = scaler.transform(test[FEATURES])

# ---------------------------------------------------------------- 2. Fit k = 1..8
models, train_sse, test_sse = {}, [], []
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X_train)
    models[k] = km
    train_sse.append(km.inertia_)
    # 3. Test: no refit; nearest training center, sum of squared distances
    d2 = ((X_test[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
    test_sse.append(d2.min(axis=1).sum())

spread = pd.DataFrame({"k": list(K_RANGE),
                       "train_spread": np.round(train_sse, 1),
                       "test_spread": np.round(test_sse, 1)})
spread["test_pct_drop_vs_prev_k"] = (
    (1 - spread["test_spread"] / spread["test_spread"].shift()) * 100).round(1)
spread.to_csv(OUT / "spread_by_k.csv", index=False)
print("\nWithin-group spread by k (standardized units):")
print(spread.to_string(index=False))

base, v1 = test_sse[0], test_sse[CHOSEN_K - 1]
print(f"\nTest spread k=1 (baseline): {base:.1f}")
print(f"Test spread k={CHOSEN_K} (v1):       {v1:.1f}  "
      f"({(1 - v1 / base) * 100:.0f}% lower than baseline)")

# Elbow plot: train and test side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
for ax, vals, title in [(axes[0], train_sse, f"Training members (n={len(train)})"),
                        (axes[1], test_sse, f"Test members (n={len(test)}, not refit)")]:
    ax.plot(list(K_RANGE), vals, marker="o", color="black")
    ax.plot(CHOSEN_K, vals[CHOSEN_K - 1], marker="o", markersize=12,
            markerfacecolor="none", markeredgecolor="red", markeredgewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Number of groups (k)")
    ax.set_xticks(list(K_RANGE))
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Within-group spread\n(sum of squared distances)")
fig.suptitle(f"Elbow at k = {CHOSEN_K} on both training and test members")
fig.tight_layout()
fig.savefig(OUT / "elbow_train_test.png", dpi=200)

# Same plot, both curves on one axis (alternative for the slide)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(list(K_RANGE), train_sse, marker="o", color="black", label=f"Training (n={len(train)})")
ax.plot(list(K_RANGE), test_sse, marker="s", color="gray", linestyle="--",
        label=f"Test (n={len(test)})")
ax.axvline(CHOSEN_K, color="red", alpha=0.4)
ax.set_xlabel("Number of groups (k)")
ax.set_ylabel("Within-group spread")
ax.set_xticks(list(K_RANGE))
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "elbow_combined.png", dpi=200)

# ---------------------------------------------------------------- Describe groups
km = models[CHOSEN_K]
centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=FEATURES)


def name_cluster(c):
    if c["months_member"] < 12 and c["visits_per_month"] < 6:
        return "Newcomers"
    if c["age"] >= 55:
        return "Seasoned Regulars"
    if c["classes_per_month"] >= 4:
        return "Class Enthusiasts"
    return "Solo Grinders"


NAMES = {i: name_cluster(centers.loc[i]) for i in centers.index}
assert len(set(NAMES.values())) == CHOSEN_K, NAMES
centers.insert(0, "segment", centers.index.map(NAMES))
centers["n_train"] = np.bincount(km.labels_, minlength=CHOSEN_K)
centers["train_avg_extras_spend"] = train.groupby(km.labels_)[SPEND].mean().values
print("\nCluster centers (training members, original units):")
print(centers.round(1).to_string(index=False))
centers.round(2).to_csv(OUT / "cluster_centers.csv", index=False)

# ---------------------------------------------------------------- 4. Decision
# A segment gets the credit if, on the training members, giving it to
# everyone in that segment nets the gym money (member by member).
train_net_by_seg = pd.Series(credit_net(train[SPEND]), index=train.index) \
    .groupby(km.labels_).sum()
GETS_CREDIT = {i: bool(train_net_by_seg.loc[i] > 0) for i in range(CHOSEN_K)}
print("\nTraining net if segment gets credit:",
      {NAMES[i]: float(v) for i, v in train_net_by_seg.items()})


def assign_member(age, visits_per_month, avg_visit_minutes, classes_per_month,
                  months_member, monthly_extras_spend=None):
    """Return (segment name, gets_credit) for one member.

    monthly_extras_spend is accepted but not used for the segment: segments
    are defined by how the member uses the gym and who they are.
    """
    x = pd.DataFrame([[age, visits_per_month, avg_visit_minutes,
                       classes_per_month, months_member]], columns=FEATURES)
    label = int(km.predict(scaler.transform(x))[0])
    return NAMES[label], GETS_CREDIT[label]


new_member = dict(age=30, visits_per_month=4, avg_visit_minutes=42,
                  classes_per_month=1, months_member=5, monthly_extras_spend=10)
seg, credit = assign_member(**new_member)
print(f"\nNew member {new_member} -> {seg}, gets credit: {credit}")

# ---------------------------------------------------------------- 5. Quantify
all_labels = km.predict(scaler.transform(df[FEATURES]))
df["segment"] = pd.Series(all_labels).map(NAMES).values
df["set"] = np.where(df.index.isin(train.index), "train", "test")
df["gets_credit_v1"] = pd.Series(all_labels).map(GETS_CREDIT).values
df["net_if_credit"] = credit_net(df[SPEND])
df["net_baseline"] = df["net_if_credit"]
df["net_v1"] = np.where(df["gets_credit_v1"], df["net_if_credit"], 0)
df.to_csv(OUT / "member_assignments.csv", index=False)

seg_table = df.groupby("segment").agg(
    members=("member_id", "size"),
    age=("age", "mean"),
    visits_per_month=("visits_per_month", "mean"),
    avg_visit_minutes=("avg_visit_minutes", "mean"),
    classes_per_month=("classes_per_month", "mean"),
    months_member=("months_member", "mean"),
    avg_extras_spend=(SPEND, "mean"),
    pct_spend_under_20=(SPEND, lambda s: (s < SPEND_THRESHOLD).mean() * 100),
    gets_credit=("gets_credit_v1", "first"),
    net_baseline=("net_baseline", "sum"),
    net_v1=("net_v1", "sum"),
).round(1)
seg_table = seg_table.sort_values("avg_extras_spend")
seg_table.to_csv(OUT / "segment_table.csv")
print("\nSegment table (all 330 members):")
print(seg_table.T.to_string())

nb, nv = df["net_baseline"].sum(), df["net_v1"].sum()
summary = pd.DataFrame({
    "plan": ["Baseline: credit to everyone", f"v1: credit to {', '.join(n for i, n in NAMES.items() if GETS_CREDIT[i])}"],
    "members_getting_credit": [len(df), int(df["gets_credit_v1"].sum())],
    "credit_cost": [CREDIT_COST * len(df), CREDIT_COST * int(df["gets_credit_v1"].sum())],
    "monthly_net": [nb, nv],
})
summary.to_csv(OUT / "value_summary.csv", index=False)
print("\nMonthly net from the credit:")
print(summary.to_string(index=False))
print(f"Improvement: ${nv - nb:+,.0f} per month (${(nv - nb) * 12:+,.0f} per year)")

# ---------------------------------------------------------------- Segment chart
# Members colored by segment; red dots = segment centers (the "average member"
# fitted on the training set). Orange is skipped so no segment color is near red.
SEG_COLORS = {"Newcomers": "#2a78d6", "Seasoned Regulars": "#1baf7a",
              "Solo Grinders": "#eda100", "Class Enthusiasts": "#e87ba4"}
rng = np.random.default_rng(SEED)
jit = lambda s, w: s + rng.uniform(-w, w, len(s))   # spreads out whole-number ties

panels = [("visits_per_month", "classes_per_month", "Visits per month", "Classes per month"),
          ("age", "months_member", "Age", "Months a member")]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (xc, yc, xl, yl) in zip(axes, panels):
    for name, color in SEG_COLORS.items():
        m = df[df["segment"] == name]
        ax.scatter(jit(m[xc], 0.3), jit(m[yc], 0.3), s=28, color=color, alpha=0.75,
                   edgecolor="white", linewidth=0.6, label=f"{name} ({len(m)})")
    ax.scatter(centers[xc], centers[yc], s=180, color="red", edgecolor="white",
               linewidth=2, zorder=5, label="Segment center")
    for _, c in centers.iterrows():
        ax.annotate(c["segment"], (c[xc], c[yc]), xytext=(9, 9),
                    textcoords="offset points", fontsize=10, fontweight="bold",
                    color="#222222", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=10)
fig.suptitle("Four member segments (all 330 members); red dots are segment centers")
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT / "segments_scatter.png", dpi=200)
print(f"\nSaved {OUT / 'segments_scatter.png'}")

# ---------------------------------------------------------------- Value chart
n_v1 = int(df["gets_credit_v1"].sum())
fig, ax = plt.subplots(figsize=(6.5, 4))
plans = ["Baseline:\ncredit to everyone", "v1:\ncredit to Newcomers only"]
vals = [nb, nv]
bars = ax.bar(plans, vals, width=0.5, color=["#bbbbbb", "#1f4e79"])
labels = [f"-${abs(nb):,.0f}/month\n{len(df)} members credited",
          f"+${nv:,.0f}/month\n{n_v1} members credited"]
for b, v, lab in zip(bars, vals, labels):
    ax.annotate(lab, (b.get_x() + b.get_width() / 2, v), ha="center",
                va="top" if v < 0 else "bottom", xytext=(0, -6 if v < 0 else 6),
                textcoords="offset points", fontsize=11)
ax.axhline(0, color="black", linewidth=1)
ax.set_ylabel("Gym's monthly net from the \\$10 credit")
ax.set_ylim(-1700, 700)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda x, _: f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"))
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "value_baseline_vs_v1.png", dpi=200)
print(f"Saved {OUT / 'value_baseline_vs_v1.png'}")
