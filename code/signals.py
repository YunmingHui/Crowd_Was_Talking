"""
Compute the monthly signals of every case stock and reference stock from the released files (../data).

A stock-month is evaluated if the stock is mentioned at least once in it. For case stocks, only mentions before the
SEC's first public action against the company count, and only months with at least one day before that action.

    mentions       number of posts and comments that mention the stock
    volume         mentions / all posts and comments of the subreddits in the month
    surge          share of the observed days of the month in the surge state of an online two-state burst detection
                   model (Kleinberg 2002) on the daily share of the stock in all posts and comments: surge rate
                   BETA x the usual share, usual share from the BASE_DAYS days before (plus one mention), entry cost
                   GAMMA x ln(number of days), state of each day from the days up to it only
    reemergence    days without any mention before the first mention of the month (counted from the start of the data
                   if the stock was not mentioned before)
    newcomers      share of the month's authors who mention the stock for the first time in the data (not computed in
                   the first BURN_IN months, where nearly every author is new)
    coordination   share of the month's mentions flagged as duplicate texts (same or near-same text by another author
                   within one hour; flag precomputed in mentions.csv.gz)
    allegations    share of the month's labelled mentions that allege misconduct (manipulation, insider trading, or both)
    bullishness    (bullish - bearish) / (bullish + bearish) among the month's labelled mentions, 0 if there is no
                   bullish or bearish mention

Usage:   python signals.py          (needs numpy and pandas)
Output:  ../data/signals_monthly.csv   one row per evaluated stock-month
"""

import os

import numpy as np
import pandas as pd

# ---------------- settings ----------------
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
FIRST_MONTH, LAST_MONTH = "2019-10", "2022-03"   # data period
BETA, GAMMA, BASE_DAYS = 2.0, 1.0, 90            # surge model
BURN_IN = 3                                      # newcomers: first months of the data without a history
ALLEGATIONS = {"manipulation", "insider", "both"}
TZ = "America/New_York"                          # the SEC's action date starts at 0:00 in this time zone
# ------------------------------------------


def online_burst(r, d, beta=BETA, gamma=GAMMA, base=BASE_DAYS):
    """Daily states (0 base, 1 surge) for r mentions among d posts and comments per day."""
    n = len(r)
    cr, cd = np.concatenate([[0], np.cumsum(r)]), np.concatenate([[0], np.cumsum(d)])
    up = gamma * np.log(n)
    tot = np.array([0.0, up])
    q = np.zeros(n, dtype=int)
    for t in range(n):
        lo = max(0, t - base)
        p0 = min((cr[t] - cr[lo] + 1) / max(cd[t] - cd[lo], 1), 0.5)
        p1 = min(beta * p0, 0.9999)
        c = np.array([-(r[t] * np.log(p) + (d[t] - r[t]) * np.log(1 - p)) for p in (p0, p1)])
        tot = np.array([min(tot[0], tot[1]), min(tot[1], tot[0] + up)]) + c
        q[t] = int(np.argmin(tot))
    return q


def main():
    cases = pd.read_csv(os.path.join(DATA, "cases.csv"))
    ref = pd.read_csv(os.path.join(DATA, "reference_stocks.csv"))
    act = pd.read_csv(os.path.join(DATA, "daily_activity.csv")).sort_values("date")
    men = pd.read_csv(os.path.join(DATA, "mentions.csv.gz"))
    # case stocks: mentions before 0:00 US Eastern time on the day of the SEC's first public action
    cut = {c: pd.Timestamp(d).tz_localize(TZ).timestamp() for c, d in zip(cases["case_id"], cases["first_sec_action"])}
    men = men[men["created_utc"] < men["stock_id"].map(cut).fillna(np.inf)].copy()

    months = pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M")
    midx = {p: i for i, p in enumerate(months)}
    days = pd.to_datetime(act["date"])
    day_m = np.array([midx[p] for p in days.dt.to_period("M")])
    items_day = act["posts_and_comments"].to_numpy(float)
    items_month = np.bincount(day_m, weights=items_day, minlength=len(months))

    # last day before the SEC's first public action (case stocks)
    action = dict(zip(cases["case_id"], pd.to_datetime(cases["first_sec_action"])))

    t = pd.to_datetime(men["created_utc"], unit="s", utc=True)
    men["m"] = [midx.get(p, -1) for p in t.dt.tz_localize(None).dt.to_period("M")]
    men = men[men["m"] >= 0]
    men["day"] = t.dt.strftime("%Y-%m-%d").to_numpy()
    men["alleg"] = men["allegation"].isin(ALLEGATIONS).astype(float)
    men["bull"] = (men["stance"] == "bullish").astype(float)
    men["bear"] = (men["stance"] == "bearish").astype(float)
    men["labelled"] = men["stance"].notna() & men["allegation"].notna()

    # newcomers: first month in which each author mentions each stock
    a = men[men["author_idx"].notna()][["stock_id", "author_idx", "m"]].drop_duplicates()
    a = a.join(a.groupby(["stock_id", "author_idx"])["m"].min().rename("first"), on=["stock_id", "author_idx"])
    a["new"] = (a["m"] == a["first"]).astype(float)
    newc = a[a["m"] >= BURN_IN].groupby(["stock_id", "m"])["new"].mean()

    lab = men[men["labelled"]].groupby(["stock_id", "m"]).agg(alleg=("alleg", "mean"), bull=("bull", "sum"),
                                                             bear=("bear", "sum"))
    per_month = men.groupby(["stock_id", "m"]).agg(mentions=("m", "size"), duplicates=("duplicate", "sum"))

    stocks = list(cases["case_id"]) + list(ref["stock_id"])
    group = {s: "case" for s in cases["case_id"]} | {s: "reference" for s in ref["stock_id"]}
    daily = men.groupby(["stock_id", "day"]).size()
    day_str = act["date"].to_numpy()
    start = pd.Timestamp(FIRST_MONTH + "-01")
    rows = []
    for s in stocks:
        if s not in per_month.index.get_level_values(0):
            continue
        pm = per_month.loc[s]
        if s in action:
            ok = (days < action[s]).to_numpy()
        else:
            ok = np.ones(len(days), dtype=bool)
        r = daily[s].reindex(day_str, fill_value=0).to_numpy(float) if s in daily.index.get_level_values(0) \
            else np.zeros(len(days))
        q = online_burst(np.minimum(r, items_day), items_day)
        surge_days = np.bincount(day_m[ok], weights=q[ok], minlength=len(months))
        observed = np.bincount(day_m[ok], minlength=len(months))
        # reemergence: silent days before the first mention of each month
        d_ = pd.to_datetime(pd.Index(daily[s].index)).sort_values()
        prev = pd.DatetimeIndex([start - pd.Timedelta(days=1)]).append(d_[:-1])
        gap = (d_ - prev).days - 1
        dm = np.array([midx[p] for p in d_.to_period("M")])
        firsts = np.r_[True, dm[1:] != dm[:-1]]
        reem = dict(zip(dm[firsts], np.asarray(gap)[firsts]))
        for m, row in pm.iterrows():
            if observed[m] == 0:
                continue
            bl = np.nan
            if (s, m) in lab.index:
                L = lab.loc[(s, m)]
                bl = (L.bull - L.bear) / (L.bull + L.bear) if L.bull + L.bear > 0 else 0.0
            rows.append({"stock_id": s, "group": group[s], "month": str(months[m]), "mentions": int(row.mentions),
                         "volume": row.mentions / items_month[m], "surge": surge_days[m] / observed[m],
                         "reemergence": reem.get(m, np.nan), "newcomers": newc.get((s, m), np.nan),
                         "coordination": row.duplicates / row.mentions,
                         "allegations": lab["alleg"].get((s, m), np.nan), "bullishness": bl})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA, "signals_monthly.csv"), index=False)
    print(f"{len(out):,} stock-months ({(out.group == 'case').sum():,} case, {(out.group == 'reference').sum():,} "
          f"reference). Saved {os.path.join(DATA, 'signals_monthly.csv')}")


if __name__ == "__main__":
    main()
