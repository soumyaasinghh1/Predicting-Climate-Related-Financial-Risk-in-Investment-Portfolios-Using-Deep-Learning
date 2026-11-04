# %% [markdown]
# # Deep Learning Climate-Scenario Stress Testing of Sector Volatility
# Task: forecast each sector's next-21-day realised volatility from daily market + macro data (4,000+ samples).
# Models: LSTM vs gated Transformer vs LSTM-Transformer hybrid, against honest benchmarks (train mean,
# persistence, HAR-RV regression, ridge). The best-performing forecaster is then stressed with NGFS scenario shocks.
# Run each `# %%` cell top to bottom (use Run All after any kernel restart).

# %% 0. Setup and configuration
import os, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import torch, torch.nn as nn
warnings.filterwarnings("ignore")

OFFLINE_DEMO = os.environ.get("DEMO") == "1"      # synthetic data, only for testing the pipeline
START, END = "2008-07-01", "2026-10-31"           # ICLN (green proxy) launched mid-2008
SECTORS = ["XLE", "XLU", "XLB", "XLI", "XLK", "XLF", "XLV", "XLP", "XLY", "ICLN"]
A = len(SECTORS)
LOOKBACK = 63                                     # trading days of history the models see (~3 months)
HORIZON = 21                                      # predict realised vol over the next 21 trading days
TRAIN_END, VAL_END = pd.Timestamp("2018-12-31"), pd.Timestamp("2021-12-31")   # test = 2022 onwards
EMBARGO = pd.Timedelta(days=30)                   # gap after each split so overlapping targets cannot leak
N_SEEDS = 3                                       # models averaged per architecture
SCENARIOS = ["Net Zero 2050", "Delayed Transition", "Current Policies"]
YEARS = [2030, 2040, 2050]                        # stress snapshots
COLORS = {"Net Zero 2050": "tab:green", "Delayed Transition": "tab:red", "Current Policies": "tab:blue"}

# --- STRESS CALIBRATION (ASSUMPTIONS, not estimates; state them in your report) ---
# Shift applied to the model's input features, in standard deviations of each feature:
C_REF = 100.0       # reference carbon price, USD/tCO2
K_BMG = -0.10       # brown-minus-green return shift per C_REF of carbon price increase
K_OIL = -0.06       # oil return shift per C_REF of carbon price increase
K_VIX = 0.15        # VIX (uncertainty) shift per 1% of GDP lost vs. today

os.makedirs("outputs", exist_ok=True); os.makedirs("data", exist_ok=True)
np.random.seed(0); torch.manual_seed(0)

# %% 1. Download DAILY market + macro data (cached in data/ so reruns are instant)
FRED = {"y10": "DGS10", "vix": "VIXCLS", "oil": "DCOILWTICO"}      # all daily series

def synthetic():                                   # volatility-clustering fake data, only to test the code
    idx = pd.bdate_range(START, END); n = len(idx); rng = np.random.default_rng(0)
    lv = np.zeros(n); idio = np.zeros((n, A))
    for t in range(1, n):
        lv[t] = 0.98 * lv[t - 1] + 0.12 * rng.normal(); idio[t] = 0.95 * idio[t - 1] + 0.08 * rng.normal(size=A)
    sig = 0.009 * np.exp(lv[:, None] + idio)
    px = pd.DataFrame(100 * np.exp(np.cumsum(sig * rng.normal(size=(n, A)), 0)), index=idx, columns=SECTORS)
    fr = pd.DataFrame({"y10": 3 + np.cumsum(rng.normal(0, 0.02, n)), "vix": 20 * np.exp(0.5 * lv),
                       "oil": 60 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))}, index=idx)
    return px, fr

def load_data():
    if OFFLINE_DEMO: return synthetic()
    if os.path.exists("data/prices_daily.csv") and os.path.exists("data/fred_daily.csv"):
        px = pd.read_csv("data/prices_daily.csv", index_col=0, parse_dates=True)
        fr = pd.read_csv("data/fred_daily.csv", index_col=0, parse_dates=True)
        return px, fr
    import yfinance as yf
    px = yf.download(SECTORS, start=START, end=END, interval="1d", auto_adjust=True, progress=False)["Close"][SECTORS].dropna()
    fr = pd.DataFrame({k: pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", na_values=".",
                                      index_col=0, parse_dates=True).iloc[:, 0] for k, sid in FRED.items()})
    px.to_csv("data/prices_daily.csv"); fr.to_csv("data/fred_daily.csv")
    return px, fr

px, fr = load_data()
print(px.shape, px.index.min().date(), px.index.max().date()); print(px.tail(3)); print(fr.tail(3))

# %% 2. NGFS scenarios -> tidy table (scenario, year, carbon, gdp_pct)
def load_ngfs(path="data/ngfs_scenarios.csv"):
    raw = pd.read_csv(path)
    if "Variable" not in raw.columns:                       # already tidy
        return raw[["scenario", "year", "carbon", "gdp_pct"]]
    yrs = [c for c in raw.columns if str(c).isdigit()]
    keys = {"Net Zero 2050": "net", "Delayed Transition": "delayed", "Current Policies": "current"}
    out = {}
    for name, kw in keys.items():
        sub = raw[raw["Scenario"].str.lower().str.contains(kw)]
        if "Region" in sub: sub = sub[sub["Region"].str.contains("World", case=False)]
        v = sub["Variable"].str.lower()
        out[name] = pd.DataFrame({"carbon": sub[v == "price|carbon"][yrs].mean(),
                                  "gdp": sub[v == "gdp|ppp"][yrs].mean()})
    rows = []
    for name, d in out.items():
        d = d.copy(); d["gdp_pct"] = (d["gdp"] / out["Current Policies"]["gdp"] - 1) * 100
        d["scenario"], d["year"] = name, d.index.astype(int)
        rows.append(d.reset_index(drop=True))
    return pd.concat(rows)[["scenario", "year", "carbon", "gdp_pct"]]

ng = load_ngfs().dropna()
assert set(SCENARIOS) <= set(ng.scenario), f"Scenarios found: {ng.scenario.unique()}"
print(ng.pivot(index="year", columns="scenario", values="carbon"))

# %% 3. Features, target (next-21-day log realised vol), windows, chronological split with embargo
fr = fr.reindex(px.index).ffill(limit=5)
ret = np.log(px).diff()
rv21 = np.log(np.sqrt((ret ** 2).rolling(21).sum()) + 1e-8)           # backward-looking realised vol (log)
rv21.columns = [f"rv_{s}" for s in SECTORS]
oil = fr.oil.where(fr.oil > 1).ffill()                                # WTI went negative in Apr-2020: mask it
macro = pd.DataFrame({"d_y10": fr.y10.diff(), "vix_l": np.log(fr.vix),
                      "oil_ret": np.log(oil).diff().clip(-0.25, 0.25),
                      "bmg": ret["XLE"] - ret["ICLN"]})                # brown minus green
feat = pd.concat([ret, macro, rv21], axis=1).dropna()
cols = list(feat.columns); F = len(cols)

# target at day t = log of realised vol over days t+1 ... t+HORIZON (never seen by the inputs at t)
fwd = np.log(np.sqrt((ret ** 2).rolling(HORIZON).sum().shift(-HORIZON)) + 1e-8)
tgt = fwd.reindex(feat.index)

trm = feat.index <= TRAIN_END
mu, sd = feat[trm].mean(), feat[trm].std()                            # standardise with TRAIN statistics only
Zv = ((feat - mu) / sd).values.astype("float32")
ymu, ysd = tgt[trm].mean().values, tgt[trm].std().values
Tz = ((tgt.values - ymu) / ysd).astype("float32")
ok = ~np.isnan(Tz).any(1)
ids = np.array([i for i in range(LOOKBACK - 1, len(Zv)) if ok[i]])
X = np.stack([Zv[i - LOOKBACK + 1:i + 1] for i in ids]); Y = Tz[ids]; dates = feat.index[ids]
tr = dates <= TRAIN_END
va = (dates > TRAIN_END + EMBARGO) & (dates <= VAL_END)
te = dates > VAL_END + EMBARGO
Xtr, Ytr, Xva, Yva, Xte, Yte = X[tr], Y[tr], X[va], Y[va], X[te], Y[te]
sd_r, mu_r = sd[SECTORS].values, mu[SECTORS].values
print(f"features={F}  train={len(Xtr)}  val={len(Xva)}  test={len(Xte)}  (trading days)")

# %% 4. Benchmarks the deep models must beat
def har_feats(X):                                   # classic HAR-RV inputs: 5-, 21-, 63-day log realised vol
    r = X[:, :, :A] * sd_r + mu_r                   # back to raw daily log returns, shape (n, L, A)
    f = lambda w: np.log(np.sqrt((r[:, -w:] ** 2).sum(1)) + 1e-8)
    return np.stack([f(5), f(21), f(LOOKBACK)], 2)  # (n, A, 3)

from sklearn.linear_model import LinearRegression, RidgeCV
Htr = har_feats(Xtr)
har = [LinearRegression().fit(Htr[:, a], Ytr[:, a]) for a in range(A)]
ridge = RidgeCV(alphas=np.logspace(1, 5, 15)).fit(Xtr.reshape(len(Xtr), -1), Ytr)

bench = {"Train mean": lambda X: np.zeros((len(X), A)),
         "Persistence": lambda X: (har_feats(X)[:, :, 1] - ymu) / ysd,       # tomorrow's vol = last 21d vol
         "HAR-RV": lambda X: np.stack([har[a].predict(har_feats(X)[:, a]) for a in range(A)], 1),
         "Ridge": lambda X: ridge.predict(X.reshape(len(X), -1))}

def score(pz, Yz):                                  # RMSE in log-vol units; R2 relative to the train-mean forecast
    p, t = pz * ysd + ymu, Yz * ysd + ymu
    mse = ((p - t) ** 2).mean()
    return np.sqrt(mse), 1 - mse / ((ymu - t) ** 2).mean()

print(pd.DataFrame({n: score(f(Xte), Yte) for n, f in bench.items()}, index=["Test RMSE", "Test R2"]).T.round(4))

# %% 5. Models
class LSTMNet(nn.Module):                         # baseline
    def __init__(s, F, A, h=32):
        super().__init__()
        s.lstm = nn.LSTM(F, h, batch_first=True); s.drop = nn.Dropout(0.3); s.head = nn.Linear(h, A)
    def forward(s, x):
        o, _ = s.lstm(x); return s.head(s.drop(o[:, -1]))

class GatedTransformer(nn.Module):                # primary model (use_lstm=True -> hybrid)
    def __init__(s, F, A, d=32, heads=4, layers=2, L=LOOKBACK, use_lstm=False, drop=0.2):
        super().__init__()
        s.gate = nn.Parameter(torch.zeros(F))     # learned per-feature gate -> interpretability
        s.use_lstm = use_lstm
        if use_lstm: s.lstm = nn.LSTM(F, d, batch_first=True)
        else: s.proj = nn.Linear(F, d)
        s.pos = nn.Parameter(torch.randn(1, L, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, d * 2, drop, batch_first=True)
        s.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        s.head = nn.Linear(d, A)
    def gates(s): return torch.sigmoid(s.gate)
    def forward(s, x):
        x = x * s.gates()
        h = s.lstm(x)[0] if s.use_lstm else s.proj(x)
        return s.head(s.enc(h + s.pos)[:, -1])

class Ensemble(nn.Module):
    def __init__(s, models): super().__init__(); s.models = nn.ModuleList(models)
    def forward(s, x): return torch.stack([m(x) for m in s.models]).mean(0)
    def gates(s): return torch.stack([m.gates() for m in s.models]).mean(0)

def train(model, epochs=100, lr=1e-3, wd=1e-3, patience=10, bs=64):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd); lossf = nn.MSELoss()
    xt, yt, xv, yv = map(torch.tensor, (Xtr, Ytr, Xva, Yva))
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(xt))
        for j in range(0, len(xt), bs):
            idx = perm[j:j + bs]; opt.zero_grad()
            loss = lossf(model(xt[idx]), yt[idx])
            if hasattr(model, "gates"): loss = loss + 0.001 * model.gates().mean()   # very mild gate sparsity
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): v = lossf(model(xv), yv).item()
        if v < best - 1e-5: best, bad, best_state = v, 0, {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state); model.eval(); return model, best

def fit_ensemble(factory, name):
    ms, vs = [], []
    for s in range(N_SEEDS):
        torch.manual_seed(s); m, v = train(factory()); ms.append(m); vs.append(v)
    print(f"{name:12s} val loss {np.mean(vs):.4f}"); return Ensemble(ms)

# %% 6. Train all three and compare with the benchmarks on the held-out test period
models = {"LSTM": fit_ensemble(lambda: LSTMNet(F, A), "LSTM"),
          "Transformer": fit_ensemble(lambda: GatedTransformer(F, A), "Transformer"),
          "Hybrid": fit_ensemble(lambda: GatedTransformer(F, A, use_lstm=True), "Hybrid")}

def predict(m, X):
    m.eval()
    with torch.no_grad(): return m(torch.tensor(X)).numpy()

rows = {n: score(f(Xte), Yte) for n, f in bench.items()}
for n, m in models.items(): rows[n] = score(predict(m, Xte), Yte)
results = pd.DataFrame(rows, index=["Test RMSE", "Test R2"]).T.round(4)
print(results); results.to_csv("outputs/model_comparison.csv")

per_sector = pd.DataFrame({n: [1 - ((predict(m, Xte)[:, a] - Yte[:, a]) ** 2).sum() / (Yte[:, a] ** 2).sum()
                               for a in range(A)] for n, m in models.items()}, index=SECTORS).round(3)
print("\nPer-sector R2 (vs z=0 train-mean forecast):\n", per_sector); per_sector.to_csv("outputs/per_sector_r2.csv")

# %% 7. Climate stress: shift the model's inputs the way each NGFS scenario would, read off the vol forecast
base_year = feat.index[-1].year + feat.index[-1].month / 12
win0 = Zv[-LOOKBACK:].copy()                                          # most recent 63 trading days
i_bmg, i_oil, i_vix = cols.index("bmg"), cols.index("oil_ret"), cols.index("vix_l")

def shocks(name, year, scale=1.0):
    d = ng[ng.scenario == name].sort_values("year")
    c = np.interp([base_year, year], d.year.values, d.carbon.values)
    g = np.interp([base_year, year], d.year.values, d.gdp_pct.values)
    dc, dg = max(c[1] - c[0], 0), g[1] - g[0]                         # change relative to today
    return dict(bmg=scale * K_BMG * dc / C_REF, oil=scale * K_OIL * dc / C_REF, vix=-scale * K_VIX * dg)

def vol_change_pct(members, name, year, scale=1.0):                  # % change in forecast vol, per member x sector
    s = shocks(name, year, scale); w = win0.copy()
    w[:, i_bmg] += s["bmg"]; w[:, i_oil] += s["oil"]; w[:, i_vix] += s["vix"]
    xb, xs = torch.tensor(win0)[None], torch.tensor(w)[None]
    with torch.no_grad():
        d = np.stack([(m.eval()(xs) - m.eval()(xb)).numpy()[0] * ysd for m in members])
    return np.expm1(d) * 100

rows = []
for mn, ens in models.items():
    for sc in SCENARIOS:
        for y in YEARS:
            ch = vol_change_pct(ens.models, sc, y)
            rows.append({"model": mn, "scenario": sc, "year": y, "avg_vol_change_pct": ch.mean(),
                         "member_std": ch.mean(1).std(), **dict(zip(SECTORS, ch.mean(0)))})
stress = pd.DataFrame(rows); stress.round(3).to_csv("outputs/stress_vol_change.csv", index=False)
print(stress[stress.model == "Transformer"][["scenario", "year", "avg_vol_change_pct", "member_std"]].round(2))

tm = models["Transformer"]                                            # sensitivity to the shock-size assumption
sens = pd.DataFrame({f"x{k} shock": {sc: vol_change_pct(tm.models, sc, YEARS[-1], k).mean() for sc in SCENARIOS}
                     for k in [1, 2, 4]}).round(2)
print(f"\nAvg forecast-vol change (%) in {YEARS[-1]} by shock multiplier:\n", sens); sens.to_csv("outputs/sensitivity.csv")

# %% 8. Charts
results.plot(kind="bar", y="Test R2", legend=False, figsize=(8, 4), title="Test R2 for next-21-day vol (vs train mean; higher is better)")
plt.axhline(0, color="k", lw=.8); plt.tight_layout(); plt.savefig("outputs/1_model_comparison.png", dpi=150); plt.show()

best = results.loc[["LSTM", "Transformer", "Hybrid"], "Test R2"].idxmax(); a = SECTORS.index("XLE")
ann = lambda z: np.exp(z * ysd[a] + ymu[a]) * np.sqrt(252 / HORIZON)
plt.figure(figsize=(10, 4)); plt.plot(dates[te], ann(Yte[:, a]), label="Actual (next-21d)", color="k", lw=1)
plt.plot(dates[te], ann(predict(models[best], Xte)[:, a]), label=f"{best} forecast", color="tab:orange")
plt.plot(dates[te], ann(bench["HAR-RV"](Xte)[:, a]), label="HAR-RV", color="tab:blue", alpha=.6)
plt.title("XLE annualised realised volatility: forecast vs actual (test period)"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("outputs/2_vol_forecast_XLE.png", dpi=150); plt.show()

t = stress[(stress.model == "Transformer") & (stress.year == YEARS[-1])].set_index("scenario")[SECTORS].T
t[SCENARIOS].plot(kind="bar", figsize=(10, 5), color=[COLORS[n] for n in SCENARIOS])
plt.title(f"Forecast vol change by sector, {YEARS[-1]} scenario vs today (%, Transformer)"); plt.axhline(0, color="k", lw=.8)
plt.tight_layout(); plt.savefig("outputs/3_sector_vol_change.png", dpi=150); plt.show()

plt.figure(figsize=(7, 4))
for sc in SCENARIOS:
    d = stress[(stress.model == "Transformer") & (stress.scenario == sc)]
    plt.errorbar(d.year, d.avg_vol_change_pct, yerr=d.member_std, marker="o", capsize=3, label=sc, color=COLORS[sc])
plt.title("Average forecast vol change (%), error bars = spread across seeds"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("outputs/4_vol_change_over_time.png", dpi=150); plt.show()

g = pd.Series(tm.gates().detach().numpy(), index=cols).sort_values()
g.plot(kind="barh", figsize=(7, 7)); plt.title("Transformer feature gates (higher = more used)")
plt.tight_layout(); plt.savefig("outputs/5_feature_gates.png", dpi=150); plt.show()
print("Saved charts + CSVs in outputs/")
