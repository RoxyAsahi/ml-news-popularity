"""Transfer law: from Upworthy (how big headline effects are; how well Jev reads them) predict Jev's pairwise accuracy on
Mashable shares, under a Gaussian model where the headline effect h adds to log shares independently of everything else."""
import numpy as np, pandas as pd
from scipy.stats import norm
rng = np.random.default_rng(0)
lgt = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
U = pd.read_csv("/home/user/jev/api_upw_random.csv")
O = pd.read_csv("/home/user/explore/mash_oof.csv"); y = O.y.to_numpy()


def upw_stats(U):
    J = lgt((U.p0 + 1 - U.p1) / 2).to_numpy(); obs = U.obs.to_numpy(); nv = U.nv.to_numpy()
    ok = np.isfinite(obs) & np.isfinite(nv) & (nv < 1)
    J, obs, nv = J[ok], obs[ok], nv[ok]
    vh = np.mean(obs ** 2) - np.mean(nv)                 # variance of the true headline contrast
    rho = np.mean(J * obs) / np.sqrt(np.mean(J ** 2) * vh)   # corr(Jev, true contrast), noise removed
    return vh, rho


def mash_pred(P, vh, rho):
    dy = y[P.a] - y[P.b]; vy = np.mean(dy ** 2)
    r = np.sqrt(vh / vy)
    return r, 0.5 + np.arcsin(r) / np.pi, 0.5 + np.arcsin(rho * r) / np.pi


vh, rho = upw_stats(U)
print(f"Upworthy: sd of true headline contrast (logit) {np.sqrt(vh):.3f}; corr(Jev, true contrast) {rho:.3f}")
B = [upw_stats(U.iloc[rng.integers(0, len(U), len(U))]) for _ in range(500)]
for mode in ["section", "day"]:
    P = pd.read_csv(f"/home/user/jev/api_mash_titles_{mode}.csv")
    p = (P.p0_title + 1 - P.p1_title) / 2; t = y[P.a] > y[P.b]
    obs_acc = np.mean((p > .5) == t)
    bo = [np.mean(((p > .5) == t)[i]) for i in [rng.integers(0, len(P), len(P)) for _ in range(500)]]
    r, ora, pred = mash_pred(P, vh, rho)
    pb = [mash_pred(P, a, b)[2] for a, b in B]; ob = [mash_pred(P, a, b)[1] for a, b in B]
    print(f"Mashable {mode}: sd of log-share contrast {np.sqrt(np.mean((y[P.a]-y[P.b])**2)):.3f}; headline share of variance r^2 {r**2:.3f}")
    print(f"   perfect headline judge would reach {ora:.3f} [{np.quantile(ob,.025):.3f}, {np.quantile(ob,.975):.3f}]")
    print(f"   predicted Jev {pred:.3f} [{np.quantile(pb,.025):.3f}, {np.quantile(pb,.975):.3f}]   observed Jev {obs_acc:.3f} [{np.quantile(bo,.025):.3f}, {np.quantile(bo,.975):.3f}]")
