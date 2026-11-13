"""U10: is the archive's saturation a property of the archive (bias) or of the linear learner (capacity)?

Confirmatory folds as in u6. Idealized archive (one random headline per training test, week-demeaned click-through)
at 4,500 headlines and at full size, fit by ridge (as in u6), gradient-boosted trees and a small neural network on
the same representation. If the ridge archive saturated because the learner is too simple, a more flexible learner
should gain from the larger archive and move closer to what randomized training reaches.
Usage: python scripts/u10_capacity.py
Output: results/tables/u10_capacity.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from newspop import upw_rank as ur
from newspop.data import ROOT

S = ur.setup("confirmatory")
Z = S.Z
rng = np.random.default_rng(3)
LEARNERS = {
    "ridge": lambda: RidgeCV(alphas=np.logspace(-1, 5, 13)),
    "boosted trees": lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, early_stopping=True,
                                                           random_state=0),
    "neural net": lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2,
                                                                       early_stopping=True, max_iter=300,
                                                                       random_state=0)),
}
rows = []
for fold, (tr, te) in enumerate(GroupKFold(5).split(np.zeros(len(S.tests)), groups=S.article)):
    score_rows = np.where(np.isin(S.tix, te))[0]
    Zte = Z[S.hid[score_rows]]
    for size in [4500, -1]:
        sub = tr if size == -1 else rng.choice(tr, size, replace=False)
        rows_a = ur.one_per_test(S, sub, rng)
        Xa, ya = Z[S.hid[rows_a]], ur.week_demeaned(S, rows_a)
        for name, make in LEARNERS.items():
            m = make().fit(Xa, ya)
            rows.append(dict(fold=fold, learner=name, size=size, **ur.evaluate(S, score_rows, m.predict(Zte), te)))
    print(fold, flush=True)
R = pd.DataFrame(rows)
R.to_csv(ROOT / "results" / "tables" / "u10_capacity.csv", index=False)
print(R.groupby(["learner", "size"])[["within_acc_sig", "lift_pct"]].mean().round(4).to_string())
