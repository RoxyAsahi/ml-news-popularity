"""Conformalized quantile regression under temporal drift + adaptive conformal inference."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR, HistGradientBoostingClassifier as HGB
from sklearn.metrics import roc_auc_score
from newspop.data import load, feature_groups
df=load(); g=feature_groups(df); allf=[c for v in g.values() for c in v]
n=len(df); a=int(.7*n); b=int(.8*n); y=np.log1p(df.shares.values); X=df[allf]; alpha=0.2
lo=HGBR(loss='quantile',quantile=alpha/2,max_iter=300,learning_rate=.05,random_state=0).fit(X[:a],y[:a])
hi=HGBR(loss='quantile',quantile=1-alpha/2,max_iter=300,learning_rate=.05,random_state=0).fit(X[:a],y[:a])
L=lo.predict(X); H=hi.predict(X); s=np.maximum(L-y,y-H)
q=np.quantile(s[a:b],np.ceil((b-a+1)*(1-alpha))/(b-a))
te=np.arange(b,n); cov_raw=((y[te]>=L[te])&(y[te]<=H[te])); cov_cqr=((y[te]>=L[te]-q)&(y[te]<=H[te]+q))
month=df.publish_date.dt.to_period('M').astype(str).values
print('target',1-alpha,'raw QR cov',cov_raw.mean().round(3),'CQR cov',cov_cqr.mean().round(3))
print(pd.DataFrame({'m':month[te],'raw':cov_raw,'cqr':cov_cqr}).groupby('m').mean().round(3).T)
w=np.exp((H[te]+q)-(L[te]-q)); print('median multiplicative width of 80% interval: x',np.median(w).round(1), ' e.g. median interval',np.round(np.expm1(np.median(L[te]-q))),'-',np.round(np.expm1(np.median(H[te]+q))))
# ACI (Gibbs & Candes 2021) online over test articles in time order, using rolling calibration scores
gamma=0.005; at=alpha; covs=[]; scores=list(s[a:b])
for t in te:
    qt=np.quantile(scores[-3000:],min(1,max(0,1-at))); c=(y[t]>=L[t]-qt)&(y[t]<=H[t]+qt); covs.append(c)
    at=at+gamma*(alpha-(0 if c else 1)); scores.append(s[t])
covs=np.array(covs); print('ACI cov',covs.mean().round(3)); print(pd.DataFrame({'m':month[te],'aci':covs}).groupby('m').mean().round(3).T)
# selective classification
lab=(df.shares>=1400).values.astype(int); m=HGB(learning_rate=.03,max_leaf_nodes=15,l2_regularization=1,max_iter=500,random_state=0).fit(X[:b],lab[:b])
p=m.predict_proba(X[b:])[:,1]; conf=np.abs(p-.5); acc=((p>.5)==lab[b:])
for cov in [1,.5,.2,.1,.05]:
    k=int(cov*len(p)); idx=np.argsort(-conf)[:k]; print(f'coverage {cov}: accuracy {acc[idx].mean():.3f}')
