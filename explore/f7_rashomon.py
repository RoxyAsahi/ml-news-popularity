"""Rashomon set: do near-equally-accurate models agree on what matters?"""
import numpy as np, pandas as pd, itertools
from sklearn.ensemble import HistGradientBoostingClassifier as HGB, RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from newspop.data import load, feature_groups
df=load(); g=feature_groups(df); allf=[c for v in g.values() for c in v]
n=len(df); a=int(.7*n); b=int(.8*n); lab=(df.shares>=1400).values.astype(int); X=df[allf]
rng=np.random.RandomState(0); models=[]
for lr,leaf,l2,it in itertools.product([.03,.1],[7,15,31],[0,1],[200,500]):
    models.append(('hgb',HGB(learning_rate=lr,max_leaf_nodes=leaf,l2_regularization=l2,max_iter=it,random_state=0)))
for leaf,mf in itertools.product([5,20,50],['sqrt',.3]): models.append(('rf',RF(n_estimators=300,min_samples_leaf=leaf,max_features=mf,n_jobs=-1,random_state=0)))
for C in [.01,.1,1]: models.append(('lr',make_pipeline(StandardScaler(),LogisticRegression(C=C,penalty='l1',solver='liblinear'))))
res=[]
groups={k:v for k,v in g.items() if v}
for name,m in models:
    m.fit(X[:a],lab[:a]); pv=m.predict_proba(X[a:b])[:,1]; va=roc_auc_score(lab[a:b],pv)
    pt=m.predict_proba(X[b:])[:,1]; te=roc_auc_score(lab[b:],pt)
    imp={}
    for gn,cols in groups.items():
        Xp=X[a:b].copy(); perm=rng.permutation(b-a); Xp[cols]=Xp[cols].values[perm]
        imp[gn]=va-roc_auc_score(lab[a:b],m.predict_proba(Xp)[:,1])
    res.append(dict(model=name,val=va,test=te,**imp)); print(name,round(va,4),round(te,4),flush=True)
R=pd.DataFrame(res); R.to_csv('explore/f7_rashomon.csv',index=False)
best=R.val.max(); S=R[R.val>=best-0.01]
print('Rashomon set size (val within 0.01):',len(S),'of',len(R)); print(S.drop(columns=['model']).describe().loc[['min','max']].round(4).T)
print(S.groupby('model').size())
