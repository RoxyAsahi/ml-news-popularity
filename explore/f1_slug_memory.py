"""Feasibility: point-in-time semantic-neighbour popularity from URL slugs."""
import numpy as np, pandas as pd, sys, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import roc_auc_score
from newspop.data import load, feature_groups
df = load(); n=len(df)
df['slug']=df.url.str.extract(r'/\d{4}/\d{2}/\d{2}/([^/]+)/?$')[0].str.replace(r'-\d+$','',regex=True)
txt=df.slug.str.replace('-',' ')
y=np.log1p(df.shares.values); day=(df.publish_date-df.publish_date.min()).dt.days.values
V=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True).fit_transform(txt)
print('vocab',V.shape)
K=20; H=[None,30,90,365]
feats={f'nb_h{h}':np.full(n,np.nan) for h in H}; feats['nb_maxsim']=np.zeros(n); feats['nb_cnt05']=np.zeros(n)
VT=V.T.tocsc(); t0=time.time()
for s in range(0,n,2000):
    e=min(n,s+2000); S=(V[s:e]@VT).toarray()
    for r,i in enumerate(range(s,e)):
        prev=np.where(day<day[i])[0]  # strictly earlier days
        if len(prev)==0: continue
        sim=S[r,prev]; top=np.argpartition(-sim,min(K,len(prev)-1))[:K]; top=top[sim[top]>0]
        if len(top)==0: continue
        j=prev[top]; w0=sim[top]
        for h in H:
            w=w0*(1 if h is None else 0.5**((day[i]-day[j])/h))
            feats[f'nb_h{h}'][i]=np.sum(w*y[j])/np.sum(w)
        feats['nb_maxsim'][i]=w0.max(); feats['nb_cnt05'][i]=(sim>0.5).sum()
print('sim done',time.time()-t0)
F=pd.DataFrame(feats); F.to_csv('explore/f1_feats.csv',index=False)
g=feature_groups(df.drop(columns=['slug'])); allf=[c for v in g.values() for c in v]
noshare=[c for c in allf if c not in g['keyword']+g['self_reference']]
lab=(df.shares>=1400).astype(int).values
a,b=int(.8*n),n  # train oldest 80%, test newest 20%
def run(cols,X=None):
    X=pd.concat([df[cols].reset_index(drop=True),F[X]] ,axis=1) if X else df[cols]
    aucs=[]
    for seed in range(3):
        m=HGB(learning_rate=.03,max_leaf_nodes=15,l2_regularization=1,max_iter=500,random_state=seed).fit(X.iloc[:a],lab[:a])
        aucs.append(roc_auc_score(lab[a:],m.predict_proba(X.iloc[a:])[:,1]))
    return np.mean(aucs)
nb=list(F.columns)
for name,c,x in [('all',allf,None),('all+nb',allf,nb),('noshare',noshare,None),('noshare+nb',noshare,nb),
                 ('noshare+nb_h30 only',noshare,['nb_h30','nb_maxsim','nb_cnt05']),('noshare+nb_hNone only',noshare,['nb_hNone','nb_maxsim','nb_cnt05']),
                 ('nb only',[],nb)]:
    print(f'{name:28s} AUC {run(c,x):.4f}')
from scipy.stats import spearmanr
for c in nb: print(c, 'spearman w/ logshares (test)', round(spearmanr(F[c][a:],y[a:],nan_policy='omit')[0],3))
print('corr nb_h30 vs kw_avg_avg', spearmanr(F.nb_h30,df.kw_avg_avg,nan_policy='omit')[0])
