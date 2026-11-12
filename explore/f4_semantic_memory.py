"""Semantic memory: point-in-time, time-decayed neighbour popularity from slug word vectors."""
import numpy as np, pandas as pd, spacy, time
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import roc_auc_score
from newspop.data import load, feature_groups
df=load(); g=feature_groups(df); allf=[c for v in g.values() for c in v]
noshare=[c for c in allf if c not in g['keyword']+g['self_reference']]
n=len(df); slug=df.url.str.extract(r'/\d{4}/\d{2}/\d{2}/([^/]+)/?$')[0].str.replace(r'-\d+$','',regex=True).str.replace('-',' ')
nlp=spacy.load('en_core_web_md',exclude=['tok2vec','tagger','parser','attribute_ruler','lemmatizer','ner','senter'])
E=np.array([d.vector for d in nlp.pipe(slug.tolist(),batch_size=512)]); E/=np.linalg.norm(E,axis=1,keepdims=True)+1e-9
np.save('explore/slug_vec.npy',E)
day=(df.publish_date-df.publish_date.min()).dt.days.values; ls=np.log1p(df.shares.values)
ch=[c for c in df if c.startswith('data_channel_is')]; chan=np.where(df[ch].sum(1)==0,6,df[ch].values.argmax(1))
# residual popularity: remove channel x month mean using only... (exploratory: full-sample mean)
month=df.publish_date.dt.to_period('M').astype(str).values
res=ls-pd.Series(ls).groupby([chan,month]).transform('mean').values
K=20; out={}
for tgt,name in [(ls,'raw'),(res,'res')]:
  for h in [None,7,30,180]:
    f=np.full(n,np.nan)
    for s in range(0,n,2000):
        S=E[s:s+2000]@E.T
        for r,i in enumerate(range(s,min(n,s+2000))):
            prev=np.where(day<day[i])[0]
            if len(prev)<K: continue
            sim=S[r,prev]; top=np.argpartition(-sim,K)[:K]; j=prev[top]; w=np.maximum(sim[top],0)**4
            if h: w=w*0.5**((day[i]-day[j])/h)
            f[i]=(w*tgt[j]).sum()/w.sum()
    out[f'{name}_h{h}']=f
F=pd.DataFrame(out); F.to_csv('explore/f4_feats.csv',index=False)
a=int(.8*n); lab=(df.shares>=1400).values.astype(int)
for c in F: print(c,'spearman vs log shares',round(spearmanr(F[c],ls,nan_policy='omit')[0],3),' vs residual',round(spearmanr(F[c],res,nan_policy='omit')[0],3))
def run(cols,extra=[]):
    X=pd.concat([df[cols],F[extra]],axis=1)
    return np.mean([roc_auc_score(lab[a:],HGB(learning_rate=.03,max_leaf_nodes=15,l2_regularization=1,max_iter=500,random_state=s).fit(X[:a],lab[:a]).predict_proba(X[a:])[:,1]) for s in range(3)])
print('noshare',run(noshare),'noshare+mem',run(noshare,list(F.columns)),'all',run(allf),'all+mem',run(allf,list(F.columns)))
