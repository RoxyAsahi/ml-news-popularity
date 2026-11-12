"""Feasibility: near-duplicate 'twin' articles (same story, close in time) -> noise floor."""
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, pearsonr
from newspop.data import load
df=load(); n=len(df)
df['slug']=df.url.str.extract(r'/\d{4}/\d{2}/\d{2}/([^/]+)/?$')[0].str.replace(r'-\d+$','',regex=True)
V=TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True).fit_transform(df.slug.str.replace('-',' '))
day=(df.publish_date-df.publish_date.min()).dt.days.values
y=np.log1p(df.shares.values)
ch=[c for c in df if c.startswith('data_channel_is')]; chan=np.where(df[ch].sum(1)==0,6,df[ch].values.argmax(1))
pairs=[]
for s in range(0,n,3000):
    S=(V[s:s+3000]@V.T).tocoo()
    m=(S.data>=0.5)&(S.col>S.row+s)
    for r,c,v in zip(S.row[m]+s,S.col[m],S.data[m]):
        if abs(day[r]-day[c])<=7: pairs.append((r,c,v))
P=pd.DataFrame(pairs,columns=['i','j','sim']); P['dt']=abs(day[P.i]-day[P.j]); P['samech']=chan[P.i]==chan[P.j]
print('pairs',len(P), 'articles involved',len(set(P.i)|set(P.j)))
for th in [0.5,0.6,0.7,0.8]:
    Q=P[P.sim>=th]
    r=pearsonr(y[Q.i],y[Q.j])[0]
    # random same-channel, same-week baseline pairs
    rng=np.random.RandomState(0); bi=[];bj=[]
    for i in Q.i:
        cand=np.where((chan==chan[i])&(abs(day-day[i])<=7))[0]; bi.append(i); bj.append(rng.choice(cand))
    rb=pearsonr(y[bi],y[bj])[0]
    # twin AUC: predict label of j from shares of i
    lab=(df.shares.values>=1400)
    auc=roc_auc_score(np.r_[lab[Q.j],lab[Q.i]],np.r_[y[Q.i],y[Q.j]])
    print(f'sim>={th}: n={len(Q)} ICC~r(log shares)={r:.3f}  baseline same-chan-week r={rb:.3f}  twin-AUC={auc:.3f} samech={Q.samech.mean():.2f}')


P.to_csv('explore/f2_pairs.csv',index=False)
