"""Anatomy of predictability with uncertainty: pairwise AUC within nested strata + twin ICC, bootstrap CIs."""
import numpy as np, pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from newspop.data import load, feature_groups
df=load(); g=feature_groups(df); allf=[c for v in g.values() for c in v]; n=len(df); ls=np.log1p(df.shares.values)
lab=(df.shares>=1400).values.astype(int)
ch=[c for c in df if c.startswith('data_channel_is')]; chan=np.where(df[ch].sum(1)==0,6,df[ch].values.argmax(1))
day=(df.publish_date-df.publish_date.min()).dt.days.values; week=day//7
sc=np.zeros(n)  # out-of-fold scores, 5 contiguous time blocks
for k in range(5):
    te=np.arange(k*n//5,(k+1)*n//5); tr=np.setdiff1d(np.arange(n),te)
    sc[te]=HGB(learning_rate=.03,max_leaf_nodes=15,l2_regularization=1,max_iter=500,random_state=0).fit(df[allf].values[tr],lab[tr]).predict_proba(df[allf].values[te])[:,1]
P=pd.read_csv('explore/f2_pairs.csv'); P=P[P.sim>=0.5]; S=P[chan[P.i]==chan[P.j]]
def strat_pairs(groups):
    I=[];J=[]
    for _,idx in pd.Series(np.arange(n)).groupby(groups):
        idx=idx.values; p=idx[lab[idx]==1]; q=idx[lab[idx]==0]
        if len(p) and len(q):
            a,b=np.meshgrid(p,q); I.append(a.ravel()); J.append(b.ravel())
    return np.concatenate(I),np.concatenate(J)
rng=np.random.RandomState(0)
def auc_ci(I,J,cl):  # cluster bootstrap over weeks of the positive article
    v=(sc[I]>sc[J])+0.5*(sc[I]==sc[J]); est=v.mean()
    W=np.unique(cl); df_=pd.DataFrame({'v':v,'c':cl}).groupby('c').v.agg(['sum','size'])
    bs=[]
    for _ in range(300):
        s=df_.loc[rng.choice(df_.index,len(df_))]; bs.append(s['sum'].sum()/s['size'].sum())
    return est,np.percentile(bs,[2.5,97.5]),len(v)
rows=[]
for nm,gr in [('pooled (all pairs)',np.zeros(n,int)),('within week',week),('within day',day),('within channel',chan),('within channel x week',chan*1000+week),('within channel x day',chan*1000+day)]:
    I,J=strat_pairs(gr)
    if len(I)>3e6: s=rng.choice(len(I),3_000_000,replace=False); I,J=I[s],J[s]
    e,ci,m=auc_ci(I,J,week[I]); rows.append((nm,e,*ci,m))
for nm,Q in [('same-story twins (any channel)',P),('same-story twins, same channel',S)]:
    d=Q[lab[Q.i]!=lab[Q.j]]; I=np.where(lab[d.i]==1,d.i,d.j); J=np.where(lab[d.i]==1,d.j,d.i)
    e,ci,m=auc_ci(I,J,week[I]); rows.append((nm,e,*ci,m))
R=pd.DataFrame(rows,columns=['stratum','pairwise_auc','lo','hi','n_pairs']); print(R.round(3).to_string()); R.to_csv('explore/f9_anatomy.csv',index=False)
res=ls-pd.Series(ls).groupby([chan,week]).transform('mean').values
for nm,Q in [('all twins',P),('same-channel twins',S)]:
    bs=[pearsonr(*(lambda q:(res[q.i],res[q.j]))(Q.iloc[rng.randint(0,len(Q),len(Q))]))[0] for _ in range(500)]
    print(nm,'ICC(residual log shares)',round(pearsonr(res[Q.i],res[Q.j])[0],3),np.percentile(bs,[2.5,97.5]).round(3),len(Q))
