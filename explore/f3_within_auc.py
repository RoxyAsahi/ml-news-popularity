"""Feasibility: what is AUC 0.74 made of? Pairwise AUC within strata (day, channel-day, story twins)."""
import numpy as np, pandas as pd, itertools
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import roc_auc_score
from newspop.data import load, feature_groups
df=load(); g=feature_groups(df); allf=[c for v in g.values() for c in v]
noshare=[c for c in allf if c not in g['keyword']+g['self_reference']]
n=len(df); a=int(.8*n); lab=(df.shares>=1400).values.astype(int); ls=np.log1p(df.shares.values)
ch=[c for c in df if c.startswith('data_channel_is')]; chan=np.where(df[ch].sum(1)==0,6,df[ch].values.argmax(1))
day=(df.publish_date-df.publish_date.min()).dt.days.values
P=pd.read_csv('explore/f2_pairs.csv'); P=P[(P.i>=a)&(P.j>=a)&(P.sim>=0.5)]
def pair_auc(score, groups):
    # AUC over pairs (pos,neg) within same group
    num=0.;den=0.
    te=np.arange(a,n)
    for _,idx in pd.Series(te).groupby(groups[te]):
        idx=idx.values; p=idx[lab[idx]==1]; q=idx[lab[idx]==0]
        if len(p)==0 or len(q)==0: continue
        d=score[p][:,None]-score[q][None,:]; num+=(d>0).sum()+.5*(d==0).sum(); den+=d.size
    return num/den, den
def twin_auc(score):
    m=lab[P.i]!=lab[P.j]; i=P.i[m].values;j=P.j[m].values
    s=np.sign(score[i]-score[j]); t=np.sign(lab[i]-lab[j]); return ((s==t).mean()+0)*1, m.sum()
for name,cols in [('all',allf),('noshare',noshare)]:
    m=HGB(learning_rate=.03,max_leaf_nodes=15,l2_regularization=1,max_iter=500,random_state=0).fit(df[cols][:a],lab[:a])
    sc=np.zeros(n); sc[a:]=m.predict_proba(df[cols][a:])[:,1]
    print(name,'overall',round(roc_auc_score(lab[a:],sc[a:]),3))
    print('  within day',[round(x,3) for x in pair_auc(sc,day)])
    print('  within channel',[round(x,3) for x in pair_auc(sc,chan)])
    print('  within channel x day',[round(x,3) for x in pair_auc(sc,chan*1000+day)])
    print('  within story twins',twin_auc(sc))
# how much does day-level context alone give? oracle day mean (leave-one-out) of log shares
dm=pd.Series(ls).groupby(day).transform('sum'); dc=pd.Series(ls).groupby(day).transform('count')
loo=((dm-ls)/(dc-1)).values; print('oracle LOO day mean AUC',round(roc_auc_score(lab[a:],loo[a:]),3))
cd=chan*1000+day; dm=pd.Series(ls).groupby(cd).transform('sum'); dc=pd.Series(ls).groupby(cd).transform('count')
loo=((dm-ls)/(dc-1).clip(lower=1)).values; ok=(dc>1).values[a:]
print('oracle LOO channel-day mean AUC',round(roc_auc_score(lab[a:][ok],loo[a:][ok]),3))
