"""Attention competition: does an article do worse on crowded days / next to strong competitors?"""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from newspop.data import load
df=load(); ls=np.log1p(df.shares.values); df['ls']=ls
ch=[c for c in df if c.startswith('data_channel_is')]; df['chan']=np.where(df[ch].sum(1)==0,6,df[ch].values.argmax(1))
df['day']=(df.publish_date-df.publish_date.min()).dt.days; df['month']=df.publish_date.dt.to_period('M').astype(str)
df['wd']=df.publish_date.dt.dayofweek
df['n_day']=df.groupby('day').ls.transform('size'); df['n_chday']=df.groupby(['day','chan']).ls.transform('size')
# competitor strength: leave-one-out mean of kw_avg_avg (pre-publication proxy) among same-day articles
s=df.groupby('day').kw_avg_avg.transform('sum'); df['comp_kw']=(s-df.kw_avg_avg)/(df.n_day-1)
s=df.groupby(['day','chan']).kw_avg_avg.transform('sum'); df['comp_kw_ch']=np.where(df.n_chday>1,(s-df.kw_avg_avg)/(df.n_chday-1).clip(lower=1),np.nan)
df['log_n_day']=np.log(df.n_day); df['log_n_chday']=np.log(df.n_chday)
for f in ['ls ~ log_n_day + C(wd) + C(month) + C(chan)',
          'ls ~ log_n_chday + log_n_day + C(wd) + C(month) + C(chan)',
          'ls ~ np.log(comp_kw) + np.log1p(kw_avg_avg) + log_n_day + C(wd) + C(month) + C(chan)',
          'ls ~ np.log1p(comp_kw_ch) + np.log1p(kw_avg_avg) + log_n_chday + C(wd) + C(month) + C(chan)']:
    m=smf.ols(f,df).fit(cov_type='cluster',cov_kwds={'groups':df.loc[df.index.isin(smf.ols(f,df).fit().model.data.row_labels),'day'] if False else None}) if False else smf.ols(f,df.dropna(subset=['comp_kw_ch']) if 'comp_kw_ch' in f else df).fit(cov_type='cluster',cov_kwds={'groups':(df.dropna(subset=['comp_kw_ch']) if 'comp_kw_ch' in f else df)['day']})
    keep=[k for k in m.params.index if not k.startswith('C(')]
    print(f); print(pd.DataFrame({'coef':m.params[keep],'lo':m.conf_int().loc[keep,0],'hi':m.conf_int().loc[keep,1]}).round(4))
