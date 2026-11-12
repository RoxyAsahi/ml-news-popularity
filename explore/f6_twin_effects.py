"""Topic-controlled natural experiment: within same-story twin pairs, which content features predict the winner?"""
import numpy as np, pandas as pd, statsmodels.api as sm
from newspop.data import load
df=load(); ls=np.log1p(df.shares.values)
P=pd.read_csv('explore/f2_pairs.csv'); P=P[P.sim>=0.5]
feat={'log_n_tokens_content':np.log1p(df.n_tokens_content),'log_n_tokens_title':np.log1p(df.n_tokens_title),
 'log_num_imgs':np.log1p(df.num_imgs),'log_num_videos':np.log1p(df.num_videos),'log_num_hrefs':np.log1p(df.num_hrefs),
 'global_subjectivity':df.global_subjectivity,'global_sentiment_polarity':df.global_sentiment_polarity,
 'title_subjectivity':df.title_subjectivity,'title_sentiment_polarity':df.title_sentiment_polarity,
 'rate_negative_words':df.rate_negative_words,'kw_avg_avg_log':np.log1p(df.kw_avg_avg.clip(lower=0)),
 'is_weekend':df.is_weekend,'num_keywords':df.num_keywords}
X=pd.DataFrame(feat); X=(X-X.mean())/X.std()
ch=[c for c in df if c.startswith('data_channel_is')]; chan=np.where(df[ch].sum(1)==0,6,df[ch].values.argmax(1))
day=(df.publish_date-df.publish_date.min()).dt.days.values
# cross-section with channel + week FE
C=pd.get_dummies(pd.Series(chan).astype(str),prefix='ch',drop_first=True).astype(float)
W=pd.get_dummies(pd.Series(day//7).astype(str),prefix='w',drop_first=True).astype(float)
cs=sm.OLS(ls,sm.add_constant(pd.concat([X,C,W],axis=1))).fit(cov_type='cluster',cov_kwds={'groups':day})
# within twins: difference regression (pair FE), cluster by first article
D=X.values[P.i]-X.values[P.j]; dy=ls[P.i]-ls[P.j]
same=(chan[P.i]==chan[P.j]).astype(float)
Dc=np.c_[D, (pd.get_dummies(pd.Series(chan[P.i]))).values[:,1:]-(pd.get_dummies(pd.Series(chan[P.j]))).values[:,1:]] if False else D
tw=sm.OLS(dy,Dc).fit(cov_type='cluster',cov_kwds={'groups':P.i.values})
out=pd.DataFrame({'cross_section':cs.params[X.columns].values,'cs_lo':cs.conf_int().loc[X.columns,0].values,'cs_hi':cs.conf_int().loc[X.columns,1].values,
                  'within_twin':tw.params[:len(X.columns)],'tw_lo':tw.conf_int()[:len(X.columns),0],'tw_hi':tw.conf_int()[:len(X.columns),1]},index=X.columns)
print(out.round(3)); print('pairs',len(P),'R2 within',round(tw.rsquared,3),'R2 cs (incl FE)',round(cs.rsquared,3))
print('var log shares',ls.var().round(3),' mean sq twin diff/2',(dy**2).mean()/2)
