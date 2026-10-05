"""Authenticity audit of the committed dataset. Run: python scripts/data_audit.py (see reports/data_audit.md)."""
import pandas as pd, numpy as np
df=pd.read_csv('load_forecast_new/delhi_interaction_enhanced_cleaned.csv',parse_dates=['datetime']).set_index('datetime').sort_index()
L=df['delhi_load']; T=df['temperature_2m (°C)']; S=df['data_source'].astype(str)
print("span",L.index.min(),L.index.max(),"rows",len(L),"dup ts",L.index.duplicated().sum())
for y,g in L.groupby(L.index.year): print(y,"peak",round(g.max()),g.idxmax(),"min",round(g.min()))
print(df.groupby([L.index.to_period('Q'),S]).size().unstack(fill_value=0))
for m in [6,1]:
  s=df[(L.index.month==m)&(S=='simulation_realistic')]; print("month",m,"sim mean by hour", s.groupby(s.index.hour)['delhi_load'].mean().round().astype(int).tolist())
for src in ['simulation_realistic','historical','0']:
  g=df[S==src]
  if len(g)<48: continue
  d=g['delhi_load'].resample('D').max().dropna(); t=g['temperature_2m (°C)'].resample('D').max().reindex(d.index)
  r=g['delhi_load']-g['delhi_load'].groupby([g.index.month,g.index.dayofweek,g.index.hour]).transform('mean')
  print(src,len(g),"peak~Tmax corr",round(d.corr(t),3),"resid acf1",round(r.autocorr(1),3),"kurt",round(r.kurt(),2))
parts=['brpl_load','bypl_load','ndpl_load','ndmc_load','mes_load']
gap=(df[parts].sum(axis=1)-L)
print("sum(discoms)-delhi by src:", gap.groupby(S).describe()[['mean','std','min','max']].round(1))
print("share brpl by src std:", (df.brpl_load/L).groupby(S).agg(['mean','std']).round(4))
print("frac non-integer load", (L%1!=0).mean())
T=df['temperature_2m (°C)']; RH=df['relative_humidity_2m (%)']; Td=df['dew_point_2m (°C)']
a,b=17.625,243.04; g=np.log(RH/100)+a*T/(b+T)
print("dew point vs Magnus formula, mean abs err by src:", (Td-b*g/(a-g)).abs().groupby(S).mean().round(2).to_dict())
h=df.index.hour
print("mean-profile peak hour: radiation", df.groupby(h).shortwave_radiation.mean().idxmax(),
      "temperature", T.groupby(h).mean().idxmax(), "load", L.groupby(h).mean().idxmax())
d=L.resample('D').min(); summer=d[d.index.month.isin([5,6,7])]
print("May-Jul days with overnight minimum < 3000 MW:", int((summer<3000).sum()), "of", len(summer))
