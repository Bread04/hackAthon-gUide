import numpy as np, pandas as pd
rng=np.random.default_rng(0); rows=[]
for uid in range(5):
    ds=pd.date_range("2024-01-01",periods=365,freq="D"); t=np.arange(365)
    rows.append(pd.DataFrame({"unique_id":f"s{uid}","ds":ds,"y":10+uid+3*np.sin(2*np.pi*t/7)+rng.normal(0,1,365)}))
df=pd.concat(rows,ignore_index=True)
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, AutoETS
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
import lightgbm as lgb
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse
H, W = 14, 4
sf = StatsForecast(models=[SeasonalNaive(season_length=7), AutoETS(season_length=7)], freq='D', n_jobs=-1)
cv_sf = sf.cross_validation(h=H, df=df, n_windows=W, step_size=H)
mlf = MLForecast(models=[lgb.LGBMRegressor(verbosity=-1)], freq='D', lags=[7, 14],
                 lag_transforms={7: [RollingMean(window_size=28)]}, date_features=['dayofweek'])
cv_ml = mlf.cross_validation(df=df, n_windows=W, h=H, step_size=H)
cv = cv_sf.merge(cv_ml.drop(columns='y'), on=['unique_id', 'ds', 'cutoff'])
print(evaluate(cv.drop(columns='cutoff'), metrics=[mae, rmse]).groupby('metric').mean(numeric_only=True))
