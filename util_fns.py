import numpy as np
from statsmodels.api import OLS

def rmse(x, y):
    return np.sqrt(np.mean((x-y)**2))

def ubrmse(x, y):
    return np.sqrt(np.mean(((x-x.mean())-(y-y.mean()))**2))

def bias(pred, true):
    return (pred-true).mean()

def mult_bias(df):
    nec_cols = ['err', 'avg_pday', 'air_temp']
    missing = [x for x in nec_cols if x not in df.columns]
    if len(missing) > 0:
        raise Exception(f'missing column(s) {missing} to compute mult bias')

    df['const'] = 1
    df['centered_avg_pday'] = df.avg_pday - df.avg_pday.mean()
    df['centered_air_temp'] = df.air_temp - df.air_temp.mean()
    
    mod = OLS(df.err, df[['centered_avg_pday', 'centered_air_temp', 'const']])
    fit = mod.fit()

    return fit.params, fit.HC1_se
