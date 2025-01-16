import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from scipy.stats import linregress
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from util_fns import rmse, ubrmse, bias, mult_bias

states = [17, 18, 19]
years = list(range(2003, 2013))

data = pickle.load(open('ndvi_ts.p', 'rb'))
#ndvi = data.loc[:, ndvi_start:ndvi_end]
#data = pd.concat((data.iloc[:, :2], ndvi), axis=1)

fy = pickle.load(open('good_data.p', 'rb'))

#counts = data.groupby(['fips', 'year']).agg('count').reset_index()
#good_counties = counts[counts.iloc[:, 3:].min(axis=1)>10][['fips', 'year']]
#data = data.merge(good_counties, on=['fips', 'year'])
data = data.groupby(['fips', 'year']).agg('mean').reset_index()
#data = data.dropna()
data = fy.merge(data, on=['fips', 'year'])
fit_start = 6

def local_slope(ts):
    res = linregress(np.arange(len(ts)), ts)

    return res.slope

def count_turns(ts, length=5):
    positive = True
    turns = 0
    for j in range(len(ts)-length):
        slope = local_slope(ts[j:j+length])
        if positive and slope < 0:
            turns += 1
            positive = False
        positive = res.slope > 0

    return turns

def find_change_ind(ts, start_offset=0, length=5):
    change_ind = -1
    for j in range(start_offset, ts.size-length):
        slope = local_slope(ts[j:j+length])
        if slope < 0:
            change_ind = j
            break

    # return index halfway through test patch
    # assumed patch has odd length
    return change_ind+length//2

def make_log_fn(a, b, c, d):
    def log_fn(t):
        return c/(1+np.exp(a+b*t))+d
    return log_fn
    
def fit_log(ts):
    def sse_fn(x):
        log_fn = make_log_fn(*x)
        return np.sum((log_fn(np.arange(len(ts)))-ts)**2)
   
    res = minimize(sse_fn, [3, -1, 0.8, 0.5],)
                   #bounds=[(0.5, 40), (-8, -0.01), (0.01, 2), (0, 1)])

    return res.x#, log_fun(*res.x)

def truncate_then_fit(ts):
    change_ind = find_change_ind(ts, 11)
    trunc_ts = ts[:change_ind+1]

    return fit_log(trunc_ts), change_ind

def threshold(pct, size, log_params):
    log_fn = make_log_fn(*log_params)
    fit_ts = log_fn(np.arange(size))
    a, b, c, d = log_params

    vi_range = fit_ts[-1]-fit_ts[0]
    target = fit_ts[0]+pct*vi_range

    return (np.log(c/(target-d)-1)-a)/b

def process_row(row):
    res = row[['fips', 'year']]
    #ts = row[[x for x in row.index if 'ndvi' in x]]
    ts = row.loc[fit_start:]
    params, change_ind = truncate_then_fit(ts)
    param_df = pd.Series(data=params, index=['a', 'b', 'c', 'd'])
    param_df['change_ind'] = change_ind
    res = pd.concat((res, param_df))
    for th in np.arange(.1, 1, .1):
        res[f't{int(th*10)}'] = threshold(th, ts.size, params)

    res['tinf'] = -params[0]/params[1]

    return res


#######################################################################


row_ls = [data.iloc[x] for x in range(data.shape[0])]

with ProcessPoolExecutor(max_workers=40) as e:
    res = e.map(process_row, row_ls)

results = pd.DataFrame(res)

#pdays = pickle.load(open('../planting_date/pdays.p', 'rb'))
#pdays = pdays[['YEAR', 'FIPS', 'sday']]
#pdays.columns = ['year', 'fips', 'avg_pday']

#results = results.merge(pdays, on=['fips', 'year'])

results = fy.merge(results, on=['fips', 'year'])
#print(results.corr().loc['avg_pday'])

def show_co_yr(fips, year):
    plt.plot(data[(data.fips==fips)&(data.year==year)].iloc[0, 5:])
    res = results[(results.fips==fips)&(results.year==year)]
    log_fn = make_log_fn(*res.iloc[0].loc[['a', 'b', 'c', 'd']])
    plt.plot(np.arange(fit_start, fit_start+res.iloc[0].change_ind),
             log_fn(np.arange(res.iloc[0].change_ind)))
    plt.show()

results['const'] = 1

for test_spec in [1, 2]:
    print(test_spec)
    pred = f'pred{test_spec}'
    group = f'group{test_spec}'
    for yr in fy.year.unique():
        for grp in range(3):
            train = results[(results.year!=yr)&(results[group]!=grp)]
            test = results[(results.year==yr)&(results[group]==grp)].copy()
    
            res = np.linalg.lstsq(train[['t5', 'const']], train.avg_pday, rcond=None)
            test.loc[:, (pred)] = res[0][0]*test.t5 + res[0][1]*test.const
            results.loc[test.index, pred] = test[pred]
    
    print(f'rmse: {rmse(results[pred], results.avg_pday):.3}')
    print(f'ubrmse: {ubrmse(results[pred], results.avg_pday):.3}')
    print(f'bias: {bias(results[pred], results.avg_pday):.3}')
    print(f'proj corr: {np.corrcoef(results.avg_pday, results.t5)[0, 1]:.3}')
    print(f'pred corr: {np.corrcoef(results.avg_pday, results[pred])[0, 1]:.3}')

    results['err'] = results[f'pred{test_spec}'] - results.avg_pday
    est, se = mult_bias(results)
    print('pred_error ~ centered_pday + air_temp + const')
    print(f'param estimates: {est.values}')
    print(f'est std errors: {se.values}')
    print('')

results.to_csv('log_preds.csv', index=False)

#for yr in fy.year.unique():
#    for grp in range(3):
#        train = results[(results.year!=yr)&(results.group2!=grp)]
#        test = results[(results.year==yr)&(results.group2==grp)].copy()
#
#        res = np.linalg.lstsq(train[['t5', 'const']], train.avg_pday, rcond=None)
#        test.loc[:, ('pred2')] = res[0][0]*test.t5 + res[0][1]*test.const
#        results.loc[test.index, 'pred2'] = test.pred2
#
#print('mixed')
#print(rmse(results.pred2, results.avg_pday))
#print(ubrmse(results.pred2, results.avg_pday))
