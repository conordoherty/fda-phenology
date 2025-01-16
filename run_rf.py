import pickle
import time
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor as rfr
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from util_fns import rmse, ubrmse, bias, mult_bias

data = pickle.load(open('clean_data.p', 'rb'))
years = data.year.unique()

input_combs = [['ndvi'], ['ndvi', 'nsdsi'], ['ndvi', 'lst'], ['ndvi', 'nsdsi', 'lst']]

test_combs = list(product(np.arange(3), years, [1, 2], input_combs))

def do_rf(comb):
    group, year, spec, inputs = comb 
    cols = [x for x in data.columns if any(y in x for y in inputs)]

    train = data[(data[f'group{spec}']!=group)&(data.year!=year)]
    test = data[(data[f'group{spec}']==group)&(data.year==year)]

    rf = rfr().fit(train[cols], train.avg_pday)
    preds = rf.predict(test[cols])
    df = test[['fips', 'year']].copy()
    df[f'predicted'] = preds
    df[f'test_spec'] = spec 
    df[f'inputs'] = '+'.join(inputs)

    return df

start = time.time()
    
#res = list(map(do_rf, test_combs))
with ProcessPoolExecutor(max_workers=40) as e:
    res = e.map(do_rf, test_combs)

res = pd.concat(list(res)).reset_index(drop=True)
#res = res[res.pred2.isna()].drop('pred2', axis=1)\
#      .merge(res[res.pred1.isna()].drop('pred1', axis=1),
#             on=['fips', 'year', 'inputs'])
res = res.merge(data[['fips', 'year', 'avg_pday', 'air_temp']], on=['fips', 'year'])
res['err'] = res.predicted-res.avg_pday

#print((time.time()-start)/60)
for test_spec in [1, 2]:
    print(test_spec)
    for inputs in ['+'.join(x) for x in input_combs]:
        print(inputs)
        mod_res = res[(res.inputs==inputs)&(res.test_spec==test_spec)].copy()

        print(f'rmse: {rmse(mod_res.predicted, mod_res.avg_pday):.3}')
        print(f'ubrmse: {ubrmse(mod_res.predicted, mod_res.avg_pday):.3}')
        print(f'bias: {bias(mod_res.predicted, mod_res.avg_pday):.3}')
        print(f'pred corr: {np.corrcoef(mod_res.avg_pday, mod_res.predicted)[0, 1]:.3}')

        mod_res['err'] = mod_res.predicted - mod_res.avg_pday
        est, se = mult_bias(mod_res)
        print('pred_error ~ centered_pday + air_temp + const')
        print(f'param estimates: {est.values}')
        print(f'est std errors: {se.values}')
        print('')

res.to_csv('rf_preds.csv', index=False)

