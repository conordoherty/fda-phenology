import pandas as pd
import numpy as np
import pickle
from util_fns import rmse, ubrmse, bias, mult_bias

fy = pickle.load(open('good_data.p', 'rb'))

preds = pd.DataFrame()
for spec in [1, 2]:
    for grp in range(3):
        group = f'group{spec}'
        for yr in range(2003, 2013):
            train = fy[(fy.year!=yr)&(fy[group]!=grp)]
            test = fy[(fy.year==yr)&(fy[group]==grp)].copy()

            test['pred'] = train.avg_pday.mean()
            test['spec'] = spec

            preds = pd.concat((preds, test[['fips', 'year', 'avg_pday', 'air_temp',
                                            'pred', 'spec']]))

for test_spec in [1, 2]:
    print(test_spec)
    spec = preds[preds.spec==test_spec].copy()
    spec['err'] = spec.pred-spec.avg_pday

    print(f'rmse: {rmse(spec.pred, spec.avg_pday):.3}')
    print(f'ubrmse: {ubrmse(spec.pred, spec.avg_pday):.3}')
    print(f'bias: {bias(spec.pred, spec.avg_pday):.3}')
    print(f'pred corr: {np.corrcoef(spec.avg_pday, spec.pred)[0, 1]:.3}')

    spec['err'] = spec.pred - spec.avg_pday
    est, se = mult_bias(spec)
    print('pred_error ~ centered_pday + air_temp + const')
    print(f'param estimates: {est.values}')
    print(f'est std errors: {se.values}')
    print('')

preds.to_csv('naive_preds.csv', index=False)

