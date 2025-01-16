import pandas as pd 
import numpy as np
import pickle
import matplotlib.pyplot as plt
import statsmodels.api as sm

from itertools import product
from fda import make_fda_vecs
from fda_aims2 import get_fda_transform
from multiprocessing import Pool
from util_fns import rmse, ubrmse, bias, mult_bias

rng = np.random.default_rng(0)
info_cols = 5

states = [17, 18, 19]

ndvi_start = 7
ndvi_end = 20

#ndvi_inds = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

lst_start = 4
lst_end = 8

nsdsi_start = 4
nsdsi_end = 8

data = pickle.load(open('ndvi_ts.p', 'rb'))
data = data[data.year>2002]
yf = data.iloc[:, :2]

ndvi = data.loc[:, ndvi_start:ndvi_end]
ndvi.columns = [str(x)+'_ndvi' for x in ndvi.columns]
#ndvi = data.loc[:, ndvi_inds]

sat = 'aqua'

dlst_data = pickle.load(open(f'{sat}_dlst_ts_lib.p', 'rb'))
dlst = dlst_data.loc[:, lst_start:lst_end]/100
dlst.columns = [str(x)+'_dlst' for x in dlst.columns]

nlst_data = pickle.load(open(f'{sat}_nlst_ts_lib.p', 'rb'))
nlst = nlst_data.loc[:, lst_start:lst_end]/100
nlst.columns = [str(x)+'_nlst' for x in nlst.columns]

lst_agg = pd.concat((yf, dlst, nlst), axis=1)\
            .groupby(['fips', 'year'])\
            .agg('mean')\
            .reset_index()

nsdsi_data = pickle.load(open('nsdsi_ts.p', 'rb'))
nsdsi = nsdsi_data.loc[:, nsdsi_start:nsdsi_end]
nsdsi.columns = [str(x)+'_nsdsi' for x in nsdsi.columns]

all_inputs = pd.concat((data.iloc[:, :2], ndvi, dlst, nlst, nsdsi), axis=1)
#all_inputs = pd.concat((data.iloc[:, :2], ndvi), axis=1)
counts = all_inputs.groupby(['fips', 'year']).agg('count').reset_index()
good_counties = counts[counts.iloc[:, 2:].min(axis=1)>=10][['fips', 'year']]

#dat = pickle.load(open('../planting_date/scaled_dat2.p', 'rb'))
#data = dat[dat.band==12].drop('band', axis=1).reset_index(drop=True)

combs = [
           ('ndvi', [ndvi]),
           ('ndvi+nsdsi', [ndvi, nsdsi]),
           ('ndvi+lst', [ndvi, dlst, nlst]),
           ('ndvi+nsdsi+lst', [ndvi, nsdsi, dlst, nlst])
        ]

# year, fips -> data.iloc[:, :2]
#data = pd.concat((data.iloc[:, :2], ndvi), axis=1)
#data = pd.concat((data.iloc[:, :2], ndvi, nsdsi), axis=1)
#data = pd.concat((data.iloc[:, :2], ndvi, dlst, nlst), axis=1)
#data = pd.concat((yf, ndvi, dlst, nlst, nsdsi), axis=1)

county_perm = rng.permutation(data.fips.unique())
all_preds = pd.DataFrame()
vec_dict = {}
for test_spec in [1, 2]:
    print('*'*30)
    if test_spec == 1:
        print('counties divided by state')
    else:
        print('counties divided randomly')
    print('*'*30)
    print('')

    for comb in combs:
        data = pd.concat(([yf]+comb[1]), axis=1)
        #data = pd.concat((yf, ndvi, dlst, nlst, nsdsi), axis=1)
        
        data = data.merge(good_counties, on=['fips', 'year'])
        data = data.groupby(['fips', 'year']).agg('mean').reset_index()
        data = data.dropna()
        
        #data2 = pickle.load(open('ndvi_test.p', 'rb'))
        #ndvi2 = data2.loc[:, ndvi_start:ndvi_end]
        #data2 = pd.concat((data2.iloc[:, :2], ndvi2), axis=1)
        ##data2 = data2.merge(lst_agg, on=['fips', 'year']).dropna()
        
        pdays = pickle.load(open('../planting_date/pdays.p', 'rb'))
        #pdays['air_temp'] = (pdays.Tmax_91_120 + pdays.Tmin_91_120) / 2
        pdays['air_temp'] = pdays.Tmax_91_120
        temp = pdays[['YEAR', 'FIPS', 'air_temp']]
        temp.columns = ['year', 'fips', 'air_temp']
        pdays = pdays[['YEAR', 'FIPS', 'sday']]
        pdays.columns = ['year', 'fips', 'avg_pday']
        
        #data = pickle.load(open('../planting_date/scaled_dat2.p', 'rb'))
        #data = data[(data.band==12)&(data.year>2002)]
        data = pdays.merge(data, on=['year', 'fips'])
        
        group1 = data.fips // 1000 - 17
        data.insert(3, 'group1', group1)
        
        num_county_groups = 3
        groups2 = np.resize(np.arange(num_county_groups), county_perm.size)
        rand_group_df = pd.DataFrame(np.vstack((county_perm, groups2)).T,
                                     columns=['fips', 'group2'])
        grp2 = data.merge(rand_group_df, on='fips')[['year', 'fips', 'group2']]\
                   .sort_values(['year', 'fips'])['group2'].to_numpy()
        data.insert(4, 'group2', grp2)
        
        years = data.year.unique()
        test_combs = list(product(np.arange(num_county_groups), years))
        
        def run_fda(test_comb, group_type):
            #print(test_comb)
            group, year = test_comb
        
            # train
            test_counties = data[f'group{group_type}'] == group
            test_year = data.year == year
            train = data[~test_counties & ~test_year]
            #cols = select_inputs(train)
            cols = np.arange(7, 21)
            #print([x for x in scols if x not in cols])
            #cols = ndvi_inds
            mean_ndvi = train.iloc[:, info_cols:].mean().to_numpy()
            #print(train.year.unique(), (train.fips // 1000).unique())
        
            train_data = train.iloc[:, info_cols:].to_numpy()
            train_data -=  mean_ndvi
            train_labels = (train.avg_pday.round()).to_numpy()
        
            #fda_vec = make_fda_vecs(train_data, train_labels, alpha=1)
            fda_vec = get_fda_transform(1, train_data, train_labels, alpha=1)
        
            if fda_vec[5] > 0:
                fda_vec = fda_vec*-1
            train_proj = train_data@fda_vec
        
            res = np.linalg.lstsq(np.concatenate((train_proj, np.ones_like(train_proj)),
                                                 axis=1),
                                  train.avg_pday, rcond=None)
            coeff, const = res[0]
        
            #test
            test = data[test_counties & test_year]
            test_data = test.iloc[:, info_cols:].to_numpy()
            test_data -= mean_ndvi
            test_proj = test_data@fda_vec
            test_pred = coeff*test_proj + const
            #print(test.year.unique(), (test.fips // 1000).unique())
        
            pred_df = test[['year', 'fips', 'avg_pday']]
            pred_df = pred_df.assign(predicted=test_pred)
            pred_df = pred_df.assign(proj=test_proj)
        
            return pred_df, fda_vec
        
        
        #plt.plot([min(ndvi_inds), max(ndvi_inds)], [0, 0], 'k--')
        #for vec in vecs:
        #    if vec[1] < 0:
        #        vec = -vec
        #    plt.plot(ndvi_inds, vec, 'b_')
        #
        #plt.show()
        
        #for test_comb in test_combs:
        #    _, vec = run_fda(test_comb)
        #    if vec[6] < 0:
        #        vec = -vec
        #    plt.plot(vec)
        
        #plt.show()
        #with Pool(2) as p:
        #    a = p.map(run_fda, test_combs)
        res = list(map(lambda x: run_fda(x, test_spec), test_combs))
        preds, vecs = zip(*res)
        
        vec_dict[(comb[0], test_spec)] = vecs

        preds = pd.concat(preds)
        preds['err'] = preds.predicted-preds.avg_pday

        print(comb[0])
        print(f'rmse: {rmse(preds.avg_pday, preds.predicted):.3}')
        print(f'ubrmse: {ubrmse(preds.avg_pday, preds.predicted):.3}')
        print(f'bias: {bias(preds.predicted, preds.avg_pday):.3}')
        print(f'proj corr: {np.corrcoef(preds.avg_pday, preds.proj)[0, 1]:.3}')
        print(f'pred corr: {np.corrcoef(preds.avg_pday, preds.predicted)[0, 1]:.3}')
        
        data = data.merge(temp, on=['fips', 'year'])
        pickle.dump(data[['fips', 'year', 'group1', 'group2', 'avg_pday',
                          'air_temp']],
                    open('good_data.p', 'wb'))
        
        preds = preds.merge(temp, on=['year', 'fips'])

        est, se = mult_bias(preds)

        print('pred_error ~ centered_pday + centered_air_temp + const')
        print(f'param estimates: {est.values}')
        print(f'est std errors: {se.values}')
        print('')
        #print(data.shape)
        #if data.shape[0] != 2054:
        #    raise Exception('data changed (n != 2054)')
    
        preds_add = preds[['fips', 'year', 'avg_pday', 'air_temp', 'err', 'predicted']].copy()
        preds_add['inputs'] = comb[0]
        preds_add['test_spec'] = test_spec
        all_preds = pd.concat((all_preds, preds_add))
        all_preds = all_preds.reset_index(drop=True)
        #plt.plot(preds.avg_pday, preds.predicted, '.')

pickle.dump(data, open('clean_data.p', 'wb'))
pickle.dump(vec_dict, open('vec_dict.p', 'wb'))
#pickle.dump(all_preds, open('all_preds.p', 'wb'))
all_preds.to_csv('fda_preds.csv', index=False)

#from seaborn import kdeplot 
#
#kde1 = kdeplot(all_preds.err, hue=all_preds['inputs'], alpha=.9,
#        palette='colorblind')
#plt.xlabel('Error (days)')
#plt.xlim(all_preds.err.min(), all_preds.err.max())
#plt.setp(kde1.get_legend().get_texts(), fontsize='10')

'''
from scipy.stats import gaussian_kde
from seaborn import color_palette

cb = color_palette('colorblind')

p = all_preds.err
n = all_preds[all_preds['inputs']=='ndvi'].err
nn = all_preds[all_preds['inputs']=='ndvi+nsdsi'].err
nl = all_preds[all_preds['inputs']=='ndvi+lst'].err
nln = all_preds[all_preds['inputs']=='ndvi+nsdsi+lst'].err


kde_ndvi = gaussian_kde(n) 
kde_ndvi_nsdsi = gaussian_kde(nn) 
kde_ndvi_lst = gaussian_kde(nl) 
kde_ndvi_lst_nsdsi = gaussian_kde(nln) 

x = np.linspace(p.min(), p.max(), 500)

ndvi_dens = kde_ndvi(x)
plt.plot(x, ndvi_dens, color=cb[0], label='ndvi')

ndvi_nsdsi_dens = kde_ndvi_nsdsi(x)
plt.plot(x, ndvi_nsdsi_dens, '--', color=cb[0], label='ndvi+nsdsi')

ndvi_lst_dens = kde_ndvi_lst(x)
plt.plot(x, ndvi_lst_dens, color=cb[1], label='ndvi+lst')

ndvi_lst_nsdsi_dens = kde_ndvi_lst_nsdsi(x)
plt.plot(x, ndvi_lst_nsdsi_dens, '--', color=cb[1], label='ndvi+lst+nsdsi')

all_dens = [ndvi_dens, ndvi_nsdsi_dens,
            ndvi_lst_dens, ndvi_lst_nsdsi_dens]

plot_max = max([x.max() for x in all_dens])*1.1
ext = max(abs(x.min()), abs(x.max()))
plt.ylim(0, plot_max)
plt.xlim(-ext, ext)

plt.plot([0, 0], [0, plot_max], '--', color='gray', linewidth='.9')

plt.xlabel('Error (days)')
plt.ylabel('Estimated density')

plt.legend()



plt.tight_layout()
plt.show()
'''


#ndvi_len = ndvi_end+1-ndvi_start
#lst_len = lst_end+1-lst_start
#plt.plot([ndvi_start, ndvi_end], [0, 0], 'k--')
#for vec in vecs:
#    plt.plot(np.arange(ndvi_start, ndvi_end+1), vec[:ndvi_len], 'b_')
##    plt.plot(np.arange(lst_start, lst_end+1), vec[ndvi_len:ndvi_len+lst_len], 'r_')
##    plt.plot(np.arange(lst_start, lst_end+1), vec[ndvi_len+lst_len:ndvi_len+2*lst_len], 'g_')
##
###plt.plot(data.iloc[:, 5:].mean())
#
#plt.show()
