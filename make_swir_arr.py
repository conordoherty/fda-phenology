import glob
import time

import numpy as np

from osgeo import gdal
from ras_utils import resample_grid
#from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

start = time.time()

data_dir = '/mnt/tiffy/conor/fda'
scale = 0.0001
nodata = -9999
timesteps = 27
size = 2400

def make_swir_arr(year):
    hdf_fns = glob.glob(f'{data_dir}/modis_sr/*MOD*A{year}*.hdf')
    
    # get unique day ids and sort
    day_ids = list(set(map(lambda x: x.split('.')[1], hdf_fns)))
    day_ids.sort()
    if len(day_ids) != timesteps:
        print(len(day_ids))
        #raise Exception(f'Missing files for year {year}')
        off = 3
    else:
        off = 0
    
    # array for terra day and night sr for each period
    terra_b6 = np.zeros((size*2, size*2, timesteps), dtype=np.float32)
    terra_b7 = np.zeros((size*2, size*2, timesteps), dtype=np.float32)
    for i, day_id in enumerate(day_ids):
        print(f'{year} {i+1} of {len(day_ids)}')
        day_hdfs = glob.glob(f'{data_dir}/modis_sr/*MOD*{day_id}*.hdf')
        for hdf in day_hdfs:
            tile = hdf.split('.')[2]
            sds = gdal.Open(hdf).GetSubDatasets()
    
            sr_qc_ds = [x[0] for x in sds if 'qc' in x[0]][0]
            sr_qc = gdal.Open(sr_qc_ds).ReadAsArray()
    
            cloud_ds = [x[0] for x in sds if 'state' in x[0]][0]
            cloud = gdal.Open(cloud_ds).ReadAsArray()
            # not cloudy or mixed
            cloud = ((cloud & 3) % 3 != 0)
            shadow = (cloud % 4) != 0
            cirrus = (cloud & (2**9+2**8) != 0)
            cloud_bad = cloud | shadow | cirrus
    
            b6_ds = [x[0] for x in sds if f'b06' in x[0]][0]
            b6 = gdal.Open(b6_ds).ReadAsArray()
    
            b7_ds = [x[0] for x in sds if f'b07' in x[0]][0]
            b7 = gdal.Open(b7_ds).ReadAsArray()
    
            b6[(sr_qc & (2**26-22**2) != 0)|(cloud_bad)] = nodata
            b7[(sr_qc & (2**30-26**2) != 0)|(cloud_bad)] = nodata
    
            row_off = 0
            if 'v05' in tile:
                row_off = 1
    
            col_off = 0
            if 'h11' in tile:
                col_off = 1
    
            terra_b6[row_off*size:(row_off+1)*size,
                       col_off*size:(col_off+1)*size, i+off] = b6
            terra_b7[row_off*size:(row_off+1)*size,
                       col_off*size:(col_off+1)*size, i+off] = b7
    
    np.save(open(f'{data_dir}/terra_swir/terra_b6_{year}.npy', 'wb'), terra_b6)
    np.save(open(f'{data_dir}/terra_swir/terra_b7_{year}.npy', 'wb'), terra_b7)
    print(f'completed {year} in {(time.time()-start)/60} minutes')


with ThreadPoolExecutor(max_workers=2) as e:
    e.map(make_swir_arr, range(2002, 2013))

#with Pool(threads=2) as pool:
#    pool.map(make_ndvi_arr, range(2002, 2013))

#for yr in range(2002, 2013):
#    make_ndvi_arr(yr)
