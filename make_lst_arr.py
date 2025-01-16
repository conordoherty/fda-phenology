import glob
import time

import numpy as np

from osgeo import gdal
from ras_utils import write_ras_same, warp_ras_grid
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor


data_dir = '/mnt/tiffy/conor/fda'
scale = 0.02
nodata = -9999
timesteps = 27
size = 2400

fn_pre = 'MYD'
sat = 'aqua'

total_time_start = time.time()
def make_lst_arr(year):
    print(f'starting {year}')
    start = time.time()
    hdf_fns = glob.glob(f'{data_dir}/modis_lst/*{fn_pre}*A{year}*.hdf')
    
    # get unique day ids and sort
    day_ids = list(set(map(lambda x: x.split('.')[1], hdf_fns)))
    day_ids.sort()
    if len(day_ids) != timesteps:
        print(len(day_ids))
        #raise Exception(f'Missing files for year {year}')
        off = 27 - len(day_ids)
    else:
        off = 0
    
    # array for terra day and night sr for each period
    terra_dlst = np.ones((size*2, size*2, timesteps), dtype=np.float32)*nodata
    terra_nlst = np.ones((size*2, size*2, timesteps), dtype=np.float32)*nodata
    for i, day_id in enumerate(day_ids):
        print(f'{year} {i+1} of {len(day_ids)}')
        day_hdfs = glob.glob(f'{data_dir}/modis_lst/*{fn_pre}*{day_id}*.hdf')
        temp_day = np.ones((size, size), dtype=np.float32)*nodata
        temp_night = np.ones((size, size), dtype=np.float32)*nodata
        for hdf in day_hdfs:
            tile = hdf.split('.')[2]
            sds = gdal.Open(hdf).GetSubDatasets()
    
            # Day lst
            day_ds = [x[0] for x in sds if 'LST_Day' in x[0]][0]
            day_arr = gdal.Open(day_ds).ReadAsArray()
            day_arr = day_arr.astype(np.float32)*scale
            day_qc_ds = [x[0] for x in sds if 'QC_Day' in x[0]][0]
            day_qc_arr = gdal.Open(day_qc_ds).ReadAsArray()

            #day_arr[day_qc_arr & 2**7 == 0 ] = nodata
            #day_arr[day_qc_arr & 3 != 0] = nodata
            day_arr[day_arr == 0] = nodata
            day_arr[day_qc_arr & 2 != 0] = nodata
            day_arr[day_qc_arr & 4 != 0] = nodata
            day_arr[day_qc_arr & (2**7) == 0] = nodata
            #day_arr[day_qc_arr & (2**7+2**6) != 2**7+2**6] = nodata

            # Night lst
            night_ds = [x[0] for x in sds if 'LST_Night' in x[0]][0]
            night_arr = gdal.Open(night_ds).ReadAsArray()
            night_arr = night_arr.astype(np.float32)*scale
            night_qc_ds = [x[0] for x in sds if 'QC_Night' in x[0]][0]
            night_qc_arr = gdal.Open(night_qc_ds).ReadAsArray()

            #night_arr[night_qc_arr & 2**7 == 0 ] = nodata
            #night_arr[night_qc_arr & 3 != 0 ] = nodata
            night_arr[night_arr == 0] = nodata
            night_arr[night_qc_arr & 2 != 0] = nodata
            night_arr[night_qc_arr & 4 != 0] = nodata
            night_arr[night_qc_arr & (2**7) == 0] = nodata
            #night_arr[night_qc_arr & (2**7+2**6) != 2**7+2**6] = nodata

            row_off = 0
            if 'v05' in tile:
                row_off = 1
    
            col_off = 0
            if 'h11' in tile:
                col_off = 1

            temp_day[row_off*1200:(row_off+1)*1200,
                     col_off*1200:(col_off+1)*1200] = day_arr
    
            temp_night[row_off*1200:(row_off+1)*1200,
                       col_off*1200:(col_off+1)*1200] = night_arr

            #if i == 8 and col_off == 1 and row_off == 0:
            #    import ipdb
            #    ipdb.set_trace()
    
        # write day
        day_coarse_ras = write_ras_same(temp_day, '', f'{data_dir}/big_grid.tif',
                                        ras_format='MEM', no_data=nodata)
        day_fine_ras = warp_ras_grid(day_coarse_ras, '',
                                     f'{data_dir}/big_grid_500.tif',
                                     resample_method='nearest',
                                     ras_format='MEM', no_data=nodata)
        terra_dlst[:, :, i+off] = day_fine_ras.ReadAsArray()

        # write night 
        night_coarse_ras = write_ras_same(temp_night, '', f'{data_dir}/big_grid.tif',
                                          ras_format='MEM', no_data=nodata)
        night_fine_ras = warp_ras_grid(night_coarse_ras, '',
                                       f'{data_dir}/big_grid_500.tif',
                                       resample_method='nearest',
                                       ras_format='MEM', no_data=nodata)

        terra_nlst[:, :, i+off] = night_fine_ras.ReadAsArray()

    
    np.save(open(f'{data_dir}/lst_arr/{sat}_dlst_{year}_lib.npy', 'wb'), terra_dlst)
    np.save(open(f'{data_dir}/lst_arr/{sat}_nlst_{year}_lib.npy', 'wb'), terra_nlst)
    print(f'completed {year} in {(time.time()-start)/60} minutes')



with ThreadPoolExecutor(max_workers=4) as e:
    e.map(make_lst_arr, range(2002, 2013))

#make_lst_arr(2006)

#with Pool(processes=4) as pool:
#    pool.map(make_lst_arr, range(2002, 2013))

#for yr in range(2002, 2013):
#    make_ndvi_arr(yr)

print(f'total time: {(time.time()-total_time_start)/60} minutes')
