from osgeo import gdal_array, gdal, osr
import datetime
import numpy as np
import gc
from numba import njit

def write_ras_same(array, new_ras_fn, template, ras_format='GTiff', no_data=None):
    """Write `array` to disk as gtiff using the same SRS/GT/dimensions as `new_raster_fn`.
    If `array` has different dimensions than the template raster, it will fail!

    array -- 2d numpy array of values to be written to disk
    new_ras_fn -- name of file to be written
    template_fn -- name of file to be used as raster template
    """
    if type(template) == str:
        template_ras = gdal.Open(template)
    else:
        template_ras = template

    #template_srs = template_ras.GetSpatialRef()
    template_srs = template_ras.GetProjectionRef()
    template_gt = template_ras.GetGeoTransform()
    template_band = template_ras.GetRasterBand(1)

    if no_data == None:
        template_nodata = template_band.GetNoDataValue()
    else:
        template_nodata = no_data

    rows, cols = array.shape
    if array.dtype == np.int64:
        raise Exception('Array type is int64 which gdal does not understand')
    d_type = gdal_array.NumericTypeCodeToGDALTypeCode(array.dtype)

    driver = gdal.GetDriverByName(ras_format)
    out_ras = driver.Create(new_ras_fn, cols, rows, 1, d_type)
    out_ras.SetGeoTransform(template_gt)
    out_band = out_ras.GetRasterBand(1)
    out_band.WriteArray(array)
    out_band.SetNoDataValue(template_nodata)
    #out_ras.SetProjection(template_srs.ExportToWkt())
    out_ras.SetProjection(template_srs)
    out_band.FlushCache()

    template_band = None
    template_ras = None

    if ras_format != 'MEM':
        out_band = None
        out_ras = None
        return 0
    else:
        return out_ras


def warp_ras_grid(in_ras, new_ras_fn, template_fn, ras_format='GTiff',
                  resample_method='lanczos', no_data='-9999.0'):
    template_ras = gdal.Open(template_fn)
    template_srs = template_ras.GetProjectionRef()
    proj = template_ras.GetProjectionRef()
    temp_gt = template_ras.GetGeoTransform()
    if type(in_ras) == str:
        in_ras = gdal.Open(in_ras)
    in_ras_nodata = in_ras.GetRasterBand(1).GetNoDataValue()

    height = template_ras.RasterYSize
    width = template_ras.RasterXSize

    new_ras = gdal.Warp(new_ras_fn,
                        in_ras,
                        format=ras_format,
                        outputBounds=(temp_gt[0], temp_gt[3]+temp_gt[5]*height, temp_gt[0]+temp_gt[1]*width, temp_gt[3]),
                        dstSRS=template_srs,
                        resampleAlg=resample_method,
                        srcNodata=in_ras_nodata,
                        dstNodata=no_data,
                        height=height,
                        width=width,
                        multithread=True)

    template_band = None
    template_ras = None

    if ras_format != 'MEM':
        new_ras = None
    else:
        return new_ras

def max_composite(temp, ras_list, new_ras_fn, ras_format='GTiff',
                  resample_method='lanczos'):
    reproj_ras_list = []
    for ras in ras_list:
        reproj_ras = gdal.Warp('',
                               ras,
                               srcSRS='EPSG:32610',
                               dstSRS='EPSG:2163',
                               format='MEM',
                               resampleAlg='lanczos',
                               multithread=True)

        grid_ras = warp_ras_grid(ras, '', temp, ras_format='MEM',
                                   resample_method='lanczos')
        reproj_ras = None
        gc.collect()
        reproj_ras_list.append(grid_ras)

    final_ras_arr = reproj_ras_list[0].ReadAsArray()
    for i in range(1, len(reproj_ras_list)):
        final_ras_arr = np.maximum(final_ras_arr, reproj_ras_list[i].ReadAsArray())

    if ras_format=='MEM':
        return write_ras_same(final_ras_arr, new_ras_fn, temp, ras_format='MEM')
    else:
        write_ras_same(final_ras_arr, new_ras_fn, temp)


# vals_arr is array with lenth n
# mask_ras_fn is filename of raster containing n pixels equal to mask_val
def write_vals_as_ras(vals_arr, new_ras_fn, mask_ras_fn, mask_val=1,
                      ras_format='GTiff', nodata=-9999):
    mask_arr = gdal.Open(mask_ras_fn).ReadAsArray()
    if (mask_arr==mask_val).sum() != vals_arr.size:
        raise Exception('size of vals_arr does not match number of pixels with '+
                        'value mask_val')

    out_ras_arr = np.ones_like(mask_arr, dtype=vals_arr.dtype)*nodata
    out_ras_arr[mask_arr==mask_val] = vals_arr

    # TODO change no_data to nodata in write_ras_same
    # but don't want to break anything right now ...
    write_ras_same(out_ras_arr, new_ras_fn, mask_ras_fn,
                   ras_format=ras_format, no_data=nodata)


@njit()
def resample_grid(arr, factor, keep_fraction=0, nodata=0):
    rows, cols = arr.shape
    if rows % factor != 0 or cols % factor != 0:
        raise Exception('Dimension not multiple of factor')

    nrows, ncols = int(rows/factor), int(cols/factor)
    new_arr = np.zeros((nrows, ncols), dtype=arr.dtype)
    for i in range(nrows):
        for j in range(ncols):
            sample = arr[i*factor:(i+1)*factor, j*factor:(j+1)*factor].flatten()
            sample_data = sample[sample != nodata]
            if sample_data.size <= keep_fraction/sample.size:
                new_arr[i, j] = nodata
            else:
                new_arr[i, j] = sample_data.mean() 

    return new_arr
