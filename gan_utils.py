import os
import pickle as cPickle
import numpy as np
from warnings import warn
from netCDF4 import Dataset
from utils import downscale_image, generate_TFRecords
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import time
import pandas as pd


def check_file(path, mode=None): 
    assert path, f'No file given for {mode}'
    assert os.path.isfile(path), f'File {path} does not exist'
    assert os.path.splitext(os.path.basename(path))[1] in ['.nc', '.json', '.tfrecord'], \
        f'Datatype {type} is not supported. Load either .nc, .json or .tfrecord'


def create_tempmaps(datapath):
    """
    Loads temperature maps from .nc file and flattens the layer dimension to return a [layers x time, lat, lon] array
    """
    # try to load .nc file and give warning if it cannot be loaded.
    try:
        # load maps, extract tempmaps, creat np.array and replace fill values -9999 with 0
        maps = Dataset(datapath, 'r', format="NETCDF4")
        tempmaps = np.array(maps["theta_xy"][:, :, :, :])
        #! NaN values are not supported in tfrecords, so we leave the fill value
        # tempmaps[tempmaps == -9999] = np.NaN
        tempmaps -= 273.15
    except Exception as e:
        warn(f'The NetCDF file at path {datapath} could not be loaded, or the temperature '
             f'variable has a name other than "theta_xy"', Warning)
        raise ValueError

    tempmaps = np.reshape(tempmaps, newshape=(tempmaps.shape[0]*tempmaps.shape[1],
                                              tempmaps.shape[2], tempmaps.shape[3]))
    tempmaps = tempmaps.astype('float64')
    # np.save(os.path.join(os.getcwd(), f'Data/{filename}.npy'), tempmaps)

    return tempmaps


def extract_temps(palmfile: Dataset):
    try:
        temps = palmfile['theta_xy']
    except IndexError:
        temps = palmfile['theta']

    temps = np.reshape(temps, newshape=(temps.shape[0]*temps.shape[1],
                                        temps.shape[2], temps.shape[3]))
    #! NaN values are not supported in tfrecords, so we leave the fill value
    # temps[np.where(temps == -9999)] = np.NaN
    temps[np.where(temps != -9999)] -= 273.15
    # flip maps to account for PALM having origin at the bottom left, not top left
    temps = np.flip(temps, axis=1)
    return temps


def extract_surfacetemps(palmfile: str):
    try:
        temps = palmfile['theta_xy']
    except IndexError:
        temps = palmfile['theta']

    surf_temps = np.zeros(shape=(temps.shape[0], temps.shape[2], temps.shape[3]))

    for time in range(temps.shape[0]):
        for idxs, _ in np.ndenumerate(temps[time, 0, :, :]):
            for layer in range(temps.shape[1]):
                if temps[time, layer, idxs[0], idxs[1]] != -9999:
                    surf_temps[time, :, :][idxs] = temps[time, layer, idxs[0], idxs[1]] - 273.15
                    break
                else:
                    continue
    # flip maps to account for PALM having origin at the bottom left, not top left
    surf_temps = np.flip(surf_temps, axis=1)

    return surf_temps


def palm_times(palmfile: Dataset):
    """
    Extracts the time vector, formatting it as a datetime. The time contained within the PALM file is given as
    minutes since origin. Additionally, a boolean vector is generated, indicating the start of the useable time
    series (certain observations are required to create the moving average).
    """
    origintime = pd.to_datetime(palmfile.origin_time)
    times_list = palmfile['time']
    times = []
    for _, time in enumerate(times_list):
        times.append(origintime + pd.Timedelta(minutes=np.round(time * 24 * 60)))
    if not times:
        raise ValueError
    return times


def create_images(imgdir, HR_array, LR_array):
    norm = Normalize(vmin=2, vmax=42)
    cmap = plt.get_cmap('viridis')
    cmap.set_bad('white', 1.)
    # cm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # set monitor DPI (https://www.infobyip.com/detectmonitordpi.php)
    # required to save images at exact resolution using plt.savefig
    mydpi = 96
    figsize_HR = (HR_array.shape[1]/mydpi, HR_array.shape[2]/mydpi)
    figsize_LR = (LR_array.shape[1]/mydpi, LR_array.shape[2]/mydpi)

    print('Generating HR and LR images')
    for idx in range(HR_array.shape[0]):
        if not os.path.isfile(os.path.join(imgdir, f'tempmap{idx}_HR.png')):
            tempmap = HR_array[idx, :, :]
            plt.figure(figsize=figsize_HR, dpi=mydpi)
            plt.imshow(norm(tempmap), cmap=cmap)
            plt.axis('off')
            plt.savefig(os.path.join(imgdir, f'tempmap{idx}_HR.png'), bbox_inches='tight', pad_inches=0, dpi=mydpi)
            plt.close('all')

        if not os.path.isfile(os.path.join(imgdir, f'tempmap{idx}_LR.png')):
            tempmap = LR_array[idx, :, :]
            plt.figure(figsize=figsize_LR, dpi=mydpi)
            plt.imshow(norm(tempmap), cmap=cmap)
            plt.axis('off')
            plt.savefig(os.path.join(imgdir, f'tempmap{idx}_LR.png'), bbox_inches='tight', pad_inches=0, dpi=mydpi)
            plt.close('all')

    # create .npy files and save
    # return create_arrays(imgdir, datapath, scalingfactor)


def create_arrays(imgdir, datapath, scalingfactor):
    # create HR imgarray
    print('Loading HR images to .npy file')
    shape = plt.imread(os.path.join(imgdir, os.listdir(imgdir)[0])).shape
    shape = (1, shape[0], shape[1], shape[2])
    imgarray_HR = np.empty(shape)

    for imgname in os.listdir(imgdir):
        if imgname.endswith('HR.png'):
            img = plt.imread(os.path.join(imgdir, imgname)).reshape(shape)
            imgarray_HR = np.append(imgarray_HR, img, axis=0)

    imgarray_HR = imgarray_HR[1:, 0:shape[1] - shape[1] % scalingfactor,
                  0:shape[2] - shape[2] % scalingfactor, 0:-1]

    imgarray_HR = imgarray_HR.astype('float64')
    np.save(os.path.join(os.getcwd(), f'{datapath}_imgs_HR.npy'), imgarray_HR)


    # create LR imgarray
    print('Loading LR images to .npy file')
    shape = plt.imread(os.path.join(imgdir, os.listdir(imgdir)[1])).shape
    shape = (1, shape[0], shape[1], shape[2])
    imgarray_LR = np.empty(shape)

    for imgname in os.listdir(imgdir):
        if imgname.endswith('LR.png'):
            img = plt.imread(os.path.join(imgdir, imgname)).reshape(shape)
            imgarray_LR = np.append(imgarray_LR, img, axis=0)

    imgarray_LR = imgarray_LR[1:, 0:shape[1] - shape[1] % scalingfactor,
                  0:shape[2] - shape[2] % scalingfactor, 0:-1]

    imgarray_LR = imgarray_LR.astype('float64')
    np.save(os.path.join(os.getcwd(), f'{datapath}_imgs_LR.npy'), imgarray_LR)
    return imgarray_HR, imgarray_LR


def adjust_dimensions(array, sf):
    """
    Adjusts an array to be scalable to the given scaling factor, meaning that all dimenions are divisible by the
    scaling factor.
    """
    return array[:, 0:array.shape[1] - array.shape[1] % sf,  0:array.shape[2] - array.shape[2] % sf, :]


def generate_LRHR_from_array(arrayHR, scalingfactor):
    if len(arrayHR.shape) == 3:
        arrayHR = arrayHR.reshape(arrayHR.shape[0], arrayHR.shape[1], arrayHR.shape[2], 1)
    elif len(arrayHR.shape) != 4:
        warn(f'Array has shape {arrayHR.shape}, which is not supported')
        raise ValueError
    imagearray_HR = adjust_dimensions(arrayHR, scalingfactor)
    imagearray_LR = lower_resolution(imagearray_HR, scalingfactor)
    # imagearray_LR = utils.downscale_image(imagearray_HR, scalingfactor)
    return imagearray_HR, imagearray_LR


def generate_LRHR(datapath, scalingfactor):
    """
    Load data from either .json or .nc to create images and .tfrecord
    """
    filename, type = os.path.splitext(os.path.basename(datapath))

    # Load HR image array either from .nc or .json
    if type == '.nc':
        imgarray_HR = create_tempmaps(datapath)
    elif type == '.json':
        with open(datapath, 'rb') as file:
            imgarray_HR = cPickle.load(file)
            file.close()
    else:
        warn(f'Data type {type} is not supported')
        raise TypeError

    if len(imgarray_HR.shape) == 3:
        imgarray_HR = imgarray_HR.reshape(imgarray_HR.shape[0], imgarray_HR.shape[1], imgarray_HR.shape[2], 1)
    elif len(imgarray_HR.shape) != 4:
        warn(f'Array has shape {imgarray_HR.shape}, which is not supported')
        raise ValueError
    
    # adjust array dimensions to be divisible by the scaling factor and then generate LR image array
    imgarray_HR = adjust_dimensions(imgarray_HR, scalingfactor)
    imgarray_LR = lower_resolution(imgarray_HR, scalingfactor)
    # imgarray_LR = utils.downscale_image(imgarray_HR, scalingfactor)

    return imgarray_HR, imgarray_LR


def check_hrlr(imgarrayHR, imgarrayLR, scalingfactor):
    """
    Check that the LR image is scaled down correctly from the HR image and have the same number of samples.
    """
    assert imgarrayHR.shape[1] == imgarrayLR.shape[1] * scalingfactor, 'LR image is not scaled down correctly'
    assert imgarrayHR.shape[2] == imgarrayLR.shape[2] * scalingfactor, 'LR image is not scaled down correctly'
    assert imgarrayHR.shape[0] == imgarrayLR.shape[0], 'HR and LR image arrays have different lengths'


def lower_resolution(hr_map, sr_factor):
    """
    Lowers the resolution of sparse arrays using a simple nanmean to avoid full nan arrays when using the 
    standard conv2D downscaling method. 
    """
    lr_map = np.zeros((hr_map.shape[0], hr_map.shape[1] // sr_factor, hr_map.shape[2] // sr_factor, hr_map.shape[3]))
    for i in range(lr_map.shape[1]):
        for j in range(lr_map.shape[2]):
            for t in range(lr_map.shape[0]):
                lr_map[t, i, j, :] = np.nanmean(hr_map[t, i*sr_factor : (i+1)*sr_factor, 
                                                       j*sr_factor : (j+1)*sr_factor, :])
    return lr_map


def load_training_data(datapathHR, datapathLR, scalingfactor):
    """
    Load training data from .npy files and adjust dimensions to be divisible by the scaling factor
    """
    imgarray_HR = np.load(datapathHR)
    imgarray_LR = np.load(datapathLR)
    check_hrlr(imgarray_HR, imgarray_LR, scalingfactor)
    imgarrayHR_train, imgarrayHR_test, imgarrayLR_train, imgarrayLR_test = train_test_split(imgarray_HR, imgarray_LR)
    return imgarrayHR_train, imgarrayHR_test, imgarrayLR_train, imgarrayLR_test


def dataprep(datapath, scalingfactor, mode=None):
    """
    Recieves datapath for either a .nc or .json file, for which it generates the HR/LR images and the .tfrecord suitable
    for GAN usage and returns the path to the saved .tfrecord.
    """
    # generate LR and HR image arrays
    filename, ext = os.path.splitext(os.path.basename(datapath))
    # set imgdir and create if necessary, then generate images
    imgdir = os.path.join(os.getcwd(), f'Images/{filename}_{scalingfactor}xSR')

    # TRAINING
    if 'train' == mode:
        tfrecordpath_train = os.path.join(os.path.dirname(datapath), f'{filename}_train.tfrecord')
        tfrecordpath_test = os.path.join(os.path.dirname(datapath), f'{filename}_test.tfrecord')
        if not os.path.isfile(tfrecordpath_train) or not os.path.isfile(tfrecordpath_test) or not os.path.isdir(imgdir):
            print(f'\nGenerating Training dataset from {filename}{ext}\n')
            imgarray_HR, imgarray_LR = generate_LRHR(datapath, scalingfactor)
            # make images if necessary
            if not os.path.isdir(imgdir):
                os.makedirs(imgdir)
                create_images(imgdir, imgarray_HR, imgarray_LR)
            LR_train, LR_test, HR_train, HR_test = train_test_split(imgarray_LR, imgarray_HR)
            generate_TFRecords(tfrecordpath_train, data_HR=HR_train, data_LR=LR_train, mode='train')
            generate_TFRecords(tfrecordpath_test, data_LR=LR_test, mode='test')
            np.save(os.path.join(os.path.dirname(datapath), f'{filename}_test_HR.npy'), HR_test)
        else:
            HR_test = np.load(os.path.join(os.path.dirname(datapath), f'{filename}_test_HR.npy'))

        return tfrecordpath_train, tfrecordpath_test, HR_test

    # INFERENCE
    elif 'inference' == mode:
        tfrecordpath = os.path.join(os.path.dirname(datapath), f'{filename}_inference.tfrecord')
        if not os.path.isfile(tfrecordpath):
            print(f'\nGenerating Inference dataset from {filename}{ext}\n')
            # TODO: THERE SHOULD BE NO HR GENERATION HERE, ONLY LR --> REWORK THIS CODE
            imgarray_HR, imgarray_LR = generate_LRHR(datapath, scalingfactor)
            np.save(os.path.join(os.path.dirname(datapath), f'{filename}_inference_HR.npy'), imgarray_HR)
            generate_TFRecords(tfrecordpath, data_LR=imgarray_LR, mode='test')
            return tfrecordpath, imgarray_HR
        else:
            imgarray_HR = np.load(os.path.join(os.path.dirname(datapath), f'{filename}_inference_HR.npy'))
            return tfrecordpath, imgarray_HR


def train_test_split(imgarrayHR, imgarrayLR, test_size=0.2):
    i = int((1 - test_size) * imgarrayLR.shape[0])
    o = np.random.permutation(imgarrayLR.shape[0])

    imgarrayLR_train, imgarrayLR_test = np.split(np.take(imgarrayLR, o, axis=0), [i])
    imgarrayHR_train, imgarrayHR_test = np.split(np.take(imgarrayHR, o, axis=0), [i])

    return imgarrayHR_train, imgarrayHR_test, imgarrayLR_train, imgarrayLR_test


def tfrecord_filename(datapath, mode):
    filename = os.path.splitext(os.path.basename(datapath))[0]
    return os.path.join(os.path.dirname(datapath), f'{filename}_{mode}.tfrecord')


def start_timer():
    global _start_time
    _start_time = time.time()


def end_timer():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    return f'{t_hour}:{t_min}:{t_sec}'
