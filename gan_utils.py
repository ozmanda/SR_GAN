import os
import _pickle as cPickle
import numpy as np
from warnings import warn
from netCDF4 import Dataset

import utils
from utils import downscale_image, generate_TFRecords
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import time


def create_tempmaps(datapath, filename, scalingfactor):
    """
    Loads temperature maps from .nc file and flattens the layer dimension to return a [layers x time, lat, lon] array
    """
    # try to load .nc file and give warning if it cannot be loaded.
    try:
        # load maps, extract tempmaps, creat np.array and replace fill values -9999 with 0
        maps = Dataset(datapath, 'r', format="NETCDF4")
        tempmaps = np.array(maps["theta_xy"][:, :, :, :])
        tempmaps[tempmaps == -9999] = np.NaN
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
    return array[:, 0:array.shape[1] - array.shape[1] % sf,  0:array.shape[2] - array.shape[2] % sf]

def generate_LRHR(datapath, scalingfactor):
    """
    Load data from either .json or .nc to create images and .tfrecord
    """
    filename, type = os.path.splitext(os.path.basename(datapath))

    # Load HR image array either from .nc or .json
    if type == '.nc':
        imgarray_HR = create_tempmaps(datapath, filename, scalingfactor)
    elif type == '.json':
        with open(datapath, 'rb') as file:
            imgarray_HR = cPickle.load(file)
            file.close()
    else:
        warn(f'Data type {type} is not supported')
        raise TypeError

    # adjust array dimensions to be divisible by the scaling factor and then generate LR image array
    imgarray_HR = adjust_dimensions(imgarray_HR, scalingfactor)
    imgarray_LR = utils.downscale_image(imgarray_HR, scalingfactor)

    return imgarray_HR, imgarray_LR


def dataprep(datapath, scalingfactor, mode=None):
    """
    Recieves datapath for either a .nc or .json file, for which it generates the HR/LR images and the .tfrecord suitable
    for GAN usage and returns the path to the saved .tfrecord.
    """
    # generate LR and HR image arrays
    filename, ext = os.path.splitext(os.path.basename(datapath))
    # set imgdir and create if necessary, then generate images
    imgdir = os.path.join(os.getcwd(), f'Images/{filename}_{scalingfactor}xSR')

    # PRETRAINING
    if 'pretrain' == mode:
        tfrecordpath = os.path.join(os.path.dirname(datapath), f'{filename}_pretrain.tfrecord')
        if not os.path.isfile(tfrecordpath):
            print(f'\nGenerating Pretraining dataset from {filename}{ext}\n')
            imgarray_HR, imgarray_LR = generate_LRHR(datapath, scalingfactor)
            generate_TFRecords(tfrecordpath, data_HR=imgarray_HR, data_LR=imgarray_LR, mode='train')
            print(f'utils: {tfrecordpath}')
        return tfrecordpath

    # TRAINING
    elif 'train' == mode:
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
            # THERE SHOULD BE NO HR GENERATION HERE, ONLY LR --> REWORK THIS CODE
            imgarray_HR, imgarray_LR = generate_LRHR(datapath, scalingfactor)
            np.save(os.path.join(os.path.dirname(datapath), f'{filename}_inference_HR.npy'), imgarray_HR)
            generate_TFRecords(tfrecordpath, data_LR=imgarray_LR, mode='test')
            return tfrecordpath, imgarray_HR
        else:
            imgarray_HR = np.load(os.path.join(os.path.dirname(datapath), f'{filename}_inference_HR.npy'))
            return tfrecordpath, imgarray_HR

def train_test_split(imgarrayLR, imgarrayHR, test_size=0.2):
    i = int((1 - test_size) * imgarrayLR.shape[0])
    o = np.random.permutation(imgarrayLR.shape[0])

    imgarrayLR_train, imgarrayLR_test = np.split(np.take(imgarrayLR, o, axis=0), [i])
    imgarrayHR_train, imgarrayHR_test = np.split(np.take(imgarrayHR, o, axis=0), [i])

    return imgarrayLR_train, imgarrayLR_test, imgarrayHR_train, imgarrayHR_test


def start_timer():
    global _start_time
    _start_time = time.time()


def end_timer():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    return f'{t_hour}:{t_min}:{t_sec}'
