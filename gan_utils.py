import os
import numpy as np
from warnings import warn
from netCDF4 import Dataset
from utils import downscale_image, generate_TFRecords
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def create_tempmaps(datapath, filename, scalingfactor):
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

    tempmaps = tempmaps[:, :,
               0:tempmaps.shape[2] - tempmaps.shape[2] % scalingfactor,
               0:tempmaps.shape[3] - tempmaps.shape[3] % scalingfactor]
    tempmaps = tempmaps.reshape((-1, tempmaps.shape[2], tempmaps.shape[3], 1))
    tempmaps = tempmaps.astype('float64')
    np.save(os.path.join(os.getcwd(), f'Data/{filename}.npy'), tempmaps)

    return tempmaps


def create_images(imgdir, datapath, tempmaps, scalingfactor):
    norm = Normalize(vmin=2, vmax=42)
    cmap = plt.get_cmap('viridis')
    cmap.set_bad('white', 1.)
    # cm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # set monitor DPI (https://www.infobyip.com/detectmonitordpi.php)
    # required to save images at exact resolution using plt.savefig
    mydpi = 96
    figsize_HR = (tempmaps.shape[1]/mydpi, tempmaps.shape[2]/mydpi)
    figsize_LR = (tempmaps.shape[1]/mydpi/scalingfactor, tempmaps.shape[2]/mydpi/scalingfactor)

    print('Generating HR and LR images')
    for idx in range(tempmaps.shape[0]):
        if not os.path.isfile(os.path.join(imgdir, f'tempmap{idx}_HR.png')):
            tempmap = tempmaps[idx, :, :, 0]
            plt.figure(figsize=figsize_HR, dpi=mydpi)
            plt.imshow(norm(tempmap), cmap=cmap)
            plt.axis('off')
            plt.savefig(os.path.join(imgdir, f'tempmap{idx}_HR.png'), bbox_inches='tight', pad_inches=0, dpi=mydpi)
            plt.close('all')

        if not os.path.isfile(os.path.join(imgdir, f'tempmap{idx}_LR.png')):
            tempmap = tempmaps[idx, :, :, :]
            tempmap = downscale_image(np.array(tempmap), scalingfactor)[0, :, :, 0]
            plt.figure(figsize=figsize_LR, dpi=mydpi)
            plt.imshow(norm(tempmap), cmap=cmap)
            plt.axis('off')
            plt.savefig(os.path.join(imgdir, f'tempmap{idx}_LR.png'), bbox_inches='tight', pad_inches=0, dpi=mydpi)
            plt.close('all')

    # create .npy files and save
    return create_arrays(imgdir, datapath, scalingfactor)


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


def dataprep(datapath, scalingfactor):
    # Check datapath and break if not valid
    if not datapath:
        warn('No relative path to the .nc training file was given.', Warning)
        raise FileNotFoundError

    filename = datapath.split('/')[-1]
    datapath = os.path.join(os.getcwd(), f'{datapath}.nc')
    if not os.path.isfile(datapath):
        warn(f'The .nc file at the path {datapath} does not exist.', Warning)
        raise FileNotFoundError

    # set imgdir and create if necessary
    imgdir = os.path.join(os.getcwd(), f'Images\\{filename}_{scalingfactor}xSR')
    if not os.path.isdir(imgdir):
        os.mkdir(imgdir)

    # check if .npy files available to generate TFRecords with - only need to check one
    if os.path.isfile(os.path.join(os.getcwd(), f'{datapath}_imgs_HR.npy')):
        imgarray_HR = np.load(os.path.join(os.getcwd(), f'{datapath}_imgs_HR.npy'))
        imgarray_LR = np.load(os.path.join(os.getcwd(), f'{datapath}_imgs_LR.npy'))

    else:
        # create tempmaps and adjust map to scaling factor size
        if not os.path.isfile(os.path.join(os.getcwd(), f'Data/{filename}_{scalingfactor}xSR.npy')):
            tempmaps = create_tempmaps(datapath, filename, scalingfactor)
        else:
            tempmaps = np.load(os.path.join(os.getcwd(), f'Data/{filename}_{scalingfactor}xSR_HR.npy'))
        imgarray_HR, imgarray_LR = create_images(imgdir, datapath, tempmaps, scalingfactor)

    # generate TFRecords from imgarrays
    print(f'\nGenerating .tfrecord from {filename}.nc\n')
    generate_TFRecords(os.path.join(os.getcwd(), f'Data/{filename}_train.tfrecord'), data_HR=imgarray_HR,
                       data_LR=imgarray_LR, mode='train')
    generate_TFRecords(os.path.join(os.getcwd(), f'Data/{filename}_test.tfrecord'), data_LR=imgarray_LR, mode='test')

    return imgarray_HR
