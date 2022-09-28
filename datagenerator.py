import os
import argparse
import pickle
from utils import generate_TFRecords, calculate_mu_sig
import numpy as np
import matplotlib.pyplot as plt
from warnings import warn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('datatype', type=str, default=None, help='Type of data (train/test)')
    parser.add_argument('--filetype', type=str, default='image', help='Either netcdf or image')
    parser.add_argument('--foldername', type=str, default=None, help='Relative path to folder containing training '
                                                                     'images or .ncfile to generate .tfrecord with')
    parser.add_argument('--imgdir', type=str, default=None, help='Folder to save images if filetype=netcdf')
    parser.add_argument('--savefile', type=str, default=None, help='Name of the savefile (without extension)')
    parser.add_argument('--savedirectory', type=str, default=None, help='Path to folder to save generated data')

    args = parser.parse_args()

    mode = args.datatype
    if mode != 'train' and mode != 'test':
        warn(f'Mode must be either train or test', Warning)
        raise ValueError

    if not args.foldername:
        warn('No data path was given.', Warning)
        raise ValueError
    else:
        datafolder = args.foldername

    if not args.savedirectory:
        warn('No save path was given a standard will be chosen and created if necessary', Warning)
        savedir = os.path.join(os.getcwd(), "Data")
        if not os.path.isdir(savedir):
            os.mkdir(savedir)
    else:
        savedir = args.savedirectory

    if not args.savefile:
        args.savefile = f'{datafolder.split("/")[-1]}'
    elif args.savefile.endswith('.tfrecord'):
        warn('Save file name should not have an extension', Warning)
        raise ValueError

    if args.filetype == 'image':
        # create img array with shape (N_batch, height, width, channels) according to PhIREGAN GitHub
        shape = plt.imread(os.path.join(datafolder, os.listdir(datafolder)[1])).shape
        newshape = (1, shape[0] - shape[0] % 5, shape[1] - shape[1] % 5, 3)
        if len(shape) == 3:
            imgarray = np.empty(shape=newshape)
        elif len(shape) == 2:
            imgarray = np.empty(shape=(1, newshape[1], newshape[2], 1))
        else:

            print(f'Reference image (pos 0) has bad shape {shape}. Images must have either 2 or 3 dimensions.')
            raise ValueError

        # iterate through all files in the data folder
        for idx, file in enumerate(os.listdir(datafolder)):

            # try to load image, file is skipped if not possible
            try:
                img = plt.imread(os.path.join(datafolder, file))
            except Exception:
                warn(f'Error loading image name {file}. Skipping this file - ensure data folder contains only images',
                     Warning)
                continue
            else:
                shp = img.shape

            # check that image is of the correct shape, if not it is skipped
            if shp != shape:
                warn(f'Image {file} shape {shp} does not correspond to standard {shape}. Skipping this image.', Warning)
                continue

            # fill array at image index
            try:
                imgarray = np.append(imgarray, img[0:newshape[1], 0:newshape[2], 0:newshape[3]].reshape(newshape),
                                     axis=0)
            except ValueError as e:
                print(img.shape)
                print(newshape)
                print(e)

    elif args.filetype == 'netcdf':
        imgdir = os.path.join(os.getcwd(), args.imgdir)



    mu_sig = calculate_mu_sig(imgarray)


# create TFRecords file according to mode given
    if mode == 'train':
        filename = f'{args.savefile}_test.tfrecord'
        filepath = os.path.join(savedir, filename)
        generate_TFRecords(filepath, imgarray, 'train', 5)

        # filename = f'{args.savefile}_mu_sig.PICKLE'
        # filepath = os.path.join(savedir, filename)
        # pickle.dump(mu_sig, open(filepath, "rb"))

        print(f'TFRecords file generated at {filepath}')

    elif mode == 'test':
        filename = f'{args.savefile}_test.tfrecord'
        filepath = os.path.join(savedir, filename)
        generate_TFRecords(filepath, imgarray, 'test', 5)

        filename = f'{args.savefile}_test_mu_sig.PICKLE'
        filepath = os.path.join(savedir, filename)
        pickle.dump(mu_sig, open(filepath, "wb"))

        print(f'TFRecords file generated at {filepath}')

