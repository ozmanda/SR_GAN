from PhIREGANs import *
from palm_tempmaps import *
from netCDF4 import Dataset
import argparse
from gan_utils import *
from utils import generate_TFRecords, calculate_mu_sig, downscale_image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

"""
This script  is a wrapper which should be used to either train a GANs from a specific PALM output file or to test an
existing GAN on a PALM output file. This is the alternative to gans.py which creates 3-channel images
  - palm_tempmaps.py
  - PhIREGANs.py


Training:
---------
The model is trained on the specified data file and the training images are subsequently tested. The trained model is 
saved in /models and the test results are saved in /data_out. 

Testing:
--------
The model located at modelpath is loaded and tested using the images generated from the PALM output file loacted at
datapath. If the images have already been generated in a previous test, use the script main.py for testing purposes.
The generated images are automatically created and saved, as well as the temperature map in .npy format.

"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Requirements
    parser.add_argument('mode', type=str, help='Usage mode, either train/test. Default is train', default='train')
    parser.add_argument('--datapath', type=str, help='relative path to the .nc file without extension', default=None)

    # Testing parameters
    parser.add_argument('--modelpath', type=str, default=None, help='the relative path to the trained model, '
                                                                    'empty if no pretrained model (training)')

    # Optional parameters
    parser.add_argument('--datatype', type=str, default='temperature', help='the name of the data type being trained, '
                                                                            'default is temperature')
    parser.add_argument('--scaling_factor', type=str, help='the superresolution factor, default is 5', default=5)
    parser.add_argument('--batchsize', type=int, help='Batch size', default=1)
    parser.add_argument('--epochs_pretrain', type=int, help='Number of epochs for pretraining, '
                                                            'default is 10', default=None)
    parser.add_argument('--epochs_train', type=int, help='Number of epochs for training, default is 10', default=None)
    parser.add_argument('--alpha_advers', type=float, help='Scaling value for the effect of the discriminator',
                        default=0.001)
    parser.add_argument('--batch_size', type=int, help='Number of images grabbed per batch', default=100)
    args = parser.parse_args()

    filename = args.datapath.split('/')[-1]

    if args.mode == 'train':
        tfrecord_path = os.path.join(os.getcwd(), f'Data/{args.datapath.split("/")[-1]}_train.tfrecord')
        if not os.path.isfile(tfrecord_path):
            imgarray_HR = dataprep(args.datapath, args.scaling_factor)
        else:
            imgarray_HR = np.load(os.path.join(os.getcwd(), f'Data/{args.datapath.split("/")[-1]}_imgs_HR.npy'))

        model_path = os.path.join(os.getcwd(), args.modelpath) if args.modelpath else None

        phiregans = PhIREGANs(data_type=args.datatype,
                              N_epochs_pretrain=args.epochs_pretrain,
                              N_epochs_train=args.epochs_train)

        model_dir = phiregans.pretrain(r=[args.scaling_factor],
                                       data_path=tfrecord_path,
                                       model_path=model_path,
                                       batch_size=args.batchsize)

        model_dir = phiregans.train(r=[args.scaling_factor],
                                    data_path=tfrecord_path,
                                    model_path=model_dir,
                                    batch_size=args.batchsize)

        data_out, data_out_path = phiregans.test(r=[args.scaling_factor],
                                                 data_path=tfrecord_path,
                                                 model_path=model_dir,
                                                 batch_size=args.batchsize)

        mse = (1/len(imgarray_HR)) * np.sum((imgarray_HR-data_out)**2)
        print(f'\n\nThe mean squared error of the model is {np.round(mse, 2)}\n\n')




    if args.mode == 'test':
        # save the .tfrecord file
        tfrecord_path = os.path.join(os.getcwd(), f'Data/{args.datapath.split("/")[-1]}_test.tfrecord')
        if not os.path.isfile(tfrecord_path):
            imgarray_HR = dataprep(args.datapath, args.scaling_factor)
        else:
            imgarray_HR = np.load(os.path.join(os.getcwd(), f'Data/{args.datapath.split("/")[-1]}_imgs_HR.npy'))

        if not args.modelpath:
            warn('No model path was given for mode test - no model could be loaded.', Warning)
            raise ValueError
        else:
            model_path = os.path.join(os.getcwd(), args.modelpath)

        # calculate mean and sd of dataset and save as .pickle file
        mu_sig = calculate_mu_sig(imgarray_HR)

        phiregans = PhIREGANs(data_type=args.datatype, mu_sig=mu_sig)

        data_out, data_out_path = phiregans.test(r=[args.scaling_factor],
                                                 data_path=tfrecord_path,
                                                 model_path=model_path,
                                                 batch_size=args.batchsize)

        mse = (1/len(imgarray_HR)) * np.sum((imgarray_HR-data_out)**2)
        print(f'\n\nThe mean squared error of the model is {np.round(mse, 2)}\n\n')


