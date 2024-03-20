import os
from SRGAN import SRGAN
from PhIREGANs import *
from palm_tempmaps import *
from netCDF4 import Dataset
import argparse
import gan_utils
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

VALID_MODES = ['pretrain', 'train', 'inference']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Mode and Data
    parser.add_argument('mode', type=str, help='Usage mode, either pretrain/train/inference. Default is train',
                        nargs='*', default=None)
    parser.add_argument('--scalingfactor', type=int, help='Scaling factor for super resolution', default=5)
    parser.add_argument('--pretraindata', type=str, help='relative path to the pretraining dataset', default=None)
    parser.add_argument('--traindata', type=str, help='relative path to the training dataset', default=None, nargs='*')
    parser.add_argument('--inferencedata', type=str, help='relative path to the test dataset', default=None)

    # Model paths
    parser.add_argument('--pretrainedmodelspath', type=str, default='models/pretrained/',
                        help='the relative path to the folder containing trained models')
    parser.add_argument('--trainedmodelspath', type=str, default='models/trained/',
                        help='the relative path to the folder containing trained models')
    parser.add_argument('--pretrainedmodel', type=str, default=None, help='path to the model to use for training')
    parser.add_argument('--trainedmodel', type=str, default=None, help='path to the model to use for inference')

    # train/test parameters
    parser.add_argument('--epochs_pretrain', type=int, help='Number of epochs of pretraining', default=10)
    parser.add_argument('--batchsize_pretrain', type=int, help='Batch size for pretraining', default=100)
    parser.add_argument('--epochs_train', type=int, help='Number of epochs of training', default=10)
    parser.add_argument('--batchsize_train', type=int, help='Batch size for training', default=100)
    parser.add_argument('--batchsize_test', type=int, help='Batch size for testing', default=100)
    parser.add_argument('--batchsize_inference', type=int, help='Batch size for inference', default=100)

    # optional parameters
    parser.add_argument('--learning_rate', type=float, help='Learning rate for gradient descent (may decrease to 1e-5 after initial training)', default=1e-4)
    parser.add_argument('--epoch_shift', type=int, help='Epoch to start at when reloading previously trained network', default=0)
    parser.add_argument('--save_every', type=int, help='Epochs between model saves', default=10)
    parser.add_argument('--print_every', type=int, help='Iterations between performance write outs', default=2)
    parser.add_argument('--alphaadvers', type=float, help='Scaling value for the effect of the discriminator',
                        default=0.001)
    args = parser.parse_args()

    assert set(args.mode).isdisjoint(VALID_MODES), 'At least one valid mode must be given.'
    if 'inference' in args.mode and 'pretrain' in args.mode and 'train' not in args.mode:
        print('Inference is not possible on a pre- but untrained model.')
        raise ValueError

    srgan = SRGAN(data_type='temperature', sf=args.scalingfactor)

    if 'pretrain' in args.mode:
        srgan.configure_pretraining(datapath=args.pretraindata, epochs=args.epochs_pretrain, batchsize=args.batchsize_pretrain, learningrate=args.learning_rate, 
                                    pretrainedmodel=args.pretrainedmodel, pretrainedmodelspath=args.pretrainedmodelspath, alphaadvers=args.alphaadvers)
        if args.pretrainedmodel:
            srgan.set_pretrained_model(args.pretrainedmodel)
        srgan.set_pretrain_data(args.pretraindata, args.scalingfactor)
        srgan.run_pretraining(args.epochs_pretrain, args.pretrainedmodelspath, args.batchsize, args.pretrainedmodel)

    if 'train' in args.mode:
        #! consider loading a previously pretrained model
        #! consider training a previously trained model --> same process?
        srgan.configure_training(datapath=args.traindata, epochs=args.epochs_train, batchsize_train=args.batchsize_train, batchsize_test=args.batchsize_test,
                                learningrate=args.learning_rate, trainedmodel=args.trainedmodel, savepath=args.trainedmodelspath)
        if args.trainedmodel:
            srgan.set_trained_model(args.trainedmodel)
        srgan.run_training(args.epochs_train, args.trainedmodelspath, args.batchsize, args.trainedmodel)

    if 'inference' in args.mode:
        if args.trainedmodel:
            srgan.set_trained_model(args.trainedmodel)
        srgan.configure_inference(datapath=args.inferencedata, batchsize=args.batchsize_inference)
        srgan.run_inference()

    srgan.write_run_info()


