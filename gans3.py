import os

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

    # Mode and Data
    parser.add_argument('mode', type=str, help='Usage mode, either pretrain/train/inference. Default is train',
                        nargs='*', default=None)
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

    # Parameters
    parser.add_argument('--scalingfactor', type=int, help='the superresolution factor, default is 5', default=5)
    parser.add_argument('--batchsize', type=int, help='Number of images grabbed per batch', default=100)
    parser.add_argument('--epochspretrain', type=int, help='Number of epochs for pretraining, default is 10',
                        default=None)
    parser.add_argument('--epochstrain', type=int, help='Number of epochs for training, default is 10', default=None)
    parser.add_argument('--alphaadvers', type=float, help='Scaling value for the effect of the discriminator',
                        default=0.001)
    args = parser.parse_args()

    # ensure that at least one valid mode has been entered, and make sure inference is not run on a pretrained model
    assert args.mode, 'At least one mode must be given.'
    if not 'pretrain' in args.mode and not 'train' in args.mode and not 'inference' in args.mode:
        print('Must provide one of the following modes: pretrain, train, inference')
        raise ValueError
    elif 'inference' in args.mode and 'pretrain' in args.mode and 'train' not in args.mode:
        print('Inference is not possible on a pre- but untrained model.')
        raise ValueError

    times = {'pretraintime': None, 'traintime': None, 'testtime': None, 'inferencetime': None}
    model_dir = None

    # PRETRAINING
    if 'pretrain' in args.mode:
        assert args.pretraindata, 'Data for pretraining must be given'
        assert os.path.isfile(args.pretraindata), f'File at {args.pretraindata} does not exist'
        assert os.path.splitext(os.path.basename(args.pretraindata))[1] in ['.nc', '.json', '.tfrecord'], \
            f'Datatype {type} is not supported. Load either .nc, .json or .tfrecord'

        # if .nc or .json is given, start the dataprep process, otherwise set datapath == tfrecordpath and continue
        print(f'main: {args.pretraindata}')
        if os.path.splitext(args.pretraindata)[1] != '.tfrecord':
            tfrecord_pretrain = dataprep(args.pretraindata, args.scalingfactor, mode='pretrain')
        else:
            tfrecord_pretrain = args.pretraindata

        # check if tfrecord exists, otherwise generate it
        # tfrecord_pretrain = os.path.join(os.getcwd(), f'{args.pretraindata}_pretrain.tfrecord')
        # if not os.path.isfile(tfrecord_pretrain):
        #     dataprep(args.pretraindata, tfrecord_pretrain, args.scalingfactor, 'pretrain')

        # initialise phiregan
        phiregans = PhIREGANs(data_type='temperature', N_epochs_pretrain=args.epochspretrain)

        # pretrain model using provided dataset, without using another pretrained model (pretrainedmodel_path==None)
        start_timer()
        print(f'main: {tfrecord_pretrain}')
        model_dir = phiregans.pretrain(r=[args.scalingfactor], model_path=args.pretrainedmodelspath,
                                       data_path=tfrecord_pretrain, batch_size=args.batchsize,
                                       pretrainedmodel_path=args.pretrainedmodel)
        times['pretraintime'] = end_timer()

    # TRAINING
    if 'train' in args.mode:
        # assert data and pretrained model (if no model was pretrained) are provided
        assert args.traindata, 'Training data must be provided'
        if not model_dir:
            assert args.pretrainedmodel, 'You must either pretrain a model before training or provide a pretrained model'
            model_dir = args.pretrainedmodel

        if len(args.traindata) == 1:
            args.traindata = args.traindata[0]
            assert os.path.splitext(args.traindata)[1] != '.tfrecord', 'If a .tfrecord is given, both train and test' \
                                                                       'files must be specified'
            assert os.path.splitext(os.path.basename(args.traindata))[1] in ['.nc', '.json'], \
                f'Datatype {os.path.splitext(os.path.basename(args.traindata))[1]} is not supported. Load either .nc, '\
                f'.json or .tfrecord'
            tfrecord_train, tfrecord_test, imgarray_HR_test = dataprep(args.traindata, args.scalingfactor, mode='train')

        elif len(args.traindata) == 2:
            types = [os.path.splitext(os.path.basename(args.traindata[0]))[1],
                     os.path.splitext(os.path.basename(args.traindata[1]))[1]]
            assert types[0] == '.tfrecord', f'Datatype {types[0]} must be a .tfrecord'
            assert types[1] == '.tfrecord', f'Datatype {types[1]} must be a .tfrecord'
            tfrecord_train, tfrecord_test = args.traindata
            filename = os.path.splitext(os.path.basename(args.traindata))[0]
            try:
                imgarray_HR_test = np.load(os.path.join(os.path.dirname(args.traindata),  f'{filename}_test_HR.npy'))
            except FileNotFoundError as e:
                warn(f'The HR image array could not be located for {filename}, reload original .nc of .json file')
                raise e

        else:
            warn(f'More than 3 arguments were given for the flat --traindata, at most 2 can be given')
            raise ValueError

        # initialise empty PhIRE-GAN and begin training
        phiregans = PhIREGANs(data_type='temperature', N_epochs_train=args.epochstrain)

        start_timer()
        model_dir = phiregans.train(r=[args.scalingfactor],
                                    data_path=tfrecord_train,
                                    model_path=args.trainedmodelspath,
                                    trainedmodelpath=model_dir,
                                    batch_size=args.batchsize)
        times['traintime'] = end_timer()

        start_timer()
        data_out, data_out_path = phiregans.test(r=[args.scalingfactor],
                                                 data_path=tfrecord_test,
                                                 model_path=model_dir,
                                                 batch_size=args.batchsize)
        times['testtime'] = end_timer()

        mse = (1 / len(imgarray_HR_test)) * np.sum((imgarray_HR_test - data_out) ** 2)
        print(f'\n\nThe mean squared error of the model is {np.round(mse, 2)}\n\n')

        infofile = open(os.path.join(os.path.dirname(model_dir), f'model_information.txt'), 'w')
        infofile.writelines([f'{phiregans.model_name} MODEL INFORMATION\n',
                             f'Training data: {args.traindata}\n',
                             f'Scaling factor: {args.scalingfactor}\n',
                             f'Batch size: {args.batchsize}\n',
                             f'Epochs:{args.epochspretrain} pretraining, {args.epochstrain} training',
                             f'Times: {times["pretraintime"]} pretraining, {times["traintime"]} training, '
                             f'{times["testtime"]} testing'])
        infofile.close()

    # INFERENCE
    if 'inference' in args.mode:
        # assert required parameters are provided and valid
        assert args.inferencedata, 'Data for pretraining must be given'
        assert os.path.isfile(args.inferencedata), f'File at {args.pretraindata} does not exist'
        assert os.path.splitext(os.path.basename(args.inferencedata))[1] in ['.nc', '.json', '.tfrecord'], \
            f'Datatype is not supported. Load either .nc, .json or .tfrecord'
        if not model_dir:
            assert args.trainedmodel, 'A trained model must be given'
            model_dir = args.trainedmodel

        # load or generate data
        if os.path.splitext(args.inferencedata) != '.tfrecord':
            tfrecord_inference, imgarray_HR = dataprep(args.inferencedata, args.scalingfactor, mode='inference')
        else:
            tfrecord_inference = args.inferencedata
            imgarray_HR = np.load(os.path.join(os.getcwd(), f'Data/{args.inferencedata.split("/")[-1]}_imgs_HR.npy'))

        # calculate mean and sd of dataset and save as .pickle file
        mu_sig = calculate_mu_sig(imgarray_HR)

        phiregans = PhIREGANs(data_type='temperature', mu_sig=mu_sig)

        start_timer()
        data_out, data_out_path = phiregans.test(r=[args.scalingfactor],
                                                 data_path=tfrecord_inference,
                                                 model_path=model_dir,
                                                 batch_size=args.batchsize)
        times['inferencetime'] = end_timer()

        infofile = open(os.path.join(data_out_path, f'model_information.txt'), 'w')
        infofile.writelines([f'{phiregans.model_name} MODEL INFORMATION\n',
                             f'Datapath: {args.inferencedata}\n',
                             f'Scaling factor: {args.scalingfactor}\n',
                             f'Batch size: {args.batchsize}\n',
                             f'Times: {times["pretraintime"]} pretraining, {times["traintime"]} training, '
                             f'{times["testtime"]} testing, {times["inferencetime"]} inference'])
        infofile.close()

        mse = (1 / len(imgarray_HR)) * np.sum((imgarray_HR - data_out) ** 2)
        print(f'\n\nThe mean squared error of the model is {np.round(mse, 2)}\n\n')



