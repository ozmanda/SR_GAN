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
    parser.add_argument('--traindata', type=str, help='relative path to the netCDF training file', default=None, nargs='*')
    parser.add_argument('--inferencedata', type=str, help='relative path to the netCDF test file', default=None)

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
    parser.add_argument('--epochspretrain', type=int, help='Number of epochs for pretraining, '
                                                            'default is 10', default=None)
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
        if os.path.splitext(args.pretraindata)[1] != '.tfrecord':
            tfrecord_pretrain = dataprep(args.pretraindata, args.scalingfactor, mode=args.mode)
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
        model_dir = phiregans.pretrain(r=[args.scalingfactor], model_path=args.pretrainedmodelspath,
                                       data_path=tfrecord_pretrain, batch_size=args.batchsize,
                                       pretrainedmodel_path=args.pretrainedmodel)
        times['pretraintime'] = end_timer()

    # TRAINING
    if 'train' in args.mode:
        # assert data is provided and load or generate if necessary
        assert args.traindata, 'Training data must be provided'
        assert os.path.basename(args.traindata).splitext()[1] in ['.nc', '.json', '.tfrecord'], \
            f'Datatype {type} is not supported. Load either .nc, .json or .tfrecord'

        # ensure pretrained model was provided if no model was pretrained
        if not model_dir:
            assert args.pretrainedmodel, 'You must either pretrain a model before training or provide a pretrained model'
            model_dir = args.pretrainedmodel

        if len(args.traindata) == 1:
            assert os.path.splitext(args.traindata[0])[1] != '.tfrecord', 'If a .tfrecord is given, both train and test' \
                                                                          'files must be specified'
            tfrecord_train, tfrecord_test = dataprep(args.traindata[0], args.scalingfactor, mode=args.mode)

        elif len(args.traindata) == 2:
            tfrecord_train, tfrecord_test = args.traindata

        else:
            warn(f'More than 3 arguments were given for the flat --traindata, at most 2 can be given')
            raise ValueError

        # initialise empty PhIRE-GAN and begin training
        phiregans = PhIREGANs(data_type='temperature', N_epochs_train=args.epochstrain)

        start_timer()
        model_dir = phiregans.train(r=[args.scalingfactor],
                                    data_path=tfrecord_train,
                                    model_path=args.trainedmodel,
                                    trainedmodelpath=model_dir,
                                    batch_size=args.batchsize)
        times['traintime'] = end_timer()

        start_timer()
        data_out, data_out_path = phiregans.test(r=[args.scalingfactor],
                                                 data_path=tfrecord_test,
                                                 model_path=model_dir,
                                                 batch_size=args.batchsize)
        times['testtime'] = end_timer()

        mse = (1/len(imgarray_HR)) * np.sum((imgarray_HR-data_out)**2)
        print(f'\n\nThe mean squared error of the model is {np.round(mse, 2)}\n\n')

        infofile = open(os.path.join(os.path.dirname(model_dir), f'model_information.txt'), 'w')
        infofile.writelines([f'{phiregans.model_name} MODEL INFORMATION\n',
                             f'Training data: {args.traindata}\n',
                             f'Scaling factor: {args.scalingfactor}\n',
                             f'Batch size: {args.batchsize}\n',
                             f'Epochs:{args.epochs_pretrain} pretraining, {args.epochs_train} training',
                             f'Times: {times[0]} pretraining, {times[1]} training, {times[2]} testing'])
        infofile.close()

    # INFERENCE
    if 'inference' in args.mode:
        # assert required parameters are provided and valid
        assert args.inferencedata, 'Data for pretraining must be given'
        assert os.path.isfile(args.inferencedata), f'File at {args.pretraindata} does not exist'
        assert os.path.basename(args.inferencedata).splitext()[1] in ['.nc', '.json', '.tfrecord'], \
            f'Datatype {type} is not supported. Load either .nc, .json or .tfrecord'

        # load or generate data
        tfrecord_inference = os.path.join(os.getcwd(), f'Data/{args.inferencedata.split("/")[-1]}_test.tfrecord')
        if not os.path.isfile(tfrecord_inference):
            imgarray_HR = dataprep(args.inferencedata, tfrecord_inference, args.scalingfactor, 'inference')
        else:
            imgarray_HR = np.load(os.path.join(os.getcwd(), f'Data/{args.inferencedata.split("/")[-1]}_imgs_HR.npy'))

        if not args.modelpath:
            warn('No model path was given for mode test - no model could be loaded.', Warning)
            raise ValueError
        else:
            model_path = args.trainedmodel

        # calculate mean and sd of dataset and save as .pickle file
        mu_sig = calculate_mu_sig(imgarray_HR)

        phiregans = PhIREGANs(data_type=args.datatype, mu_sig=mu_sig)

        start_timer()
        data_out, data_out_path = phiregans.test(r=[args.scalingfactor],
                                                 data_path=tfrecord_inference,
                                                 model_path=model_path,
                                                 batch_size=args.batchsize)
        times['inferencetime'] = end_timer()

        infofile = open(os.path.join(data_out_path, f'model_information.txt'), 'w')
        infofile.writelines([f'{phiregans.model_name} MODEL INFORMATION\n',
                             f'Datapath: {args.inferencedata}\n',
                             f'Scaling factor: {args.scalingfactor}\n',
                             f'Batch size: {args.batchsize}\n',
                             f'Times: {times[0]} pretraining, {times[1]} training, {times[2]} testing'])
        infofile.close()

        mse = (1/len(imgarray_HR)) * np.sum((imgarray_HR-data_out)**2)
        print(f'\n\nThe mean squared error of the model is {np.round(mse, 2)}\n\n')



