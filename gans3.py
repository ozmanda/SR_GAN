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
    parser.add_argument('--traindata', type=str, help='relative path to the netCDF training file without .nc extension',
                        default=None)
    parser.add_argument('--testdata', type=str, help='relative path to the netCDF test file without .nc extension',
                        default=None)

    # Model paths
    parser.add_argument('--pretrainedmodelspath', type=str, default='models/pretrained/',
                        help='the relative path to the folder containing trained models')
    parser.add_argument('--trainedmodelspath', type=str, default='models/trained/',
                        help='the relative path to the folder containing trained models')
    parser.add_argument('--pretrainedmodelname', type=str, default=None, help='name of the model to use for training')
    parser.add_argument('--trainedmodelname', type=str, default=None, help='name of the model to use for inference')

    # Parameters
    parser.add_argument('--scalingfactor', type=str, help='the superresolution factor, default is 5', default=5)
    parser.add_argument('--batchsize', type=int, help='Number of images grabbed per batch', default=100)
    parser.add_argument('--epochspretrain', type=int, help='Number of epochs for pretraining, '
                                                            'default is 10', default=None)
    parser.add_argument('--epochstrain', type=int, help='Number of epochs for training, default is 10', default=None)
    parser.add_argument('--alphaadvers', type=float, help='Scaling value for the effect of the discriminator',
                        default=0.001)
    args = parser.parse_args()

    # ensure that at least one valid mode has been entered, and make sure inference is not run on a pretrained model
    assert args.mode, 'At least one mode must be given.'
    if not 'pretrain' in args.mode or 'train' in args.mode or 'inference' in args.mode:
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

        # check if tfrecord exists, otherwise generate it
        tfrecord_pretrain = os.path.join(os.getcwd(), f'{args.pretraindata}_pretrain.tfrecord')
        if not os.path.isfile(tfrecord_pretrain):
            _ = dataprep(args.pretraindata, tfrecord_pretrain, args.scalingfactor, 'pretrain')

        # initialise phiregan
        phiregans = PhIREGANs(data_type='temperature',
                              N_epochs_pretrain=args.epochspretrain)

        # pretrain model using provided dataset, without using another pretrained model (model_path==None)
        start_timer()
        model_dir = phiregans.pretrain(r=[args.scalingfactor],
                                       data_path=tfrecord_pretrain,
                                       model_path=args.pretrainedmodelspath,
                                       batch_size=args.batchsize)
        times['pretraintime'] = end_timer()


    # TRAINING
    if 'train' in args.mode:
        # determine if model must be pretrained
        if not model_dir:
            assert args.pretrainedmodelname, 'You must either pretrain a model before training or provide a pretrained model'
            model_dir = os.path.join(args.pretrainedmodelspath, args.pretrainedmodelname)

        # assert data is provided and load or generate if necessary
        assert args.traindata, 'Training data must be provided'
        tfrecord_train = os.path.join(os.getcwd(), f'Data/{args.traindata.split("/")[-1]}_train.tfrecord')
        if not os.path.isfile(tfrecord_train):
            imgarray_HR = dataprep(args.datapath, args.scaling_factor)
        else:
            imgarray_HR = np.load(os.path.join(os.getcwd(), f'Data/{args.datapath.split("/")[-1]}_imgs_HR.npy'))

        start_timer()
        model_dir = phiregans.train(r=[args.scalingfactor],
                                    data_path=tfrecord_train,
                                    model_path=model_dir,
                                    batch_size=args.batchsize)
        times['traintime'] = end_timer()

        data_out, data_out_path = phiregans.test(r=[args.scalingfactor],
                                                 data_path=tfrecord_train,
                                                 model_path=model_dir,
                                                 batch_size=args.batchsize)

        mse = (1/len(imgarray_HR)) * np.sum((imgarray_HR-data_out)**2)
        print(f'\n\nThe mean squared error of the model is {np.round(mse, 2)}\n\n')

        infofile = open(os.path.join(data_out_path, f'model_information.txt'), 'w')
        infofile.writelines([f'{phiregans.model_name} MODEL INFORMATION\n',
                             f'Training data: {args.datapath}\n',
                             f'Scaling factor: {args.scaling_factor}\n',
                             f'Batch size: {args.batchsize}\n',
                             f'Epochs:{args.epochs_pretrain} pretraining, {args.epochs_train} training',
                             f'Times: {times[0]} pretraining, {times[1]} training, {times[2]} testing'])
        infofile.close()




    # INFERENCCE
    if 'inference' in args.mode:
        # assert data is provided and load or generate if necessary
        tfrecord_inference = os.path.join(os.getcwd(), f'Data/{args.datapath.split("/")[-1]}_test.tfrecord')
        if not os.path.isfile(tfrecord_inference):
            imgarray_HR = dataprep(args.datapath, tfrecord_inference, args.scaling_factor,
                                   'inference', return_array=True)
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

        start_timer()
        data_out, data_out_path = phiregans.test(r=[args.scaling_factor],
                                                 data_path=tfrecord_inference,
                                                 model_path=model_path,
                                                 batch_size=args.batchsize)
        times['inferencetime'] = end_timer()

        infofile = open(os.path.join(data_out_path, f'model_information.txt'), 'w')
        infofile.writelines([f'{phiregans.model_name} MODEL INFORMATION\n',
                             f'Datapath: {args.datapath}\n',
                             f'Scaling factor: {args.scaling_factor}\n',
                             f'Batch size: {args.batchsize}\n',
                             f'Inference Time: {time}'])
        infofile.close()

        mse = (1/len(imgarray_HR)) * np.sum((imgarray_HR-data_out)**2)
        print(f'\n\nThe mean squared error of the model is {np.round(mse, 2)}\n\n')



