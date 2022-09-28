import pickle
from warnings import warn
from PhIREGANs import *
from palm_tempmaps import *
from netCDF4 import Dataset
import argparse
from utils import generate_TFRecords, calculate_mu_sig

"""
This script  is a wrapper which should be used to either train a GANs from a specific PALM output file or to test an
existing GAN on a PALM output file. The following scripts are used in this wrapper:
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
    parser.add_argument('--datapath', type=str, help='relative path to the netCDF file without extension', default=None)

    # Testing parameters
    parser.add_argument('--modelpath', type=str, default=None, help='the relative path to the trained model, '
                                                                    'empty if no pretrained model (training)')

    # Optional parameters
    # parser.add_argument('--datatype', type=str, default='temperature', help='the name of the data type being trained, '
    #                                                                         'default is temperature')
    parser.add_argument('--scaling_factor', type=str, help='the superresolution factor, default is 5', default=5)
    parser.add_argument('--batchsize', type=int, help='Batch size', default=1)
    parser.add_argument('--epochs_pretrain', type=int, help='Number of epochs for pretraining, '
                                                            'default is 10', default=None)
    parser.add_argument('--epochs_train', type=int, help='Number of epochs for training, default is 10', default=None)


    args = parser.parse_args()

    # region LOAD PALM DATA
    # Check datapath and break if not valid
    if not args.datapath:
        warn('No relative path to the .nc training file was given.', Warning)
        raise FileNotFoundError

    filename = args.datapath.split('/')[-1]
    datapath = os.path.join(os.getcwd(), f'{args.datapath}.nc')

    if not os.path.isfile(datapath):
        warn(f'The .nc file at the path {datapath} does not exist.', Warning)
        raise FileNotFoundError

    # try to load .nc file and give warning if it cannot be loaded.
    try:
        # load maps, extract tempmaps, create np.array and replace fill values -9999 with 0
        maps = Dataset(datapath, 'r', format="NETCDF4")
        origindate = maps.origin_time.split(" ")[0]
        tempmaps = np.array(maps["theta_xy"][:, :, :, :]) - 273.15
        tempmaps[tempmaps == -9999] = 0
    except Exception as e:
        warn(f'The NetCDF file at path {datapath} could not be loaded, or the temperature '
             f'variable has been renamed', Warning)
        raise ValueError

    # create tempmaps and adjust map to scaling factor size
    if not os.path.isfile(os.path.join(os.getcwd(), f'Data/{filename}.npy')):
        tempmaps = tempmaps[:, :,
                            0:tempmaps.shape[2] - tempmaps.shape[2] % args.scaling_factor,
                            0:tempmaps.shape[3] - tempmaps.shape[3] % args.scaling_factor]
        tempmaps = tempmaps.reshape((-1, tempmaps.shape[2], tempmaps.shape[3], 1))
        tempmaps = tempmaps.astype('float64')
        np.save(os.path.join(os.getcwd(), f'Data/{filename}.npy'), tempmaps)
    else:
        tempmaps = np.load(os.path.join(os.getcwd(), f'Data/{filename}.npy'))


    # endregion


    if args.mode == 'train':
        tfrecord_path = os.path.join(os.getcwd(), f'Data/{filename}_train.tfrecord')
        if not os.path.isfile(tfrecord_path):
            print(f'\nGenerating .tfrecord from {filename}.nc')
            generate_TFRecords(tfrecord_path, tempmaps, 'train', args.scaling_factor)

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

        mse = (1/len(tempmaps)) * np.sum((tempmaps-data_out)**2)
        print(f'\n\nThe mean squared error of the model is {np.round(mse, 2)}\n\n')




    if args.mode == 'test':
        # save the .tfrecord file
        tfrecord_path = os.path.join(os.getcwd(), f'Data/{filename}_test.tfrecord')
        if not os.path.isfile(tfrecord_path):
            print(f'\nGenerating .tfrecord from {filename}.nc\n')
            generate_TFRecords(tfrecord_path, tempmaps, 'test', args.scaling_factor)

        if not args.modelpath:
            warn('No model path was given for mode test - no model could be loaded.', Warning)
            raise ValueError
        else:
            model_path = os.path.join(os.getcwd(), args.modelpath)

        # calculate mean and sd of dataset and save as .pickle file
        mu_sig = calculate_mu_sig(tempmaps)
        pickle.dump(mu_sig, open(os.path.join(os.getcwd(), f'Data/{filename}_mu_sig.PICKLE'), 'wb'))

        phiregans = PhIREGANs(data_type=args.datatype, mu_sig=mu_sig)

        data_out, data_out_path = phiregans.test(r=[args.scaling_factor],
                                                 data_path=tfrecord_path,
                                                 model_path=model_path,
                                                 batch_size=args.batchsize)

        mse = (1/len(tempmaps)) * np.sum((tempmaps-data_out)**2)
        print(f'\n\nThe mean squared error of the model is {np.round(mse, 2)}\n\n')


