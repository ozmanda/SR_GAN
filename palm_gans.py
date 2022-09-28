import pickle
from warnings import warn
from PhIREGANs import *
from palm_tempmaps import *
from netCDF4 import Dataset
import argparse
from utils import generate_TFRecords, calculate_mu_sig
from imgout import save_outmaps

"""
This script  is a wrapper which should be used to either train a GANs from a specific PALM output file or to test an
existing GAN on a PALM output file. The following scripts are used in this wrapper:
  - palm_tempmaps.py
  - datagenerator.py
  - PhIREGANs.py
  - imgout


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
    parser.add_argument('--savedir', type=str, help='Relative path to folder where images should be saved',
                        default='Images')
    parser.add_argument('--foldername', type=str, help='Foldername for new images', default=None)
    parser.add_argument('--datatype', type=str, default='temperature', help='the name of the data type being trained, '
                                                                            'default is temperature')
    parser.add_argument('--animation', type=bool, help='True if an animation should be created, False otherwise. '
                                                       'Default is True', default=False)
    parser.add_argument('--surfacemaps', type=bool, default=False, help='Determines if surfacemaps should be generated.'
                                                                        ' Default is False')
    parser.add_argument('--scaling_factor', type=str, help='the superresolution factor, default is 5', default=5)
    parser.add_argument('--batchsize', type=int, help='Batch size', default=1)


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

    # define path for saved images and create the directory if it does not already exist
    # if args.imgpath:
    #     imgpath = os.path.join(os.getcwd(), f'{args.savedir}/{filename}')
    # else:
    #     imgpath = os.path.join(os.getcwd(), f'Images/{filename}')
    # if not os.path.isdir(imgpath):
    #     os.mkdir(imgpath)

    # try to load .nc file and give warning if it cannot be loaded
    try:
        maps = Dataset(datapath, 'r', format="NETCDF4")
        origindate = maps.origin_time.split(" ")[0]
        maps = maps["theta_xy"][:, :, :, :]
    except Exception as e:
        warn(f'The NetCDF file at path {datapath} could not be loaded, or the temperature '
             f'variable has been renamed', Warning)
        raise ValueError

    # create tempmaps and adjust map to scaling factor size
    tempmaps = generate_maps(datapath, args.animation, args.surfacemaps)
    newshape = (tempmaps.shape[0], tempmaps.shape[1] - tempmaps.shape[1] % args.scaling_factor,
                tempmaps.shape[2] - tempmaps.shape[2] % args.scaling_factor, 1)
    tempmaps = tempmaps[:, 0:newshape[1], 0:newshape[2]]
    if len(tempmaps.shape) == 3:
        tempmaps = np.reshape(tempmaps, newshape=newshape)

    # endregion


    if args.mode == 'train':
        tfrecord_path = os.path.join(os.getcwd(), f'{args.savedir}/{filename}_train.tfrecord')
        if not os.path.isfile(tfrecord_path):
            generate_TFRecords(tfrecord_path, tempmaps, 'train', args.scaling_factor)

        model_path = os.path.join(os.getcwd(), args.modelpath) if args.modelpath else None

        phiregans = PhIREGANs(data_type=args.datatype, mu_sig=None)

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

        data_out_path = os.path.join(os.getcwd(), f'data_out/{data_out_path.split("/")[-1]}')
        date = data_out_path.split("-")[1]
        save_outmaps(data_out, data_out_path, date)




    if args.mode == 'test':
        # save the .tfrecord file
        tfrecord_path = os.path.join(os.getcwd(), f'{args.savedir}/{filename}_test.tfrecord')
        if not os.path.isfile(tfrecord_path):
            generate_TFRecords(tfrecord_path, tempmaps, 'test', args.scaling_factor)

        if not args.modelpath:
            warn('No model path was given for mode test - no model could be loaded.', Warning)
            raise ValueError
        else:
            model_path = os.path.join(os.getcwd(), args.modelpath)

        # calculate mean and sd of dataset and save as .pickle file
        mu_sig = calculate_mu_sig(tempmaps)
        pickle.dump(mu_sig, open(os.path.join(os.getcwd(), f'{args.savedir}/{filename}_mu_sig.PICKLE'), 'wb'))

        phiregans = PhIREGANs(data_type=args.datatype, mu_sig=mu_sig)

        data_out, data_out_path = phiregans.test(r=[args.scaling_factor],
                                                 data_path=tfrecord_path,
                                                 model_path=model_path,
                                                 batch_size=args.batchsize)

        data_out_path = os.path.join(os.getcwd(), f'data_out/{data_out_path.split("/")[-1]}')
        date = data_out_path.split("-")[1]
        save_outmaps(data_out, data_out_path, date)

