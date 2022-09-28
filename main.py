""" @author: Karen Stengel """
import argparse
import pickle
from warnings import warn
from PhIREGANs import *
from utils import calculate_mu_sig

# WIND - LR-MR
# -------------------------------------------------------------

# data_type = 'wind'
# data_path = 'example_data/wind_LR-MR.tfrecord'
# model_path = 'models/wind_lr-mr/trained_gan/gan'
# r = [2, 5]
# mu_sig = [[0.7684, -0.4575], [4.9491, 5.8441]]

# WIND - MR-HR
# -------------------------------------------------------------
'''
data_type = 'wind'
data_path = 'example_data/wind_MR-HR.tfrecord'
model_path = 'models/wind_mr-hr/trained_gan/gan'
r = [5]
mu_sig=[[0.7684, -0.4575], [5.02455, 5.9017]]
'''

# SOLAR - LR-MR
# -------------------------------------------------------------
'''
data_type = 'solar'
data_path = 'example_data/solar_LR-MR.tfrecord'
model_path = 'models/solar_lr-mr/trained_gan/gan'
r = [5]
mu_sig=[[344.3262, 113.7444], [370.8409, 111.1224]]
'''

# SOLAR - MR-HR
# -------------------------------------------------------------
'''
data_type = 'solar'
data_path = 'example_data/solar_MR-HR.tfrecord'
model_path = 'models/solar_mr-hr/trained_gan/gan'
r = [5]
mu_sig = [[344.3262, 113.7444], [386.9283, 117.9627]]
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('mode', type=str, help='Usage mode, either train/test', default=None)
    parser.add_argument('--datatype', type=str, help='the name of the data type being trained, defualt is temperature',
                        default='temperature')
    parser.add_argument('--modelpath', type=str, help='the path to the trained model, empty if no pretrained model',
                        default=None)
    parser.add_argument('--scaling_factor', type=str, help='the superresolution factor, default is 5', default=[5])
    parser.add_argument('--datapath', type=str, help='relative path to the .tfrecord file without extension',
                        default=None)
    args = parser.parse_args()

    mode = args.mode
    data_type = args.datatype
    r = args.scaling_factor
    model_path = os.path.join(os.getcwd(), args.modelpath) if args.modelpath else None
    if args.datapath:
        data_path = os.path.join(os.getcwd(), f'{args.datapath}.tfrecord')
        pickledat = os.path.join(os.getcwd(), f'{args.datapath}_mu_sig.PICKLE')
    else:
        warn('No data path was given.', Warning)
        raise FileNotFoundError

    # if you want to pass values for mu_sig --> change here!
    mu_sig = None


    if mode == 'train':
        phiregans = PhIREGANs(data_type=data_type, mu_sig=mu_sig)

        model_dir = phiregans.pretrain(r=r,
                                       data_path=data_path,
                                       model_path=model_path,
                                       batch_size=1)

        model_dir = phiregans.train(r=r,
                                    data_path=data_path,
                                    model_path=model_dir,
                                    batch_size=1)

        _ = phiregans.test(r=r,
                       data_path=data_path,
                       model_path=model_dir,
                       batch_size=1)

    elif mode == 'test':
        if not os.path.isfile(pickledat):
            warn('Mean and standard have not been previously saved and will be generated now')

        else:
            mu_sig = pickle.load(open(os.path.join(data_path, pickledat), 'rb'))

        phiregans = PhIREGANs(data_type=data_type, mu_sig=mu_sig)

        _ = phiregans.test(r=r,
                       data_path=data_path,
                       model_path=model_path,
                       batch_size=1)
