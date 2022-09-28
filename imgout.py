import os
import argparse
import numpy as np
from utils import downscale_image
from warnings import warn
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def save_outmaps(outfile, savedir, date):
    norm = Normalize(vmin=2, vmax=42)

    for imgidx in range(outfile.shape[0]):
        img = outfile[imgidx, :, :].copy()
        img[img < 0] = 0
        plt.imshow(norm(img), cmap='viridis')
        plt.axis('off')
        plt.savefig(os.path.join(savedir, f'{date}_tempmap{imgidx}_SR.png'), bbox_inches='tight', pad_inches=0)

        img = downscale_image(img, 5)
        img[img < 0] = 0
        plt.imshow(norm(img[0]), cmap='viridis')
        plt.axis('off')
        plt.savefig(os.path.join(savedir, f'{date}_tempmap{imgidx}_LR.png'), bbox_inches='tight', pad_inches=0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('outfile', type=str, help='the name of the folder containing the dataSR.npy file to be '
                                                  'processed.', default=None)
    parser.add_argument('--savefolder', type=str, help='Folder where images should be saved to. Default is the same '
                                                       'folder containing the dataSR.npy file.', default=None)

    args = parser.parse_args()

    dataout = os.path.join(os.getcwd(), "data_out")
    savedir = os.path.join(dataout, args.savefolder) if args.savefolder else os.path.join(dataout, args.outfile)
    dataout = os.path.join(dataout, f'{args.outfile}/dataSR.npy')
    if not os.path.isfile(dataout):
        warn(f'The file dataSR.npy could not be found in the folder {args.outfile}.')
        raise FileNotFoundError

    outfile = np.load(dataout)
    date = args.outfile.split('-')[1]

    save_outmaps(outfile, savedir, date)
