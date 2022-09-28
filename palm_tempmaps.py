import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from netCDF4 import Dataset
from warnings import warn


def generate_maps(filepath, animation=False, surfacemaps=False, savepath=os.path.join(os.getcwd(), 'Images')):
    maps = Dataset(filepath, "r", format="NETCDF4")
    date = maps.origin_time.split(" ")[0]

    foldername = f'tempmaps_{date}'
    savepath = os.path.join(savepath, foldername)

    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    maps = maps["theta_xy"][:, :, :, :]

    if surfacemaps:
        savepath = os.path.join(savepath, "Layermaps")
        os.mkdir(savepath)

    if len(maps.shape) == 3:
        tempmaps = np.zeros(shape=(maps.shape[0], maps.shape[1], maps.shape[2]))
        vmin = np.nanmin(maps)
        vmax = np.nanmax(maps)

        for mapidx in range(maps.shape[0]):
            map = maps[mapidx, :, :]
            # flip maps to account for PALM having origin at the bottom left, not top left
            tempmaps[mapidx, :, :] = np.flip(map, axis=0)

            if not os.path.isfile(os.path.join(savepath, f'{date}_tempmap{mapidx}.png')):
                plt.imsave(os.path.join(savepath, f'{date}_tempmap{mapidx}.png'), np.flip(map, axis=0),
                           cmap='viridis', vmin=vmin, vmax=vmax)
        if animation:
            animate(savepath, f'{date}_tempmap')

    elif len(maps.shape) == 4:
        tempmaps = np.zeros(shape=(maps.shape[0]*maps.shape[1], maps.shape[2], maps.shape[3]))
        vmin = np.nanmin(maps)
        vmax = np.nanmax(maps)

        for layer in range(maps.shape[1]):
            picdir = os.path.join(savepath, f'Layer_{layer + 1}')

            if not os.path.isdir(picdir):
                os.mkdir(picdir)

            for mapidx in range(maps.shape[0]):
                map = maps[mapidx, layer, :, :]
                tempmaps[mapidx*layer, :, :] = map
                tempmaps[tempmaps < 0] = 0

                if not os.path.isfile(os.path.join(picdir, f'{date}_Layer-{layer}_tempmap{mapidx}.png')):
                    # flip maps to account for PALM having origin at the bottom left, not top left
                    plt.imsave(os.path.join(picdir, f'{date}_Layer-{layer}_tempmap{mapidx}.png'), np.flip(map, axis=0),
                               cmap='viridis', vmin=vmin, vmax=vmax)

            if animation:
                animate(picdir, f'{date}_Layer-{layer}_tempmap')

        if surfacemaps:
            _ = generate_surfacemaps(maps, date, savepath, animation)

    else:
        warn(f'The data loaded has an incorrect shape, it must be either 3 or 4, the data has shape {maps.shape}',
             Warning)
        raise ValueError

    return tempmaps


def generate_surfacemaps(maps, date, imgdir, animation):
    surfacedir = os.path.join(imgdir, "Surfacemaps")
    os.mkdir(surfacedir)

    surfacemap = np.empty(shape=(maps.shape[0], maps.shape[2], maps.shape[3]))

    for idx, _ in np.ndenumerate(surfacemap[1, :, :]):
        for layer in range(maps.shape[1]):
            if not np.isnan(maps[1, layer, idx[0], idx[1]]):
                surfacemap[:, idx[0], idx[1]] = maps[:, layer, idx[0], idx[1]]
                break

    # colour spread hard-coded to enable decoding after the GAN
    vmin = 42
    vmax = 2

    tempmaps = np.zeros(shape=(maps.shape[0], maps.shape[2], maps.shape[3]))

    for mapidx in range(maps.shape[0]):
        map = surfacemap[mapidx, :, :]
        tempmaps[mapidx, :, :] = map

        plt.imsave(os.path.join(surfacedir, f'{date}_surfacetemp{mapidx}.png'), map,
                   cmap='viridis', vmin=vmin, vmax=vmax)
    if animation:
        animate(surfacedir, f'{date}_surfacemap')

    return tempmaps


def animate(imgdir, filename):
    frames = [Image.open(os.path.join(imgdir, imagename)) for imagename in os.listdir(imgdir) if
              imagename.endswith(".png")]
    frame = frames[0]
    frame.save(os.path.join(imgdir, f'{filename}_animation.GIF'), format="GIF",
               append_images=frames, save_all=True, duration=1000, loop=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inferencefile', type=str, help='Relative path to file for which maps should be created')
    parser.add_argument('--savefolder', type=str, help='Folder where images should be saved',
                        default='Images')
    parser.add_argument('--foldername', type=str, help='Foldername for new images', default=None)
    parser.add_argument('--animation', type=bool, help='True if an animation should be created, False otherwise. Default'
                                                     'is True', default=False)
    parser.add_argument('--surfacemaps', type=bool, help='Determines if surfacemaps should be generated. Defualt is '
                                                         'False', default=False)

    args = parser.parse_args()

    imgpath = os.path.join(os.getcwd(), args.savefolder)
    filepath = os.path.join(os.getcwd(), args.inferencefile)
    animation = args.animation
    surfacemaps = args.surfacemaps

    _ = generate_maps(filepath, animation, surfacemaps, imgpath)
