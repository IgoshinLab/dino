import skimage.external.tifffile as tiffreader
import os
import numpy as np


def resize_img(img_dir, out_dir, size, idx):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    img = tiffreader.imread(img_dir)
    new_img = img[idx, ::]
    img_size = np.shape(new_img)
    cp = [img_size[0] // 2, img_size[1] // 2]
    new_img = new_img[cp[0] - size // 2: cp[0] + size // 2, cp[1] - size // 2: cp[1] + size // 2]
    name = img_dir.split('/')[-1].replace(' ', '_')
    tiffreader.imsave(os.path.join(out_dir, name), new_img)
    a = 1
    return 0

resize_img("/mnt/data/ShimketsLab/movie/AG1102 071517/AG1102 071517.tif", "/mnt/data/ShimketsLab/img", 1000, -1)