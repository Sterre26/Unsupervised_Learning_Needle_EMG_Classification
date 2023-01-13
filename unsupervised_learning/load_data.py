import numpy as np
from PIL import Image
from tqdm import tqdm

def load_mels(file_list):
    """ Loads mel spectrograms (.png files) from directory which are of size 128x128 """

    x_train = np.empty((0,128,128,1))
    
    for file in tqdm(file_list):
        im = Image.open(file)
        im = np.array(im)
        

        im = im.reshape(-1, 128, 128, 1).astype('float32')

        # im_sliced = im[:, :, 0:128] # same shape as mnist dataset 
        # im_sliced = im_sliced.reshape(-1, 128, 128, 1).astype('float32')
        im = im/255.
        x_train = np.append(x_train, im, axis=0)

    print('Mels set:', x_train.shape, '\n')

    return x_train 