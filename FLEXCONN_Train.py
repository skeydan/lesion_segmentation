from __future__ import print_function, division

import argparse
import os
import random
import time
import sys
import nibabel as nib
import numpy as np
import statsmodels.api as sm
from keras import backend
from keras.engine import Input, Model
from keras.layers import Conv2D, concatenate
from keras.optimizers import Adam
from scipy import ndimage
from scipy.signal import argrelextrema


backend.set_image_data_format = 'channels_last'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def pad_image(vol, padsize):
    dim = vol.shape
    padsize = np.asarray(padsize, dtype=int)
    dim2 = dim + 2 * padsize
    temp = np.zeros(dim2, dtype=np.float16)
    temp[padsize:dim[0] + padsize, padsize:dim[1] + padsize, padsize:dim[2] + padsize] = vol
    return temp


def normalize_image(vol, contrast):

    temp = vol[np.nonzero(vol)].astype(float)

    q = np.percentile(temp, 99)
    temp = temp[temp <= q]
    temp = temp.reshape(-1, 1)
    bw = q / 80
    print("99th quantile is %.4f, gridsize = %.4f" % (q, bw))
    
    kde = sm.nonparametric.KDEUnivariate(temp)
    
    kde.fit(kernel='gau', bw=bw, gridsize=80, fft=True)
    x_mat = 100.0 * kde.density
    y_mat = kde.support
    
    indx = argrelextrema(x_mat, np.greater)
    indx = np.asarray(indx, dtype=int)
    heights = x_mat[indx][0]
    peaks = y_mat[indx][0]
    peak = 0.00
    print("%d peaks found." % (len(peaks)))
    
    # norm_vol = vol
    if contrast.lower() == "t1":
        peak = peaks[-1]
        print("Peak found at %.4f for %s" % (peak, contrast))
        # norm_vol = vol/peak
        # norm_vol[norm_vol > 1.25] = 1.25
        # norm_vol = norm_vol/1.25
    elif contrast.lower() in ['t2', 'pd', 'fl']:
        peak_height = np.amax(heights)
        idx = np.where(heights == peak_height)
        peak = peaks[idx]
        print("Peak found at %.4f for %s" % (peak, contrast))
        # norm_vol = vol / peak
        # norm_vol[norm_vol > 3.5] = 3.5
        # norm_vol = norm_vol / 3.5
    else:
        print("Contrast must be either T1,T2,PD, or FL. You entered %s. Returning 0." % contrast)
    
    # return peak, norm_vol
    return peak


def get_patches(invol1, invol2, mask, patchsize):
    rng = random.SystemRandom()
    dsize = np.floor(patchsize / 2).astype(np.int)
    
    indx = np.array(np.nonzero(mask))
    num_patches = len(indx[0])
    
    print("Number of patches used  = %d " % num_patches)
    randindx = rng.sample(range(num_patches), num_patches)
    newindx = np.ndarray((3, num_patches), dtype=np.int)
    for i in range(num_patches):
        for j in range(3):
            newindx[j, i] = indx[j, randindx[i]]
    
    matsize = (num_patches, patchsize[0], patchsize[1], 1)
    
    blurmask = ndimage.filters.gaussian_filter(mask.astype(np.float32), sigma=(1, 1, 1))
    blurmask[blurmask < 0.0001] = 0
    blurmask = blurmask * 100  # Just to have reasonable looking error values during training, otherwise
    # the error values become too small
    
    t1_patches = np.zeros(matsize, dtype=np.float32)
    fl_patches = np.zeros(matsize, dtype=np.float32)
    mask_patches = np.zeros(matsize, dtype=np.float32)
    
    for i in range(0, num_patches):
        x = newindx[0, i]
        y = newindx[1, i]
        z = newindx[2, i]
        
        t1_patches[i, :, :, 0] = invol1[x - dsize[0]:x + dsize[0] + 1, y - dsize[1]:y + dsize[1] + 1, z]
        fl_patches[i, :, :, 0] = invol2[x - dsize[0]:x + dsize[0] + 1, y - dsize[1]:y + dsize[1] + 1, z]
        mask_patches[i, :, :, 0] = blurmask[x - dsize[0]:x + dsize[0] + 1, y - dsize[1]:y + dsize[1] + 1, z]
    
    return t1_patches, fl_patches, mask_patches


def get_model():
    ds = 2
    t1 = Input((None, None, 1))
    conv1a = Conv2D(128, (3, 3), activation='relu', padding='same')(t1)
    conv2a = Conv2D(128 // ds, (5, 5), activation='relu', padding='same')(conv1a)
    conv3a = Conv2D(128 // (ds * 2), (3, 3), activation='relu', padding='same')(conv2a)
    conv4a = Conv2D(128 // (ds ** 3), (5, 5), activation='relu', padding='same')(conv3a)
    conv5a = Conv2D(128 // (ds ** 4), (3, 3), activation='relu', padding='same')(conv4a)
    
    fl = Input((None, None, 1))
    conv1b = Conv2D(128, (3, 3), activation='relu', padding='same')(fl)
    conv2b = Conv2D(128 // ds, (5, 5), activation='relu', padding='same')(conv1b)
    conv3b = Conv2D(128 // (ds * 2), (3, 3), activation='relu', padding='same')(conv2b)
    conv4b = Conv2D(128 // (ds ** 3), (5, 5), activation='relu', padding='same')(conv3b)
    conv5b = Conv2D(128 // (ds ** 4), (3, 3), activation='relu', padding='same')(conv4b)
    
    concat = concatenate([conv5a, conv5b], axis=-1)
    
    conv1c = Conv2D(128, (3, 3), activation='relu', padding='same')(concat)
    conv2c = Conv2D(128 // ds, (5, 5), activation='relu', padding='same')(conv1c)
    conv3c = Conv2D(128 // (ds * 2), (3, 3), activation='relu', padding='same')(conv2c)
    conv4c = Conv2D(128 // (ds ** 3), (5, 5), activation='relu', padding='same')(conv3c)
    conv5c = Conv2D(128 // (ds ** 4), (3, 3), activation='relu', padding='same')(conv4c)
    
    conv_last = Conv2D(1, (3, 3), activation='relu', padding='same')(conv5c)
    
    model = Model(inputs=[t1, fl], outputs=conv_last)
    model.compile(optimizer=Adam(lr=0.0001), loss='mean_squared_error')
    
    return model


def main(atlas_dir, numatlas, patchsize, out_dir):
    batchsize = 128
    
    patchsize = np.array(patchsize)
    padsize = np.max(patchsize + 1) / 2
    num_patches = 0
    for i in range(numatlas):
        num_patches += int(nib.load(os.path.join(atlas_dir, "atlas" + str(i + 1) + "_mask.nii.gz")).get_data().sum())
    
    print("Total number of lesion patches = " + str(num_patches))
    ident = time.strftime("%d-%m-%Y") + "_" + time.strftime("%H-%M-%S")
    print("Unique ID is %s " % ident)
    
    patch_str = str(int(patchsize[0])) + "x" + str(int(patchsize[1]))
    outname = os.path.join(out_dir, "FLEXCONNmodel2D_" + patch_str + "_" + ident + ".h5")
    print("Trained model will be written at %s" % outname)

    matsize = (num_patches, patchsize[0], patchsize[1], 1)
    t1_patches = np.zeros(matsize, dtype=np.float32)
    fl_patches = np.zeros(matsize, dtype=np.float32)
    mask_patches = np.zeros(matsize, dtype=np.float32)
    
    count2 = 0
    count1 = 0
    for i in range(0, numatlas):
        t1name = os.path.join(atlas_dir, "atlas" + str(i + 1) + "_" + "T1.nii.gz")
        print("Reading %s" % t1name)
        t1 = nib.load(t1name).get_data().astype(np.float32)

        
        flname = os.path.join(atlas_dir, "atlas" + str(i + 1) + "_" + "FL.nii.gz")
        print("Reading %s" % flname)
        fl = nib.load(flname).get_data().astype(np.float32)
        
        maskname = os.path.join(atlas_dir, "atlas" + str(i + 1) + "_" + "mask.nii.gz")
        print("Reading %s" % maskname)
        mask = nib.load(maskname).get_data().astype(np.float32)
        
        # Normalize the images
        t1 = np.array(t1 / normalize_image(t1, 'T1'), dtype=np.float32)
        fl = np.array(fl / normalize_image(fl, 'FL'), dtype=np.float32)
        
        dim = t1.shape
        print("Image size = %d x %d x %d " % (dim[0], dim[1], dim[2]))
        
        padded_t1 = pad_image(t1, padsize)
        padded_fl = pad_image(fl, padsize)
        padded_mask = pad_image(mask, padsize)
        
        t1_patches_a, fl_patches_a, mask_patches_a = get_patches(padded_t1, padded_fl, padded_mask, patchsize)
        
        dim = t1_patches_a.shape
        count2 = count1 + dim[0]
        print("Atlas %d : indices [%d,%d]" % (i + 1, count1, count2 - 1))
        t1_patches[count1:count2, :, :, :] = t1_patches_a
        fl_patches[count1:count2, :, :, :] = fl_patches_a
        mask_patches[count1:count2, :, :, :] = mask_patches_a
        count1 = count1 + dim[0]

    t1_patches = np.asarray(t1_patches,dtype=np.float16)
    fl_patches = np.asarray(fl_patches, dtype=np.float16)
    mask_patches = np.asarray(mask_patches, dtype=np.float16)
    
    print("Total number of patches collected = " + str(count2))
    
    print("Size of the input matrix is " + str(mask_patches.shape))
    model = get_model()
    model.fit([t1_patches, fl_patches], mask_patches, batch_size=batchsize, epochs=10, verbose=1, validation_split=0.2)
    
    print("Model is written at " + outname)
    model.save(outname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training for Fast Lesion Extraction using '
                                                 'Convolutional Neural Networks (FLEXCONN)')
    
    parser.add_argument('--atlasdir', required=True, type=str,
                        help='Atlas directory containing atlasXX_T1.nii, atlasXX_FL.nii, atlasXX_mask.nii.gz etc. '
                             'XX=1,2,3... All atlas images must be in axial RAI orientation, or whatever orientation '
                             'FLAIR has the highest in-plane resolution. For example, the associated training data  '
                             'from ISBI2015 lesion segmentation challenge has axial 2D FLAIR acquired with 1x1x4 mm^3. '
                             'Atlas T1 and FLAIR images must be coregistered and have same dimensions.')
    parser.add_argument('--natlas', required=True, type=int,
                        help='Number of atlases to be used. Atlas directory must contain these many atlases.')
    parser.add_argument('--psize', required=True,
                        help='Patch size, e.g. 35x35 or 31x31 (2D). Patch sizes are separated by x. '
                             'Note that 2D patches are employed because usually FLAIR images are acquired 2D. '
                             'Future releases will include full support for 3D patches. ')
    parser.add_argument('--outdir', required=True, help='Output directory where the trained models are written.')
    parser.add_argument('--gpu', type=str, help='Choice for GPU. Use the integer ID for the GPU. Use "cpu" to use CPU.')
    
    results = parser.parse_args()
    outdir = os.path.abspath(os.path.expanduser(results.outdir))
    
    atlasdir = os.path.abspath(os.path.expanduser(results.atlasdir))
    
    print("Keras parameters are as follows -- ")
    print("Backend           : " + backend.backend())
    print("Float             : " + backend.floatx())
    print("Image Data Format : " + backend.image_data_format())
    
    if backend.floatx() != 'float32':
        print("WARNING: Data type should be float32 to save on memory.")
    
    if results.gpu == 'cpu':
        # To run prediction only on CPU, uncomment the following two lines
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("Using CPU")
    elif results.gpu is not None:
        # Change gpu id to run on different gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = results.gpu
        print("Using GPU id " + str(os.environ["CUDA_VISIBLE_DEVICES"]))
    
    print('Atlas Directory     =', atlasdir)
    print('Number of atlases   =', results.natlas)
    print('Patch size          =', results.psize)
    print('Output Directory    =', outdir)
    
    psize = [int(item) for item in results.psize.split('x')]
    
    print('Patch size          = %d x %d ' % (psize[0], psize[1]))
    
    if not os.path.isdir(outdir):
        print("Output directory does not exist. I will create it.")
        os.makedirs(outdir)
    
    main(atlasdir, results.natlas, psize, outdir)
