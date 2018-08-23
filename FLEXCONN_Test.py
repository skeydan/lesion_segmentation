from __future__ import division, print_function

import argparse
import os
import shutil
import tempfile
import time

import h5py
import nibabel as nib
import numpy as np
import statsmodels.api as sm
from keras.models import load_model
from keras import backend
from scipy import ndimage
from scipy.signal import argrelextrema


backend.set_image_data_format = 'channels_last'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def apply_model(image1, image2, pred_model):

    dim = image1.shape
    dim_2d = (1, dim[0], dim[1], 1)

    slice_2d_t1 = np.zeros(dim_2d, dtype=np.float16)
    slice_2d_fl = np.zeros(dim_2d, dtype=np.float16)
    output_image = np.zeros(dim, dtype=np.float16)

    for k in range(dim[2]):
        slice_2d_t1[0, :, :, 0] = image1[:, :, k]
        slice_2d_fl[0, :, :, 0] = image2[:, :, k]
        pred = pred_model.predict([slice_2d_t1, slice_2d_fl])
        output_image[:, :, k] = pred[0, :, :, 0]

    return output_image


def normalize_image(vol, contrast):
    temp = vol[np.nonzero(vol)].astype(float)
    q = np.percentile(temp, 99)
    temp = temp[temp <= q]
    temp = temp.reshape(-1, 1)
    bw = q / 80
    print("99th quantile is %.4f, gridsize = %.4f" % (q, bw))

    kde = sm.nonparametric.KDEUnivariate(temp)

    kde.fit(kernel='gau', bw=bw, gridsize=80, fft=True)
    x_mat = 100.0*kde.density
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


def suffix(num):
    suffix_dict = {1: 'st', 2: 'nd', 3: 'rd'}
    return suffix_dict.get(num, 'th')


def split_filename(input_path):
    dirname = os.path.dirname(input_path)
    basename = os.path.basename(input_path)
    
    base_arr = basename.split('.')
    ext = base_arr[-1]
    if ext == 'gz':
        ext = '.'.join(base_arr[-2:])
    if ext != '':
        ext = '.' + ext
    return dirname, basename[:-len(ext)], ext


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction with Fast Lesion Extraction using Convolutional '
                                                 'Neural Networks (FLEXCONN)')
    
    parser.add_argument('--models', type=str, required=True, nargs='+',
                        help='Learnt models (.h5) files. Multiple models are accepted, e.g. training separately '
                             'with two sets of lesion masks (from ISBI2015 challenge) as provided with this code.')
    parser.add_argument('--t1', type=str, required=True,
                        help='Subject T1 Image (skullstripped, bias-corrected). Since the training is 2D, make sure '
                             'the test image is properly oriented, i.e. the in-plane has the highest native '
                             'resolution. E.g. the training images are axial because their native resolution is 1x1x4 '
                             'mm^3 in axial RAI orientation.')
    parser.add_argument('--flair', type=str, required=True,
                        help='Subject FLAIR Image (skullstripped, bias-corrected), '
                             'must be registered to T1 and have same orientation as T1.')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Output directory where the resultant membership and mask are written')
    parser.add_argument('--gpu', type=str, help='Choice for GPU. Either an integer for the GPU. Use "cpu" to use CPU.')
    
    results = parser.parse_args()

    if results.gpu == 'cpu':
        # To run prediction only on CPU, uncomment the following two lines
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("Using CPU")
    elif results.gpu is not None:
        # Change gpu id to run on different gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = results.gpu
        print("Using GPU id " + str(os.environ["CUDA_VISIBLE_DEVICES"]))
    
    im1 = os.path.abspath(os.path.expanduser(results.t1))
    im2 = os.path.abspath(os.path.expanduser(results.flair))
    
    results.outdir = os.path.abspath(os.path.expanduser(results.outdir))
    _, base, _ = split_filename(im1)
    outname1 = os.path.join(results.outdir, base + "_LesionMembership.nii.gz")
    outname2 = os.path.join(results.outdir, base + "_LesionMask.nii.gz")
    
    model_files = results.models
    models = []
    for i in range(len(model_files)):
        models.append(''.join(model_files[i]))
    
    print("%d models found at" % (len(models)))
    for i in range(len(models)):
        models[i] = os.path.abspath(os.path.expanduser(models[i]))
        print(models[i])
    
    tmpdir = tempfile.mkdtemp()
    print("Temporary directory :" + tmpdir)
    
    print("T1 image         = " + im1)
    print("FLAIR image      = " + im2)
    print("Output directory = " + results.outdir)
    
    img_obj1 = nib.load(im1)
    img_obj2 = nib.load(im2)
    
    im_size = img_obj1.shape
    
    vol1 = img_obj1.get_data()
    vol2 = img_obj2.get_data()
    
    vol1 /= normalize_image(vol1, 't1')
    vol2 /= normalize_image(vol2, 'fl')
    
    outvol = np.zeros(im_size + (len(models),))
    
    newmodels = []
    
    for i in range(len(models)):
        src = models[i]
        name = str(i+1) + '.h5'
        dst = os.path.join(tmpdir, name)
        shutil.copy(src, dst)
        newmodels.append(dst)
    
    print('Predicting memberships.')
    for t in range(len(models)):
        start = time.time()
        model = load_model(newmodels[t])
    
        # copying the models to the temporary dir and delete "optimizer_weights" flags is
        # necessary when working with tensorflow version <=1.2.
        with h5py.File(newmodels[t], 'a') as f:
            if 'optimizer_weights' in f.keys():
                del f['optimizer_weights']
    
        mem = apply_model(vol1, vol2, model)
        outvol[:, :, :, t] = mem
    
        elapsed = time.time() - start
    
        print("Time taken for %d%s model = %.2f seconds" % (t + 1, suffix(t + 1), elapsed))
    
    outvol = np.mean(outvol, axis=3)/100.0
    
    # save the whole membership
    print("Writing " + outname1)
    nib.Nifti1Image(outvol, img_obj1.affine, img_obj1.header).to_filename(outname1)
    
    print("Thresholding memberships at 0.34 and removing 18-connected objects with volume <27 voxels.")
    thr = 0.34
    seg = np.zeros_like(outvol)
    seg[outvol > thr] = 1
    se = ndimage.morphology.generate_binary_structure(3, 1)
    label, ncomp = ndimage.label(seg, structure=se)
    unique, counts = np.unique(label, return_counts=True)
    for j, unq in enumerate(unique):
        if counts[j] < 27:
            label[label == unq] = 0
    label[label > 0] = 1
    print("Writing " + outname2)
    nib.Nifti1Image(label, img_obj1.affine, img_obj1.header).to_filename(outname2)
    shutil.rmtree(tmpdir)
