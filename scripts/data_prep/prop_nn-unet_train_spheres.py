"""
This script prepares a dataset generated by PolNet to serve as input data to training nn-UNet for segmentation
considering that all input features are spheres
    Input:
        - A directory with the output of a PolNet dataset (see script data_gen/all_features.py)
        - Labels dictionary to match each structural with its label, labels not included in this dictionary are
          considered background
        - The radius of the sphere modeling the features
    Ouput:
        - A new directory with the structure required by nn-UNet for training a model for semantic segmentation
"""
__author__ = 'Antonio Martinez-Sanchez'

import os
import shutil
import nrrd
import json
import time

import numpy as np
import scipy as sp
import pandas

from polnet import lio

ROOT_DIR = '/home/martinez/workspace/data'
in_csv = ROOT_DIR + '/../pycharm_proj/polnet/data/data_generated/all_v11/tomos_motif_list.csv'
out_dir = ROOT_DIR + '/nn-unet/raw'
dataset_id = '011'
dataset_suffix = 'sinth11_all_v11'
fg_labels = {'membrane': (1, 2, 3), 'microtuble': (4,), 'actin': (5,), 'ribo': (6, 11, 12),
             'cprots': tuple(np.arange(7, 11).tolist() + np.arange(13, 26).tolist()),  'mb_prot': tuple(range(26, 35))}
fg_radii = {'membrane': 25, 'microtuble': 60, 'actin': 40, 'ribo': 60,
             'cprots': 25,  'mb_prot': 50} # in A
v_size_decimal = 3 # number of decimal to cut the voxel size precision
VOI_OFFS =  ((4,496), (4,496), (50,250)) # None (default)

# Parsing tomograms filenames from the CSV file
df = pandas.read_csv(in_csv, delimiter='\t')
tomos = set(df['Tomo3D'].tolist())
segs = dict()
for tomo in tomos:
    tomo_path, tomo_fname = os.path.split(tomo)
    segs[tomo] = tomo_path + '/tomo_lbls_' + tomo_fname.split('_')[2] + '.mrc'
assert len(tomos) == len(segs.keys())

# Create the dataset in nn-UNet format
out_dataset = out_dir + '/Dataset' + dataset_id + '_' + dataset_suffix
if os.path.exists(out_dataset):
    shutil.rmtree(out_dataset)
os.mkdir(out_dataset)
imgs_tr_dir, lbls_ts_dir = out_dataset + '/imagesTr', out_dataset + '/labelsTr'
os.mkdir(imgs_tr_dir)
os.mkdir(lbls_ts_dir)
out_labels = {'background': 0}
for tomo_id, tomo_in in enumerate(tomos):
    print('Processing tomogram:', tomo_in)
    tomo_df = df[df['Tomo3D'] == tomo_in]
    tomo = lio.load_mrc(tomo_in)
    seg_post = np.zeros(shape=tomo.shape, dtype=np.uint8)
    v_sizes = lio.read_mrc_v_size(tomo_in)
    v_sizes = (round(float(v_sizes[0]), v_size_decimal), round(float(v_sizes[1]), v_size_decimal),
               round(float(v_sizes[2]), v_size_decimal))
    if (v_sizes[0] != v_sizes[1]) or (v_sizes[0] != v_sizes[1]):
        print('\t-WARNING: this tomograms cannot pre processes due to its anisotropic voxel size ', v_sizes)
        continue
    v_size_i = 1. / v_sizes[0]
    for i, key in enumerate(fg_labels.keys()):
        print('\tProcessing label:', key)
        hold_lbl = i + 1
        feat_df = tomo_df[tomo_df['Label'].isin(fg_labels[key])]
        tomo_centers = np.ones(shape=tomo.shape, dtype=bool)
        x_coords, y_coords, z_coords = feat_df['X'].to_numpy(), feat_df['Y'].to_numpy(), feat_df['Z'].to_numpy()
        # coords = np.stack((x_coords, y_coords, z_coords), axis=1)
        # coords = np.round(coords * v_size_i).astype(int)
        coords = (np.round(x_coords * v_size_i).astype(int), np.round(y_coords * v_size_i).astype(int),
                  np.round(z_coords * v_size_i).astype(int))
        np.put(tomo_centers, np.ravel_multi_index(coords, dims=tomo_centers.shape), False, mode='clip')
        # for j in range(len(feat_df.index)):
        #     x, y, z = feat_df.iloc[j]['X'] * v_size_i, feat_df.iloc[j]['Y'] * v_size_i, feat_df.iloc[j]['Z'] * v_size_i
        #     x, y, z = int(round(x)), int(round(y)), int(round(z))
        #     if (x >= 0) and (y >= 0) and (z >= 0) and \
        #             (x < tomo_centers.shape[0]) and (y < tomo_centers.shape[1]) and (z < tomo_centers.shape[2]):
        #         tomo_centers[x, y, z] = 0
        tomo_dsts = sp.ndimage.distance_transform_edt(tomo_centers).astype(np.float32)
        seg_post[tomo_dsts <= (fg_radii[key] * v_size_i)] = hold_lbl
        # lio.write_mrc(tomo_dsts, imgs_tr_dir + '/hold.mrc')
        # lio.write_mrc(seg_post, imgs_tr_dir + '/hold_2.mrc')
        out_labels[key] = hold_lbl
        out_labels[key] = hold_lbl
    nrrd.write(imgs_tr_dir + '/tomo_' + str(tomo_id).zfill(3) + '_0000.nrrd', tomo)
    if VOI_OFFS is not None:
        off_mask = np.zeros(shape=seg_post.shape, dtype=seg_post.dtype)
        off_mask[VOI_OFFS[0][0]:VOI_OFFS[0][1], VOI_OFFS[1][0]:VOI_OFFS[1][1], VOI_OFFS[2][0]:VOI_OFFS[2][1]] = 1
        seg_post *= off_mask
    nrrd.write(lbls_ts_dir + '/tomo_' + str(tomo_id).zfill(3) + '.nrrd', seg_post)

# Json configuration file
dict_json = {
    'channel_names': {'0': 'rescale_to_0_1'},
    'labels': out_labels,
    'numTraining': len(tomos),
    'file_ending': '.nrrd'
}
with open(out_dataset + '/dataset.json', 'w') as outfile:
    outfile.write(json.dumps(dict_json, indent=4))

print('Successfully terminated. (' + time.strftime("%c") + ')')
