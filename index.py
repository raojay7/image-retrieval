# -*- coding: utf-8 -*-
# Author: yongyuan.name
import os
import h5py
import numpy as np
import argparse
from sklearn.decomposition import PCA

from extract_cnn_vgg16_keras import VGGNet


ap = argparse.ArgumentParser()
ap.add_argument("-database", required = True,
	help = "Path to database which contains images to be indexed")
ap.add_argument("-index", required = True,
	help = "Name of index file")
args = vars(ap.parse_args())


'''
 Returns a list of filenames for all jpg images in a directory. 
'''
def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]


'''
 Extract features and index the images
'''
if __name__ == "__main__":


    db = args["database"]
    img_list = get_imlist(db)
    
    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")
    
    feats = []
    names = []

    model = VGGNet()
    for i, img_path in enumerate(img_list):
        norm_feat = model.extract_feat(img_path)
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name)
        print(img_name)
        print("extracting feature from image No. %d , %d images in total" %((i+1), len(img_list)))

    # print(names)
    feats = np.array(feats)

    #pca
    # pca = PCA(n_components=2048,svd_solver='auto')
    # feats=pca.fit_transform(feats)

    print(feats.shape[0])
    print(feats.shape[1])
    # directory for storing extracted features
    output = args["index"]
    
    print("--------------------------------------------------")
    print("      writing feature extraction results ...")
    print("--------------------------------------------------")


    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data = feats)
    # h5f.create_dataset('dataset_2', data = names) 会报错
    h5f.create_dataset('dataset_2', data = np.string_(names))
    h5f.close()
