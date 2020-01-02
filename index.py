# -*- coding: utf-8 -*-

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import h5py
import numpy as np
import argparse

from preprocess_images import mac_regions

from extract_cnn_keras import VGGNet
from extract_cnn_keras import ResNet
# ap = argparse.ArgumentParser()
# ap.add_argument("-database", required = True,
# 	help = "Path to holiday which contains images to be indexed")
# ap.add_argument("-index", required = True,
# 	help = "Name of index file")
# args = vars(ap.parse_args())


'''
 Returns a list of filenames for all jpg images in a directory. 
'''
def get_imlist(path):
    imlist=[os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
    # search(os.path.abspath(path), '.jpg',imlist)  # jpg格式查找 #'.'全部文件  '123'
    return sorted(imlist)

'''
 Extract features and index the images
'''
if __name__ == "__main__":


    # db = args["database"]
    # img_list = get_imlist(db)
    img_list=get_imlist("oxbuild_images")
    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")
    
    feats = [[],[],[],[]]
    names = [[],[],[],[]]

    model = VGGNet()
    # model=ResNet()
    for i, img_path in enumerate(img_list):
        regions=mac_regions(600,600,4)
        norm_feat = model.extract_feat(img_path,regions)
        img_name = os.path.split(img_path)[1]
        print(norm_feat.shape)
        feats[0].append(norm_feat[:512])

        for item in range(4):
            feats[1].append(norm_feat[512*(item+1):512*(item+2)])
        for item in range(9):
            feats[2].append(norm_feat[512*(item+5):512*(item+6)])
        for item in range(16):
            feats[3].append(norm_feat[512*(item+14):512*(item+15)])

        print("---------------")
        # print(feats[0])
        print("img_name--------")
        print(img_name)
        names[0].append(img_name)
        for item in range(1,4):
            for j in range(item+1):
                for k in range(item+1):
                    img_name = os.path.splitext(img_name)[0]
                    names[item].append(img_name+ "_" +str(j)+str(k)+".jpg")
        # print(img_name)
        print("extracting feature from image No. %d , %d images in total" %((i+1), len(img_list)))

    # print(names)
    feats = np.array(feats)



    # print(feats.shape[0])
    # print(feats.shape[1])
    # directory for storing extracted features
    # output = args["index"]
    for i in range(4):
        output="oxbuildtest_"+str(i)+".h5"
        print("--------------------------------------------------")
        print("      writing feature extraction results"+str(i)+ "...")
        print("--------------------------------------------------")


        h5f = h5py.File(output, 'w')
        # print(feats[i])
        h5f.create_dataset('dataset_1', data = feats[i])
        # print("feats:")
        # print(np.array(feats[i]).shape)
        # h5f.create_dataset('dataset_2', data = names) 会报错
        # print(names[i])
        h5f.create_dataset('dataset_2', data = np.string_(names[i]))
        h5f.close()
