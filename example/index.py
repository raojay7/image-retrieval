# -*- coding: utf-8 -*-

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import h5py
import numpy as np
from extract_cnn_keras import VGGNet
from extract_cnn_keras import ResNet
from utils.preprocess_images import mac_regions
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

    img_list=get_imlist("E:/PycharmProjects/image-retrieval/data/test/rparis6k/jpg")
    # img_list=get_imlist("../test/")

    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")

    feats = []
    names = []

    model = VGGNet()
    # model=ResNet()
    #if L=1

    # for i, img_path in enumerate(img_list):
    #     norm_feat = model.extract_feat(img_path)
    #     img_name = os.path.split(img_path)[1]
    #     feats.append(norm_feat)
    #     print("img_name--------")
    #     print(img_name)
    #     names.append(img_name)
    #     # print(img_name)
    #     print("extracting feature from image No. %d , %d images in total" % ((i + 1), len(img_list)))

    #if L=2
    L=2
    # regions=mac_regions(1024,1024,L,32)
    regions=mac_regions(600,600,L,18)
    for i, img_path in enumerate(img_list):
        norm_feats = model.extract_feat(img_path,regions)
        img_name = os.path.split(img_path)[1]
        count=0
        for j in range(L):
            for k in range(L):
                feats.append(np.squeeze(norm_feats[count]))
                count=count+1
                print("img_name--------")
                img_name = os.path.splitext(img_name)[0]
                subimage_name=img_name+'_'+str(j)+str(k)+'.jpg'
                print(subimage_name)
                names.append(subimage_name)
        # print(img_name)
        print("extracting feature from image No. %d , %d images in total" % ((i + 1), len(img_list)))

    # print(names)


    feats = np.array(feats)



    # directory for storing extracted features
    # output = args["index"]
    output="paris1.h5"
    print("--------------------------------------------------")
    print("      writing feature extraction results"+ "...")
    print("--------------------------------------------------")


    h5f = h5py.File(output, 'w')
    # print(feats[i])
    h5f.create_dataset('dataset_1', data = feats)
    # print("feats:")
    # print(np.array(feats[i]).shape)
    # h5f.create_dataset('dataset_2', data = names) 会报错

    h5f.create_dataset('dataset_2', data = np.string_(names))
    h5f.close()
