# -*- coding: utf-8 -*-
# Author: yongyuan.name
from extract_cnn_vgg16_keras import VGGNet

import numpy as np
import h5py




def getResult(query):
    # read in indexed images' feature vectors and corresponding image names
    h5f = h5py.File("featureCNN.h5", 'r')
    # feats = h5f['dataset_1'][:]
    feats = h5f['dataset_1'][:]
    print(feats)
    imgNames = h5f['dataset_2'][:]
    print(imgNames)
    h5f.close()

    #不用输入模型重复计算
    queryVec=feats[imgNames.tolist().index(np.string_(query))]
    scores = np.dot(queryVec, feats.T)
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]
    print(rank_ID)
    print(rank_score)

    # number of top retrieved images to show
    maxres = 10
    imlist = [imgNames[index].decode() for i, index in enumerate(rank_ID[0:maxres])]
    print("top %d images in order are: " % maxres, imlist)
    return imlist


def writeResult():
    with open("resultfile.dat","w",encoding='utf-8') as f:
        begin=100000
        step=100
        #得到所有query图
        for i in range(500):
            queryname=str(begin+step*i)
            queryname=queryname+".jpg"

            f.write(queryname+" ")
            resultList=getResult(queryname)
            for j in range(len(resultList)):
                f.write(str(j)+" ")
                f.write(resultList[j]+" ")
            f.write("\r\n")


writeResult()