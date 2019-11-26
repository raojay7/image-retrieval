# -*- coding: utf-8 -*-
# Author: yongyuan.name
from extract_cnn_vgg16_keras import VGGNet

import numpy as np
import h5py
from sklearn.decomposition import PCA
from sklearn import preprocessing

#参数配置
featurename="featureCNN.h5"

def getResult(query,feats,imgNames):


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


def postProcess(feats):
    # 在这里正则
    # l2norm之前已经进行了一次
    # pca
    pca = PCA(n_components=512, svd_solver='auto', whiten=True)
    feats = pca.fit_transform(feats)
    # l2renorm
    # preprocessing.normalize(feats, norm='l2')
    return feats

def writeResult():
    with open("resultfile.dat","w",encoding='utf-8') as f:
        begin=100000
        step=100
        #实现得到数据库的vector
        # read in indexed images' feature vectors and corresponding image names
        h5f = h5py.File(featurename, 'r')
        # feats = h5f['dataset_1'][:]
        feats = h5f['dataset_1'][:]
        # print(feats)

        #进行后处理
        feats=postProcess(feats)

        imgNames = h5f['dataset_2'][:]  # 这里是带后缀的
        # print(imgNames)
        h5f.close()

        #得到所有query图
        for i in range(500):
            queryname=str(begin+step*i)
            queryname=queryname+".jpg"

            f.write(queryname+" ")
            resultList=getResult(queryname,feats,imgNames)
            for j in range(len(resultList)):
                f.write(str(j)+" ")
                f.write(resultList[j]+" ")
            f.write("\r\n")


writeResult()