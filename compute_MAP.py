# -*- coding: utf-8 -*-
# Author: yongyuan.name

import os
import numpy as np
import h5py
from sklearn.decomposition import PCA
from sklearn import preprocessing

#参数配置
featurename1="featureCNN.h5"
featurename2="featureCNN1.h5"
featurename3="featureCNN2.h5"
featurename4="featureCNN3.h5"

querySize=500
datasetSize=1491 #原始数据库大小
def getResult(query,feats,imgNames):
    #不用输入模型重复计算
    queryVec=feats[imgNames.tolist().index(np.string_(query))]
    scores = np.dot(queryVec, feats.T)


    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]
    # number of top retrieved images to show
    maxres = 10
    imlist = [imgNames[index].decode() for i, index in enumerate(rank_ID[0:maxres])]
    print("top %d images in order are: " % maxres, imlist)
    return imlist

def getLayerScore(query,feats,imgNames,L):

    if L==1 :
        queryVec = feats[imgNames.tolist().index(np.string_(query))]
        npfinalScore = np.dot(queryVec, feats.T)
        return npfinalScore

    scoresList = []  # 得到query图片的分数向量
    finalScore = []
    totalSubOfLayer=L*L
    for k in range(totalSubOfLayer):
        scoresList.append([])  # 每个子分数的大小都是原始分数的1/L*L
    img_name = os.path.splitext(query)[0]

    now = 0
    for i in range(L):
        for j in range(L):
            # 计算每个子图的feature分数
            subPatchName = img_name + "_" + str(i) + str(j) + ".jpg"
            queryVec = feats[imgNames.tolist().index(np.string_(subPatchName))]
            scores = np.dot(queryVec, feats.T)

            max = float('-inf')  # 未做归一化，值的范围要这样设置
            for k in range(len(scores)):
                # 取subpatch的最大值
                if (max < scores[k]):
                    max = scores[k]
                if ((k + 1) % totalSubOfLayer == 0):
                    scoresList[now].append(max)
                    max = float('-inf')
            now = now + 1
    # 计算平均值，放入原始图片1491大小的列表中
    for i in range(datasetSize):
        ave = 0.0
        for j in range(totalSubOfLayer):
            ave += scoresList[j][i]
        ave /= totalSubOfLayer #测试sum
        finalScore.append(ave)
    npfinalScore = np.array(finalScore)
    return npfinalScore



#得到根据每层特征的排序
def getLayerResult(query,feats,imgNames,baseimgNames,L):
    npfinalScore=getLayerScore(query,feats,imgNames,L)
    return getResultList(npfinalScore,baseimgNames)

def getResultList(npfinalScore,baseimgNames):
    # 再次计算分数
    rank_ID = np.argsort(npfinalScore)[::-1]
    rank_score = npfinalScore[rank_ID]

    # number of top retrieved images to show
    maxres = 10
    imlist = [baseimgNames[index].decode() for i, index in enumerate(rank_ID[0:maxres])]
    print("top %d images in order are: " % maxres, imlist)
    return imlist

def postProcess(feats):
    # 在这里正则
    # l2norm之前已经进行了一次
    # pca
    pca = PCA(n_components=512, svd_solver='auto', whiten=True)
    # 使用oxford来训练
    h5f = h5py.File("featureCNN3.h5", 'r')
    feat_train = h5f['dataset_1'][:]
    pca.fit_transform(feat_train)
    feats = pca.transform(feats)
    h5f.close()
    # l2renorm
    feats=preprocessing.normalize(feats, norm='l2')

    return feats

def writeResult():
    with open("resultfile.dat","w",encoding='utf-8') as f:
        begin=100000
        step=100
        #实现得到数据库的vector
        # read in indexed images' feature vectors and corresponding image names
        h5f1 = h5py.File(featurename1, 'r')
        # h5f2 = h5py.File(featurename2, 'r')
        # h5f3 = h5py.File(featurename3, 'r')
        h5f4 = h5py.File(featurename4, 'r')


        feats1 = h5f1['dataset_1'][:]

        # print(feats1.shape)
        # feats2 = h5f2['dataset_1'][:]
        # feats3 = h5f3['dataset_1'][:]
        feats4 = h5f4['dataset_1'][:]

        #进行后处理
        # feats1=postProcess(feats1)
        # feats2=postProcess(feats2)
        # feats3=postProcess(feats3)
        feats4=postProcess(feats4)

        # 这里是带后缀的裁剪图片
        imgNames1 = h5f1['dataset_2'][:]
        # imgNames2 = h5f2['dataset_2'][:]
        # imgNames3 = h5f3['dataset_2'][:]
        imgNames4 = h5f4['dataset_2'][:]


        h5f1.close()
        # h5f2.close()
        # h5f3.close()
        h5f4.close()

        #得到所有query图
        for i in range(querySize):
            queryname=str(begin+step*i)
            queryname=queryname+".jpg"

            f.write(queryname+" ")
            # resultList=getResult(queryname,feats,imgNames)
            # resultList=getLayerResult(queryname,feats,imgNames,baseImageNames)

            # sc1=getLayerScore(queryname,feats1,imgNames1,1)
            # sc2=getLayerScore(queryname,feats2,imgNames2,2)
            # sc3=getLayerScore(queryname,feats3,imgNames3,3)
            sc4=getLayerScore(queryname,feats4,imgNames4,4)
            resultList=getResultList(sc4,imgNames1)
            for j in range(len(resultList)):
                f.write(str(j)+" ")
                f.write(resultList[j]+" ")
            f.write("\r\n")

writeResult()