# -*- coding: utf-8 -*-
# Author: yongyuan.name

import os
import numpy as np
import h5py
from sklearn.decomposition import PCA
from sklearn import preprocessing

#参数配置
baseImageName="featureCNN.h5"
featurename="featureCNN2.h5"
L=3
totalSubOfLayer=L*L
querySize=500
datasetSize=1491 #原始数据库大小
def getResult(query,feats,imgNames):


    #不用输入模型重复计算
    queryVec=feats[imgNames.tolist().index(np.string_(query))]
    print(queryVec)
    scores = np.dot(queryVec, feats.T)
    # print(scores)
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]
    # print(rank_ID)
    # print(rank_score)

    # number of top retrieved images to show
    maxres = 10
    imlist = [imgNames[index].decode() for i, index in enumerate(rank_ID[0:maxres])]
    print("top %d images in order are: " % maxres, imlist)
    return imlist

#输入都是原图的图片，比如100000.jpg
def getLayerResult(query,feats,imgNames,baseimgNames):
    scoresList=[]#得到query图片的分数向量
    finalScore=[]
    for k in range(totalSubOfLayer):
        scoresList.append([])#每个子分数的大小都是原始分数的1/L*L
    img_name = os.path.splitext(query)[0]

    now=0
    for i in range(L):
        for j in range(L):
            # 计算每个子图的feature分数
            subPatchName = img_name + "_" + str(i) + str(j) + ".jpg"
            queryVec = feats[imgNames.tolist().index(np.string_(subPatchName))]
            print(queryVec.shape)
            scores = np.dot(queryVec, feats.T)
            print("scores:"+subPatchName)
            print(scores[:1000])
            # print(scores[1000:2000])
            # print(scores[2000:3000])
            # print(scores[3000:4000])
            # print(scores[4000:5000])
            # print(scores[5000:])


            max=float('-inf') #未做归一化，值的范围要这样设置
            for k in range(len(scores)):
                #取subpatch的最大值
                if(max<scores[k]):
                    max=scores[k]
                if((k+1)%totalSubOfLayer==0):
                    scoresList[now].append(max)
                    max=float('-inf')
            now=now+1
    #计算平均值，放入原始图片1491大小的列表中
    for i in range(datasetSize):
        ave=0.0
        for j in range(totalSubOfLayer):
            ave+=scoresList[j][i]
        ave/=totalSubOfLayer
        finalScore.append(ave)

    print(finalScore)
    npfinalScore=np.array(finalScore)
    #再次计算分数
    rank_ID = np.argsort(npfinalScore)[::-1]
    print(rank_ID)
    rank_score = npfinalScore[rank_ID]
    print(rank_score)

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
    feats = pca.fit_transform(feats)
    # l2renorm
    preprocessing.normalize(feats, norm='l2')
    return feats

#等待解决bug顺序不一致
def writeResult():
    with open("resultfile.dat","w",encoding='utf-8') as f:
        begin=100000
        step=100
        #实现得到数据库的vector
        # read in indexed images' feature vectors and corresponding image names
        h5f = h5py.File(featurename, 'r')
        h5fbase = h5py.File(baseImageName, 'r')
        feats = h5f['dataset_1'][:]
        print(feats.shape)

        #进行后处理
        feats=postProcess(feats)
        imgNames = h5f['dataset_2'][:]  # 这里是带后缀的裁剪图片


        # print(imgNames[:1000])
        # print(imgNames[1000:])

        baseImageNames=h5fbase['dataset_2'][:]


        h5f.close()
        h5fbase.close()
        #得到所有query图
        for i in range(querySize):
            queryname=str(begin+step*i)
            queryname=queryname+".jpg"

            f.write(queryname+" ")
            # resultList=getResult(queryname,feats,imgNames)
            resultList=getLayerResult(queryname,feats,imgNames,baseImageNames)

            for j in range(len(resultList)):
                f.write(str(j)+" ")
                f.write(resultList[j]+" ")
            f.write("\r\n")


writeResult()