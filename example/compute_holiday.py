# -*- coding: utf-8 -*-

import os
import numpy as np
import h5py
from sklearn.decomposition import PCA
from sklearn import preprocessing

#参数配置
featurename1="../gem_res_holiday_1.h5"
featurename2="../gem_res_holiday_2.h5"
featurename3="../gem_res_holiday_3.h5"


querySize=500
datasetSize=1491 #原始数据库大小

def postProcess(feats):
    # 在这里正则
    # l2norm之前已经进行了一次
    # pca
    pca = PCA(n_components=8, svd_solver='auto', whiten=True)
    # 使用oxford来训练
    h5f = h5py.File("../gem_res_ox_3.h5", 'r')
    feat_train = h5f['dataset_1'][:]
    pca.fit_transform(feat_train)
    feats = pca.transform(feats)
    # h5f.close()
    # l2renorm
    feats=preprocessing.normalize(feats, norm='l2')
    return feats
# 实现得到数据库的vector
h5f1 = h5py.File(featurename1, 'r')
h5f2 = h5py.File(featurename2, 'r')
h5f3 = h5py.File(featurename3, 'r')

feats1 = h5f1['dataset_1'][:]
feats2 = h5f2['dataset_1'][:]
feats3 = h5f3['dataset_1'][:]

# print("---------feats1---------")
# print(feats1.shape)
# print("---------feats2---------")
# print(feats2)
# print("---------feats3---------")
# print(feats3)




# 进行后处理
feats1 = postProcess(feats1)
feats2 = postProcess(feats2)
feats3 = postProcess(feats3)


# 这里是带后缀的裁剪图片
imgNames1 = h5f1['dataset_2'][:]
imgNames2 = h5f2['dataset_2'][:]
imgNames3 = h5f3['dataset_2'][:]


h5f1.close()
h5f2.close()
h5f3.close()

def getLayerImgNames(L):
    if L==1:
        return imgNames1
    elif L==2:
        return imgNames2
    elif L==3:
        return imgNames3

def getLayerFeats(L):
    if L==1:
        return feats1
    elif L==2:
        return feats2
    elif L==3:
        return feats3

def getResult(query,feats,imgNames):
    #不用输入模型重复计算
    queryVec=feats[imgNames.tolist().index(np.string_(query))]
    scores = np.dot(queryVec, feats.T)


    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]
    # number of top retrieved images to show
    maxres = datasetSize
    imlist = [imgNames[index].decode() for i, index in enumerate(rank_ID[0:maxres])]
    print("top %d images in order are: " % maxres, imlist[0:10])
    return imlist

def getImageScore(Lr,Lq,query):
    subScores=[]
    for i in range(1,Lq+1):
        if i==1:
            npfinalScore=getsubqueryScore(Lr, query, getLayerFeats(i), getLayerImgNames(i))
            for item in range(datasetSize):
                npfinalScore[item]=npfinalScore[item]*Lr
            subScores.append(npfinalScore)
            continue
        img_name = os.path.splitext(query)[0]
        for j in range(i):
            for k in range(i):
                subPatchName = img_name + "_" + str(j) + str(k) + ".jpg"
                npfinalScore=getsubqueryScore(Lr, subPatchName, getLayerFeats(i), getLayerImgNames(i))
                for item in range(datasetSize):
                    #改变子块权重
                    npfinalScore[item] = npfinalScore[item]
                subScores.append(npfinalScore)
    finalScore=[0.0,]*datasetSize
    for i in range(datasetSize):
        for j in range(len(subScores)):
            #相加
            finalScore[i]+=subScores[j][i]
        finalScore[i]=finalScore[i]#/len(subScores)
    return np.array(finalScore)
# 得到query图片的当前子块分数向量
def getsubqueryScore(Lr,subquery,qfeats,qimgNames):
    finalScore = []
    layerScores = []
    i = Lr
    while i > 0:
        totalSubOfLayer = i * i
        # 计算每个查询图的最小feature分数
        queryVec = qfeats[qimgNames.tolist().index(np.string_(subquery))]
        scores = np.dot(queryVec, getLayerFeats(i).T)

        max = float('-inf')  # 未做归一化，值的范围要这样设置
        #重设分数
        score = []
        for k in range(len(scores)):
            # 得到当前layer块中的最相似的子块
            if (max < scores[k]):
                max = scores[k]
            if ((k + 1) % totalSubOfLayer == 0):
                score.append(max)
                max = float('-inf')
        i = i - 1
        #加入每层的最终分数
        layerScores.append(score)
    #根据每层的分数得到查询图片的最终分数
    for j in range(datasetSize):
        max = float('-inf')
        for i in range(Lr):
            if(max<layerScores[i][j]):
                max=layerScores[i][j]
        finalScore.append(max)
    npfinalScore = np.array(finalScore)
    return npfinalScore

def getResultList(npfinalScore,baseimgNames):
    # 再次计算分数
    rank_ID = np.argsort(npfinalScore)[::-1]
    rank_score = npfinalScore[rank_ID]

    # number of top retrieved images to show
    maxres = 1491
    imlist = [baseimgNames[index].decode() for i, index in enumerate(rank_ID[0:maxres])]
    print("top %d images in order are: " % maxres, imlist)
    return imlist



def writeResult():
    with open("resultfile.dat","w",encoding='utf-8') as f:
        begin=100000
        step=100
        #得到所有query图
        for i in range(querySize):
            queryname=str(begin+step*i)
            queryname=queryname+".jpg"

            f.write(queryname+" ")
            resultList=getResultList(getImageScore(3,1,queryname),getLayerImgNames(1))

            for j in range(len(resultList)):
                f.write(str(j)+" ")
                f.write(resultList[j]+" ")
            f.write("\n")

writeResult()