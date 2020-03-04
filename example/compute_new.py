# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import h5py
from sklearn.decomposition import PCA
from sklearn import preprocessing
from utils.general import htime,get_data_root
from utils.testdataset import configdataset
from utils.evaluate import compute_map_and_print
featurename1="../gem_res_rox_1.h5"
featurename2="../gem_res_rox_2.h5"
featurename3="../gem_res_rox_3.h5"
featurename4="../mytest.h5"


datasetSize=4993 #原始数据库大小
# datasetSize=6322 #原始数据库大小
querysize=70

def postProcess(feats):
    # 在这里正则
    # l2norm之前已经进行了一次
    # pca
    pca = PCA(n_components=2048, svd_solver='auto', whiten=True)
    # 使用oxford来训练
    h5f = h5py.File("../gem_res_rox_3.h5", 'r')
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
h5f4 = h5py.File(featurename4, 'r')

#得到query的名字以及对应的向量
qfeats1=h5f1["dataset_3"][:]
qnames1=h5f1["dataset_4"][:]
qfeats2=h5f2["dataset_3"][:]
qnames2=h5f2["dataset_4"][:]
qfeats3=h5f3["dataset_3"][:]
qnames3=h5f3["dataset_4"][:]



feats1 = h5f1['dataset_1'][:]
feats2 = h5f2['dataset_1'][:]
feats3 = h5f3['dataset_1'][:]
feats4 = h5f4['dataset_1'][:]
# print("---------feats1---------")
# print(feats1.shape)
# print("---------feats2---------")
# print(feats2)
# print("---------feats3---------")
# print(feats3)
# print("---------feats4---------")
# print(feats4)



# 进行后处理
# feats1 = postProcess(feats1)
# feats2 = postProcess(feats2)
# feats3 = postProcess(feats3)
#
# qfeats1 = postProcess(qfeats1)
# qfeats2 = postProcess(qfeats2)
# qfeats3 = postProcess(qfeats3)


# feats4 = postProcess(feats4)

# 这里是带后缀的裁剪图片
imgNames1 = h5f1['dataset_2'][:]
imgNames2 = h5f2['dataset_2'][:]
imgNames3 = h5f3['dataset_2'][:]
imgNames4 = h5f4['dataset_2'][:]


h5f1.close()
h5f2.close()
h5f3.close()
h5f4.close()
def getLayerImgNames(L):
    if L==1:
        return imgNames1
    elif L==2:
        return imgNames2
    elif L==3:
        return imgNames3
    elif L==4:
        return imgNames4
def getLayerFeats(L):
    if L==1:
        return feats1
    elif L==2:
        return feats2
    elif L==3:
        return feats3
    elif L==4:
        return feats4
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

def getQueryVecByname(queryname,layer):
    queryVecs=np.array(0)
    qimgNames=np.array(0)
    if layer==1:
        queryVecs=qfeats1
        qimgNames=qnames1
    if layer==2:
        queryVecs=qfeats2
        qimgNames=qnames2
    if layer==3:
        queryVecs=qfeats3
        qimgNames=qnames3
    queryVec = queryVecs[qimgNames.tolist().index(np.string_(queryname))]
    return queryVec


def getImageScore(Lr,Lq,query):
    subScores=[]
    for i in range(1,Lq+1):
        if i==1:
            npfinalScore=getsubqueryScore(Lr, query,getQueryVecByname(query,i), getLayerImgNames(i))
            for item in range(datasetSize):
                npfinalScore[item]=npfinalScore[item]*(pow(Lr-i+1,1))
            subScores.append(npfinalScore)
            continue
        img_name = os.path.splitext(query)[0]
        for j in range(i):
            for k in range(i):
                subPatchName = img_name + "_" + str(j) + str(k) + ".jpg"
                npfinalScore=getsubqueryScore(Lr, subPatchName, getQueryVecByname(subPatchName,i), getLayerImgNames(i))
                for item in range(datasetSize):
                    #改变子块权重
                    npfinalScore[item] = npfinalScore[item]*(pow(Lr-i+1,1))
                subScores.append(npfinalScore)
    finalScore=[0.0,]*datasetSize
    for i in range(datasetSize):
        for j in range(len(subScores)):
            #相加
            finalScore[i]+=subScores[j][i]
        finalScore[i]=finalScore[i]#/len(subScores)
    return np.array(finalScore)

# 得到query图片的当前子块分数向量
def getsubqueryScore(Lr,subquery,queryVec,qimgNames):
    finalScore = []
    layerScores = []
    i = Lr

    while i > 0:
        totalSubOfLayer = i * i
        # 计算每个查询图的最小feature分数
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
    imlist = [baseimgNames[index].decode() for i, index in enumerate(rank_ID[0:])]
    print("top %d images in order are: " % datasetSize, imlist[0:10])
    return imlist


def compute_dataset_map():


    dataset="roxford5k"
    # dataset="rparis6k"

    #得到所有query图
    # prepare config structure for the test dataset
    print(get_data_root())
    cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
    images = [cfg['im_fname'](cfg, i) for i in range(cfg['n'])]
    qimages = [cfg['qim_fname'](cfg, i) for i in range(cfg['nq'])]
    try:
        bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]
    except:
        bbxs = None  # for holidaysmanrot and copydays


    print("eval start:")
    start = time.time()

    scores=[]
    for i in range(len(qnames1)):
        qimg_name = qnames1[i]
        print(qimg_name)
        scores.append(getImageScore(1,1, qimg_name.decode()))#解码成str
    scores=np.array(scores)
    scores=scores.T
    ranks = np.argsort(-scores, axis=0)

    compute_map_and_print(dataset, ranks, cfg['gnd'])

    print('>>: total time: {}'.format(htime(time.time() - start)))


compute_dataset_map()