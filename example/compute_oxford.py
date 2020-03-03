# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import h5py
from sklearn.decomposition import PCA
from sklearn import preprocessing
from utils.general import htime

featurename1="../mytest.h5"
featurename2="../gem_res_ox_2.h5"
featurename3="../gem_res_ox_3.h5"
featurename4="../gem_res_ox_1.h5"

#6392 paris
datasetSize=5063 #原始数据库大小
# datasetSize=6392 #原始数据库大小
querysize=55

def postProcess(feats):
    # 在这里正则
    # l2norm之前已经进行了一次
    # pca
    pca = PCA(n_components=2048, svd_solver='auto', whiten=True)
    # 使用oxford来训练
    h5f = h5py.File("gem_res_paris_1.h5", 'r')
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


def getImageScore(Lr,Lq,query):
    subScores=[]
    for i in range(1,Lq+1):
        if i==1:
            npfinalScore=getsubqueryScore(Lr, query, getLayerFeats(i), getLayerImgNames(i))
            for item in range(datasetSize):
                npfinalScore[item]=npfinalScore[item]*(pow(Lr-i+1,1))
            subScores.append(npfinalScore)
            continue
        img_name = os.path.splitext(query)[0]
        for j in range(i):
            for k in range(i):
                subPatchName = img_name + "_" + str(j) + str(k) + ".jpg"
                npfinalScore=getsubqueryScore(Lr, subPatchName, getLayerFeats(i), getLayerImgNames(i))
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
    imlist = [baseimgNames[index].decode() for i, index in enumerate(rank_ID[0:])]
    print("top %d images in order are: " % datasetSize, imlist[0:10])
    return imlist


def writeResult():


    #得到所有query图
    oxford_querynames=[
        "all_souls_000013","all_souls_000026","oxford_002985","all_souls_000051","oxford_003410",
        "ashmolean_000058","ashmolean_000000","ashmolean_000269","ashmolean_000007","ashmolean_000305",
        "balliol_000051","balliol_000187","balliol_000167","balliol_000194","oxford_001753",
        "bodleian_000107","oxford_002416","bodleian_000108","bodleian_000407","bodleian_000163",
        "christ_church_000179","oxford_002734","christ_church_000999","christ_church_001020","oxford_002562",
        "cornmarket_000047","cornmarket_000105","cornmarket_000019","oxford_000545","cornmarket_000131",
        "hertford_000015","oxford_001752","oxford_000317","hertford_000027","hertford_000063",
        "keble_000245","keble_000214","keble_000227","keble_000028","keble_000055",
        "magdalen_000078","oxford_003335","magdalen_000058","oxford_001115","magdalen_000560",
        "pitt_rivers_000033","pitt_rivers_000119","pitt_rivers_000153","pitt_rivers_000087","pitt_rivers_000058",
        "radcliffe_camera_000519","oxford_002904","radcliffe_camera_000523","radcliffe_camera_000095","bodleian_000132"
    ]
    paris_querynames=[
        "paris_defense_000605","paris_defense_000331","paris_defense_000216","paris_defense_000056","paris_defense_000254",
        "paris_general_002985", "paris_general_001729", "paris_eiffel_000266", "paris_general_002645","paris_general_002391",
        "paris_invalides_000355", "paris_invalides_000072", "paris_invalides_000490", "paris_invalides_000229", "paris_invalides_000360",
        "paris_louvre_000081", "paris_louvre_000135", "paris_louvre_000050", "paris_louvre_000035", "paris_louvre_000139",
        "paris_moulinrouge_000667", "paris_moulinrouge_000868", "paris_moulinrouge_000657", "paris_moulinrouge_000794","paris_moulinrouge_000004",
        "paris_museedorsay_000527", "paris_museedorsay_000012", "paris_museedorsay_000897", "paris_museedorsay_000564","paris_museedorsay_000878",
        "paris_notredame_000256", "paris_notredame_000965", "paris_notredame_000390", "paris_general_003117","paris_notredame_000581",
        "paris_pantheon_000466", "paris_pantheon_000520", "paris_pantheon_000232", "paris_pantheon_000547","paris_pantheon_000339",
        "paris_pompidou_000432", "paris_pompidou_000444", "paris_pompidou_000252", "paris_pompidou_000471","paris_pompidou_000636",
        "paris_sacrecoeur_000162", "paris_sacrecoeur_000417", "paris_sacrecoeur_000237", "paris_sacrecoeur_000586","paris_sacrecoeur_000437",
        "paris_triomphe_000369", "paris_triomphe_000016", "paris_triomphe_000135", "paris_triomphe_000149","paris_defense_000038",
    ]
    start = time.time()
    print("eval start:")
    for i in range(querysize):
        with open("ranked_"+str(i+1)+".txt", "w", encoding='utf-8') as f:
            print(oxford_querynames[i])
            # resultList=getResult(querynames[i]+".jpg",feats1,imgNames1)
            finalscore=[]
            resultList=getResultList(getImageScore(1,1,oxford_querynames[i]+".jpg"),getLayerImgNames(1))
            for j in range(len(resultList)):
                f.write(resultList[j].split(".")[0]+"\n")
    print('>>: total time: {}'.format(htime(time.time() - start)))


writeResult()