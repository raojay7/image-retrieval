# -*- coding: utf-8 -*-
import os
import numpy as np
import h5py
from sklearn.decomposition import PCA
from sklearn import preprocessing


featurename1="oxford.h5"
featurename4="oxford3_vgg.h5"

datasetSize=5063 #原始数据库大小
querysize=55
def getResult(query,feats,imgNames):
    #不用输入模型重复计算
    queryVec=feats[imgNames.tolist().index(np.string_(query))]
    scores = np.dot(queryVec, feats.T)


    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]
    # number of top retrieved images to show
    maxres = datasetSize
    imlist = [imgNames[index].decode() for i, index in enumerate(rank_ID[0:maxres])]
    # print("top %d images in order are: " % maxres, imlist)
    return imlist

def getLayerScore(query,feats,imgNames,L):

    if(L==1):
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
    imlist = [baseimgNames[index].decode() for i, index in enumerate(rank_ID[0:])]
    # print("top %d images in order are: " % datasetSize, imlist)
    return imlist

def postProcess(feats):
    # 在这里正则
    # l2norm之前已经进行了一次
    # pca
    pca = PCA(n_components=128, svd_solver='auto', whiten=True)
    # 使用oxford来训练
    h5f = h5py.File("oxford_vgg.h5", 'r')
    feat_train = h5f['dataset_1'][:]
    pca.fit_transform(feat_train)
    feats = pca.transform(feats)
    h5f.close()
    # l2renorm
    feats=preprocessing.normalize(feats, norm='l2')

    return feats

def writeResult():

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
        querynames=[
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
        for i in range(querysize):
            with open("ranked_"+str(i+1)+".txt", "w", encoding='utf-8') as f:
                print(querynames[i])
                resultList=getResult(querynames[i]+".jpg",feats1,imgNames1)
                # resultList=getLayerResult(queryname,feats,imgNames,baseImageNames)

                # sc2=getLayerScore(queryname,feats2,imgNames2,2)
                # sc3=getLayerScore(queryname,feats3,imgNames3,3)
                # sc4=getLayerScore(querynames[i],feats4,imgNames4,4)
                # resultList=getResultList(sc4,imgNames1)
                for j in range(len(resultList)):
                    if j==0: continue
                    f.write(resultList[j].split(".")[0]+"\n")
writeResult()