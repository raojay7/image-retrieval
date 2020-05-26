# Image Retrieval Engine Based on Keras

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](../LICENSE)


## 环境

```python
In [1]: import keras
Using Tensorflow backend.
```

keras 2.2.2版本经过测试可用。推荐Python 3.5.

此外需要numpy, matplotlib, os, h5py, argparse. 推荐使用anaconda安装

### 使用

- 步骤一

`python index.py -database <path-to-dataset> -index <name-for-output-index>`

- 步骤二

`python query_online.py -query <path-to-query-image> -index <path-to-index-flie> -result <path-to-images-for-retrieval>`

```sh
├── database 图像数据集
├── extract_cnn_vgg16_keras.py 使用预训练vgg16模型提取图像特征
|── index.py 对图像集提取特征，建立索引
├── query_online.py 库内搜索
└── README.md
```
- 步骤三

如果使用空间搜索，首先使用`preprocess_images.py`进行处理，生成原图的子块；

再利用步骤一继续抽取相应的子图特征

- 步骤四

利用`compute_MAP.py`，改变layrer和数据库即可得到map计算文件`resultfile.dat`，格式可参照
[holiday数据集](http://lear.inrialpes.fr/people/jegou/data.php)，最后使用holiday的eval工具计算
#### 示例

```sh
# 对database文件夹内图片进行特征提取，建立索引文件featureCNN.h5
python index.py -database database -index featureCNN.h5

# 使用database文件夹内001_accordion_image_0001.jpg作为测试图片，在database内以featureCNN.h5进行近似图片查找，并显示最近似的3张图片
python query_online.py -query database/001_accordion_image_0001.jpg -index featureCNN.h5 -result database
```


### Goal

- [x] 重新用flask写web界面，已完成。

### 论文推荐

[**awesome-cbir-papers**](https://github.com/willard-yuan/awesome-cbir-papers)

### 问题汇总

- `query_online.py` line 28报错，将`index.py` line 62注释，使用line 61.需要传入numpy的数组



### 注意事项
- 数据库内的名字要按字符串顺序排序
