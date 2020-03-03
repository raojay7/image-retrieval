import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#select search方法获取图片区域块



'''
 Returns a list of filenames for all jpg images in a directory. 
'''
def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]


def mac_regions(W, H, L,featuremap_h):
    regions = []
    h = H
    w = W
    xmin = 0
    xmax = int(w)
    ymin = 0
    ymax = int(h)
    for every in range(1,L+1):
        if every==1 :
            R = np.array([ymin, xmin, ymax, xmax], dtype=np.int)
            R=R * featuremap_h / H
            regions.append(R)
        for l in range(2,every+1):
            wl = (2 * w) / (l + 1)
            hl = (2 * h) / (l + 1)
            s = max(wl, hl)

            for i in range(1, l + 1):
                for j in range(1, l + 1):
                    bw = int((w - wl) / (l - 1))
                    bh = int((h - hl) / (l - 1))
                    centerx = int(wl / 2 + (i - 1) * bw)
                    centery = int(hl / 2 + (j - 1) * bh)
                    xstart = int(centerx - s / 2)
                    xend = int(centerx + s / 2)
                    ystart = int(centery - s / 2)
                    yend = int(centery + s / 2)
                    if (xstart < xmin):
                        xstart = 0
                    if (ystart < ymin):
                        ystart = 0
                    if (xend > xmax):
                        xend = xmax
                    if (yend > ymax):
                        yend = ymax
                    R = np.array([ystart,xstart,yend,xend], dtype=np.int)
                    #将此时的区域映射到featuremap上去，这里最后一层的卷积是18*18
                    R=R*featuremap_h/H
                    regions.append(R)
    regions = np.asarray(regions)
    return regions

def get_subimages():
    #配置
    db="./oxbuild_images/"
    # db="./paris/"
    L=3 #改变L来得到不同的空间layer


    # 加载图像并显示
    img_list = get_imlist(db)
    print("--------------------------------------------------")
    print("         get sub-patch starts")
    print("--------------------------------------------------")
    for i, img_path in enumerate(img_list):
        image = cv2.imread(img_path)
        img_name = os.path.split(img_path)[1]#包含后缀
        img_name=os.path.splitext(img_name)[0]
        print(img_name)
        print("get sub-patch from image No. %d , %d images in total" % ((i + 1), len(img_list)))

        sp=image.shape
        h=sp[0]
        w=sp[1]
        xmin=0
        xmax=int(w)
        ymin=0
        ymax=int(h)
        for l in range(2,L+1):
            wl=(2*w)/(l+1)
            hl=(2*h)/(l+1)
            s=max(wl,hl)
            for i in range(1,l+1):
                for j in range(1,l+1):
                    bw=int((w-wl)/(l-1))
                    bh=int((h-hl)/(l-1))
                    centerx=int(wl/2+(i-1)*bw)
                    centery=int(hl/2+(j-1)*bh)
                    xstart=int(centerx-s/2)
                    xend=int(centerx+s/2)
                    ystart=int(centery-s/2)
                    yend=int(centery+s/2)
                    if(xstart<xmin):
                        xstart=0
                    if(ystart<ymin):
                        ystart=0
                    if(xend>xmax):
                        xend=xmax
                    if(yend>ymax):
                        yend=ymax
                    cropImg=image[ystart:yend,xstart:xend]
                    cv2.imwrite("ox2/"+img_name+"_"+str(i-1)+str(j-1)+".jpg",cropImg) #保存到指定目录00左上，01左下10右上，11右下

    print("process ok")

get_subimages()