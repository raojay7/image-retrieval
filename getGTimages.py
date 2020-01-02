import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#Oxford or Paris dataset extract from ground truth

'''
 Returns a list of filenames for all jpg images in a directory. 
'''
def get_GTlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('query.txt')]
#配置
db="C:/test/gt_files_170407/"
#db="C:/test/paris_120310/"

gtList = get_GTlist(db)


print("--------------------------------------------------")
print("starts")
print("--------------------------------------------------")
for i, gtpath in enumerate(gtList):
    with open(gtpath, "r", encoding='utf-8') as f:
        # print(f.readline())
        contents=f.readline().rstrip("\n")
        every=contents.split(" ")
        print(every[1:])
        img_path="E:/oxbuild_images/"+every[0][5:]+".jpg"
        # img_path = "E:/paris/paris/" + every[0] + ".jpg"
        image = cv2.imread(img_path)
        sp = image.shape
        imgh = sp[0]
        imgw = sp[1]

        img_name = os.path.split(img_path)[1]#包含后缀
        img_name=os.path.splitext(img_name)[0]#不包含后缀
        print("get gt from "+img_name)
        ystart=0
        yend=0
        xstart=0
        xend=0
        xstart=int(float(every[1]))
        ystart=int(float(every[2]))
        xend=int(float(every[3]))
        yend=int(float(every[4]))
        w=xend-xstart
        h=yend-ystart
        if w<h:
            left=int((h-w)/2)
            if (xstart-left>=0 and xend+left<=imgw):
                xstart=xstart-left
                xend=xend+left

            elif (xstart-2*left>=0 and xend+left>imgw):
                xstart=xstart-2*left
            elif (xstart-left< 0 and xend+2*left<=imgw):
                xend=xend+2*left

        elif w>h:
            top = int((w - h) / 2)
            if ystart - top >= 0 and yend + left <= imgh:
                ystart = ystart - top
                yend = yend + top
            elif ystart-2*top>=0 and yend+top>=imgh:
                ystart=ystart-2*top
            elif ystart-top< 0 and yend+2*top<=imgh:
                yend=yend+2*top
        print(xstart,ystart,xend,yend)
        cropImg=image[ystart:yend,xstart:xend]
        # dst=cv2.resize(cropImg,(600,600))
        cv2.imwrite("new_image/"+img_name+".jpg",cropImg) #保存到指定目录
print("process ok")
