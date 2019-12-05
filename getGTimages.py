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
        image = cv2.imread(img_path)
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
        cropImg=image[ystart:yend,xstart:xend]
        cv2.imwrite("new_image/"+img_name+".jpg",cropImg) #保存到指定目录
print("process ok")
