import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#select search方法获取图片区域块



'''
 Returns a list of filenames for all jpg images in a directory. 
'''
def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
#配置
db="./database/"
L=4 #改变L来得到不同的空间layer



# 加载图像并显示

img_list = get_imlist(db)


print("--------------------------------------------------")
print("         get sub-patch starts")
print("--------------------------------------------------")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#显示为BGR
# plt.title("Original Image")
# plt.imshow(image)
# plt.show()


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
                cv2.imwrite("database3/"+img_name+"_"+str(i-1)+str(j-1)+".jpg",cropImg) #保存到指定目录00左上，01左下10右上，11右下

print("process ok")
