# # # from keras.applications.vgg16 import VGG16
# # # from keras.preprocessing import image
# # # from keras.applications.vgg16 import preprocess_input
# # # from keras.models import Model
# # # import numpy as np
# # #
# # # base_model = VGG16(weights='imagenet')
# # # model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
# # #
# # # img_path = './holiday/100000.jpg'
# # # img = image.load_img(img_path, target_size=(224, 224))
# # # x = image.img_to_array(img)
# # # x = np.expand_dims(x, axis=0)
# # # x = preprocess_input(x)
# # #
# # # block4_pool_features = model.predict(x)
# # # print(block4_pool_features)
# # # from preprocess_images import mac_regions
# # #
# # # regions=mac_regions(600,600,3)
# # # print(regions.shape)
# # # print(regions)
# # # import numpy as np
# # # ans=np.array([
# # #     [[[1,2,3]]],
# # #     [[[4,5,6]]],
# # #     [[[7, 8, 9]]],
# # #     # [[7],[8],[9]]
# # # ])
# #
# #
# # #测试roi函数
# # def test_roi():
# #     from RoiPooling import RoiPooling
# #     import numpy as np
# #     import matplotlib.pyplot as plt
# #     import matplotlib.patches as patches
# #
# #     mode = 'tf'
# #     h = 18
# #     w = 18
# #     if mode == 'tf':
# #         feature_map = np.zeros((h,w,512))
# #     elif mode == 'th':
# #         feature_map = np.zeros((512,h,w))
# #     for i in range(h):
# #         for j in range(w):
# #             if np.random.rand() < 0.1:
# #                 if mode == 'tf':
# #                     for k in range(512):
# #                         feature_map[i,j,k] = np.random.rand()
# #
# #     roi_batch = np.array([[0,0,2,2],[2,2,5,5]])
# #
# #     print(feature_map.shape)
# #     roi_pooled = RoiPooling(mode=mode).get_pooled_rois(feature_map, roi_batch)
# #     print(roi_pooled.shape)
# #
# #     listfeat=feature_map.tolist()
# #     for k in range(5):
# #         for i in range(18):
# #             for j in range(18):
# #                 print("{:.2f}\t".format(listfeat[i][j][k]),end="")
# #             print("")
# #         print("---------------")
# #
# #
# #     print(roi_pooled)
# #     # roilist=roi_pooled.tolist()
# #     # for i in range(2):
# #     #     print(roilist[i][:][:])
# #     if mode=='tf':
# #         _, ax = plt.subplots(2)
# #         ax[0].imshow(feature_map[...,0])
# #         xmin, ymin, xmax, ymax = roi_batch[0]
# #         ax[0].add_patch(patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, edgecolor='r', facecolor='none', linewidth=1))
# #         ax[1].imshow(roi_pooled[0,...,0])
# #         plt.show()

# import os
# import cv2
# import matplotlib.pyplot as plt
# from example.index import get_imlist
# # 使用matplotlib展示多张图片
# def matplotlib_multi_pic1(imgs,name):
#     for i in range(len(imgs)):
#         img = cv2.imread(imgs[i])
#
#         b, g, r = cv2.split(img)
#         img2 = cv2.merge([r, g, b])
#         #行，列，索引
#         plt.subplot(2,10,i+1)
#         plt.imshow(img2)
#
#         plt.gcf().set_size_inches(18.5, 5.5)
#         plt.xticks([])
#         plt.yticks([])
#         plt.savefig('E:\\PycharmProjects\\image-retrieval\\result\\'+str(name)+'.png',dpi=100)
#     # plt.show()
#
# if __name__ == "__main__":
#
#     ox_expand="E:\\PycharmProjects\\image-retrieval\\new_image\\oxford_expand_gt\\"
#     ox_source="E:\\PycharmProjects\\image-retrieval\\new_image\\oxford_source_gt\\"
#
#     paris_expand = "E:\\PycharmProjects\\image-retrieval\\new_image\\paris_expand_gt\\"
#     paris_source = "E:\\PycharmProjects\\image-retrieval\\new_image\\paris_source_gt\\"
#
#     expand_img_list = get_imlist(paris_expand)
#     source_img_list= get_imlist(paris_source)
#     imgs=[]
#     s_imgs=[]
#     ex_imgs=[]
#     num=0
#     for i in range(len(expand_img_list)):
#         expand=cv2.imread(expand_img_list[i])
#         source=cv2.imread(source_img_list[i])
#
#         if expand.shape!=source.shape:
#             s_imgs.append(source_img_list[i])
#             ex_imgs.append(expand_img_list[i])
#             num=num+1
#             if num==10:
#                 break
#     imgs=s_imgs+ex_imgs
#     matplotlib_multi_pic1(imgs,"paris")
