import cv2
import numpy as np
import os

#'seg-modify'
#'C:/Users/zhouc/Desktop/python/data_orginal 2/seg'

folder = 'C:/Users/zhouc/Desktop/python/data_orginal 2/seg-modify'
newfolder = 'C:/Users/zhouc/Desktop/python/data_orginal 2/ann-modify'

for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    #img = cv2.resize(img, (600, 600))
    #cv2.imwrite(folder + "/" + file + ".png", img)
    file = os.path.splitext(filename)[0]
    ann_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for i in range(img.shape[0]):  # for every pixel:
        for j in range(img.shape[1]):
            if img[i,j][1]==128 :#green
                #print("green", img[i, j])
                ann_img[i, j] = 1
                #print("green", ann_img[i, j])
            elif img[i,j][2]==128:#red
                #print("red", img[i, j])
                ann_img[i, j] = 0
                #print("red", ann_img[i, j])
            elif (img[i, j] == [0, 0, 0]).all():#black
                #print("black", img[i, j])
                ann_img[i, j] = 2
                #print("black", ann_img[i, j])
    cv2.imwrite(newfolder +"/"+ file+".png", ann_img)
    print("Finished writing "+file+".png in the folder.")
