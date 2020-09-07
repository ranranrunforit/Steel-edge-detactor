import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import numpy as np
import os

#apply Crop, Flip and GaussianBlur transformation randomly.
seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)), # blur images with a sigma of 0 to 3.0
    iaa.Flipud(1), # vertically flip 100% of all images
    iaa.AddToSaturation((-30, 30)), # add random values to the saturation of images.
    iaa.AddToBrightness((-30, 30)), # add to the brightness channels of input images.
    iaa.AddToHue((-30, 30)), # add random values to the hue of images.
])

#'-modify'
folder = 'C:/Users/zhouc/Desktop/python/data_orginal 2/images-modify'
newfolder = 'C:/Users/zhouc/Desktop/python/data_orginal 2/seg-modify'


def augment_seg(img, seg):
    aug_det = seq.to_deterministic()
    image_aug = aug_det.augment_image(img)

    segmap = ia.SegmentationMapsOnImage(seg, shape=img.shape)
    segmap_aug = aug_det.augment_segmentation_maps(segmap)
    segmap_aug = segmap_aug.get_arr()

    cv2.imwrite(folder + "/" + file + "_6.png", image_aug)
    cv2.imwrite(newfolder + "/" + file + "_6.png", segmap_aug)
    print("Finished writing " + file + ".png in two folders.")
    #return image_aug, segmap_aug

for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    file = os.path.splitext(filename)[0]
    seg = cv2.imread(os.path.join(newfolder,filename))
    augment_seg(img.copy(), seg.copy())