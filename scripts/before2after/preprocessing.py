import csv
import numpy as np
import os
from sklearn.model_selection import train_test_split
import urllib.request

from sklearn.cluster import MiniBatchKMeans
import cv2

def get_filepaths_in_folder(folder):

    directory = os.fsencode(folder)

    filenames = []

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        filenames.append(os.path.join(str(directory.decode("utf-8")), str(filename)))

    return filenames

def to_greyscale(img_folder, target_folder=None):

    print("Grey-Scaling Images")

    filepaths = get_filepaths_in_folder(img_folder)

    # create target folders if not exist
    if target_folder is not None:
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

    for i, single_filepath in enumerate(filepaths):

        # open the image
        cur_image = cv2.imread(single_filepath)

        # convert to greyscale
        gray = cv2.cvtColor(cur_image, cv2.COLOR_BGR2GRAY)

        if target_folder is None:

            # overwrite old image
            cv2.imwrite(single_filepath, gray)

        else:

            file_name = single_filepath.split("/")[-1]
            target_path = target_folder + "/" + file_name

            cv2.imwrite(target_path, gray)

        print("Processed image {}/{}\r".format(i+1, len(filepaths)), end="")  

def color_quantization(folder, color_count, target_folder):

    filepaths = get_filepaths_in_folder(folder)

    # create target folders if not exist
    if target_folder is not None:
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

    print("Quantizing Images")

    for i, single_filepath in enumerate(filepaths):
    
        # load the image and grab its width and height
        image = cv2.imread(single_filepath)
        (h, w) = image.shape[:2]
        
        # convert the image from the RGB color space to the L*a*b*
        # color space -- since we will be clustering using k-means
        # which is based on the euclidean distance, we'll use the
        # L*a*b* color space where the euclidean distance implies
        # perceptual meaning
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # reshape the image into a feature vector so that k-means
        # can be applied
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        
        # apply k-means using the specified number of clusters and
        # then create the quantized image based on the predictions
        clt = MiniBatchKMeans(n_clusters = color_count)
        labels = clt.fit_predict(image)
        quant = clt.cluster_centers_.astype("uint8")[labels]
        
        # reshape the feature vectors to images
        quant = quant.reshape((h, w, 3))
        
        # convert from L*a*b* to RGB
        quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
        
        if target_folder is None:

            # overwrite old image
            cv2.imwrite(single_filepath, quant)

        else:
            file_name = single_filepath.split("/")[-1]
            target_path = target_folder + "/" + file_name

            cv2.imwrite(target_path, quant)

        print("Processed image {}/{}\r".format(i+1, len(filepaths)), end="")
    
if __name__ == "__main__":

    base_source = "datasets/before2after"
 
    # color count for quantization
    col_count = 8

    color_quantization(base_source + "/A/train", col_count)
    color_quantization(base_source + "/A/val", col_count)
    color_quantization(base_source + "/A/test", col_count)

    color_quantization(base_source + "/B/train", col_count)
    color_quantization(base_source + "/B/val", col_count)
    color_quantization(base_source + "/B/test", col_count)

    to_greyscale(base_source + "/A/train")
    to_greyscale(base_source + "/A/val")
    to_greyscale(base_source + "/A/test")

    to_greyscale(base_source + "/B/train")
    to_greyscale(base_source + "/B/val")
    to_greyscale(base_source + "/B/test")
    
    
