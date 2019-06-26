import csv
import numpy as np
import os
from sklearn.model_selection import train_test_split
import urllib.request


before_image_urls = "/data/users/sstauden/pytorch-CycleGAN-and-pix2pix/datasets/before2after/img_urls/before.csv"
after_image_urls = "/data/users/sstauden/pytorch-CycleGAN-and-pix2pix/datasets/before2after/img_urls/after.csv"
filtered_image_index_file = "/data/users/sstauden/pytorch-CycleGAN-and-pix2pix/datasets/before2after/img_urls/filter.txt"

folder_path_a = "/data/users/sstauden/pytorch-CycleGAN-and-pix2pix/datasets/before2after/A"
folder_path_b = "/data/users/sstauden/pytorch-CycleGAN-and-pix2pix/datasets/before2after/B"

train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1 

def read_url_file(file):

    url_list = []

    with open(file) as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            url_list.append(row[0])

        print("Read {} image urls from {}".format(len(url_list), file))

        return np.array(url_list)

def read_filter_file(file):

    filter_indices = []

    with open(file, "rb") as txt_file:
        
        for line in txt_file:
            filter_indices.append(int(line))

        print("Read {} image filter indeices from {}".format(len(filter_indices), file))

    return np.array(filter_indices)

def filter_urls(url_list, url_filtered_indices):
    return np.delete(url_list, url_filtered_indices)

def apply_splitting(url_list_a, url_list_b, train_ratio, val_ratio, test_ratio):

    assert(abs(train_ratio + val_ratio + test_ratio - 1) < 0.001)

    train_urls_a, rem_a, train_ulrs_b, rem_b = train_test_split(url_list_a, url_list_b, test_size=1-train_ratio, random_state=42)
    
    test_size = test_ratio / float(val_ratio + test_ratio)
    val_urls_a, test_urls_a, val_urls_b, test_urls_b = train_test_split(rem_a, rem_b, test_size=test_size, random_state=42)

    return (train_urls_a, train_ulrs_b), (val_urls_a, val_urls_b), (test_urls_a, test_urls_b)

def download_images(url_list_a, url_list_b, img_name_prefix, target_folder_a, target_folder_b):

    assert (len(url_list_a) == len(url_list_b))

    # create target folders if not exist
    if not os.path.exists(target_folder_a):
        os.makedirs(target_folder_a)

    if not os.path.exists(target_folder_b):
        os.makedirs(target_folder_b)

    cur_img_id = 0

    for counter in range(len(url_list_a)):

        # create commong file name
        file_ending = url_list_a[counter].split(".")[-1]
        file_name = "{}_{}.{}".format(img_name_prefix, cur_img_id, file_ending)

        # try downloading the files
        try:
            img_a = urllib.request.urlopen(url_list_a[counter])
            img_b = urllib.request.urlopen(url_list_b[counter])
        
        except:
            # skip when there are problems
            print("Could not open one of the files. Skipping")
            continue 
        
        # save images when everything is fine
        with  open(target_folder_a + "/" + file_name, "wb") as file_out_a:
            file_out_a.write(img_a.read())

        with  open(target_folder_b + "/" + file_name, "wb") as file_out_b:
            file_out_b.write(img_b.read())

        print("Processed image {}/{}".format(counter + 1, len(url_list_a)), end="")

        cur_img_id += 1

if __name__ == "__main__":
    
    # load image urls
    before_urls = read_url_file(before_image_urls)
    after_urls = read_url_file(after_image_urls)

    # load filter indices
    filter_indices = read_filter_file(filtered_image_index_file)

    # filter urls
    before_urls = filter_urls(before_urls, filter_indices)
    after_urls = filter_urls(after_urls, filter_indices)

    # split urls in train, val, test sets
    train_urls, val_urls, test_urls = apply_splitting(before_urls, after_urls, train_ratio, val_ratio, test_ratio)

    # create target folder names
    train_target_a = folder_path_a + "/train"
    val_target_a = folder_path_a + "/val"
    test_target_a = folder_path_a + "/test"

    train_target_b = folder_path_b + "/train"
    val_target_b = folder_path_b + "/val"
    test_target_b = folder_path_b + "/test"

    # download images to target folders
    download_images(train_urls[0], train_urls[1], "train", train_target_a, train_target_b)
    download_images(val_urls[0], val_urls[1], "val", val_target_a, val_target_b)
    download_images(test_urls[0], test_urls[1], "test", test_target_a, test_target_b)
