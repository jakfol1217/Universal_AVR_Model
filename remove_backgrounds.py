from transformers import pipeline
from PIL import Image
import os
import sys
import shutil
from tqdm import tqdm
import glob

MODEL = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

def remove_background_bongard_hoi(dataset_path, save_path):
    dataset_dirs = os.listdir(os.path.join(dataset_path, "hake"))
    files = []
    for dir in dataset_dirs:
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            files.extend(glob.glob(os.path.join(dataset_path, "hake", dir, "**", ext), recursive=True))
    for file_path in tqdm(files):
        im = Image.open(file_path)
        im_no_bg = MODEL(im)
        dir_path, file = os.path.split(file_path)
        new_filename = file.split(".")[0] + ".png"
        path = os.path.join(save_path, dir_path.replace(dataset_path + "/", ""))
        os.makedirs(path, exist_ok=True)
        im_no_bg.save(os.path.join(path, new_filename))
    os.makedirs(os.path.join(save_path, "bongard_hoi_release"), exist_ok=True)
    for file in os.listdir(os.path.join(dataset_path, "bongard_hoi_release")):
        shutil.copyfile(os.path.join(dataset_path, "bongard_hoi_release", file), os.path.join(save_path, "bongard_hoi_release", file))

def remove_background_bongard_hoi_simple(dataset_path, save_path):
    files = os.listdir(dataset_path)
    for file in tqdm(files):
        im = Image.open(os.path.join(dataset_path, file))
        im_no_bg = MODEL(im)
        new_filename = file.split(".")[0] + ".png"
        im_no_bg.save(os.path.join(save_path, new_filename))

def remove_background_vasr(dataset_path, save_path):
    image_dataset_path = os.path.join(dataset_path, "images_512")
    files = os.listdir(image_dataset_path)
    
    for file in tqdm(files):
        im = Image.open(os.path.join(image_dataset_path, file))
        im_no_bg = MODEL(im)
        new_filename = file.split(".")[0] + ".png"
        save_path_images = os.path.join(save_path, "images_512")
        os.makedirs(save_path_images, exist_ok=True)
        im_no_bg.save(os.path.join(save_path_images, new_filename))
    for file_path in glob.glob(os.path.join(dataset_path, "*.csv")):
        _, file = os.path.split(file_path)
        shutil.copyfile(file_path, os.path.join(save_path, file))


if __name__ == "__main__":
    args = sys.argv
    # args[0] = current file
    # args[1] = function name
    # args[2:] = function args : (*unpacked)
    # args[3] = save_path
    os.makedirs(args[3], exist_ok=True)
    globals()[args[1]](*args[2:])