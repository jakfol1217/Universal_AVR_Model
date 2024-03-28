import h5py
import os
import json
from PIL import Image
import numpy as np
from tqdm import tqdm

# TODO: modify/add dataset classes that will process h5py versions of datasets

# Bongard HOI

# Bongard LOGO
def h5pyfy_bongard_logo(bongard_logo_path, h5py_path, compress=True):
    """
    Function for transforming the Bongard LOGO dataset into h5py format. It creates 3 files:
    Bongard_LOGO_val.hy, Bongard_LOGO_train.hy and Bongard_LOGO_test.hy, each containing different dataset split.
    Each file contains problems indexed with integers as strings (0, 1, 2, etc.) Each problem contains "grp_1" and "grp_2",
    which are 2 groups making up the Bonngard problem. Each group contains 7 images of shape 512x512x3. Each group is a
    single array 7x512x512x3.
    Args:
    bongard_logo_path -- path in which the Bongard_LOGO dataset is stored
    h5py_path -- path to which the new HDF5 files are to be saved.
    compress -- whether to compress the underlying numpy arrays. Uncompressed files are rather large (9GB for just val dataset),
    while after compression its just 200 mb. Significantly slows down the process of creating the files.
    It seems that it does not affect the file access.
    """
    with open(os.path.join(bongard_logo_path, "ShapeBongard_V2_split.json")) as f:
        bongard_logo_split = json.load(f)
    val_files = bongard_logo_split['val']
    train_files = bongard_logo_split['train']
    test_files = []
    for key in bongard_logo_split.keys():
        if "test" in key:
            test_files += bongard_logo_split[key]

    def create_bongard_logo_h5py(stage, files):
        """
        Function that transforms a given split into h5py
        Args:
        stage: val, train or test
        files: bongard logo files involved in that stage, in a form of an array
        """
        with h5py.File(os.path.join(h5py_path, "bongard_logo_" + stage + ".hy"), "w") as f:
            for i, file in enumerate(tqdm(files)):
                path = os.path.join(bongard_logo_path, files[i][:2], "images", files[i])
                grp = f.create_group(str(i), track_order=True)
                for j in range(2):
                    images = []
                    for file in os.listdir(os.path.join(path, str(j))):
                        img = Image.open(os.path.join(path, str(j), file))
                        images.append(np.array(img))
                    images = np.array(images)
                    if compress:
                        grp.create_dataset("grp_" + str(j + 1), data=images, compression="gzip")
                    else:
                        grp.create_dataset("grp_" + str(j + 1), data=images)

    print("Creating val dataset...")
    create_bongard_logo_h5py("val", val_files)
    print("Creating train dataset...")
    create_bongard_logo_h5py("train", train_files)
    print("Creating test dataset...")
    create_bongard_logo_h5py("test", test_files)

# CLEVR
# Ill try to actually download it

# DOPT

# VAEC
# already in HDF5

# dSprites
# unnecessary since its already 1 file

# g-set

# i-Raven
# already in Eden

# MNS

# PGM

# SVRT

# VAP/LABC

# VASR

# Sandia

# KiloGram

# ARC




