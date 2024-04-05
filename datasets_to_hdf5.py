import h5py
import os
import json
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob

# TODO: modify/add dataset classes that will process h5py versions of datasets

# Bongard HOI
# I decided not to go the 1 index 1 problem route due to possibly repeating images making the already
# large dataset even bigger. Here is a simple translation of data folders into single hdf5 files
def h5pyfy_bongard_hoi(bongard_hoi_path, h5py_path, compress = True):
    """
    Translates the Bongard HOI data folder into a h5py file. The resulting file recreates all the paths in the
    original dataset, making it possible to still use the json files.
    WARNING: the resulting files are extremely large in size to make it somewhat manageable,
    the folders are split into separate files.
    Args:
    bongard_hoi_path -- path in which the Bongard_HOI dataset is stored
    h5py_path -- path to which the new HDF5 files are to be saved.
    compress -- whether to compress the underlying numpy arrays.
    """

    for dr in os.listdir(bongard_hoi_path):
        with h5py.File(os.path.join(h5py_path, "bongard_hoi_" + dr + ".hy"), "w") as f:
            images = []
            for ext in ["*.jpg", "*.png", "*.jpeg"]:
                images.extend(glob.glob(os.path.join(bongard_hoi_path, dr, "**", ext), recursive=True))
            print(f"Converting folder: {dr}")
            for img in tqdm(images):
                img_array = np.array(Image.open(img))
                file_path = img.replace(bongard_hoi_path, "").replace("\\","/")[1:]
                file_path_1, file_path_2 = file_path.rsplit("/", 1)
                grp = f.require_group(file_path_1)
                if compress:
                    grp.create_dataset(file_path_2, data=img_array, compression="gzip")
                else:
                    grp.create_dataset(file_path_2, data=img_array)

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
def h5pyfy_clevr(clevr_path, h5py_path, compress=True):
    """
    Function for transforming the CLEVR dataset into h5py format. It creates 3 files:
    CLEVR_val.hy, CLEVR_train.hy and CLEVR_test.hy, each containing different dataset split.
    Each file contains problems indexed with integers as strings (0, 1, 2, etc.) Each problem contains "data" of size
    16x240x320x3 and "target" os size 1.
    Args:
    clevr_path -- path in which the CLEVR dataset is stored
    h5py_path -- path to which the new HDF5 files are to be saved.
    compress -- whether to compress the underlying numpy arrays.
    """

    def create_clevr_h5py(stage):
        """
        Function that transforms a given split into h5py
        Args:
        stage: val, train or test
        """

        with h5py.File(os.path.join(h5py_path, "clevr_" + stage + ".hy"), "w") as f:
            files = []
            for dr in os.listdir(clevr_path):
                files.extend(glob.glob(os.path.join(clevr_path, dr, f"*{stage}*.npz")))

            for i, file in enumerate(tqdm(files)):
                grp = f.create_group(str(i), track_order=True)
                data = np.load(file)
                if compress:
                    grp.create_dataset("data", data=data['image'], compression="gzip")

                else:
                    grp.create_dataset("data", data=data['image'])
                grp.create_dataset("target", data=data['target'])

    print("Creating test dataset...")
    create_clevr_h5py("test")
    print("Creating train dataset...")
    create_clevr_h5py("train")
    print("Creating val dataset...")
    create_clevr_h5py("val")

# DOPT
# Its already 3 files

# VAEC
# already in HDF5

# dSprites
# unnecessary since its already 1 file

# g-set
def h5pyfy_gset(gset_path, h5py_path, compress=True):
    """
    Function for transforming the G-set dataset into h5py format. It creates 1 file:
    gset.hy.
    This file contains problems indexed with integers as strings (0, 1, 2, etc.) Each problem contains "test" of size
    8x50x50x4, "answers" of size 5x50x50x4  and "target" of size 1.
    Args:
    gset_path -- path in which the G-set dataset is stored
    h5py_path -- path to which the new HDF5 files are to be saved.
    compress -- whether to compress the underlying numpy arrays.
    """

    def split_gset_answer(image):
        """
        Function for splitting the answers image into singular panels.
        """
        images = []
        for window in range(0, image.size[0], 50):
            images.append(np.array(image)[:, window:window + 50])
        return images

    def split_gset_context(image):
        """
        Function for splitting the context image into singular panels.
        """

        images = []
        for window_w in range(0, image.size[0], 50):
            for window_h in range(0, image.size[1], 50):
                images.append(np.array(image)[window_w:window_w + 50, window_h:window_h + 50])
        return images[:-1]

    targets = pd.read_csv(os.path.join(gset_path, "answers.csv"), header=None)
    with h5py.File(os.path.join(h5py_path, "gset.hy"), "w") as f:
        for ext in ["answers", "test"]:
            print(f"Creating {ext}...")
            for file in tqdm(glob.glob(os.path.join(gset_path, "*" + ext + ".png"))):
                idx = file.replace('\\', '/').rsplit('/', 1)[1][:5]
                if idx == "00000":
                    idx = "0"
                else:
                    idx = idx.lstrip("0")
                grp = f.require_group(idx)
                img = Image.open(file)
                if ext == "answers":
                    img_split = split_gset_answer(img)
                else:
                    img_split = split_gset_context(img)
                    grp.create_dataset("target", data=targets[0][int(idx)])

                if compress:
                    grp.create_dataset(ext, data=np.array(img_split), compression="gzip")
                else:
                    grp.create_dataset(ext, data=np.array(img_split))



# i-Raven
# already in Eden

# MNS

def h5pyfy_mns(mns_path, h5py_path, compress=True):
    """
    Function for transforming the MNS dataset into h5py format. It creates 3 files:
    MNS_val.hy, MNS_train.hy and MNS_test.hy, each containing different dataset split.
    Each file contains problems indexed with integers as strings (0, 1, 2, etc.) Each problem contains "data" of size 3x160x160
    and "target" os size 1.
    Args:
    mns_path -- path in which the MNS dataset is stored
    h5py_path -- path to which the new HDF5 files are to be saved.
    compress -- whether to compress the underlying numpy arrays.
    """

    def create_mns_h5py(stage):
        """
        Function that transforms a given split into h5py
        Args:
        stage: val, train or test
        """
        full_path = os.path.join(mns_path, stage + "_set")
        with h5py.File(os.path.join(h5py_path, "mns_" + stage + "_set.hy"), "w") as f:
            for i, file in enumerate(tqdm(os.listdir(full_path))):
                grp = f.create_group(str(i), track_order=True)
                data = np.load(os.path.join(full_path, file))
                if compress:
                    grp.create_dataset("data", data=data['image'], compression="gzip")

                else:
                    grp.create_dataset("data", data=data['image'])
                grp.create_dataset("target", data=data['target'])
                grp.attrs['filename'] = file

    print("Creating val dataset...")
    create_mns_h5py("val")
    print("Creating train dataset...")
    create_mns_h5py("train")
    print("Creating test dataset...")
    create_mns_h5py("test")

# PGM

def h5pyfy_pgm(pgm_path, h5py_path, compress=True):
    """
    Function for transforming the PGM dataset into h5py format. It creates 3 files:
    PGM_val.hy, PGM_train.hy and PGM_test.hy, each containing different dataset split.
    Each file contains problems indexed with integers as strings (0, 1, 2, etc.) Each problem contains "image" of size
    16x160x160 and "target" os size 1.
    The function is intended to be used on 1 regime at a time (so 3 files per regime), as the regimes are quite large
    and diverse.
    Args:
    pgm_path -- path in which the PGM dataset is stored
    h5py_path -- path to which the new HDF5 files are to be saved.
    compress -- whether to compress the underlying numpy arrays.
    """
    regime = pgm_path.replace("\\", "/").rsplit('/', 1)[1]

    def create_pgm_h5py(stage):
        """
        Function that transforms a given split into h5py
        Args:
        stage: val, train or test
        """

        with h5py.File(os.path.join(h5py_path, "_".join(["pgm", regime, stage]) + ".hy"), "w") as f:
            files = glob.glob(os.path.join(pgm_path, f"*{stage}*.npz"))

            for i, file in enumerate(tqdm(files)):
                grp = f.create_group(str(i), track_order=True)
                data = np.load(file)
                image = data['image'].reshape(16, 160, 160)
                if compress:
                    grp.create_dataset("data", data=image, compression="gzip")

                else:
                    grp.create_dataset("data", data=image)
                grp.create_dataset("meta_target", data=data['meta_target'])
                grp.create_dataset("relation_structure", data=data['relation_structure'])
                grp.create_dataset("relation_structure_encoded", data=data['relation_structure_encoded'])
                grp.attrs['filename'] = file.replace("\\", "/").rsplit("/", 1)[1]
                grp.create_dataset("target", data=data['target'])

    print("Creating val dataset...")
    create_pgm_h5py("val")
    print("Creating train dataset...")
    create_pgm_h5py("train")
    print("Creating test dataset...")
    create_pgm_h5py("test")
# SVRT

def h5pyfy_svrt(svrt_path, h5py_path, compress = True):
    """
    Function for transforming the SVRT dataset into h5py format. It creates 1 file:
    svrt.hy
    This file contains the whole dataset, mimicking the way it is stored originally. It contains 23 class groups, and
    each class group is further split into 2 additional groups, each containing 100 images -- so each group contains
    a numpy array of size 100x128x128x3.
    Args:
    svrt_path -- path in which the SVRT dataset is stored
    h5py_path -- path to which the new HDF5 files are to be saved.
    compress -- whether to compress the underlying numpy arrays.
    """
    with h5py.File(os.path.join(h5py_path, "svrt.hy"), "w") as f:
        for fldr in os.listdir(svrt_path):
            print(f"Converting folder: {fldr}")
            grp = f.create_group(fldr)
            photos_0 = []
            photos_1 = []
            for file in tqdm(glob.glob(os.path.join(svrt_path, fldr, "*.png"))):
                if file.rsplit("_",2)[1] == "0":
                    photos_0.append(np.array(Image.open(file)))
                else:
                    photos_1.append(np.array(Image.open(file)))
            if compress:
                grp.create_dataset("group_0", data=np.array(photos_0), compression="gzip")
                grp.create_dataset("group_1", data=np.array(photos_1), compression="gzip")
            else:
                grp.create_dataset("group_0", data=np.array(photos_0))
                grp.create_dataset("group_1", data=np.array(photos_1))

# VAP/LABC

def h5pyfy_labc(labc_path, h5py_path, compress=True):
    """
    Function for transforming the LABC dataset into h5py format. It creates 3 files:
    LABC_val.hy, LABC_train.hy and LABC_test.hy, each containing different dataset split.
    Each file contains problems indexed with integers as strings (0, 1, 2, etc.) Each problem contains "image" of size
    9x160x160 and "target" os size 1.
    The function is intended to be used on 1 regime at a time (so 3 files per regime), as the regimes are quite large
    and diverse.
    Args:
    labc_path -- path in which the LABC dataset is stored
    h5py_path -- path to which the new HDF5 files are to be saved.
    compress -- whether to compress the underlying numpy arrays.
    """
    regime = labc_path.replace("\\", "/").rsplit('/', 1)[1]

    def create_labc_h5py(stage):
        """
        Function that transforms a given split into h5py
        Args:
        stage: val, train or test
        """

        with h5py.File(os.path.join(h5py_path, "_".join(["labc", regime, stage]) + ".hy"), "w") as f:
            files = glob.glob(os.path.join(labc_path, f"*{stage}*.npz"))

            for i, file in enumerate(tqdm(files)):
                grp = f.create_group(str(i), track_order=True)
                data = np.load(file)
                image = data['image'].reshape(-1, 160, 160)
                if compress:
                    grp.create_dataset("data", data=image, compression="gzip")


                else:
                    grp.create_dataset("data", data=image)
                grp.create_dataset("relation_structure", data=data['relation_structure'])
                grp.create_dataset("relation_structure_encoded", data=data['relation_structure_encoded'])
                grp.create_dataset("target", data=data['target'])
                grp.attrs['filename'] = file.replace("\\", "/").rsplit("/", 1)[1]

    print("Creating val dataset...")
    create_labc_h5py("val")
    print("Creating train dataset...")
    create_labc_h5py("train")
    print("Creating test dataset...")
    create_labc_h5py("test")
# VASR

# Sandia

# KiloGram
# I don't think we will use it

# ARC




