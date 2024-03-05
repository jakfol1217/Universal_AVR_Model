import json
import os

import h5py
import numpy as np

GLOBAL_ROOT = "/avr/datasets/"
GLOBAL_TARGET_ROOT = "/home/kaminskia/studies/Universal_AVR_Model/src/data/samples/"


def copy_bongard_hoi():
    ROOT = os.path.join(GLOBAL_ROOT, "bongard_hoi/bongard_hoi_release/")
    DATA_ROOT = os.path.join(GLOBAL_ROOT, "bongard_hoi/hake/")
    TARGET_ROOT = os.path.join(GLOBAL_TARGET_ROOT, "bongard_hoi/bongard_hoi_release/")
    TARGET_DATA_ROOT = os.path.join(GLOBAL_TARGET_ROOT, "bongard_hoi/hake/")

    files = os.listdir(ROOT)

    for file in files:
        with open(os.path.join(ROOT, file), "r") as f:
            sample_config = json.load(f)[:2]

        images = [
            z["im_path"]
            for x in sample_config
            for y in x
            for z in y
            if isinstance(z, dict)
        ]
        images_source = [os.path.join(DATA_ROOT, image) for image in images]
        images_target = [os.path.join(TARGET_DATA_ROOT, image) for image in images]

        for image_source, image_target in zip(images_source, images_target):
            os.makedirs(os.path.dirname(image_target), exist_ok=True)
            os.system(f"cp {image_source} {image_target}")

        os.makedirs(TARGET_ROOT, exist_ok=True)
        with open(os.path.join(TARGET_ROOT, file), "w") as f:
            f.write(json.dumps(sample_config))


def copy_bongard_logo():
    DATA_ROOT = os.path.join(GLOBAL_ROOT, "bongard_logo/ShapeBongard_V2/")
    ANNOTATION_PATH = os.path.join(
        GLOBAL_ROOT, "bongard_logo/ShapeBongard_V2/ShapeBongard_V2_split.json"
    )
    TARGET_ANNOTATION = os.path.join(
        GLOBAL_TARGET_ROOT, "bongard_logo/ShapeBongard_V2/ShapeBongard_V2_split.json"
    )
    TARGET_DATA_ROOT = os.path.join(GLOBAL_TARGET_ROOT, "bongard_logo/ShapeBongard_V2/")

    with open(ANNOTATION_PATH, "r") as f:
        annotations = json.load(f)
    # print(annotations.keys())
    # for key, value in annotations.items():
    #     print(key, len(value))
    # val 900
    # test_ff 600
    # test_hd_comb 400
    # train 9300
    # test_bd 480
    # test_hd_novel 320
    annotations_sample = {k: v[:10] for k, v in annotations.items()}

    for value in annotations_sample.values():
        for sample in value:
            sample_root = os.path.join(DATA_ROOT, sample[:2], "images", sample)
            # Copy recursively directory
            os.makedirs(
                os.path.join(TARGET_DATA_ROOT, sample[:2], "images"), exist_ok=True
            )
            os.system(
                f"cp -r {sample_root} {os.path.join(TARGET_DATA_ROOT, sample[:2], 'images')}"
            )

    os.makedirs(os.path.dirname(TARGET_ANNOTATION), exist_ok=True)
    with open(TARGET_ANNOTATION, "w") as f:
        f.write(json.dumps(annotations_sample))


def copy_dopt():
    DATA_ROOT = os.path.join(GLOBAL_ROOT, "dopt/")
    TARGET_DATA_ROOT = os.path.join(GLOBAL_TARGET_ROOT, "dopt/")

    files = os.listdir(DATA_ROOT)

    for file in files:
        os.makedirs(TARGET_DATA_ROOT, exist_ok=True)
        data = np.load(os.path.join(DATA_ROOT, file), mmap_mode="r")
        # print(data.shape) # (500, 20, 64, 64)
        sample_data = data[0:10]
        np.save(os.path.join(TARGET_DATA_ROOT, file), sample_data)

    for file in files:
        data = np.load(os.path.join(DATA_ROOT, file), mmap_mode="r")
        data_copy = np.load(os.path.join(TARGET_DATA_ROOT, file), mmap_mode="r")
        assert np.array_equal(data[0:10], data_copy)


def copy_vaec():
    DATA_ROOT = os.path.join(GLOBAL_ROOT, "vaec/datasets/")
    TARGET_DATA_ROOT = os.path.join(GLOBAL_TARGET_ROOT, "vaec/datasets/")
    files = os.listdir(DATA_ROOT)

    for file in files:
        with h5py.File(os.path.join(DATA_ROOT, file)) as f:
            os.makedirs(TARGET_DATA_ROOT, exist_ok=True)
            with h5py.File(os.path.join(TARGET_DATA_ROOT, file), "w") as fout:
                for i, obj in enumerate(f.keys()):
                    if i >= 10:
                        break
                    # fout.create_group(key)
                    f.copy(obj, fout)
            # print(len(f))  # 19040
            # print(
            #     f["0"].keys()
            # )  # <KeysViewHDF5 ['ABCD', 'analogy_dim', 'dist', 'imgs', 'latent_class', 'not_D']>
            # print(f["0"]["ABCD"].shape)  # (4,)
            # print(f["0"]["ABCD"].shape)  # (4,)
            # print(
            #     f["0"]["analogy_dim"]
            # )  # <HDF5 dataset "analogy_dim": shape (), type "<i8">
            # print(f["0"]["dist"])  # <HDF5 dataset "dist": shape (), type "<i8">
            # print(f["0"]["imgs"].shape)  # 7, 128, 128, 3
            # print(f["0"]["latent_class"].shape)  # (7, 4)
            # print(f["0"]["not_D"].shape)  # (6,)


copy_bongard_hoi()
copy_bongard_logo()
copy_dopt()
copy_vaec()
