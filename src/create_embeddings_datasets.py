import argparse
import datetime
import os

import h5py
import timm
import torch
from tqdm import tqdm

from model.avr_datasets import (
    HOI_VITdataset,
    LOGOdataset_vit,
    LOGOdataset,
    VAECdataset_vit,
    VAECdataset,
    # VAECSamplesDataset,
    VASR_VITdataset,
)
from model.avr_datasets_h5py import (
    LOGOSamplesDataset_h5py,
)

# bongard_logo_h5_images, vaec_images
from model.models.resnet18 import Encoder, BasicBlockEnc


def transform_to_h5py_embeddings(
    model,
    out_path: str,
    dataset: torch.utils.data.Dataset,
    chunk_size: int = 4,
    **h5dataset_kwargs,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # counter = 0
    num_images_per_task = dataset[0][0].shape[0]

    n_features = model.num_features if hasattr(model, "num_features") else 512
    with h5py.File(os.path.join(out_path), "w") as f:
        data_dataset = f.create_dataset(
            "data",
            shape=(len(dataset), num_images_per_task, n_features),
            dtype="f",
            chunks=(chunk_size, num_images_per_task, n_features),
            **h5dataset_kwargs,
        )
        labels_dataset = f.create_dataset(
            "labels",
            shape=(len(dataset)),
            dtype="i8",
            chunks=(chunk_size),
            **h5dataset_kwargs,
        )
        m=torch.nn.AvgPool2d(2, (1,1))

        for i in tqdm(range(len(dataset)), total=len(dataset)):
            images, labels = dataset[i]
            with torch.no_grad():
                embeddings = model(images.to("cuda"))
            # print(m(embeddings).squeeze((2,3)).shape)
            # data_dataset[i, :] = embeddings.cpu().detach().numpy()
            data_dataset[i, :] = m(embeddings).squeeze((2,3)).cpu().detach().numpy()
            labels_dataset[i] = labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/avr/datasets")
    parser.add_argument("--out_root", type=str, default="/avr/datasets")
    parser.add_argument("--model_name", type=str, default="vit_large_patch32_384")
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--run_logo", action="store_true")
    parser.add_argument("--run_hoi", action="store_true")
    parser.add_argument("--run_vaec", action="store_true")
    parser.add_argument("--run_vasr", action="store_true")
    parser.add_argument('--custom_model', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    data_root = args.data_root
    out_root = args.out_root
    model_name = args.model_name
    chunk_size = args.chunk_size
    torch.set_float32_matmul_precision("medium")
    torch.set_default_device("cuda")

    if args.custom_model:
        model = Encoder.load_from_checkpoint("/app/model_checkpoints/912500/epoch=0-step=2083.ckpt", block=BasicBlockEnc, layers=[2, 2, 2, 2])
        model=model.to("cuda")
        # print(model)
        model_name="resnet18_custom"
        # raise NotImplementedError("TODO: ...")
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=0)

    if args.run_logo:
        for dataset_type in (
            "train",
            "val",
            "test_ff",
            "test_hd_comb",
            "test_bd",
            "test_hd_novel",
        ):
            print(
                f"[{datetime.datetime.now()}] Processing LOGOdataset_vit: {dataset_type}"
            )
            if args.custom_model:
                # logo = LOGOSamplesDataset_h5py(
                #     data_path=f"{data_root}/h5py/bongard_logo",
                #     dataset_type=dataset_type,
                #     img_size=64,
                # )
                logo = LOGOdataset(
                    # data_path=f"{data_root}/h5py/bongard_logo",
                    # annotation_path="",
                    data_path=f"{data_root}/bongard_logo/ShapeBongard_V2/",
                    annotation_path=f"{data_root}/bongard_logo/ShapeBongard_V2/ShapeBongard_V2_split.json",
                    dataset_type=dataset_type,
                    img_size=64,
                )
            else:
                logo = LOGOdataset_vit(
                    data_path=f"{data_root}/bongard_logo/ShapeBongard_V2/",
                    annotation_path=f"{data_root}/bongard_logo/ShapeBongard_V2/ShapeBongard_V2_split.json",
                    dataset_type=dataset_type,
                    model_name=model_name,
                )

            transform_to_h5py_embeddings(
                model=model,
                out_path=f"{out_root}/{model_name}/bongard_logo/{dataset_type}.hy",
                dataset=logo,
                chunk_size=chunk_size,
                compression="lzf",
            )
    if args.run_hoi:
        for dataset_type in (
            "bongard_hoi_train.json",
            "bongard_hoi_test_seen_obj_seen_act.json",
            "bongard_hoi_val_seen_obj_unseen_act.json",
            "bongard_hoi_val_unseen_obj_unseen_act.json",
            "bongard_hoi_val_seen_obj_seen_act.json",
            "bongard_hoi_val_unseen_obj_seen_act.json",
            "bongard_hoi_test_unseen_obj_seen_act.json",
            "bongard_hoi_test_seen_obj_unseen_act.json",
            "bongard_hoi_test_unseen_obj_unseen_act.json",
        ):
            print(
                f"[{datetime.datetime.now()}] Processing HOI_VITdataset: {dataset_type}"
            )
            hoi = HOI_VITdataset(
                data_path=f"{data_root}/bongard_hoi/hake/",
                annotation_path=f"{data_root}/bongard_hoi/bongard_hoi_release/",
                dataset_type=dataset_type,
                model_name=model_name,
            )

            transform_to_h5py_embeddings(
                model=model,
                out_path=f"{out_root}/{model_name}/bongard_hoi/{dataset_type.split('.')[0]}.hy",
                dataset=hoi,
                chunk_size=chunk_size,
                compression="lzf",
            )

    if args.run_vaec:
        for dataset_type in (
            "analogy_train.hy",
            "analogy_test1.hy",
            "analogy_test2.hy",
            "analogy_test3.hy",
            "analogy_test4.hy",
            "analogy_test5.hy",
            "analogy_scale_train.hy",
            "analogy_scale_test1.hy",
            "analogy_scale_test2.hy",
            "analogy_scale_test3.hy",
            "analogy_scale_test4.hy",
            "analogy_scale_test5.hy",
        ):
            print(
                f"[{datetime.datetime.now()}] Processing VAECdataset_vit: {dataset_type}"
            )
            if args.custom_model:
                vaec = VAECdataset(
                    data_path=f"{data_root}/vaec/datasets/",
                    dataset_type=dataset_type,
                    img_size=64,
                )
            else:
                vaec = VAECdataset_vit(
                    data_path=f"{data_root}/vaec/datasets/",
                    dataset_type=dataset_type,
                    model_name=model_name,
                )

            transform_to_h5py_embeddings(
                model=model,
                out_path=f"{out_root}/{model_name}/vaec/{dataset_type.split('.')[0]}.hy",
                dataset=vaec,
                chunk_size=chunk_size,
                compression="lzf",
            )

    if args.run_vasr:
        for dataset_type in ("train", "dev"):
            print(
                f"[{datetime.datetime.now()}] Processing VASR_VITdataset: {dataset_type}"
            )
            vasr = VASR_VITdataset(
                data_path=f"{data_root}/vasr",
                dataset_type=dataset_type,
                model_name=model_name,
            )

            transform_to_h5py_embeddings(
                model=model,
                out_path=f"{out_root}/{model_name}/vasr/{dataset_type}.hy",
                dataset=vasr,
                chunk_size=chunk_size,
                compression="lzf",
            )
