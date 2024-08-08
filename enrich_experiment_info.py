import os
import pickle
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

import wandb
from src.wandb_agent import WandbAgent

load_dotenv()

# to function
# get_history
def get_history(
    run: wandb.apis.public.runs.Run,
    return_train: bool = False,
    return_val: bool = True,
    cache: bool = True,
) -> list[dict[str, Any]]:
    # return_train xor return_val
    assert bool(return_train) != bool(return_val)
    mode = "train" if return_train else "val"
    useful_keys = [
        _key
        for _key in run.summary.keys()
        if not _key.startswith("gradients/")
        and _key not in ["graph_0", "_wandb"]
        and (_key in ["_step", "trainer/global_step"] or not _key.endswith("_step"))
    ]

    os.makedirs("history/.cache", exist_ok=True)
    if cache and os.path.exists(f"history/.cache/{mode}_{run.id}.pkl"):
        # pickle load
        with open(f"history/.cache/{mode}_{run.id}.pkl", "rb") as f:
            hist = pickle.load(f)
    else:
        if return_train:
            hist = run.scan_history(keys=[_k for _k in useful_keys if "val/" not in _k])
        else:
            hist = run.scan_history(
                keys=[_k for _k in useful_keys if "train/" not in _k]
            )
        hist = list(hist)
        # save pickle
        with open(f"history/.cache/{mode}_{run.id}.pkl", "wb") as f:
            pickle.dump(hist, f)

    return hist


def get_histories(
    runs: list[dict[int, wandb.apis.public.runs.Run]], **kwargs
) -> pd.DataFrame:

    hist = [get_history(run, **kwargs) for run in runs.values()]
    hist = [
        item | {"SLURM_ID": _id}
        for (sublist, _id) in zip(hist, runs.keys())
        for item in sublist
    ]
    return pd.DataFrame(hist)


def extract_best_metric(runs, val_data, train_data) -> dict:
    columns = set(val_data.columns) | set(train_data.columns)
    loss_columns = [col for col in columns if "loss" in col]
    # loss_columns = [col for col in loss_columns if "val" in col]


    out = {}
    for col in loss_columns:
        df = val_data if "val" in col else train_data
        best_val = df.sort_values(col, ascending=True).iloc[0]

        out[col] = {
            "best_loss": float(best_val[col]),
            "best_loss_epoch": int(best_val["epoch"]),
            "best_loss_step": int(best_val["trainer/global_step"]),
            "best_slurm_id": int(best_val["SLURM_ID"]),
        }

    return out


def extract_accuracy_metric(runs, val_data, train_data) -> dict:
    columns = set(val_data.columns) | set(train_data.columns)
    loss_columns = [col for col in columns if "accuracy" in col]
    # loss_columns = [col for col in loss_columns if "val" in col]


    out = {}
    for col in loss_columns:
        df = val_data if "val" in col else train_data
        best_val = df.sort_values(col, ascending=False).iloc[0]

        out[col] = {
            "best_loss": float(best_val[col]),
            "best_loss_epoch": int(best_val["epoch"]),
            "best_loss_step": int(best_val["trainer/global_step"]),
            "best_slurm_id": int(best_val["SLURM_ID"]),
        }

    return out

#  'val/loss',
#  '_runtime',
#  'train/bongard_hoi_images/loss_epoch',
#  'train/vasr_images/loss_epoch',
#  '_step',
#  'epoch',
#  '_timestamp',
#  'val/vasr_images/loss',
#  'val/bongard_hoi_images/loss'
#  'SLURM_ID'


def extract_url(runs, train_data, val_data) -> dict:
    return {"wandb_urls": [run.url for run in runs.values()]}

# always calculate best metric, max epoch, etc...
common_info = [
    # lambda dict[slurm_id, wanbdb.run], valid_history, train -> dict
    extract_url,
    extract_best_metric,
    extract_accuracy_metric,
    lambda _, __, val_data: {"max_epoch": int(val_data["epoch"].max())},
]

additional_info_config = {
    # based on experiment_nm and/or test_nm define lambda expressions of fields to calculate
    # lambda dict[slurm_id, wanbdb.run], valid_history, train -> dict
    "experiment_nm": {
        # "": lambda ...
    },
    "test_nm": {"lr_check": [lambda r, _, __: dict(lr=next(iter(r.values())).config["lr"])]},
}


def main():
    # Load the config file
    with open("experiments.yml", "r") as f:
        exps = yaml.safe_load(f)

    wandb_agent = WandbAgent(project_name="AVR_universal")

    def extract_wandb_id(id):
        try:
            wandb_id = WandbAgent.extract_wandb_id(
                id, log_dir="/home2/faculty/akaminski/Universal_AVR_Model/logs"
            )
        except FileNotFoundError:
            wandb_id = WandbAgent.extract_wandb_id(
                id, log_dir="/home2/faculty/jfoltyn/Universal_AVR_Model/logs"
            )
        return wandb_id

    out = {"experiments": [None] * len(exps["experiments"])}
    for ix, experiment in tqdm(enumerate(exps["experiments"]), total=len(exps["experiments"])):
        try:
            run_ids = {_id: extract_wandb_id(_id) for _id in experiment["slurm_id"]}
            # DEBUG:
            # run_ids = {835796: "gzkhftm7", 835797: "fl7x59k9", 835798: "qtgsn8eg"}
            try:
                runs = {
                    _id: wandb_agent.get_run_by_id(run_id) for _id, run_id in run_ids.items()
                }
            except Exception:
                current_key = wandb_agent.api.api_key
                potential_key1, potential_key2 = os.environ["JF_WANDB_API_KEY"], os.environ["AK_WANDB_API_KEY"]
                new_key = potential_key1 if current_key == potential_key2 else potential_key2
                wandb_agent = WandbAgent(project_name="AVR_universal", api_key=new_key)
                runs = {
                    _id: wandb_agent.get_run_by_id(run_id) for _id, run_id in run_ids.items()
                }

            train_history = get_histories(
                runs, return_train=True, return_val=False, cache=True
            )
            valid_history = get_histories(
                runs, return_train=False, return_val=True, cache=True
            )

            # add common info
            for info in common_info:
                try:
                    experiment.update(
                        info(runs, valid_history, train_history)
                    )
                except Exception:
                    print(f"Failed to calculate {info.__name__} for experiment with slurm ids: {experiment['slurm_id']}")
            # add additional info
            exp_nm = experiment["experiment_nm"]
            test_nm = experiment["test_nm"]

            for info in additional_info_config["experiment_nm"].get(exp_nm, []):
                if "additional_inforamations" not in experiment:
                    experiment["additional_inforamations"] = {}
                try:
                    experiment["additional_inforamations"].update(
                        info(runs, valid_history, train_history)
                    )
                except Exception:
                    print(f"Failed to calculate {info.__name__} for experiment {exp_nm} with slurm ids: {experiment['slurm_id']}")

            for info in additional_info_config["test_nm"].get(test_nm, []):
                if "additional_inforamations" not in experiment:
                    experiment["additional_inforamations"] = {}
                try:
                    experiment["additional_inforamations"].update(
                        info(runs, valid_history, train_history)
                    )
                except Exception:
                    print(f"Failed to calculate {info.__name__} for experiment (test={test_nm}) with slurm ids: {experiment['slurm_id']}")
        except Exception:
            print(f"Unexpected error occured when processing experiment with slurm ids: {experiment.get('slurm_id')}")
            experiment.update({"failed": True})
        out["experiments"][ix] = experiment

        # DEBUG
        # break

    # save to yml
    with open("enriched_experiments.yml", "w") as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False)
    return


if __name__ == "__main__":
    main()
