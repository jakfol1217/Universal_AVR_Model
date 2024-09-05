import os
import pickle
from itertools import cycle
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

import wandb
from src.wandb_agent import WandbAgent

load_dotenv()


def extract_wandb_id(id):
    paths = [
        "/mnt/evafs/groups/mandziuk-lab/akaminski/logs",
        "/home2/faculty/akaminski/Universal_AVR_Model/logs",
        "/home2/faculty/jfoltyn/Universal_AVR_Model/logs",
    ]
    for path in paths:
        try:
            wandb_id = WandbAgent.extract_wandb_id(id, log_dir=path)
            return wandb_id
        except FileNotFoundError:
            continue

    raise FileNotFoundError(f"Could not find wandb_id for {id}")

def get_history(
    run: wandb.apis.public.runs.Run,
    return_train: bool = False,
    return_val: bool = True,
    cache: bool = True,
) -> list[dict[str, Any]]:
    # return_train xor return_val
    assert bool(return_train) != bool(return_val) or bool(return_val) == False
    if return_train:
        mode="train"
    elif return_val:
        mode="val"
    else:
        mode="test"
    
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
            hist = run.scan_history(keys=[_k for _k in useful_keys if "val/" not in _k and "test/" not in _k])
        elif return_val:
            hist = run.scan_history(
                keys=[_k for _k in useful_keys if "train/" not in _k and "test/" not in _k]
            )
        else:
            test_keys = [_k for _k in useful_keys if "train/" not in _k and "val/" not in _k]
            test_tasks = [_k for _k in test_keys if _k.startswith("test/")]
            common_keys = [_k for _k in test_keys if _k not in test_tasks]
            hist = []
            # TODO: transposition here or later?
            for test_task in test_tasks:
                _hist = run.scan_history(
                    keys=common_keys+[test_task]
                )
                hist.extend(list(_hist))
        hist = list(hist)
        # save pickle
        with open(f"history/.cache/{mode}_{run.id}.pkl", "wb") as f:
            pickle.dump(hist, f)

    return hist


def get_histories(
    runs: list[dict[int, wandb.apis.public.runs.Run]], **kwargs
) -> pd.DataFrame:
    distinct_runs = {v.url: (k,v) for k,v in runs.items()}
    distinct_runs = {k:v for k,v in distinct_runs.values()}
    hist = [get_history(run, **kwargs) for run in distinct_runs.values()]
    hist = [
        item | {"SLURM_ID": _id}
        for (sublist, _id) in zip(hist, distinct_runs.keys())
        for item in sublist
    ]
    return pd.DataFrame(hist)


def extract_best_metric(runs, val_data, train_data, test_data) -> dict:
    columns = set(val_data.columns) | set(train_data.columns) | set(test_data.columns)
    loss_columns = [col for col in columns if "loss" in col]
    # loss_columns = [col for col in loss_columns if "val" in col]


    out = {}
    for col in loss_columns:
        df = val_data if "val" in col else train_data if "train" in col else test_data
        df[col] = df[col].astype(float)
        best_val = df.sort_values(col, ascending=True).iloc[0]

        out[col] = {
            "best_loss": float(best_val[col]),
            "best_loss_epoch": int(best_val["epoch"]),
            "best_loss_step": int(best_val["trainer/global_step"]),
            "best_slurm_id": int(best_val["SLURM_ID"]),
        }

    return out


def extract_accuracy_metric(runs, val_data, train_data, test_data) -> dict:
    columns = set(val_data.columns) | set(train_data.columns) | set(test_data.columns)
    loss_columns = [col for col in columns if "accuracy" in col]
    # loss_columns = [col for col in loss_columns if "val" in col]


    out = {}
    for col in loss_columns:
        df = val_data if "val" in col else train_data if "train" in col else test_data
        df[col] = df[col].astype(float)
        best_val = df.sort_values(col, ascending=False).iloc[0]

        out[col] = {
            "best_loss": float(best_val[col]),
            "best_loss_epoch": int(best_val["epoch"]),
            "best_loss_step": int(best_val["trainer/global_step"]),
            "best_slurm_id": int(best_val["SLURM_ID"]),
        }

    return out

def extract_best_loss_model_with_accuracy(runs, val_data, train_data, test_data) -> dict:
    columns = set(val_data.columns) # | set(train_data.columns) | set(test_data.columns)
    # loss_columns = [col for col in columns if col.startswith("val/") and col.endswith("/loss")]
    if len([col for col in columns if "accuracy" in col]) == 0:
        return {}

    # TODO: add searching for approximate checkpoint to best models per dataset
    loss_columns = ["val/loss"]

    model_path = "/mnt/evafs/groups/mandziuk-lab/akaminski/model_checkpoints"
    slurm_ids = runs.keys()

    out = {}
    for col in loss_columns:
        val_data[col] = val_data[col].astype(float)
        best_val = val_data.sort_values(col, ascending=True).iloc[0]
        # TODO: dataset mappings? (e.g. vasr_images -> vasr) (may here it should stay like that (to easier match corresponding dataset yaml))

        filename=f"epoch={best_val['epoch']:.0f}-step={best_val['trainer/global_step']+1:.0f}.ckpt"

        best_ckpt=None
        for slurm_id in slurm_ids:
            if os.path.exists(os.path.join(model_path, str(slurm_id), filename)):
                best_ckpt = os.path.join(model_path, str(slurm_id), filename)
                break

        if best_ckpt is None:
            raise FileNotFoundError(f"Best model checkpoint {filename} not found for slurm_id {slurm_ids}")

        dataset = col.split("/")[1]
        out[f"{dataset}/best_ckpt"] = best_ckpt

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


def extract_url(runs, val_data, train_data, test_data) -> dict:
    return {"wandb_urls": list(set([run.url for run in runs.values()]))}

# always calculate best metric, max epoch, etc...
common_info = [
    # lambda dict[slurm_id, wanbdb.run], valid_history, train -> dict
    extract_url,
    extract_best_metric,
    extract_accuracy_metric,
    lambda r, val_data, *_: {"max_epoch": int(val_data["epoch"].max())},
    extract_best_loss_model_with_accuracy,
    # datasets not working very well - config is overwritten by tests (only tasks should change)
    lambda r, *_: {"datasets": list(next(iter(r.values())).config["data"]["tasks"].keys())}
]

additional_info_config = {
    # based on experiment_nm and/or test_nm define lambda expressions of fields to calculate
    # lambda dict[slurm_id, wanbdb.run], valid_history, train -> dict
    "experiment_nm": {
        # "": lambda ...
    },
    "test_nm": {"lr_check": [lambda r, *_: dict(lr=next(iter(r.values())).config["lr"])]},
}


def main():
    # Load the config file
    with open("experiments.yml", "r") as f:
        exps = yaml.safe_load(f)

    apis_keys = [os.environ["JF_WANDB_API_KEY"], os.environ["AK_WANDB_API_KEY"]]
    wandb_agents = [WandbAgent(project_name="AVR_universal", api_key=key) for key in apis_keys]
    wandb_agents_cycle = cycle(wandb_agents)

    wandb_agent = next(wandb_agents_cycle)

    out = {"experiments": [None] * len(exps["experiments"])}
    for ix, experiment in tqdm(enumerate(exps["experiments"]), total=len(exps["experiments"])):
        try:
            run_ids = {_id: extract_wandb_id(_id) for _id in experiment["slurm_id"]}
            # DEBUG:
            # run_ids = {835796: "gzkhftm7", 835797: "fl7x59k9", 835798: "qtgsn8eg"}
            runs = None
            for _ in range(len(wandb_agents)):
                try:
                    runs = {
                        _id: wandb_agent.get_run_by_id(run_id) for _id, run_id in run_ids.items()
                    }
                    break
                except Exception:
                    wandb_agent = next(wandb_agents_cycle)
            if runs == None:
                raise Exception(f"Experiment {experiment['slurm_id']} was not found")

            cache_flg = experiment.get("cache", True)
            train_history = get_histories(
                runs, return_train=True, return_val=False, cache=cache_flg
            )
            valid_history = get_histories(
                runs, return_train=False, return_val=True, cache=cache_flg
            )
            test_history = get_histories(
                runs, return_train=False, return_val=False, cache=cache_flg
            )

            # add common info
            for info in common_info:
                try:
                    experiment.update(
                        info(runs, valid_history, train_history, test_history)
                    )
                except Exception as e:
                    print(f"Failed to calculate {info.__name__} for experiment with slurm ids: {experiment['slurm_id']}. Error: {e}")
            # add additional info
            exp_nm = experiment["experiment_nm"]
            test_nm = experiment["test_nm"]

            for info in additional_info_config["experiment_nm"].get(exp_nm, []):
                if "additional_inforamations" not in experiment:
                    experiment["additional_inforamations"] = {}
                try:
                    experiment["additional_inforamations"].update(
                        info(runs, valid_history, train_history, test_history)
                    )
                except Exception as e:
                    print(f"Failed to calculate {info.__name__} for experiment {exp_nm} with slurm ids: {experiment['slurm_id']}. Error: {e}")

            for info in additional_info_config["test_nm"].get(test_nm, []):
                if "additional_inforamations" not in experiment:
                    experiment["additional_inforamations"] = {}
                try:
                    experiment["additional_inforamations"].update(
                        info(runs, valid_history, train_history, test_history)
                    )
                except Exception as e:
                    print(f"Failed to calculate {info.__name__} for experiment (test={test_nm}) with slurm ids: {experiment['slurm_id']}. Error: {e}")
        except Exception as e:
            print(f"Unexpected error occured when processing experiment with slurm ids: {experiment.get('slurm_id')}. Error: {e}")
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
