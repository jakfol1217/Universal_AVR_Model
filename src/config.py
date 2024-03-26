from omegaconf import OmegaConf


def calculate_partition_type(timeout_min: int):
    if timeout_min < 60*24:
        return "short"
    else:
        return "long"

OmegaConf.register_new_resolver("calculate_partition_type", calculate_partition_type)
