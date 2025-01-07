import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def list_has_common_element(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return len(set1.intersection(set2)) > 0


def calculate_scale_num(input, row_block, col_block):
    if len(input.shape) > 2:
        input = input.reshape(-1, input.shape[2])
    elif len(input.shape) == 2:
        pass
    else:
        raise ValueError(f"input shape {input.shape} does not match for block cut, {input}")
    M, N = input.shape[0], input.shape[1]

    if row_block == -1:
        row_block = M
    if col_block == -1:
        col_block = N

    return input.numel() / (row_block * col_block)


def quant_get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK") or 0)


def format_string_with_condition(
    input_string,
    condition_config,
    symm,
    bits,
    blocksize_config,
    input_pad=20,
):
    padded_string = input_string.ljust(input_pad)
    output_string = padded_string

    for k, v in condition_config.items():
        if v:
            output_string = output_string + k.ljust(10) + "True".ljust(6) + "".ljust(6)
        else:
            output_string = output_string + k.ljust(10) + "".ljust(6) + "False".ljust(6)

    output_string = output_string + f"Symm {symm}".ljust(10)

    for k, v in bits.items():
        output_string = output_string + f"{k} bit".ljust(10) + v.ljust(10)
    for k, v in blocksize_config.items():
        output_string += f"{k}: {v}".ljust(15)

    return output_string


def print_warning(sentence):
    print("*" * (len(sentence) + 4))
    print(f"* {sentence} *")
    print("*" * (len(sentence) + 4))


def check_nan_inf(tensor, check_nan, check_inf):
    if check_nan:
        contain_nan = torch.isnan(tensor).any()
    else:
        contain_nan = False
    if check_inf:
        contain_inf = torch.isinf(tensor).any()
    else:
        contain_inf = False
    return contain_nan, contain_inf


def move_torch_to_numpy(tensor):
    if tensor is None:
        return None

    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.detach().float().numpy()


def flatten_to_1d(tensor):
    if tensor is None:
        return None

    return tensor.reshape(-1)


def get_uniform_bin(tensor, num_bins, blank=0.05):
    bin_arr = np.linspace(
        tensor.min() - (tensor.max() - tensor.min()) * blank,
        tensor.max() + (tensor.max() - tensor.min()) * blank,
        num_bins,
        endpoint=True,
    )
    return bin_arr


def determine_log_scale_hist(counts, threshold_ratio=3):
    max_count = np.max(counts)
    third_max_count = np.partition(counts, -3)[-3]

    if max_count >= threshold_ratio * third_max_count:
        return True
    else:
        return False


def print_list_with_separator(lst):
    separator = "-" * 30

    for item in lst:
        print(item, item.dtype)
        print(separator)


def save_tensor(tensor, RQtensor, Qtensor, fb, aw, layer_name):
    visualize_path = os.path.join("visualize", aw, fb)
    file_name = f"{layer_name}.pt"
    os.makedirs(visualize_path, exist_ok=True)
    torch.save(
        {"tensor": tensor, "RQtensor": RQtensor, "Qtensor": Qtensor, "fb": fb, "aw": aw, "layer_name": layer_name},
        os.path.join(visualize_path, file_name),
    )
    print(f"{aw}   {fb}   {layer_name} saved!")


def visualize_distribution(pt_path):
    print(pt_path)
    saved_tensor = torch.load(pt_path, map_location="cpu")
    # os.remove(pt_path)

    tensor = saved_tensor["tensor"]
    RQtensor = saved_tensor["RQtensor"]
    Qtensor = saved_tensor["Qtensor"]
    fb = saved_tensor["fb"]
    aw = saved_tensor["aw"]
    layer_name = saved_tensor["layer_name"]

    # visualize_path = os.path.join("visualize", aw, fb, layer_name)
    # file_name = "distribution.png"
    # os.makedirs(visualize_path, exist_ok=True)
    visualize_path = os.path.join("visualize", aw, fb)
    file_name = f"{layer_name}.png"
    os.makedirs(visualize_path, exist_ok=True)

    # MSE = (tensor - Qtensor).norm().item()
    tensor, RQtensor, Qtensor = move_torch_to_numpy(tensor), move_torch_to_numpy(RQtensor), move_torch_to_numpy(Qtensor)
    tensor, RQtensor, Qtensor = flatten_to_1d(tensor), flatten_to_1d(RQtensor), flatten_to_1d(Qtensor)

    fig, axs = plt.subplots(3, 2, figsize=(120, 80))
    plt.rcParams["font.size"] = 80
    for ax in axs.flatten():
        ax.tick_params(axis="both", labelsize=80)

    num_bins = 1000
    # Tensor distribution - original
    if tensor is not None:
        axs[0, 0].hist(tensor, bins=num_bins, color="blue", alpha=0.5)
        axs[0, 0].set_title(f"Original Distribution of tensor, {tensor.dtype}")

        # Tensor distribution - log scale
        axs[0, 1].hist(tensor, bins=num_bins, color="blue", alpha=0.5)
        axs[0, 1].set_yscale("log")
        axs[0, 1].set_title(f"Log Scale Distribution of tensor, {tensor.dtype}")
        axs[0, 1].set_xlabel("use log scale")

    # Qtensor distribution - original
    if RQtensor is not None:
        axs[1, 0].hist(RQtensor, bins=num_bins, color="red", alpha=0.5)
        axs[1, 0].set_title(f"Original Distribution of RQtensor, {Qtensor.dtype}")

        # Qtensor distribution - log scale
        axs[1, 1].hist(RQtensor, bins=num_bins, color="red", alpha=0.5)
        axs[1, 1].set_yscale("log")
        axs[1, 1].set_title(f"Log Scale Distribution of RQtensor, {Qtensor.dtype}")
        axs[1, 1].set_xlabel("use log scale")

    # Qtensor distribution - original
    if Qtensor is not None:
        Q_outlier = np.max(np.abs(Qtensor))
        axs[2, 0].hist(Qtensor, bins=num_bins, color="red", alpha=0.5)
        axs[2, 0].set_title(f"Original Distribution of Qtensor, {Qtensor.dtype}")
        # axs[2, 0].set_xlim(-Q_outlier, Q_outlier)

        # Qtensor distribution - log scale
        axs[2, 1].hist(Qtensor, bins=num_bins, color="red", alpha=0.5)
        axs[2, 1].set_yscale("log")
        axs[2, 1].set_title(f"Log Scale Distribution of Qtensor, {Qtensor.dtype}")
        axs[2, 1].set_xlabel("use log scale")
        # axs[2, 1].set_xlim(-Q_outlier, Q_outlier)

    plt.tight_layout()
    plt.savefig(os.path.join(visualize_path, file_name))
    plt.close(fig)
    print(f"{aw}   {fb}   {layer_name} distribution finish!")

    exit(0)


