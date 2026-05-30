import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MPLCONFIGDIR = REPO_ROOT / "record" / "_matplotlib_cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_HOME = REPO_ROOT / "record" / "_cache"
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, Subset

from utils.aggregate_block.model_trainer_generate import generate_cls_model
import utils.save_load_attack as save_load_attack
from utils.trainer_cls import given_dataloader_test


def add_common_args(parser):
    parser.add_argument("--result_file", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--json", action="store_true")


def resolve_device(requested):
    if requested == "mps" and not torch.backends.mps.is_available():
        print("MPS is not available; falling back to CPU.", file=sys.stderr)
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.", file=sys.stderr)
        return "cpu"
    return requested


def record_dir_for(result_file):
    return Path("record") / result_file


def infer_num_classes(state_dict):
    for key in ("linear.weight", "fc.weight", "classifier.weight"):
        weight = state_dict.get(key)
        if weight is not None and weight.ndim >= 2:
            return weight.shape[0]
    for key, weight in state_dict.items():
        if key.endswith(".weight") and weight.ndim == 2:
            return weight.shape[0]
    raise ValueError("Could not infer num_classes from model state dict.")


def subset_if_needed(dataset, max_samples):
    if max_samples is None:
        return dataset
    return Subset(dataset, range(min(max_samples, len(dataset))))


def load_model_and_datasets(result_file):
    attack_path = record_dir_for(result_file) / "attack_result.pt"
    if not attack_path.exists():
        raise FileNotFoundError(f"Missing attack result: {attack_path}")
    record_dir = record_dir_for(result_file)

    original_torch_load = torch.load
    original_dataset_loader = save_load_attack.dataset_and_transform_generate

    def compatible_torch_load(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        kwargs.setdefault("weights_only", False)
        loaded = original_torch_load(*args, **kwargs)
        rewrite_backdoor_paths(loaded, record_dir)
        return loaded

    def local_dataset_loader(args):
        args.dataset_path = str(resolve_local_dataset_path(args.dataset, args.dataset_path))
        return original_dataset_loader(args)

    try:
        torch.load = compatible_torch_load
        save_load_attack.dataset_and_transform_generate = local_dataset_loader
        with contextlib.redirect_stdout(sys.stderr):
            attack_result = save_load_attack.load_attack_result(str(attack_path))
    finally:
        torch.load = original_torch_load
        save_load_attack.dataset_and_transform_generate = original_dataset_loader

    state_dict = attack_result["model"]
    model = generate_cls_model(
        attack_result["model_name"],
        num_classes=infer_num_classes(state_dict),
    )
    model.load_state_dict(state_dict)
    return model, attack_result["clean_test"], attack_result["bd_test"]


def rewrite_backdoor_paths(loaded, record_dir):
    if not isinstance(loaded, dict):
        return

    for state_key, folder_name in (
        ("bd_test", "bd_test_dataset"),
        ("bd_train", "bd_train_dataset"),
    ):
        state = loaded.get(state_key)
        if not isinstance(state, dict):
            continue

        local_folder = record_dir / folder_name
        state["save_folder_path"] = str(local_folder)
        container = state.get("bd_data_container")
        if isinstance(container, dict):
            container["save_folder_path"] = str(local_folder)
            data_dict = container.get("data_dict", {})
            for entry in data_dict.values():
                if isinstance(entry, dict) and "path" in entry:
                    entry["path"] = str(local_backdoor_image_path(entry["path"], local_folder, folder_name))


def local_backdoor_image_path(saved_path, local_folder, folder_name):
    parts = Path(saved_path).parts
    if folder_name in parts:
        idx = parts.index(folder_name)
        return local_folder.joinpath(*parts[idx + 1:])
    return local_folder / Path(saved_path).name


def resolve_local_dataset_path(dataset, original_path):
    if dataset != "cifar10":
        return original_path

    candidates = [
        Path(original_path),
        REPO_ROOT / "data" / "cifar10",
        REPO_ROOT / "data",
        REPO_ROOT.parent / "Distillation" / "data",
    ]
    for candidate in candidates:
        if (candidate / "cifar-10-batches-py").exists() or (candidate / "cifar-10-python.tar.gz").exists():
            return candidate
    return original_path


def evaluate_model(model, clean_dataset, bd_dataset, args):
    clean_dataset = subset_if_needed(clean_dataset, args.max_samples)
    bd_dataset = subset_if_needed(bd_dataset, args.max_samples)

    clean_loader = DataLoader(
        clean_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    bd_loader = DataLoader(
        bd_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    criterion = nn.CrossEntropyLoss()
    clean_metrics = given_dataloader_test(
        model,
        clean_loader,
        criterion,
        non_blocking=False,
        device=args.device,
    )[0]
    bd_metrics = given_dataloader_test(
        model,
        bd_loader,
        criterion,
        non_blocking=False,
        device=args.device,
    )[0]

    return {
        "clean_acc": clean_metrics["test_acc"],
        "asr": bd_metrics["test_acc"],
        "backdoor_present": bd_metrics["test_acc"] >= args.threshold,
        "max_samples": args.max_samples,
        "device": args.device,
    }


def floating_state_tensors(model):
    return [
        tensor.detach().cpu()
        for tensor in model.state_dict().values()
        if torch.is_tensor(tensor) and tensor.is_floating_point()
    ]


def cpu_state_dict(model):
    return {
        key: tensor.detach().cpu() if torch.is_tensor(tensor) else tensor
        for key, tensor in model.state_dict().items()
    }


def zero_fraction(model):
    total = 0
    zeros = 0
    for tensor in floating_state_tensors(model):
        total += tensor.numel()
        zeros += (tensor == 0).sum().item()
    return zeros / total if total else 0.0


def serialized_state_size(model):
    buffer = io.BytesIO()
    torch.save(cpu_state_dict(model), buffer)
    return buffer.tell()


def magnitude_prune(model, amount):
    parameters = [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    prune.global_unstructured(
        parameters,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    for module, name in parameters:
        prune.remove(module, name)


def channel_prune(model, amount):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            prune.remove(module, "weight")


def apply_compression(model, mode, amount):
    if mode == "magnitude_prune":
        magnitude_prune(model, amount)
    elif mode == "channel_prune":
        channel_prune(model, amount)
    else:
        raise ValueError(f"Unsupported compression mode: {mode}")


def write_json_result(result, use_json):
    if use_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
