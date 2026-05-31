from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFENSE_DIR = REPO_ROOT / "defense"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DEFENSE_DIR) not in sys.path:
    sys.path.insert(0, str(DEFENSE_DIR))

import utils.save_load_attack as save_load_attack
from compression_eval_utils import resolve_local_dataset_path, rewrite_backdoor_paths
from utils.metric import clean_accuracy, attack_success_rate, robust_accuracy
from utils.save_load_attack import generate_cls_model

def dynamic_quantize(
    model: nn.Module,
    layer_types: set[type] = {nn.Linear, nn.LSTM, nn.GRU, nn.RNNCell},
    dtype: torch.dtype = torch.qint8,
) -> nn.Module:
    model = copy.deepcopy(model).eval()
    return torch.quantization.quantize_dynamic(model, layer_types, dtype=dtype)


def _calibrate(model: nn.Module, data_loader: Iterable) -> None:
    """Run a small calibration pass so observers collect activation statistics."""
    model.eval()
    with torch.inference_mode():
        for batch in data_loader:
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            model(inputs)


def static_quantize(
    model: nn.Module,
    calibration_loader: Iterable,
    qconfig: tq.QConfig | None = None,
    integer_only_io: bool = False,
) -> nn.Module:
    model = copy.deepcopy(model).eval()

    if qconfig is None:
        qconfig = tq.get_default_qconfig("x86")

    model.qconfig = qconfig  # type: ignore[assignment]
    tq.prepare(model, inplace=True)
    _calibrate(model, calibration_loader)
    tq.convert(model, inplace=True)

    if integer_only_io:
        model._integer_io = True  # type: ignore[assignment]

    return model

def float16_quantize(model: nn.Module) -> nn.Module:
    model = copy.deepcopy(model).eval()
    return model.half()


def int16_activation_int8_weight_quantize(
    model: nn.Module,
    calibration_loader: Iterable,
) -> nn.Module:
    model = copy.deepcopy(model).eval()

    act_observer = tq.MinMaxObserver.with_args(
        dtype=torch.qint32,
        qscheme=torch.per_tensor_symmetric,
    )
    weight_observer = tq.PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
    )
    qconfig = tq.QConfig(activation=act_observer, weight=weight_observer)

    model.qconfig = qconfig  # type: ignore[assignment]
    tq.prepare(model, inplace=True)
    _calibrate(model, calibration_loader)
    tq.convert(model, inplace=True)
    return model

def quantize(
    model: nn.Module,
    method: str,
    *,
    calibration_loader: Iterable | None = None,
    example_inputs: tuple | None = None,
    **kwargs,
) -> nn.Module | torch.jit.ScriptModule:
    method = method.lower()

    if method == "dynamic":
        return dynamic_quantize(model, **kwargs)

    if method == "static":
        assert calibration_loader is not None, "calibration_loader required for static PTQ"
        return static_quantize(model, calibration_loader, **kwargs)

    if method == "float16":
        return float16_quantize(model)

    if method == "16x8":
        assert calibration_loader is not None, "calibration_loader required for 16x8 PTQ"
        return int16_activation_int8_weight_quantize(model, calibration_loader)

    raise ValueError(
        f"Unknown quantization method: {method!r}. "
        f"Choose from: 'none', 'dynamic', 'static', 'float16', '16x8'."
    )


class QuantWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = model
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        return self.dequant(self.model(self.quant(x)))


def resolve_record_dir(modelpath: str) -> Path:
    path = Path(modelpath).expanduser()
    if path.name == "attack_result.pt":
        path = path.parent
    if path.is_absolute():
        return path
    candidate = REPO_ROOT / "record" / path
    if candidate.exists():
        return candidate
    return (Path.cwd() / path).resolve()


def resolve_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser()
    if output_dir.is_absolute():
        return output_dir
    return (REPO_ROOT / output_dir).resolve()


def load_attack_result_relative(attack_result_path: Path, record_dir: Path):
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
        return save_load_attack.load_attack_result(str(attack_result_path))
    finally:
        torch.load = original_torch_load
        save_load_attack.dataset_and_transform_generate = original_dataset_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", type=str, default="16x8", help="one of: dynamic, static, float16, 16x8")
    parser.add_argument("--modelpath", type=str, default="cifar10_preactresnet18_badnet_0_1", help="name of folder in record directory")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "quantization" / "outputs" / "torch_ao")
    args = parser.parse_args()
    OPTION = args.option.lower()

    record_dir = resolve_record_dir(args.modelpath)
    attack_result_path = record_dir / "attack_result.pt"
    if not attack_result_path.exists():
        raise FileNotFoundError(f"Missing attack result: {attack_result_path}")

    artifact = load_attack_result_relative(attack_result_path, record_dir)
    
    model = generate_cls_model(artifact['model_name'], num_classes=10)  
    model.load_state_dict(artifact['model'])  
    model.eval()

    clean_dataset = artifact['clean_test']
    indices = random.sample(range(len(clean_dataset)), k=200)
    clean_calibration_loader = DataLoader(Subset(clean_dataset, indices), batch_size=1)

    original_model = copy.deepcopy(model).eval()
    wrapped = QuantWrapper(original_model).eval()
    

    if OPTION == "dynamic":
        quantized_naive = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
        dtype=torch.qint8
        )
        quantized_naive.eval()

        quantized_model = quantized_naive
    elif OPTION == "static":
        static_quant_model = quantize(wrapped, 'static', calibration_loader=clean_calibration_loader)
        quantized_model = static_quant_model
    elif OPTION == "float16":
        float16_quant_model = quantize(wrapped, 'float16', calibration_loader=clean_calibration_loader)
        quantized_model = float16_quant_model
    elif OPTION == "16x8":
        experimental_quant_model = quantize(wrapped, '16x8', calibration_loader=clean_calibration_loader)
        quantized_model = experimental_quant_model
    else:
        raise ValueError(f"Unknown option: {args.option}")

    if OPTION in ["float16", "16x8"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            quantized_model = quantized_model.half().to(device)
        else:
            quantized_model = quantized_model.to(device)
    else:
        device = torch.device("cpu")
        quantized_model = quantized_model.to(device)

    quantized_model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    loader_clean = torch.utils.data.DataLoader(artifact["clean_test"], batch_size=128, shuffle=False, num_workers=0)
    loader_bd    = torch.utils.data.DataLoader(artifact["bd_test"],    batch_size=128, shuffle=False, num_workers=0)
    print(artifact["bd_test"].dataset.root)
    true_c, pred_c = [], []
    total_loss_c = 0.0
    total_samples_c = 0
    with torch.no_grad():
        for inputs, labels, *_ in loader_clean:
            inputs, labels = inputs.to(device), labels.to(device)
            if OPTION in ["float16", "16x8"] and device.type == "cuda":
                inputs = inputs.half()
            logits = quantized_model(inputs)
            total_loss_c += criterion(logits, labels).item()
            total_samples_c += labels.size(0)

            pred_c.append(torch.argmax(logits, dim=1).cpu().numpy())
            true_c.append(labels.cpu().numpy())
    cacc = clean_accuracy(np.concatenate(pred_c), np.concatenate(true_c))
    avg_loss_clean = total_loss_c / total_samples_c

    true_b, pred_b, ori_b = [], [], []
    total_loss_b = 0.0
    total_samples_b = 0
    with torch.no_grad():
        for inputs, labels, *other_info in loader_bd:
            inputs, labels = inputs.to(device), labels.to(device)
            if OPTION in ["float16", "16x8"] and device.type == "cuda":
                inputs = inputs.half()
            logits = quantized_model(inputs)
            total_loss_b += criterion(logits, labels).item()
            total_samples_b += labels.size(0)

            pred_b.append(torch.argmax(logits, dim=1).cpu().numpy())
            true_b.append(labels.cpu().numpy())
            ori_b.append(other_info[2].cpu().numpy())
    asr = attack_success_rate(np.concatenate(pred_b), np.concatenate(true_b))
    ra  = robust_accuracy(np.concatenate(pred_b), np.concatenate(ori_b))
    avg_loss_bd = total_loss_b / total_samples_b

    line = f"OPTION={OPTION}\nC-Acc: {cacc:.4f}\nC-Loss: {avg_loss_clean:.4f}\nASR: {asr:.4f}\nRA: {ra:.4f}\nBD-Loss: {avg_loss_bd:.4f}"
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / f"{record_dir.name}_{OPTION}_out.txt"
    with open(outfile, "w") as fil:
        fil.write(line)

if __name__ == "__main__":
    main()
