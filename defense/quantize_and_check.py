import copy
import json
from pathlib import Path

import torch
import torch.nn as nn

from compression_eval_utils import (
    add_common_args,
    argparse,
    cpu_state_dict,
    evaluate_model,
    load_model_and_datasets,
    record_dir_for,
    resolve_device,
    serialized_state_size,
    write_json_result,
)


def quantize_tensor_symmetric(tensor, bits):
    if bits >= 32:
        return tensor.detach().clone()
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    max_abs = tensor.detach().abs().max()
    if max_abs == 0:
        return tensor.detach().clone()
    scale = max_abs / qmax
    quantized = torch.clamp(torch.round(tensor.detach() / scale), qmin, qmax)
    return quantized * scale


def quantize_model_weights(model, bits):
    quantized_params = 0
    total_float_params = 0
    weighted_abs_error = 0.0

    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            original = module.weight.detach().clone()
            quantized = quantize_tensor_symmetric(original, bits)
            module.weight.copy_(quantized)

            numel = original.numel()
            quantized_params += numel
            weighted_abs_error += (quantized - original).abs().mean().item() * numel

        for tensor in model.state_dict().values():
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                total_float_params += tensor.numel()

    return {
        "quantized_parameter_fraction": quantized_params / total_float_params if total_float_params else 0.0,
        "mean_abs_weight_error": weighted_abs_error / quantized_params if quantized_params else 0.0,
        "estimated_weight_size_ratio": bits / 32,
    }


def bits_label(bits):
    return f"{bits}bit"


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--mode", default="weight_quantize", choices=["weight_quantize"])
    parser.add_argument("--bits", required=True, type=int)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_save_model", action="store_true")
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    model, clean_dataset, bd_dataset = load_model_and_datasets(args.result_file)
    before_model = copy.deepcopy(model)
    before = evaluate_model(before_model, clean_dataset, bd_dataset, args)
    before_size = serialized_state_size(model)

    quant_stats = quantize_model_weights(model, args.bits)

    after = evaluate_model(model, clean_dataset, bd_dataset, args)
    after_size = serialized_state_size(model)

    result_dir = Path(args.output_dir) if args.output_dir else record_dir_for(args.result_file)
    result_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = result_dir / f"quantization_{args.mode}_{bits_label(args.bits)}.json"
    defense_path = result_dir / f"quantized_result_{args.mode}_{bits_label(args.bits)}.pt"

    result = {
        "result_file": args.result_file,
        "mode": args.mode,
        "bits": args.bits,
        "max_samples": args.max_samples,
        "before": before,
        "after": after,
        "delta": {
            "clean_acc": after["clean_acc"] - before["clean_acc"],
            "asr": after["asr"] - before["asr"],
        },
        "persistence_ratio": after["asr"] / before["asr"] if before["asr"] else None,
        "quantization": quant_stats,
        "serialized_state_size_bytes": {
            "before": before_size,
            "after": after_size,
            "ratio": after_size / before_size if before_size else None,
        },
        "save_path": str(metrics_path),
        "quantized_result_path": None if args.no_save_model else str(defense_path),
    }

    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not args.no_save_model:
        torch.save({"model": cpu_state_dict(model)}, defense_path)

    write_json_result(result, args.json)


if __name__ == "__main__":
    main()
