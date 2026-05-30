import copy
import json
from pathlib import Path
import torch

from compression_eval_utils import (
    add_common_args,
    apply_compression,
    argparse,
    cpu_state_dict,
    evaluate_model,
    load_model_and_datasets,
    record_dir_for,
    resolve_device,
    serialized_state_size,
    write_json_result,
    zero_fraction,
)


def amount_label(amount):
    return str(amount).replace(".", "_")


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--mode", required=True, choices=["magnitude_prune", "channel_prune"])
    parser.add_argument("--amount", required=True, type=float)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_save_model", action="store_true")
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    model, clean_dataset, bd_dataset = load_model_and_datasets(args.result_file)
    before_model = copy.deepcopy(model)
    before = evaluate_model(before_model, clean_dataset, bd_dataset, args)
    before_size = serialized_state_size(model)
    before_zero_fraction = zero_fraction(model)

    apply_compression(model, args.mode, args.amount)

    after = evaluate_model(model, clean_dataset, bd_dataset, args)
    after_size = serialized_state_size(model)
    after_zero_fraction = zero_fraction(model)

    result_dir = Path(args.output_dir) if args.output_dir else record_dir_for(args.result_file)
    result_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = result_dir / f"compression_{args.mode}_{amount_label(args.amount)}.json"
    defense_path = result_dir / f"defense_result_{args.mode}_{amount_label(args.amount)}.pt"

    result = {
        "result_file": args.result_file,
        "mode": args.mode,
        "amount": args.amount,
        "max_samples": args.max_samples,
        "before": before,
        "after": after,
        "delta": {
            "clean_acc": after["clean_acc"] - before["clean_acc"],
            "asr": after["asr"] - before["asr"],
        },
        "persistence_ratio": after["asr"] / before["asr"] if before["asr"] else None,
        "parameter_sparsity": {
            "before": {"zero_fraction": before_zero_fraction},
            "after": {"zero_fraction": after_zero_fraction},
        },
        "serialized_state_size_bytes": {
            "before": before_size,
            "after": after_size,
            "ratio": after_size / before_size if before_size else None,
        },
        "save_path": str(metrics_path),
        "defense_result_path": None if args.no_save_model else str(defense_path),
    }

    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not args.no_save_model:
        torch.save({"model": cpu_state_dict(model)}, defense_path)

    write_json_result(result, args.json)


if __name__ == "__main__":
    main()
