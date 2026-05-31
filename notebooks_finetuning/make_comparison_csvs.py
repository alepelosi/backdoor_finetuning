import argparse
import csv
import json
import re
from pathlib import Path


ATTACKS = [
    "badnet",
    "blended",
    "bpp",
    "inputaware",
    "lf",
    "sig",
    "ssba",
    "wanet",
]

FIELDNAMES = [
    "poison_percent",
    "poison_fraction",
    "attack",
    "method",
    "method_detail",
    "result_file",
    "clean_acc_before",
    "asr_before",
    "ra_before",
    "clean_acc_after",
    "asr_after",
    "ra_after",
    "clean_acc_delta",
    "asr_delta",
    "persistence_ratio",
    "backdoor_present_after",
    "setting_name",
    "setting_value",
    "size_ratio_after",
    "extra_metric_name",
    "extra_metric_value",
    "source_file",
]

METHOD_ORDER = {
    "finetuning": 0,
    "conv_lora": 1,
    "distillation": 2,
    "pruning": 3,
    "quantization": 4,
}


def default_results_root():
    return Path(__file__).resolve().parents[2] / "results"


def as_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clean_name(path_or_name):
    name = Path(str(path_or_name)).name
    return re.sub(r"-\\d{8}T\\d{6}Z-\\d+-\\d+$", "", name)


def infer_attack(text):
    lowered = str(text).lower()
    for attack in ATTACKS:
        if re.search(rf"(^|[^a-z0-9]){re.escape(attack)}([^a-z0-9]|$)", lowered):
            return attack
    return ""


def infer_poison_fraction(path_or_text):
    text = str(path_or_text).lower().replace(".", "_")
    parts = [str(part).lower() for part in Path(str(path_or_text)).parts]
    if "output_finetuning_10" in parts or "0.1" in parts:
        return "0.1"
    if "output_finetuning_1" in parts or "0.01" in parts:
        return "0.01"
    if "0_01" in text or "poison_01" in text:
        return "0.01"
    if "0_1" in text or "poison_1" in text:
        return "0.1"
    return ""


def poison_percent(poison_fraction):
    if poison_fraction == "0.01":
        return "1"
    if poison_fraction == "0.1":
        return "10"
    return ""


def bool_from_asr(asr, threshold):
    value = as_float(asr)
    if value is None:
        return ""
    return value >= threshold


def ratio(after, before):
    after_value = as_float(after)
    before_value = as_float(before)
    if after_value is None or before_value in (None, 0.0):
        return ""
    return after_value / before_value


def delta(after, before):
    after_value = as_float(after)
    before_value = as_float(before)
    if after_value is None or before_value is None:
        return ""
    return after_value - before_value


def base_row(path, method, method_detail, result_file, threshold):
    poison_fraction = infer_poison_fraction(path)
    return {
        "poison_percent": poison_percent(poison_fraction),
        "poison_fraction": poison_fraction,
        "attack": infer_attack(result_file or path),
        "method": method,
        "method_detail": method_detail,
        "result_file": result_file or clean_name(path.parent),
        "clean_acc_before": "",
        "asr_before": "",
        "ra_before": "",
        "clean_acc_after": "",
        "asr_after": "",
        "ra_after": "",
        "clean_acc_delta": "",
        "asr_delta": "",
        "persistence_ratio": "",
        "backdoor_present_after": "",
        "setting_name": "",
        "setting_value": "",
        "size_ratio_after": "",
        "extra_metric_name": "",
        "extra_metric_value": "",
        "source_file": str(path),
    }


def finalize_row(row, threshold):
    row["clean_acc_delta"] = delta(row["clean_acc_after"], row["clean_acc_before"])
    row["asr_delta"] = delta(row["asr_after"], row["asr_before"])
    row["persistence_ratio"] = row["persistence_ratio"] or ratio(row["asr_after"], row["asr_before"])
    row["backdoor_present_after"] = row["backdoor_present_after"] or bool_from_asr(row["asr_after"], threshold)
    if not row["attack"]:
        row["attack"] = infer_attack(row["result_file"])
    if not row["poison_fraction"]:
        row["poison_fraction"] = infer_poison_fraction(row["result_file"])
        row["poison_percent"] = poison_percent(row["poison_fraction"])
    return row


def load_json(path):
    return json.loads(path.read_text())


def collect_distillation(results_root, threshold):
    rows = []
    for path in sorted((results_root / "Distillation").glob("*/distillation_metrics.json")):
        data = load_json(path)
        result_file = data.get("run_name") or clean_name(path.parent)
        teacher = data.get("teacher", {})
        student = data.get("student_final", {})
        row = base_row(path, "distillation", "student_final", result_file, threshold)
        row.update({
            "clean_acc_before": teacher.get("clean_acc"),
            "asr_before": teacher.get("asr"),
            "ra_before": teacher.get("ra"),
            "clean_acc_after": student.get("clean_acc"),
            "asr_after": student.get("asr"),
            "ra_after": student.get("ra"),
            "setting_name": "epochs",
            "setting_value": data.get("settings", {}).get("epochs"),
        })
        rows.append(finalize_row(row, threshold))
    return rows


def collect_finetuning(results_root, threshold):
    rows = []
    for path in sorted((results_root / "finetuning").glob("output_finetuning_*/*/finetuning_metrics.json")):
        data = load_json(path)
        result_file = data.get("run_name") or clean_name(path.parent)
        baseline = data.get("baseline", {})
        final = data.get("final", {})
        row = base_row(path, "finetuning", "full_finetune", result_file, threshold)
        row.update({
            "clean_acc_before": baseline.get("ACC"),
            "asr_before": baseline.get("ASR"),
            "ra_before": baseline.get("RA"),
            "clean_acc_after": final.get("ACC"),
            "asr_after": final.get("ASR"),
            "ra_after": final.get("RA"),
            "setting_name": "epochs",
            "setting_value": data.get("settings", {}).get("epochs") or final.get("epoch"),
        })
        rows.append(finalize_row(row, threshold))
    return rows


def collect_lora(results_root, threshold):
    rows = []
    for path in sorted((results_root / "LORA").glob("*/*/conv_lora_summary.json")):
        data = load_json(path)
        result_file = data.get("run_name") or clean_name(path.parent)
        row = base_row(path, "conv_lora", "conv_lora", result_file, threshold)
        row.update({
            "clean_acc_before": data.get("before_clean_acc"),
            "asr_before": data.get("before_asr"),
            "clean_acc_after": data.get("final_clean_acc"),
            "asr_after": data.get("final_asr"),
            "setting_name": "epochs",
            "setting_value": data.get("epochs"),
        })
        rows.append(finalize_row(row, threshold))
    return rows


def collect_pruning(results_root, threshold):
    rows = []
    for path in sorted((results_root / "Compression (pruning)").glob("*.csv")):
        with path.open(newline="") as handle:
            for source_row in csv.DictReader(handle):
                if not any(source_row.values()):
                    continue
                result_file = source_row.get("result_file") or clean_name(path)
                mode = source_row.get("compression_mode") or ""
                amount = source_row.get("amount") or ""
                detail = mode if not amount else f"{mode}_{amount}"
                row = base_row(path, "pruning", detail, result_file, threshold)
                row.update({
                    "clean_acc_before": source_row.get("clean_acc_before"),
                    "asr_before": source_row.get("asr_before"),
                    "clean_acc_after": source_row.get("clean_acc_after"),
                    "asr_after": source_row.get("asr_after"),
                    "persistence_ratio": source_row.get("persistence_ratio"),
                    "backdoor_present_after": source_row.get("backdoor_present_after"),
                    "setting_name": "amount" if amount else "",
                    "setting_value": amount,
                    "size_ratio_after": source_row.get("state_size_ratio"),
                    "extra_metric_name": "zero_fraction_after",
                    "extra_metric_value": source_row.get("zero_fraction_after"),
                })
                rows.append(finalize_row(row, threshold))
    return rows


def collect_quantization(results_root, threshold):
    rows = []
    for path in sorted((results_root / "Quantization").glob("*.csv")):
        with path.open(newline="") as handle:
            for source_row in csv.DictReader(handle):
                if not any(source_row.values()):
                    continue
                result_file = source_row.get("result_file") or clean_name(path)
                mode = source_row.get("quantization_mode") or ""
                bits = source_row.get("bits") or ""
                detail = mode if not bits else f"{mode}_{bits}bit"
                row = base_row(path, "quantization", detail, result_file, threshold)
                row.update({
                    "clean_acc_before": source_row.get("clean_acc_before"),
                    "asr_before": source_row.get("asr_before"),
                    "clean_acc_after": source_row.get("clean_acc_after"),
                    "asr_after": source_row.get("asr_after"),
                    "persistence_ratio": source_row.get("persistence_ratio"),
                    "backdoor_present_after": source_row.get("backdoor_present_after"),
                    "setting_name": "bits" if bits else "",
                    "setting_value": bits,
                    "size_ratio_after": source_row.get("estimated_weight_size_ratio"),
                    "extra_metric_name": "mean_abs_weight_error",
                    "extra_metric_value": source_row.get("mean_abs_weight_error"),
                })
                rows.append(finalize_row(row, threshold))
    return rows


def row_sort_key(row):
    attack_index = ATTACKS.index(row["attack"]) if row["attack"] in ATTACKS else len(ATTACKS)
    return (
        attack_index,
        row["attack"],
        METHOD_ORDER.get(row["method"], 99),
        row["method_detail"],
        str(row["setting_value"]),
    )


def collect_rows(results_root, threshold):
    rows = []
    rows.extend(collect_finetuning(results_root, threshold))
    rows.extend(collect_lora(results_root, threshold))
    rows.extend(collect_distillation(results_root, threshold))
    rows.extend(collect_pruning(results_root, threshold))
    rows.extend(collect_quantization(results_root, threshold))
    return [row for row in rows if row["poison_percent"] in {"1", "10"}]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=default_results_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    results_root = args.results_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else results_root
    rows = collect_rows(results_root, args.threshold)

    one_percent = sorted([row for row in rows if row["poison_percent"] == "1"], key=row_sort_key)
    ten_percent = sorted([row for row in rows if row["poison_percent"] == "10"], key=row_sort_key)

    out_1 = output_dir / "comparison_poison_1_percent.csv"
    out_10 = output_dir / "comparison_poison_10_percent.csv"
    write_csv(out_1, one_percent)
    write_csv(out_10, ten_percent)

    print(f"Wrote {len(one_percent)} rows to {out_1}")
    print(f"Wrote {len(ten_percent)} rows to {out_10}")


if __name__ == "__main__":
    main()
