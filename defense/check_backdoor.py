from compression_eval_utils import (
    add_common_args,
    argparse,
    evaluate_model,
    load_model_and_datasets,
    resolve_device,
    write_json_result,
)


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    model, clean_dataset, bd_dataset = load_model_and_datasets(args.result_file)
    result = evaluate_model(model, clean_dataset, bd_dataset, args)
    write_json_result(result, args.json)


if __name__ == "__main__":
    main()
