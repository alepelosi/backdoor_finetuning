# Security and Privacy of ML: Backdoor Fine-Tuning Experiments

This repository contains the code and notebooks used for our Security and Privacy of Machine Learning project on backdoor behavior under fine-tuning and model compression techniques.

The project is based on BackdoorBench-style CIFAR-10 attack checkpoints. We evaluate whether backdoors remain active after applying several post-training or fine-tuning techniques, including:

- standard fine-tuning sweeps
- Conv-LoRA fine-tuning
- knowledge distillation
- pruning
- simulated weight quantization

## Model Files

The model archives used in our experiments are available here:

[Google Drive model folder](https://drive.google.com/drive/folders/17BJAKc7mx77sGd3LtQNHtTrUAss7MR1x?usp=sharing)

Download the relevant `.zip` files and place them in the corresponding notebook workspace under `notebooks_finetuning/`, for example:

```text
notebooks_finetuning/pruning/
notebooks_finetuning/quantization/
notebooks_finetuning/distillation/
notebooks_finetuning/convlora/
```

You can also place model zips directly inside `notebooks_finetuning/`; the local notebooks search there as a fallback.

Generated data, extracted archives, outputs, and checkpoints are ignored by Git.

## Requirements

Create and activate a Python environment, then install the packages needed by the notebooks:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy pillow tqdm pyyaml pandas matplotlib jupyter ipykernel
python -m ipykernel install --user --name backdoor-finetuning --display-name "Backdoor Fine-Tuning"
```

On Apple Silicon, PyTorch should use MPS when available. The notebooks set:

```python
PYTORCH_ENABLE_MPS_FALLBACK=1
```

so unsupported MPS operations can fall back to CPU.

## Notebooks

All local notebooks are in:

```text
notebooks_finetuning/
```

They use relative paths and automatically locate the cloned repository root. Start Jupyter from either the repository root or from `notebooks_finetuning/`.

### `pruning.ipynb`

Runs pruning-based compression checks on a selected backdoored model.

It evaluates:

- baseline clean accuracy and ASR
- magnitude pruning at multiple sparsity levels
- channel pruning at multiple sparsity levels

Outputs are written to:

```text
notebooks_finetuning/pruning/outputs/
```

### `quantization.ipynb`

Runs simulated post-training weight quantization on all selected CIFAR-10 model zips by default.

It evaluates:

- baseline clean accuracy and ASR
- weight quantization at 8, 6, 4, 3, and 2 bits
- whether the backdoor remains present after quantization

This is simulated weight-only quantization: Conv2d and Linear weights are rounded to lower precision and then evaluated in floating point. It does not implement full deployment int8 inference with activation quantization.

Outputs are written to:

```text
notebooks_finetuning/quantization/outputs/
```

### `distillation.ipynb`

Runs knowledge distillation from a backdoored teacher model into a student model.

It evaluates whether the student retains:

- clean accuracy
- attack success rate
- backdoor behavior from the teacher

Outputs are written to:

```text
notebooks_finetuning/distillation/outputs/
```

### `conv_lora.ipynb`

Runs Conv-LoRA fine-tuning on a selected backdoored model.

It evaluates:

- clean accuracy before and after fine-tuning
- ASR before and after fine-tuning
- whether Conv-LoRA adaptation weakens or preserves the backdoor

Outputs are written to:

```text
notebooks_finetuning/convlora/outputs/
```

### `finetuning_sweep_colab.ipynb`

Google Colab notebook for running a fine-tuning sweep.

This notebook intentionally contains Colab-specific paths such as `/content/...` and Google Drive mounting code. Use it in Colab rather than as a local notebook.

## Helper Scripts

The pruning and quantization notebooks call small helper scripts in `defense/`:

```text
defense/check_backdoor.py
defense/compress_and_check.py
defense/quantize_and_check.py
defense/compression_eval_utils.py
```

These scripts load BackdoorBench attack results, evaluate clean accuracy and ASR, and write JSON metrics consumed by the notebooks.

## Running A Local Notebook

1. Clone the repository.
2. Install the requirements above.
3. Download the desired model zips from the Google Drive folder.
4. Put the zips in the matching folder under `notebooks_finetuning/`.
5. Start Jupyter:

```bash
jupyter notebook
```

6. Open the notebook you want to run.
7. Select the `Backdoor Fine-Tuning` kernel.
8. Adjust the first configuration cell if needed.
9. Run the notebook from top to bottom.

Most notebooks write results under their own `outputs/` folder. These outputs are ignored by Git.

## Notes

- Clean accuracy measures normal CIFAR-10 classification performance.
- ASR means attack success rate.
- A backdoor is considered present when ASR is above the configured threshold.
- Extreme compression, such as 2-bit quantization, can destroy the whole classifier. In that case a lower ASR does not necessarily mean a useful defense, because clean accuracy may also collapse.

