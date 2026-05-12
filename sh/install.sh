#!/usr/bin/env sh
set -e

if [ "$(uname)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    echo "Detected macOS arm64; installing Apple Silicon-compatible dependencies."
    python -m pip install --upgrade pip wheel setuptools
    python -m pip install \
        torch==2.4.1 \
        torchvision==0.19.1 \
        keras==2.7.0 \
        opencv-python==4.5.5.64 \
        pandas==1.3.1 \
        Pillow==9.5.0 \
        scikit-learn==1.1.3 \
        scikit-image==0.21.0 \
        tqdm==4.61.0 \
        PyYAML==6.0.1 \
        protobuf==3.19.6 \
        tensorboard==2.7.0 \
        kornia==0.5.0 \
        imageio==2.35.1 \
        matplotlib==3.5.1 \
        scipy==1.10.1 \
        pytorch-wavelet

    # for visualization only
    python -m pip install \
        seaborn==0.11.2 \
        shap==0.40.0 \
        grad-cam==1.3.9 \
        omnixai==1.2.3 \
        plotly==5.11.0 \
        umap-learn==0.5.3 \
        graphviz \
        hiddenlayer==0.3 \
        PyHessian==0.1 \
        "torchmetrics[image]==1.5.2" \
        pytorch-wavelets
    python -m pip install -U git+https://github.com/szagoruyko/pytorchviz.git@master
    exit 0
fi

python -m pip install torch==1.11+cu113 torchvision==0.12 torchaudio==0.11.0 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install keras==2.7.0
python -m pip install opencv-python==4.5.5.64
python -m pip install pandas==1.3.1
python -m pip install Pillow==8.2.0
python -m pip install scikit-learn==0.24.2
python -m pip install scikit-image==0.18.1
python -m pip install tqdm==4.61.0
python -m pip install pyyaml==5.4.1
python -m pip install tensorboard==2.7.0
python -m pip install Kornia==0.5.0
python -m pip install imageio==2.18.0
python -m pip install matplotlib==3.5.1
python -m pip install scipy==1.3.1
python -m pip install pytorch-wavelet

# for visualization only
python -m pip install seaborn==0.11.2
## Shapely Value
python -m pip install shap==0.40.0
## Grad-CAM
python -m pip install grad-cam==1.3.9
## Feature Map & Feature Visualization
python -m pip install omnixai==1.2.3
python -m pip install plotly==5.11.0
## UMAP
python -m pip install umap-learn==0.5.3
## Network Structure
python -m pip install graphviz
python -m pip install hiddenlayer==0.3
python -m pip install -U git+https://github.com/szagoruyko/pytorchviz.git@master

## Landscape
python -m pip install PyHessian==0.1

## Quality
python -m pip install "torchmetrics[image]"

python -m pip install pytorch-wavelets
