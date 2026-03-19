# Advanced CNN to Classify Minecraft Images
Khushi Patel, Leland Weeks, Sriram Suvaidhar \
CS 615 Final Project \
March 19, 2026

Classify Minecraft block screenshots into 60 block types \
Uses the **MiDaS** dataset (36,000 images, Minecraft v1.17.1) \
Trains a custom CNN from scratch — no pretrained weights


## Setup

We recommend using Google Collab to train and evaluate our model \
https://colab.research.google.com/drive/1NlGioW1rDLuqBWLtrcxHjDnmxogNDY29?usp=sharing


## Usage

**Train with default settings (1 kernel, 3x3 kernel size, 1 layer, 50 epochs, 0.001 learning rate):** \
`python3 src/main.py --mode train`

**Evaluate with default settings (1 kernel, 3x3 kernel size, 1 layer):** \
`python3 src/main.py --mode evaluate`

**Recreate our best model:** \
`python3 src/main.py --mode train --kernels 32 --kernel_size 3 --layers 1 --epochs 200 --lr 0.001`

**Evaluate our best model:** \
`python3 src/main.py --mode evaluate --kernels 32 --kernel_size 3 --layers 1`

The `--kernels`, `--kernel_size`, and `--layers` flags determine which saved model file is loaded. Use the same values for training and evaluation.

The `--epochs` and `--lr`, flags are only needed for training and do not affect file names.


## Training Output

- Models saved to `models/`
- Loss plots saved to `plots/`


## Example Training Output

```
Testing the Training Pipeline...
Using device: cuda
Mode: train, Color: True, Kernels: 32, Kernel Size: 3, Layers: 1
Model Filename:  model_color_32k_3x3_1l.pt
Loss Plot Filename:  plot_color_32k_3x3_1l.png
Confusion Matrix Filename:  cm_color_32k_3x3_1l.png
Setting up the data loaders...
Number of training samples: 32400
Number of test samples: 3600
Training with pareamters: Epochs: 200, Learning Rate: 0.001
Epoch [1/200], Loss: 9.1922, Time: 31.72s
...
Epoch [200/200], Loss: 0.0001, Time: 28.07s
```

## Evaluation Output

- Heatmaps saved to `plots/`

## Example Evaluation Output
```
Testing the Evaluation Pipeline...
Using device: cuda
Mode: evaluate, Color: True, Kernels: 32, Kernel Size: 3, Layers: 1
Model Filename:  model_color_32k_3x3_1l.pt
Loss Plot Filename:  plot_color_32k_3x3_1l.png
Confusion Matrix Filename:  cm_color_32k_3x3_1l.png
Setting up the data loaders...
Number of training samples: 32400
Number of test samples: 3600
Final Test Accuracy: 48.81%
```

## Project Structure

```
project/
├── csv/
├── images/
├── models/
├── notebooks/
├── plots/
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── main.py
└── README.md
```
