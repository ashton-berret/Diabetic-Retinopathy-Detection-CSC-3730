Diabetic retinopathy, a complication associated with diabetes, affects the eys and can lead to blindness if not diagnosed. This condition rsults from damage to the blood vessels inside the retina. Diabetic retinopathy is a leading cause of blindness among adults.

The goal of this project is to implement a machine learning model that can accurately predict the presence of diabetic retinopathy using retinal images.We will be using the [diabetic retinopathy](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized/) dataset from kaggle, consisting of over 35,000 1024x1024 retinal scans. Below is an example of an individual retinal scan.

# How to install?


I highly recommend using a python environment solution like [miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)

This guide will presume you're using a Linux or Mac OS.

We will first need to create a conda environment: using `conda create <env name>`.

Activate the conda environment with `conda activate <env name>`.

We will not install our packages, first run `conda install tensorflow-gpu`, secondly run `conda install jupyter pandas=1.5.2 numpy=1.20.3 matplotlib scikit-learn h5py`.

We will be using [this dataset](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized/). The jupyter notebook in this repo is formatted with the data in the following directory setup:

```
├── dataset
│   ├── resized.h5
│   ├── resized_train
│   ├── resized_train_cropped
│   ├── trainLabels_cropped.csv
│   └── trainLabels.csv
└── Project.ipynb
```