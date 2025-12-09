# Image Colorization Using U-Net + Perceptual Loss

## Overview

This project implements **automatic image colorization** using a **U-Net architecture** trained on the **Labeled Faces in the Wild (LFW)** dataset.
The model takes **grayscale images** and predicts their **colorized RGB versions** using a combination of:

* **Pixel-wise L1 loss**
* **Perceptual loss using VGG16 feature maps**

This hybrid loss encourages realistic textures and more natural colors compared to pixel-only approaches.

---

## Key Features

* U-Net model with dynamic input support
* Perceptual loss leveraging pretrained VGG16
* Training pipeline using TensorFlow & TensorFlow Datasets
* Visualization of predictions during training
* Colorization of unseen grayscale images
* Model checkpoint saving for future inference

---

## Installation

Before running the notebook, install the required packages:

```bash
pip install tensorflow tensorflow_datasets opencv-python-headless matplotlib
```

---

## Model Architecture

The model uses a **U-Net encoder–decoder** structure with:

* Downsampling via strided convolutions
* Upsampling via transpose convolutions
* Skip connections for better spatial feature retention
* Output layer predicting 3-channel RGB images

The addition of **perceptual loss** improves realism by comparing deep feature representations instead of raw pixels alone.

---

## Dataset

We use the **LFW (Labeled Faces in the Wild)** dataset via TensorFlow Datasets:

* Images are resized to **128×128**
* Converted to LAB format internally (L for grayscale input)
* Model learns to predict the **AB color channels**

---

## Training

The training loop:

1. Loads and batches LFW images
2. Generates grayscale/color pairs
3. Feeds grayscale images into U-Net
4. Computes:

   * L1 pixel loss
   * Perceptual loss via VGG16
5. Backpropagates gradients using Adam optimizer
6. Periodically visualizes outputs

---

## Results & Visualization

The notebook includes code to:

* Display grayscale → predicted color → ground truth triplets
* Test the trained model on **unseen external images**
* Save final outputs

---

## Testing on New Images

You can load any local grayscale image and run:

```python
pred = model.predict(grayscale_image)
plt.imshow(pred)
```

The model will generate a fully colorized version.

---

## Saving the Model

The notebook also supports saving:

```python
model.save('colorizer_dynamic_unet.h5')
```

This enables easy reloading for inference without retraining.

---

## Authors

**Developed by:**

* **Aldric Pinto**
* **Trevor Hitchcock**

Deep Learning, University of New Haven

Fall 2025
