# Carpet Anomaly Detection

A deep learning project for detecting and localizing defects in carpet images. Models were trained using defect-free images and evaluated on images containing both good carpets and several defect categories.

> Project report: https://drive.google.com/file/d/1H4j4f1DrkYIWc6MOrpVgAXiAavpGfH9K/view?usp=sharing
> Code status:The report contains script listings in its appendix. Upload the actual, cleaned, runnable source files only after checking them; do not present this README as evidence that code has been packaged or verified.

## Problem

Manual inspection of industrial textures can be slow and inconsistent. This project explores automated anomaly detection for carpet inspection, including image-level defect classification and pixel-level heatmap visualization.

## Dataset and setup

The report describes a dataset with 280 good images for training. Its test split contains 28 good images and 106 defective images across five categories:

| Test category | Images |
|---|---:|
| Good | 28 |
| Cut | 17 |
| Hole | 17 |
| Metal contamination | 17 |
| Thread | 19 |
| Color | 19 |

Images were 1024 × 1024 pixels. Training used only good carpet images; defective images were reserved for testing. The report does not identify a public dataset URL or its redistribution terms, so this repository should not include the dataset unless you have permission to share it.

## Methods

The report compares these anomaly-detection approaches:

- **PaDiM** — ResNet-50 features and a Gaussian model of normal feature distributions.
- **SPADE** — WideResNet-50 backbone.
- **FastFlow** — EfficientNet-B4 backbone and normalizing flows.
- **EfficientAD** — EfficientNetV2-S backbone.
- **PatchCore** — ResNet-50 features and a memory bank of normal patches.

The report also includes **CAE** and **VAE** experiments using custom convolutional networks. These are shown separately in the results table because the report's stated primary comparison focuses on the five methods above.

The images were initially resized to 256 × 256 for early experiments. The report says the final experiments used the original 1024 × 1024 resolution to retain fine texture details and improve heatmap localization.

## Reported image-level results

| Model | Backbone | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---:|---:|---:|---:|---:|
| PaDiM | ResNet-50 | 83.76% | 0.88 | 0.84 | 0.85 | 0.90 |
| SPADE | WideResNet-50 | 77.77% | 0.82 | 0.78 | 0.79 | 0.83 |
| FastFlow | EfficientNet-B4 | 78.63% | 0.79 | 0.79 | 0.72 | 0.74 |
| EfficientAD | EfficientNetV2-S | 75.21% | 0.58 | 0.75 | 0.65 | 0.70 |
| PatchCore | ResNet-50 | 76.06% | 0.70 | 0.76 | 0.67 | 0.73 |
| CAE | Custom ConvNet | 75.21% | 0.58 | 0.75 | 0.65 | 0.56 |
| VAE | Custom ConvNet | 75.21% | 0.58 | 0.75 | 0.65 | 0.29 |

**Evaluation note:** The report describes choosing the anomaly threshold to maximize accuracy on the test set, and reports those results. The threshold and accuracy are therefore based on the labeled test set; these figures should not be described as an unbiased held-out estimate of generalization. A stronger follow-up evaluation would choose the threshold using a separate validation set and report final metrics on an untouched test set.

The report identifies PaDiM as the best overall result in this experiment. Its table reports 83.76% accuracy and 0.90 AUC. These values are specific to the report's dataset and evaluation procedure.

## Heatmaps

The project generated anomaly maps and image overlays to visualize locations with higher anomaly scores. The report describes maps normalized to [0, 1], colored with the JET colormap, and blended with the original image. Add a few representative, non-sensitive examples under `results/` and link them here after confirming they are the final outputs from the project.

<!-- Example after adding an image:
![Example anomaly heatmap](results/heatmaps/padim_example.png)
-->

## Repository layout

Suggested layout (include only files you actually have):

```text
carpet-anomaly-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── report/
│   └── DLVS_SA_Report.pdf
├── src/
│   ├── padim.py
│   ├── spade.py
│   ├── fastflow.py
│   ├── efficientad.py
│   └── patchcore.py
└── results/
    ├── heatmaps/
    ├── roc_curves/
    └── reports/
```

The filenames above are examples for organizing your own source files; the report does not provide verified standalone files with these names. Add CAE/VAE source files only if you have the actual scripts.

## Files to upload

- **Report:** Upload the final PDF as `report/DLVS_SA_Report.pdf` if you are allowed to share it publicly. Check that it contains no private information or restricted material.
- **Code:** Prefer the original scripts used to generate the reported results. The PDF appendix has code listings, but PDF text extraction can disrupt indentation and line breaks. Do not copy those listings into `.py` files without checking and correcting them, and do not claim the scripts run until you have verified them.
- **Results:** Select a small set of representative heatmaps/overlays and the report's ROC curves or classification reports, if available as separate files. Keep their labels and model association clear. Avoid uploading every generated image unless useful.
- **Dataset:** Do not upload the dataset unless you own it or have explicit permission to redistribute it. The README records the split counts from the report without including image data.
- **Model weights:** Upload `.pth` or `.pkl` files only if you have them, can share them, and have checked their size and licensing constraints. They are optional.

## Dependencies

A `requirements.txt` is supplied with package names inferred from the appendix imports. The report does not state package versions or a verified Python version, so versions are intentionally unpinned. The actual environment may require additional compatible packages depending on the final scripts and hardware.

Install the listed packages with:

```bash
python -m pip install -r requirements.txt
```

The report's scripts use PyTorch and torchvision, NumPy, OpenCV, Pillow, scikit-learn, Matplotlib, tqdm, and joblib. CUDA availability depends on your machine and matching PyTorch installation.

## Limitations and next steps

- Test-set threshold selection can make reported classification accuracy optimistic; use validation data for threshold selection in a future evaluation.
- Verify and clean the original scripts before presenting them as reproducible code.
- The report proposes improving the CAE/VAE experiments with pretrained encoders, full-resolution or multi-scale inputs, improved loss functions, regularization, augmentation, adaptive thresholding, and post-processing.
- Future directions listed in the report include real-time industrial integration, other surfaces, transformer-based anomaly detection, and edge deployment.

## Acknowledgment

This project was submitted as a special assignment for **Deep Learning for Vision Systems** at the Institute of Technology, Nirma University, in November 2025.
