# Deblur2
A Deep learning model for image deblurring using ResUNet with attention modules and trajectory guidance. The model proposed is built upon a GAN framework, consisting of a trajectory-guided U-Net generator (TraUNetGenerator) and a multi-scale and multi-branch discriminator (MultiScaleDiscriminator). The generator then predicts per-pixel motion vectors and blur confidence maps to geometrically pre-correct the blurry input before residual refinement, leading to a pixel-accurate and perceptually sharp restored image. The discriminator measures realism in two complementary manners: a patch-based branch at full, half and quarter spatial resolutions measure local texture fidelity; and a frequency domain branch on FFT magnitude and phase measures missing high frequency content suppressed by blur.

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Azares-hara/Deblur2.git
cd Deblur2
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

This project provides a deep learning pipeline for image deblurring using PyTorch with custom loss functions and attention modules. The workflow is designed for Kaggle/Colab environments and includes:

- Environment setup with CUDA-enabled PyTorch and image quality libraries
- Synthetic blur generation from GOPRO dataset for training
- Modular training scripts supporting ResUNetGenerator and TraUNetGenerator
- Integration of advanced losses (Frequency, LPIPS, Neighbor Similarity, Adversarial)
- Experiment management with checkpoint resume, TensorBoard logging, and result saving
- Utilities for inference on large images using patch-based tiling

The repository demonstrates staged experimentation with multiple architectures, loss schedules, and evaluation metrics (PSNR, SSIM, LPIPS, NIQE, BRISQUE).

```bash
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118
```

# Training (minimal run)
```bash
# Train ResUNetGenerator for 10 epochs
python main.py --model ResUNetGenerator \
    --batch_size 4 --end_epoch 10 \
    --data_root /kaggle/input/GOPRO_Large
```

## Results
<img width="598" height="516" alt="Screenshot 2026-09-11 184644" src="https://github.com/user-attachments/assets/4e38da72-399b-45a8-84b0-b5580c5d3d94" />

<img width="576" height="480" alt="Screenshot 2026-09-11 184700" src="https://github.com/user-attachments/assets/7179d868-c698-48b5-acc4-c3ef406ecc41" />

## Credits
Based on GOPRO dataset and inspired by recent blind deblurring research.
