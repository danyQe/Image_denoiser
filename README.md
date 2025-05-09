# Image Denoising Application

A Streamlit-based application for automatic noise detection and image denoising using pre-trained TensorFlow models.

## Features
- Automatic noise type classification (Gaussian, Speckle, Salt and Pepper, Poisson, Multiplicative, JPEG, Quantization)
- Patch-based denoising for handling large images efficiently
- Manual noise type selection option
- Performance metrics: PSNR and SSIM calculation
- CPU-only execution (no GPU required)

## Prerequisites
- Python 3.11
- Windows 10 (tested) or any OS with Python 3.11
- CPU-only environment

## Installation
1. Clone this repository:
   ```powershell
   git clone https://github.com/your-username/image-denoiser.git
   cd image-denoiser
   ```
2. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install required packages:
   ```powershell
   pip install -r requirements.txt
   ```
   If you don't have a `requirements.txt`, install manually:
   ```powershell
   pip install streamlit opencv-python-headless numpy tensorflow keras keras-hub scikit-image
   ```

## Models
Place the following pre-trained models in the `models/` directory:
- `noise_classifier.keras` — noise detection model
- `ridnet_gaussian.keras` — Gaussian noise denoiser
- `ridnet_speckle.keras` — Speckle noise denoiser
- `ridnet_salt_and_pepper.keras` — Salt and Pepper noise denoiser
- `ridnet_poisson.keras` — Poisson noise denoiser
- `ridnet_multiplicative.keras` — Multiplicative noise denoiser
- `ridnet_jpeg.keras` — JPEG noise denoiser
- `ridnet_quantization.keras` — Quantization noise denoiser

## Usage
1. Run the Streamlit app:
   ```powershell
   streamlit run app.py
   ```
2. In the web interface:
   - Upload an image (JPEG or PNG).
   - (Optional) Check "Manual noise selection" and choose noise types.
   - Click "Denoise Image".
   - View the original and denoised images side by side.
   - Review detected or applied noise types and PSNR/SSIM metrics.

## Project Structure
```
.
├── app.py
├── models/
│   ├── noise_classifier.keras
│   ├── ridnet_gaussian.keras
│   ├── ridnet_speckle.keras
│   └── ...
├── requirements.txt
└── README.md
```

## License
This project is licensed under the MIT License. 