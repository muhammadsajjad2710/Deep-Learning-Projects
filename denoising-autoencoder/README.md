# Denoising Autoencoder

Remove noise from images with a convolutional autoencoder.

Runs locally with PyTorch on CPU. No paid APIs and no dataset downloads.

## Train

```powershell
cd D:\AI-Agents\Deep-Learning-Projects\denoising-autoencoder
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train
```

## Tests

```powershell
python -m pytest -q
```
