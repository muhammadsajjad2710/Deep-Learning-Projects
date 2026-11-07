# U-Net Image Segmentation

Segment shapes in synthetic images with a U-Net.

Runs locally with PyTorch on CPU. No paid APIs and no dataset downloads.

## Train

```powershell
cd D:\AI-Agents\Deep-Learning-Projects\unet-segmentation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train
```

## Tests

```powershell
python -m pytest -q
```
