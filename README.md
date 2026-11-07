# Deep Learning Projects

Ten classic **deep learning** projects in PyTorch. Each one trains a small model locally, reports a metric, and includes tests. No paid APIs, no GPU, and no dataset downloads.

## Projects

| # | Project | Task |
| --- | --- | --- |
| 1 | [mnist-cnn](mnist-cnn/) | CNN digit classification |
| 2 | [cifar10-cnn](cifar10-cnn/) | CNN color image classification |
| 3 | [fashion-mnist-cnn](fashion-mnist-cnn/) | CNN clothing classification |
| 4 | [lstm-sentiment](lstm-sentiment/) | LSTM review sentiment |
| 5 | [denoising-autoencoder](denoising-autoencoder/) | Convolutional denoising autoencoder |
| 6 | [lstm-forecasting](lstm-forecasting/) | LSTM sequence forecasting |
| 7 | [dcgan-images](dcgan-images/) | DCGAN image generation |
| 8 | [unet-segmentation](unet-segmentation/) | U-Net semantic segmentation |
| 9 | [neural-style-transfer](neural-style-transfer/) | Neural style transfer |
| 10 | [transformer-classifier](transformer-classifier/) | Transformer text classification |

## Quick start

```powershell
cd D:\AI-Agents\Deep-Learning-Projects\mnist-cnn
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m src.train
python -m pytest -q
```
