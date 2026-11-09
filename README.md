# Carbon Footprint of Deep Learning Models

Measuring the energy use and CO2 emissions of training and running deep learning models
for image classification, alongside their accuracy.

**Status: ongoing research (work in progress).** No results are published yet.

## Tools
- Energy / CO2: [`codecarbon`](https://github.com/mlco2/codecarbon)
- Model complexity: [`thop`](https://github.com/Lyken17/pytorch-OpCounter)
- Models via [`timm`](https://github.com/huggingface/pytorch-image-models), PyTorch
- Hardware: NVIDIA RTX 5060

## Quick start
```bash
pip install -r requirements.txt
python train.py --model resnet18 --epochs 5 --seed 0
python summarize.py
```
Each run prints the CO2 emitted and appends a row to `results/results.csv`.

## License
MIT
