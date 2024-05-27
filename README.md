# SOLMformer
Reproducible code for SOLMformer, a metadata-aware categorical  time series generative model. 
```
SOLMformer/
├── README.md
├── requirements.txt
├── data/
├── models/
│   ├── __init__.py
│   ├── gpt.py
│   └── model.py
├── train/
│   ├── __init__.py
│   ├── train_baseline.py
│   └── train_solmformer.py
└── utils/
    ├── __init__.py
    ├── BPIDataset.py
    └── Evaluator.py
```

## Setup

- `export PYTHONPATH="$PWD:${PYTHONPATH}"` if you get issues with importing items from one directory into another. 
- `pip install -r requirements.txt` to install dependencies.

## Data
You will need to download datasets 
* [BPI Challenge 2020i: International Declarations](https://data.4tu.nl/ndownloader/items/91fd1fa8-4df4-4b1a-9a3f-0116c412378f/versions/1)
* [BPI Challenge 2017](https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884)
* [BPI Challenge 2015 - 1](https://data.4tu.nl/collections/BPI_Challenge_2015/5065424)
* [BPI Challenge 2013](https://data.4tu.nl/articles/dataset/BPI_Challenge_2013_incidents/12693914/1)
* [BPI Challenge 2012](https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204/1)

After downloading, unzip and put the `.xes` file in the `data` directory.
The following train/eval instructions assume that the file data/{dataset-name}.xes exists.

## Training and Evaluating Models

### Baseline Train/Eval
Run the command
```
python train/baseline.py --dataset-name={dataset-name}
```
### S-former Train/Eval

```python train/baseline.py --include-metadata --dataset-name={dataset-name}
```
### SOLMformer-1T Train/Eval

#### 1T-Next Activity
```
python train/solmformer.py
--logits_weight=1.0 --durations_weight=0.0 --dataset-name={dataset-name}
```
#### 1T-Next Duration
```
python train/solmformer.py --logits_weight=0.0 --durations_weight=1.0 --dataset-name={dataset-name}
```

#### 1T-Time Remaining
```
python train/solmformer.py --logits_weight=0.0 --durations_weight=0.0 --dataset-name={dataset-name}
```

### SOLMformfer Train/Eval
```
python train/solmformer.py
--logits_weight={w1}
--durations_weight={w2}
--dataset-name={dataset-name}
```