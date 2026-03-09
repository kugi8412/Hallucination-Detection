# Pure NumPy Word2Vec
This repo implements from scratch Word2Vec training (SGNS and CBOW) with a focus on clarity, numerical stability, and practical training features. It demonstrates the forward/backward math, negative sampling, sparse updates, and commonly used optimizer/regularization techniques.

## Key features
- Skip-Gram with Negative Sampling (SGNS) and CBOW
- Negative sampling with unigram^0.75 distribution
- Sparse SGD with per-parameter momentum
- L2 regularization (weight decay)
- Dropout on input embeddings
- Gradient clipping
- Numerically stable loss
- Built-in analytic vs numeric gradient checker

## Requirements & installation
```
pip install numpy pandas
```

## Data layout
Place your training CSVs under ```data/```. By default the training script expects data/MedHallu.csv and a column named query. CSV files must include a header row.

## Quick start
```
python train.py --input data/MedHallu.csv --column query --method skipgram --epochs 3
```

```
python train.py \
  --input data/MedHallu.csv \
  --column context \
  --method cbow \
  --epochs 5 \
  --lr 0.025 \
  --batch_size 64 \
  --embed_dim 100 \
  --vocab_limit 10000 \
  --k_neg 10 \
  --dropout 0.1 \
  --momentum 0.9 \
  --weight_decay 1e-5 \
  --clip_grad 5.0 \
  --output models/weights_cbow.pkl
```

### Important parameters
```
--method => skipgram or cbow.
--embed_dim => embedding dimension (e.g. 50, 100).
--vocab_limit => max vocabulary size; rare words are mapped to <UNKNOWN>.
--window_size => context radius (e.g. 2 means 2 left + 2 right).
--k_neg => number of negative samples per positive pair.
--dropout => inverted dropout probability applied to the input embedding.
--momentum, --weight_decay, --clip_grad => optimizer and regularization controls.
--output => path to save trained weights.
```
