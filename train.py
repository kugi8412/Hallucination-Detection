#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train.py

"""
This module provides functions for training a Word2Vec model.
It includes a main training loop that iterates over the dataset
for a specified number of epochs, computes the loss, and updates
the model weights using mini-batch gradient descent.

Example usage:
python train.py --method skipgram --epochs 1 --batch_size 16 --embed_dim 30 --vocab_limit 1000 --output fast_skipgram.pkl
python train.py --method cbow --epochs 5 --lr 0.05 --dropout 0.05 --weight_decay 1e-5 --momentum 0.50 --output cbow.pkl
"""

import os
import pickle
import argparse

import numpy as np
from typing import List

from word2vec import Word2VecNumPy
from utils import load_data_and_build_vocab, gradient_check_pair, evaluate_cosine_similarity


def train_word2vec(model: Word2VecNumPy,
                   sentences_idx: List[List[int]],
                   window_size: int,
                   unigram_table: np.ndarray,
                   batch_size: int) -> float:
    total_loss = 0.0
    pair_count = 0
    current_batch = 0

    for sentence in sentences_idx:
        L = len(sentence)
        if L < 2:
            continue

        for i in range(L):
            center = sentence[i]
            start = max(0, i - window_size)
            end = min(L, i + window_size + 1)
            if model.method == 'skipgram':
                for j in range(start, end):
                    if i == j:
                        continue

                    context = sentence[j]
                    total_loss += model.process_sgns(center, context, unigram_table)
                    pair_count += 1
                    current_batch += 1
            else:
                contexts = [sentence[j] for j in range(start, end) if i != j]
                if not contexts:
                    continue

                total_loss += model.process_cbow(contexts, center, unigram_table)
                pair_count += 1
                current_batch += 1

            if current_batch >= batch_size:
                model.apply_batch_gradients()
                current_batch = 0

    if current_batch > 0:
        model.apply_batch_gradients()

    return total_loss / max(1, pair_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure NumPy Word2Vec (SGNS/CBOW) with Gradient Check")
    parser.add_argument("--input", type=str, default="data/MedHallu.csv", help="Input dataset path")
    parser.add_argument("--column", type=str, default="query", help="Target column for training")
    parser.add_argument("--method", type=str, choices=['skipgram', 'cbow'], default="skipgram", help="Model architecture")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size for gradient accumulation")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability (0.0 to 1.0)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="L2 regularization penalty")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD Momentum factor (0.0 to 1.0)")
    parser.add_argument("--clip_grad", type=float, default=5.0, help="Gradient clipping value (0.0 to disable)")
    parser.add_argument("--vocab_limit", type=int, default=5001, help="Maximum vocabulary size (including <UNKNOWN>)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--window_size", type=int, default=2, help="Context window size")
    parser.add_argument("--k_neg", type=int, default=5, help="Number of negative samples for SGNS")
    parser.add_argument("--embed_dim", type=int, default=50, help="Dimensionality of word embeddings")
    parser.add_argument("--output", type=str, default="weights.pkl", help="Output file path for model weights (.pkl/.npz)")
    
    args = parser.parse_args()

    np.random.seed(args.seed)

    sentences_idx, word2idx, unigram_table = load_data_and_build_vocab(
        args.input, args.column, vocab_limit=args.vocab_limit
    )

    print(f"[INFO]: Dataset loaded with independent rows: {len(sentences_idx)}")
    print(f"[INFO]: Vocabulary size: {len(word2idx)}")
    
    model = Word2VecNumPy(
        vocab_size=len(word2idx), 
        embed_dim=args.embed_dim, 
        method=args.method, 
        lr=args.lr, 
        k_neg=args.k_neg,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        clip_grad=args.clip_grad
    )

    test_center = 1 if len(word2idx) > 1 else 0
    test_context = 2 if len(word2idx) > 2 else 0

    check_results = gradient_check_pair(model,
                                        center=test_center,
                                        context=test_context,
                                        unigram_table=unigram_table
                                      )
    
    # Checking numerically stability
    w1_ok = np.allclose(check_results['analytic_w1'],
                        check_results['numeric_w1'],
                        rtol=1e-3,
                        atol=1e-4
                        )

    w2_ok = np.allclose(check_results['analytic_w2'],
                        check_results['numeric_w2'],
                        rtol=1e-3,
                        atol=1e-4
                        )
    
    assert w1_ok, "Gradient check failed for W1 (center word embedding)!"
    assert w2_ok, "Gradient check failed for W2 (context word embedding)!"
    
    if not (w1_ok and w2_ok):
        print("[WARNING]: Gradient check failed! The analytical gradients may differ from numeric approximation.")
    
    for epoch in range(args.epochs):
        loss: float = train_word2vec(model, sentences_idx, args.window_size, unigram_table, args.batch_size)
        print(f"Epoch {epoch + 1}/{args.epochs} completed | Avg Loss: {loss:.4f}")
        
    print(f"\n[INFO]: Saving model weights to {args.output}.")
    model_state = {
        'W1': model.W1,
        'W2': model.W2,
        'word2idx': word2idx,
        'hyperparams': vars(args)
    }
    
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(model_state, f)

    # Example evaluation of cosine similarity
    test_pairs = [
        ("men", "women"),
        ("hospital", "patients"),
        ("pregnant", "women")
    ]
    
    evaluate_cosine_similarity(model, word2idx, test_pairs)

    print("[SUCCESS]: Training complete and model saved.")


if __name__ == "__main__":
    main()
