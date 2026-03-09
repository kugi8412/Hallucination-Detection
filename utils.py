#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils.py

"""
This module provides functions for loading and preprocessing text data
for training a Word2Vec model.
"""

import re
import copy

import numpy as np
import pandas as pd

from collections import Counter
from typing import List, Dict, Tuple

from word2vec import Word2VecNumPy


def load_data_and_build_vocab(filepath: str,
                              column_name: str,
                              vocab_limit: int = 5001
                              ) -> Tuple[List[List[int]], Dict[str, int], np.ndarray]:
    df = pd.read_csv(filepath)
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in {filepath}")

    sentences_raw = df[column_name].dropna().astype(str).tolist()
    word_counts = Counter()
    sentences_words: List[List[str]] = []
    for text in sentences_raw:
        clean_text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
        words = clean_text.split()
        if words:
            sentences_words.append(words)
            word_counts.update(words)

    most_common = word_counts.most_common(vocab_limit - 1)
    vocab = ["<UNKNOWN>"] + [w for w, c in most_common]
    word2idx = {w: i for i, w in enumerate(vocab)}
    freqs = np.array([count for _, count in most_common], dtype=np.float32)
    freqs = np.insert(freqs, 0, 1.0)  # unknown token count = 1
    unigram_dist = (freqs ** 0.75) / np.sum(freqs ** 0.75)
    unigram_table = np.random.choice(len(vocab), size=100000, p=unigram_dist)
    sentences_idx = [[word2idx.get(w, 0) for w in words] for words in sentences_words]
    return sentences_idx, word2idx, unigram_table


def gradient_check_pair(model: Word2VecNumPy,
                        center: int,
                        context: int,
                        unigram_table: np.ndarray,
                        eps: float = 1e-3
                        ) -> Dict[str, np.ndarray]:
    """ Performs a finite-difference check for W1[center] and W2[context].
    Temporarily forces k=0 to avoid stochastic negative sampling during the check.
    """
    model_clone = copy.deepcopy(model)
    orig_k = model_clone.k
    orig_dropout = model_clone.dropout
    model_clone.k = 0
    model_clone.dropout = 0.0
    model_clone.clip_grad = 0.0

    _ = model_clone.process_sgns(center, context, unigram_table)
    analytic_w1 = model_clone.grad_W1_acc.get(center, np.zeros(model_clone.N, dtype=np.float32)).copy()
    analytic_w2 = model_clone.grad_W2_acc.get(context, np.zeros(model_clone.N, dtype=np.float32)).copy()

    model_clone.k = orig_k
    model_clone.dropout = orig_dropout

    model_ref = copy.deepcopy(model)
    model_ref.k = 0
    model_ref.dropout = 0.0
    model_ref.clip_grad = 0.0

    def loss_on_model(m):
        return m.loss_pair(center, context, unigram_table)

    # numeric grad for W1[center]
    numeric_w1 = np.zeros(model.N, dtype=np.float32)
    for i in range(model.N):
        orig = model_ref.W1[center, i]
        model_ref.W1[center, i] = orig + eps
        lp = loss_on_model(model_ref)
        model_ref.W1[center, i] = orig - eps
        lm = loss_on_model(model_ref)
        numeric_w1[i] = (lp - lm) / (2 * eps)
        model_ref.W1[center, i] = orig

    # numeric grad for W2[context]
    numeric_w2 = np.zeros(model.N, dtype=np.float32)
    for i in range(model.N):
        orig = model_ref.W2[context, i]
        model_ref.W2[context, i] = orig + eps
        lp = loss_on_model(model_ref)
        model_ref.W2[context, i] = orig - eps
        lm = loss_on_model(model_ref)
        numeric_w2[i] = (lp - lm) / (2 * eps)
        model_ref.W2[context, i] = orig

    return {
        'analytic_w1': analytic_w1,
        'numeric_w1': numeric_w1,
        'analytic_w2': analytic_w2,
        'numeric_w2': numeric_w2
    }


def evaluate_cosine_similarity(model: Word2VecNumPy,
                               word2idx: Dict[str, int],
                               pairs: List[Tuple[str, str]]
                               ) -> None:
    """ Computes and prints the cosine similarity for given pairs of words."""
    for w1, w2 in pairs:
        # Checking Dict
        if w1 not in word2idx or w2 not in word2idx:
            print(f"  -> Similarity({w1}, {w2}) = Word(s) out of vocabulary (OOV)!")
            continue

        v1 = model.W1[word2idx[w1]]
        v2 = model.W1[word2idx[w2]]
        sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        
        print(f"  -> Similarity({w1:<10}, {w2:<10}) = {sim:>7.4f}")
