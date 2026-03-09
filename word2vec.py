#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# word2vec.py

"""
This module implements a pure NumPy version of the Word2Vec algorithm,
supporting both Skip-Gram with Negative Sampling (SGNS) and
Continuous Bag of Words (CBOW) architectures. It includes
features such as dropout regularization, L2 weight decay,
momentum updates, and gradient clipping to enhance training
stability and performance.
"""

import numpy as np

from typing import Tuple, List, Dict, Optional


class Word2VecNumPy:
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 method: str = 'skipgram',
                 lr: float = 0.01,
                 k_neg: int = 5,
                 dropout: float = 0.0,
                 weight_decay: float = 0.0,
                 momentum: float = 0.1,
                 clip_grad: float = 5.0
                 ):
        self.V: int = int(vocab_size)
        self.N: int = int(embed_dim)
        self.method: str = method.lower()
        self.lr: float = float(lr)
        self.k: int = int(k_neg)
        self.dropout: float = float(dropout)
        self.weight_decay: float = float(weight_decay)
        self.momentum: float = float(momentum)
        self.clip_grad: float = float(clip_grad)

        # Xavier initialization
        limit: float = np.sqrt(2.0 / (self.V + self.N))
        self.W1: np.ndarray = np.random.uniform(-limit, limit, (self.V, self.N)).astype(np.float32)
        self.W2: np.ndarray = np.random.uniform(-limit, limit, (self.V, self.N)).astype(np.float32)

        # Momentum buffers
        if self.momentum > 0.0:
            self.v_W1: np.ndarray = np.zeros_like(self.W1, dtype=np.float32)
            self.v_W2: np.ndarray = np.zeros_like(self.W2, dtype=np.float32)
        else:
            self.v_W1 = None
            self.v_W2 = None

        # Sparse gradient accumulators
        self.grad_W1_acc: Dict[int, np.ndarray] = {}
        self.grad_W2_acc: Dict[int, np.ndarray] = {}
        self.grad_W1_count: Dict[int, int] = {}
        self.grad_W2_count: Dict[int, int] = {}

    # Numerically stable sigmoid
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))

    def apply_dropout(self, h: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.dropout == 0.0:
            return h, None

        keep_prob: float = 1.0 - self.dropout
        mask: np.ndarray = np.random.binomial(1, keep_prob, size=h.shape).astype(np.float32)
        h_drop: np.ndarray = (h * mask) / keep_prob
        return h_drop, mask

    def backward_dropout(self, dh_drop: np.ndarray,
                         mask: Optional[np.ndarray]
                         ) -> np.ndarray:
        if self.dropout == 0.0 or mask is None:
            return dh_drop

        keep_prob = 1.0 - self.dropout
        return (dh_drop * mask) / keep_prob

    def _clip_grad(self, g: np.ndarray) -> np.ndarray:
        if self.clip_grad <= 0.0:
            return g

        # Clip by L2 norm
        norm = np.linalg.norm(g)
        if norm > 0 and norm > self.clip_grad:
            return g * (self.clip_grad / norm)
        return g

    def accumulate_grad(self, acc_dict: Dict[int, np.ndarray],
                        count_dict: Dict[int, int],
                        index: int,
                        grad: np.ndarray) -> None:
        # Clip Per Vector
        g = self._clip_grad(grad)
        if index in acc_dict:
            acc_dict[index] += g
            count_dict[index] += 1
        else:
            acc_dict[index] = g.copy()
            count_dict[index] = 1

    def apply_single_weight_update(self, W: np.ndarray,
                                   v_W: Optional[np.ndarray],
                                   index: int,
                                   grad: np.ndarray,
                                   lr: float
                                   ) -> None:
        current_grad = grad.copy()
        if self.weight_decay > 0.0:
            current_grad = current_grad + self.weight_decay * W[index]

        if self.momentum > 0.0 and v_W is not None:
            v_W[index] = self.momentum * v_W[index] + current_grad
            W[index] -= lr * v_W[index]
        else:
            W[index] -= lr * current_grad

    def apply_batch_gradients(self) -> None:
        # Apply updates for W2
        for idx, grad_sum in self.grad_W2_acc.items():
            avg_grad = grad_sum / max(1, self.grad_W2_count.get(idx, 1))
            self.apply_single_weight_update(self.W2, self.v_W2, idx, avg_grad, self.lr)

        # Apply updates for W1
        for idx, grad_sum in self.grad_W1_acc.items():
            avg_grad = grad_sum / max(1, self.grad_W1_count.get(idx, 1))
            self.apply_single_weight_update(self.W1, self.v_W1, idx, avg_grad, self.lr)

        self.grad_W1_acc.clear()
        self.grad_W2_acc.clear()
        self.grad_W1_count.clear()
        self.grad_W2_count.clear()

    def loss_pair(self, center: int, context: int, unigram_table: np.ndarray) -> float:
        h = self.W1[center]
        targets = [context]
        for _ in range(self.k):
            neg = np.random.choice(unigram_table)
            while neg == center or neg == context:
                neg = np.random.choice(unigram_table)

            targets.append(int(neg))

        targets_arr = np.array(targets, dtype=np.int32)
        Vt = self.W2[targets_arr]
        z = Vt.dot(h)
        z_pos = z[0]
        z_neg = z[1:]
        pos_loss = np.logaddexp(0.0, -z_pos)
        neg_loss = np.sum(np.logaddexp(0.0, z_neg)) if z_neg.size > 0 else 0.0
        return float(pos_loss + neg_loss)

    def process_sgns(self, center: int, context: int, unigram_table: np.ndarray) -> float:
        h = self.W1[center]
        h_drop, mask = self.apply_dropout(h)

        # Sample targets (1 positive + k negatives)
        targets = [context]
        for _ in range(self.k):
            neg = np.random.choice(unigram_table)
            # Avoid trivial negatives
            tries = 0
            while (neg == center or neg == context) and tries < 10:
                neg = np.random.choice(unigram_table)
                tries += 1

            targets.append(int(neg))
        targets_arr = np.array(targets, dtype=np.int32)

        Vt = self.W2[targets_arr]
        z = Vt.dot(h_drop)
        z_pos = z[0]
        z_neg = z[1:]
        pos_loss = np.logaddexp(0.0, -z_pos)
        neg_loss = np.sum(np.logaddexp(0.0, z_neg)) if z_neg.size > 0 else 0.0
        loss = float(pos_loss + neg_loss)

        # Backward
        pred = self.sigmoid(z)
        labels = np.zeros_like(pred, dtype=np.float32)
        labels[0] = 1.0
        error = pred - labels
        grad_Vt = np.outer(error, h_drop)
        dh_drop = np.dot(error, Vt)

        # Accumulate W2 gradients
        for i, t in enumerate(targets_arr):
            self.accumulate_grad(self.grad_W2_acc, self.grad_W2_count, int(t), grad_Vt[i])

        # Accumulate W1 gradient (center)
        dh = self.backward_dropout(dh_drop, mask)
        self.accumulate_grad(self.grad_W1_acc, self.grad_W1_count, int(center), dh)

        return loss

    def process_cbow(self, contexts: List[int], center: int, unigram_table: np.ndarray) -> float:
        if len(contexts) == 0:
            return 0.0

        h = np.mean(self.W1[contexts], axis=0)
        h_drop, mask = self.apply_dropout(h)
        targets = [center]
        for _ in range(self.k):
            neg = np.random.choice(unigram_table)
            tries = 0
            while (neg == center or neg in contexts) and tries < 10:
                neg = np.random.choice(unigram_table)
                tries += 1

            targets.append(int(neg))

        targets_arr = np.array(targets, dtype=np.int32)

        Vt = self.W2[targets_arr]
        z = Vt.dot(h_drop)
        z_pos = z[0]
        z_neg = z[1:]
        pos_loss = np.logaddexp(0.0, -z_pos)
        neg_loss = np.sum(np.logaddexp(0.0, z_neg)) if z_neg.size > 0 else 0.0
        loss = float(pos_loss + neg_loss)
        pred = self.sigmoid(z)
        labels = np.zeros_like(pred, dtype=np.float32)
        labels[0] = 1.0
        error = pred - labels
        grad_Vt = np.outer(error, h_drop)
        dh_drop = np.dot(error, Vt)

        for i, t in enumerate(targets_arr):
            self.accumulate_grad(self.grad_W2_acc, self.grad_W2_count, int(t), grad_Vt[i])

        dh = self.backward_dropout(dh_drop, mask)
        grad_w1 = dh / len(contexts)

        for c in contexts:
            self.accumulate_grad(self.grad_W1_acc, self.grad_W1_count, int(c), grad_w1.copy())

        return loss
