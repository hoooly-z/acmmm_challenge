#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to evaluate performance on Task 2: Event-Based Image Retrieval.
This evaluates retrieval performance using standard metrics like
Recall@K, Precision@K, mAP, and NDCG.
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any, Set, Tuple


def setup_logging(log_level="INFO"):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_predictions(prediction_file: str) -> Dict[str, Any]:
    """Load prediction results from JSON file."""
    with open(prediction_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    return predictions


def load_references(reference_file: str) -> Dict[str, Any]:
    """Load reference data from JSON file."""
    with open(reference_file, "r", encoding="utf-8") as f:
        references = json.load(f)
    return references


def precision_at_k(retrieved_items: List[str], relevant_items: Set[str], k: int) -> float:
    """Calculate Precision@K."""
    if k == 0:
        return 0.0
    
    # Get the top-k retrieved items
    top_k_items = retrieved_items[:k]
    
    # Count the number of relevant items in the top-k
    num_relevant = sum(1 for item in top_k_items if item in relevant_items)
    
    # Calculate Precision@K
    return num_relevant / k


def recall_at_k(retrieved_items: List[str], relevant_items: Set[str], k: int) -> float:
    """Calculate Recall@K."""
    if not relevant_items:
        return 0.0
    
    # Get the top-k retrieved items
    top_k_items = retrieved_items[:k]
    
    # Count the number of relevant items in the top-k
    num_relevant = sum(1 for item in top_k_items if item in relevant_items)
    
    # Calculate Recall@K
    return num_relevant / len(relevant_items)


def average_precision(retrieved_items: List[str], relevant_items: Set[str]) -> float:
    """Calculate Average Precision (AP)."""
    if not relevant_items:
        return 0.0
    
    precisions = []
    num_relevant = 0
    
    for i, item in enumerate(retrieved_items):
        rank = i + 1
        if item in relevant_items:
            num_relevant += 1
            precisions.append(num_relevant / rank)
    
    if not precisions:
        return 0.0
    
    # Average Precision
    return sum(precisions) / len(relevant_items)


def ndcg_at_k(retrieved_items: List[str], relevant_items: Dict[str, float], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K)."""
    if not relevant_items:
        return 0.0
    
    # Get the top-k retrieved items
    top_k_items = retrieved_items[:k]
    
    # Calculate DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i, item in enumerate(top_k_items):
        rank = i + 1
        rel = relevant_items.get(item, 0.0)
        # Using 2^rel - 1 for the gain and log2(rank + 1) for the discount
        dcg += (2 ** rel - 1) / np.log2(rank + 1)
    
    # Calculate ideal DCG
    ideal_ranking = sorted(relevant_items.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / np.log2(rank + 1) for rank, rel in enumerate(ideal_ranking, 1))
    
    if idcg == 0:
        return 0.0
    
    # Calculate NDCG
    return dcg / idcg


def evaluate_retrieval_metrics(
    predictions: List[Dict], 
    references: List[Dict],
    k_values: List[int] = [1, 5, 10, 20, 50]
) -> Dict[str, float]:
    """Evaluate standard retrieval metrics: Precision@K, Recall@K, mAP, NDCG@K."""
    logger = logging.getLogger(__name__)
    
    results = {}
    
    # Initialize metric accumulators
    precision_at_ks = {k: [] for k in k_values}