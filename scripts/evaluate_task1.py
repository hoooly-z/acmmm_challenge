#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to evaluate performance on Task 1: Event-Enhanced Image Captioning.
This evaluates captioning quality using standard metrics like BLEU, METEOR, ROUGE-L, and CIDEr.
It also evaluates event relevance using semantic similarity metrics.
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
from typing import Dict, List, Any

# Import evaluation libraries
try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.spice.spice import Spice
except ImportError:
    print("Warning: pycocoevalcap not found. Please install it for full evaluation.")
    Bleu, Meteor, Rouge, Cider, Spice = None, None, None, None, None

from src.encoders.text_encoder import TextEncoder


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


def format_for_coco_eval(predictions: Dict[str, Any], references: Dict[str, Any]):
    """Format predictions and references for COCO evaluation."""
    # Format predictions
    formatted_preds = {}
    for item in predictions:
        image_id = str(item["image_id"])
        caption = item["caption"]
        formatted_preds[image_id] = [caption]
    
    # Format references
    formatted_refs = {}
    for item in references:
        image_id = str(item["image_id"])
        if "captions" in item:  # Multiple reference captions
            formatted_refs[image_id] = item["captions"]
        else:  # Single reference caption
            formatted_refs[image_id] = [item["caption"]]
    
    return formatted_preds, formatted_refs


def evaluate_captioning_metrics(predictions: Dict, references: Dict) -> Dict[str, float]:
    """Evaluate standard captioning metrics: BLEU, METEOR, ROUGE-L, CIDEr."""
    results = {}
    
    if Bleu is not None:
        scorer = Bleu(4)
        score, _ = scorer.compute_score(references, predictions)
        for i, s in enumerate(score):
            results[f"BLEU-{i+1}"] = s
    
    if Meteor is not None:
        scorer = Meteor()
        score, _ = scorer.compute_score(references, predictions)
        results["METEOR"] = score
    
    if Rouge is not None:
        scorer = Rouge()
        score, _ = scorer.compute_score(references, predictions)
        results["ROUGE_L"] = score
    
    if Cider is not None:
        scorer = Cider()
        score, _ = scorer.compute_score(references, predictions)
        results["CIDEr"] = score
    
    if Spice is not None:
        scorer = Spice()
        score, _ = scorer.compute_score(references, predictions)
        results["SPICE"] = score
    
    return results


def evaluate_event_relevance(predictions: List[Dict], references: List[Dict], device="cuda:0") -> Dict[str, float]:
    """Evaluate event relevance using semantic similarity between generated captions and event descriptions."""
    logger = logging.getLogger(__name__)
    
    # Initialize text encoder
    text_encoder = TextEncoder(model_name="sentence-transformers/all-mpnet-base-v2", device=device)
    
    # Extract prediction captions and reference event descriptions
    pred_captions = []
    event_descriptions = []
    
    for pred_item in predictions:
        image_id = pred_item["image_id"]
        
        # Find matching reference item
        ref_item = next((item for item in references if item["image_id"] == image_id), None)
        if ref_item is None:
            logger.warning(f"No reference found for image ID: {image_id}")
            continue
        
        pred_captions.append(pred_item["caption"])
        event_descriptions.append(ref_item["event_description"])
    
    # Encode texts
    logger.info("Encoding captions and event descriptions...")
    with torch.no_grad():
        caption_embeddings = text_encoder.encode_batch(pred_captions)
        event_embeddings = text_encoder.encode_batch(event_descriptions)
    
    # Compute cosine similarity
    similarities = []
    for cap_emb, evt_emb in zip(caption_embeddings, event_embeddings):
        # Normalize embeddings
        cap_emb = cap_emb / cap_emb.norm()
        evt_emb = evt_emb / evt_emb.norm()
        
        # Compute cosine similarity
        similarity = torch.dot(cap_emb, evt_emb).item()
        similarities.append(similarity)
    
    # Compute event relevance metrics
    event_relevance = {
        "event_similarity_mean": float(np.mean(similarities)),
        "event_similarity_median": float(np.median(similarities)),
        "event_similarity_std": float(np.std(similarities))
    }
    
    return event_relevance


def evaluate_task1(prediction_file: str, reference_file: str, output_file: str = None, device: str = "cuda:0"):
    """Evaluate performance on Task 1."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading predictions from {prediction_file}")
    predictions = load_predictions(prediction_file)
    
    logger.info(f"Loading references from {reference_file}")
    references = load_references(reference_file)
    
    # Evaluate captioning metrics
    logger.info("Evaluating standard captioning metrics...")
    formatted_preds, formatted_refs = format_for_coco_eval(predictions, references)
    captioning_metrics = evaluate_captioning_metrics(formatted_refs, formatted_preds)
    
    # Evaluate event relevance
    logger.info("Evaluating event relevance...")
    event_metrics = evaluate_event_relevance(predictions, references, device)
    
    # Combine results
    results = {
        "captioning_metrics": captioning_metrics,
        "event_relevance_metrics": event_metrics
    }
    
    # Calculate overall score (customize as needed)
    # For example, weighted average of CIDEr and event similarity
    if "CIDEr" in captioning_metrics:
        cider_weight = 0.6
        event_weight = 0.4
        overall_score = (captioning_metrics["CIDEr"] * cider_weight + 
                        event_metrics["event_similarity_mean"] * event_weight)
        results["overall_score"] = float(overall_score)
    
    # Print results
    logger.info("Evaluation results:")
    for category, metrics in results.items():
        if isinstance(metrics, dict):
            logger.info(f"  {category}:")
            for metric, value in metrics.items():
                logger.info(f"    {metric}: {value:.4f}")
        else:
            logger.info(f"  {category}: {metrics:.4f}")
    
    # Save results if output file is specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results saved to {output_file}")
    
    return results


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Task 1: Event-Enhanced Image Captioning")
    
    parser.add_argument(
        "--predictions", 
        type=str, 
        required=True,
        help="Path to prediction file"
    )
    
    parser.add_argument(
        "--references", 
        type=str, 
        required=True,
        help="Path to reference file"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Path to save evaluation results"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0",
        help="Device to use for computation"
    )
    
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Run evaluation
    evaluate_task1(
        prediction_file=args.predictions,
        reference_file=args.references,
        output_file=args.output,
        device=args.device
    )


if __name__ == "__main__":
    main()