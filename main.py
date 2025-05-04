#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the EVENTA challenge solution.
This script handles command-line arguments and executes the appropriate pipeline.
"""

import argparse
import os
import yaml
import logging
from pathlib import Path

from src.pipelines.task1_pipeline import EventEnhancedCaptioningPipeline
from src.pipelines.task2_pipeline import EventBasedImageRetrievalPipeline


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


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main function to parse arguments and run the appropriate pipeline."""
    parser = argparse.ArgumentParser(description="EVENTA Challenge Solution")
    
    parser.add_argument(
        "--task", 
        type=str, 
        required=True,
        choices=["task1", "task2", "both"],
        help="Which task to run: task1 (Event-Enhanced Captioning), task2 (Event-Based Image Retrieval), or both"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--data_dir", 
        type=str, 
        help="Directory containing the input data"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs",
        help="Directory to save outputs"
    )
    
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0",
        help="Device to use for computation"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=None,
        help="Batch size for processing (overrides config)"
    )
    
    parser.add_argument(
        "--evaluate", 
        action="store_true",
        help="Whether to run evaluation after prediction"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.batch_size:
        config["batch_size"] = args.batch_size
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the appropriate pipeline
    if args.task in ["task1", "both"]:
        logger.info("Running Task 1: Event-Enhanced Image Captioning")
        task1_pipeline = EventEnhancedCaptioningPipeline(
            config=config,
            output_dir=str(output_dir / "task1"),
            device=args.device
        )
        task1_pipeline.run()
        
        if args.evaluate:
            from scripts.evaluate_task1 import evaluate_task1
            evaluate_task1(
                prediction_file=str(output_dir / "task1" / "predictions.json"),
                reference_file=config["task1"]["reference_file"],
                output_file=str(output_dir / "task1" / "evaluation_results.json")
            )
    
    if args.task in ["task2", "both"]:
        logger.info("Running Task 2: Event-Based Image Retrieval")
        task2_pipeline = EventBasedImageRetrievalPipeline(
            config=config,
            output_dir=str(output_dir / "task2"),
            device=args.device
        )
        task2_pipeline.run()
        
        if args.evaluate:
            from scripts.evaluate_task2 import evaluate_task2
            evaluate_task2(
                prediction_file=str(output_dir / "task2" / "predictions.json"),
                reference_file=config["task2"]["reference_file"],
                output_file=str(output_dir / "task2" / "evaluation_results.json")
            )
    
    logger.info(f"All tasks completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()