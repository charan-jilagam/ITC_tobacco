"""
config_loader.py  –  Load configuration from config.json
"""
import json
import os
import logging

logger = logging.getLogger(__name__)


def load_config(config_path='config.json'):
    """Load and return configuration dict from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    logger.info(f"Loaded config from {config_path}")
    return config
