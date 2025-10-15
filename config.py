#!/usr/bin/env python3
"""
Configuration for the MCP server and web interface
"""

import os
from pathlib import Path

# Base configuration
BASE_DIR = Path("/opt/mcp-data-server")
IMAGES_DIR = BASE_DIR / "images"
TEMPLATES_DIR = BASE_DIR / "templates"
PLUGINS_DIR = BASE_DIR / "plugins"
DATASETS_DIR = BASE_DIR / "datasets"

# LLM Configuration
LLM_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "model": os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
    "enabled": os.getenv("LLM_ENABLED", "true").lower() == "true",
    "fallback_to_rules": os.getenv("LLM_FALLBACK_TO_RULES", "true").lower() == "true"
}

# MCP Configuration
MCP_CONFIG = {
    "mcp_host": os.getenv("MCP_HOST", "0.0.0.0"),
    "mcp_port": int(os.getenv("MCP_PORT", 8188)),
    "mcp_base_url": os.getenv("MCP_BASE_URL", "http://127.0.0.1:8188")
}

WEB_CONFIG = {
    "web_host": os.getenv("WEB_HOST", "0.0.0.0"),
    "web_port": int(os.getenv("WEB_PORT", 8187)),
    "mcp_server_url": os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8188")
}

# Dataset configuration
DATASET_CONFIG = {
    "auto_discover": True,
    "supported_formats": [".json", ".csv"],
    "default_limit": 50,
    "max_limit": 1000
}

# Model configuration
MODEL_CONFIG = {
    "default_model": "baseline_classifier",
    "supported_types": ["classification", "detection", "segmentation"],
    "cache_results": True,
    "max_batch_size": 100
}

# MCP Protocol configuration
MCP_PROTOCOL_CONFIG = {
    "server_name": "AIFARMS Extensible MCP Server",
    "server_version": "2.0.0",
    "server_description": "Extensible MCP server for AIFARMS datasets and models",
    "capabilities": {
        "resources": True,
        "tools": True,
        "prompts": False,
        "agents": False
    }
}

# Ensure directories exist
for directory in [TEMPLATES_DIR, PLUGINS_DIR, DATASETS_DIR]:
    directory.mkdir(exist_ok=True)
