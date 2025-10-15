#!/usr/bin/env python3
"""
Shared data models for the MCP server and web interface
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel

class DatasetType(str, Enum):
    WILDLIFE = "wildlife"
    PLANTS = "plants"
    PESTS = "pests"
    CUSTOM = "custom"

class ModelType(str, Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    CUSTOM = "custom"

@dataclass
class SearchRequest:
    """Search request model"""
    query: str = ""
    category_filter: List[str] = field(default_factory=list)
    species_filter: List[str] = field(default_factory=list)
    time_filter: List[str] = field(default_factory=list)
    season_filter: List[str] = field(default_factory=list)
    action_filter: List[str] = field(default_factory=list)
    plant_state_filter: List[str] = field(default_factory=list)
    collection_filter: List[str] = field(default_factory=list)
    limit: int = 50
    offset: int = 0

@dataclass
class SearchResponse:
    """Search response model"""
    results: List[Dict[str, Any]]
    total_count: int
    page: int
    total_pages: int
    limit: int
    offset: int
    query: str
    filters_applied: Dict[str, Any]

@dataclass
class ImageResult:
    """Image result model"""
    id: str
    collection: str
    category: str
    image_url: str
    metadata: Dict[str, Any]
    relevance_score: Optional[float] = None
    score_details: Optional[Dict[str, Any]] = None

@dataclass
class FilterOptions:
    """Available filter options"""
    categories: List[str]
    species: List[str]
    times: List[str]
    seasons: List[str]
    actions: List[str]
    plant_states: List[str]
    collections: List[str]

@dataclass
class DatasetInfo:
    """Dataset information"""
    name: str
    type: DatasetType
    description: str
    total_images: int
    collections: List[str]
    available_filters: FilterOptions
    metadata: Dict[str, Any]

@dataclass
class ModelInfo:
    """Model information"""
    name: str
    type: ModelType
    description: str
    version: str
    supported_datasets: List[str]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class InferenceRequest:
    """Model inference request"""
    dataset_name: str
    model_name: str
    image_ids: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InferenceResult:
    """Model inference result"""
    model_name: str
    dataset_name: str
    results: List[Dict[str, Any]]
    processing_time: float
    metadata: Dict[str, Any]

# Pydantic models for API serialization
class SearchRequestAPI(BaseModel):
    query: str = ""
    category: List[str] = []
    species: List[str] = []
    time: List[str] = []
    season: List[str] = []
    action: List[str] = []
    plant_state: List[str] = []
    collection: List[str] = []
    limit: int = 50
    offset: int = 0

class InferenceRequestAPI(BaseModel):
    dataset_name: str
    model_name: str
    image_ids: List[str]
    parameters: Dict[str, Any] = {}
