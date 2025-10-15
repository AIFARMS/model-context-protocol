#!/usr/bin/env python3
"""
Dataset registry for managing different datasets
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from models import DatasetInfo, FilterOptions, DatasetType
from config import BASE_DIR, DATASETS_DIR

@dataclass
class Dataset:
    """Dataset definition"""
    name: str
    type: DatasetType
    description: str
    file_path: Path
    collections: List[str] = field(default_factory=list)
    total_images: int = 0
    available_filters: Optional[FilterOptions] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class DatasetRegistry:
    """Registry for managing datasets"""
    
    def __init__(self):
        self.datasets: Dict[str, Dataset] = {}
        self.images_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._discover_datasets()
    
    def _discover_datasets(self):
        """Auto-discover datasets from the datasets directory"""
        if not DATASETS_DIR.exists():
            print(f"⚠️  Datasets directory not found: {DATASETS_DIR}")
            return
        
        # Look for MCP data files in the BASE_DIR (where the server is running)
        mcp_files = list(BASE_DIR.glob("*_mcp_data.json"))
        print(f"🔍 Looking for MCP files in: {BASE_DIR}")
        print(f"🔍 Found {len(mcp_files)} MCP files: {[f.name for f in mcp_files]}")
        
        for mcp_file in mcp_files:
            try:
                self._load_dataset_from_file(mcp_file)
            except Exception as e:
                print(f"⚠️  Failed to load dataset from {mcp_file}: {e}")
    
    def _load_dataset_from_file(self, file_path: Path):
        """Load a dataset from an MCP data file"""
        print(f"📁 Loading dataset from: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract dataset name from filename
            dataset_name = file_path.stem.replace('_mcp_data', '')
            print(f"📁 Dataset name: {dataset_name}")
            
            # Determine dataset type
            if any(term in dataset_name.lower() for term in ['bobcat', 'coyote', 'deer', 'wildlife', 'animal']):
                dataset_type = DatasetType.WILDLIFE
            elif any(term in dataset_name.lower() for term in ['plant', 'tree', 'flower', 'strawberry']):
                dataset_type = DatasetType.PLANTS
            elif any(term in dataset_name.lower() for term in ['pest', 'insect', 'disease']):
                dataset_type = DatasetType.PESTS
            else:
                dataset_type = DatasetType.CUSTOM
            
            # Load images and extract metadata
            images = data.get('images', [])
            print(f"📁 Found {len(images)} images in dataset")
            
            collections = list(set(img.get('collection', dataset_name) for img in images))
            
            # Extract available filters
            available_filters = self._extract_filters(images)
            
            dataset = Dataset(
                name=dataset_name,
                type=dataset_type,
                description=f"Dataset from {file_path.name}",
                file_path=file_path,
                collections=collections,
                total_images=len(images),
                available_filters=available_filters,
                metadata={
                    "source_file": str(file_path),
                    "last_modified": file_path.stat().st_mtime
                }
            )
            
            self.datasets[dataset_name] = dataset
            self.images_cache[dataset_name] = images
            
            print(f"✅ Successfully loaded dataset: {dataset_name} ({len(images)} images, {len(collections)} collections)")
            
        except Exception as e:
            print(f"❌ Error loading dataset from {file_path}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _extract_filters(self, images: List[Dict[str, Any]]) -> FilterOptions:
        """Extract available filters from images"""
        categories = set()
        species = set()
        times = set()
        seasons = set()
        actions = set()
        plant_states = set()
        collections = set()
        
        for img in images:
            metadata = img.get('metadata', {})
            collections.add(img.get('collection', 'unknown'))
            
            # Extract category
            if 'category' in img:
                categories.add(img['category'])
            
            # Extract time information
            time_info = metadata.get('time', '').lower()
            if 'night' in time_info or 'dark' in time_info:
                times.add('night')
            elif 'day' in time_info or 'morning' in time_info or 'afternoon' in time_info:
                times.add('day')
            elif 'dawn' in time_info or 'sunrise' in time_info:
                times.add('dawn')
            elif 'dusk' in time_info or 'sunset' in time_info:
                times.add('dusk')
            
            # Extract other filters
            if 'action' in metadata:
                actions.add(metadata['action'].lower())
            if 'season' in metadata:
                seasons.add(metadata['season'].lower())
            if 'plant_state' in metadata:
                plant_states.add(metadata['plant_state'].lower())
        
        return FilterOptions(
            categories=sorted(list(categories)),
            species=sorted(list(collections)),
            times=sorted(list(times)),
            seasons=sorted(list(seasons)),
            actions=sorted(list(actions)),
            plant_states=sorted(list(plant_states)),
            collections=sorted(list(collections))
        )
    
    def get_dataset(self, name: str) -> Optional[Dataset]:
        """Get a dataset by name"""
        return self.datasets.get(name)
    
    def get_all_datasets(self) -> Dict[str, DatasetInfo]:
        """Get all datasets as DatasetInfo objects"""
        return {
            name: DatasetInfo(
                name=dataset.name,
                type=dataset.type,
                description=dataset.description,
                total_images=dataset.total_images,
                collections=dataset.collections,
                available_filters=dataset.available_filters,
                metadata=dataset.metadata
            )
            for name, dataset in self.datasets.items()
        }
    
    def get_images(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Get images from a dataset"""
        return self.images_cache.get(dataset_name, [])
    
    def search_dataset(self, dataset_name: str, query: str = "", filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search within a specific dataset"""
        if dataset_name not in self.images_cache:
            return []
        
        images = self.images_cache[dataset_name]
        
        # Apply filters if provided
        if filters:
            # Implementation of filtering logic
            # This can be enhanced based on your specific needs
            pass
        
        # Apply search query if provided
        if query.strip():
            # Implementation of search logic
            # This can be enhanced based on your specific needs
            pass
        
        return images
    
    def add_dataset(self, name: str, file_path: Path, dataset_type: DatasetType = DatasetType.CUSTOM):
        """Add a new dataset"""
        try:
            self._load_dataset_from_file(file_path)
            print(f"✅ Added new dataset: {name}")
        except Exception as e:
            print(f"❌ Failed to add dataset {name}: {e}")
            raise
    
    def remove_dataset(self, name: str):
        """Remove a dataset"""
        if name in self.datasets:
            del self.datasets[name]
            if name in self.images_cache:
                del self.images_cache[name]
            print(f"🗑️  Removed dataset: {name}")
