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
from dataset_adapter import DatasetAdapterRegistry, DatasetAdapter

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
    
    def __init__(self, adapter_registry: Optional[DatasetAdapterRegistry] = None):
        self.datasets: Dict[str, Dataset] = {}
        self.images_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.adapter_registry = adapter_registry or DatasetAdapterRegistry()
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
            
            # Determine dataset type - check metadata first, then fallback to filename
            dataset_type = self._determine_dataset_type(data, dataset_name)
            print(f"📁 Dataset type: {dataset_type.value}")
            
            # Get appropriate adapter for this dataset type
            adapter = self.adapter_registry.get_adapter(dataset_type)
            
            # Load items (images or other data)
            items = data.get('images', data.get('items', data.get('data', [])))
            print(f"📁 Found {len(items)} items in dataset")
            
            # Normalize items using adapter
            normalized_items = [adapter.normalize_item(item) for item in items]
            
            # Extract collections using adapter
            collections = adapter.get_collections(normalized_items)
            
            # Extract available filters using adapter
            available_filters = adapter.extract_filters(normalized_items)
            
            # Get description from data or use default
            description = data.get('description', data.get('dataset_description', f"Dataset from {file_path.name}"))
            
            dataset = Dataset(
                name=dataset_name,
                type=dataset_type,
                description=description,
                file_path=file_path,
                collections=collections,
                total_images=len(normalized_items),
                available_filters=available_filters,
                metadata={
                    "source_file": str(file_path),
                    "last_modified": file_path.stat().st_mtime,
                    "schema": data.get('schema', {}),
                    "adapter_type": type(adapter).__name__
                }
            )
            
            self.datasets[dataset_name] = dataset
            self.images_cache[dataset_name] = normalized_items
            
            print(f"✅ Successfully loaded dataset: {dataset_name} ({len(normalized_items)} items, {len(collections)} collections)")
            
        except Exception as e:
            print(f"❌ Error loading dataset from {file_path}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _determine_dataset_type(self, data: Dict[str, Any], dataset_name: str) -> DatasetType:
        """Determine dataset type from metadata or filename"""
        # Check if dataset type is explicitly specified in the data
        if 'dataset_type' in data:
            type_str = data['dataset_type'].lower()
            try:
                return DatasetType(type_str)
            except ValueError:
                pass
        
        # Check schema metadata
        schema = data.get('schema', {})
        if 'type' in schema:
            type_str = schema['type'].lower()
            try:
                return DatasetType(type_str)
            except ValueError:
                pass
        
        # Fallback to filename-based detection (legacy behavior)
        name_lower = dataset_name.lower()
        if any(term in name_lower for term in ['bobcat', 'coyote', 'deer', 'wildlife', 'animal']):
            return DatasetType.WILDLIFE
        elif any(term in name_lower for term in ['plant', 'tree', 'flower', 'strawberry']):
            return DatasetType.PLANTS
        elif any(term in name_lower for term in ['pest', 'insect', 'disease']):
            return DatasetType.PESTS
        else:
            return DatasetType.CUSTOM
    
    def get_adapter_for_dataset(self, dataset_name: str) -> Optional[DatasetAdapter]:
        """Get the adapter for a specific dataset"""
        dataset = self.datasets.get(dataset_name)
        if dataset:
            return self.adapter_registry.get_adapter(dataset.type)
        return None
    
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
        """Search within a specific dataset using the appropriate adapter"""
        if dataset_name not in self.images_cache:
            return []
        
        items = self.images_cache[dataset_name]
        dataset = self.datasets.get(dataset_name)
        
        if not dataset:
            return []
        
        # Get adapter for this dataset type
        adapter = self.adapter_registry.get_adapter(dataset.type)
        
        # Apply filters if provided
        if filters:
            items = [item for item in items if adapter.matches_filters(item, filters)]
        
        # Apply search query if provided
        if query.strip():
            items = [item for item in items if adapter.matches_query(item, query)]
        
        return items
    
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
