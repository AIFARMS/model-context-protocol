#!/usr/bin/env python3
"""
Dataset Adapter System for Extensible Dataset Type Support

This module provides an adapter pattern for different dataset types,
allowing the MCP server to handle datasets with different schemas and structures.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
import json
from models import DatasetType, FilterOptions


@dataclass
class DatasetSchema:
    """Schema definition for a dataset type"""
    dataset_type: DatasetType
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    filter_fields: List[str] = field(default_factory=list)
    metadata_fields: List[str] = field(default_factory=list)
    description: str = ""


class DatasetAdapter(ABC):
    """Abstract base class for dataset adapters"""
    
    def __init__(self, schema: DatasetSchema):
        self.schema = schema
    
    @abstractmethod
    def extract_filters(self, items: List[Dict[str, Any]]) -> FilterOptions:
        """Extract available filter options from dataset items"""
        pass
    
    @abstractmethod
    def validate_item(self, item: Dict[str, Any]) -> bool:
        """Validate that an item conforms to this dataset type's schema"""
        pass
    
    @abstractmethod
    def normalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize an item to a standard format for searching"""
        pass
    
    @abstractmethod
    def matches_query(self, item: Dict[str, Any], query: str) -> bool:
        """Check if an item matches a search query"""
        pass
    
    @abstractmethod
    def matches_filters(self, item: Dict[str, Any], filters: Dict[str, List[str]]) -> bool:
        """Check if an item matches the given filters"""
        pass
    
    def get_collections(self, items: List[Dict[str, Any]]) -> List[str]:
        """Extract collection names from items (default implementation)"""
        collections = set()
        for item in items:
            collection = item.get('collection') or item.get('category') or item.get('type')
            if collection:
                collections.add(collection)
        return sorted(list(collections))


class SpeciesObservationAdapter(DatasetAdapter):
    """Adapter for species observation datasets (default/legacy)"""
    
    def __init__(self):
        schema = DatasetSchema(
            dataset_type=DatasetType.WILDLIFE,
            required_fields=['id', 'collection'],
            optional_fields=['category', 'metadata'],
            filter_fields=['category', 'species', 'time', 'season', 'action', 'plant_state', 'collection'],
            metadata_fields=['species', 'action', 'time', 'season', 'scene', 'weather', 'date', 'description'],
            description="Species observation dataset with wildlife, plants, or pest observations"
        )
        super().__init__(schema)
    
    def extract_filters(self, items: List[Dict[str, Any]]) -> FilterOptions:
        """Extract filters for species observation datasets"""
        categories = set()
        species = set()
        times = set()
        seasons = set()
        actions = set()
        plant_states = set()
        collections = set()
        
        for item in items:
            metadata = item.get('metadata', {})
            collections.add(item.get('collection', 'unknown'))
            
            # Extract category
            if 'category' in item:
                categories.add(item['category'])
            
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
            
            # Extract species from collection or metadata
            species_name = item.get('collection') or metadata.get('species', '')
            if species_name:
                species.add(species_name.lower())
        
        return FilterOptions(
            categories=sorted(list(categories)),
            species=sorted(list(species)),
            times=sorted(list(times)),
            seasons=sorted(list(seasons)),
            actions=sorted(list(actions)),
            plant_states=sorted(list(plant_states)),
            collections=sorted(list(collections))
        )
    
    def validate_item(self, item: Dict[str, Any]) -> bool:
        """Validate species observation item"""
        return 'id' in item and ('collection' in item or 'category' in item)
    
    def normalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize species observation item"""
        normalized = item.copy()
        # Ensure collection field exists
        if 'collection' not in normalized:
            normalized['collection'] = normalized.get('category', 'unknown')
        return normalized
    
    def matches_query(self, item: Dict[str, Any], query: str) -> bool:
        """Check if item matches search query"""
        query_lower = query.lower()
        
        # Check collection name
        if query_lower in item.get("collection", "").lower():
            return True
        
        # Check metadata description
        metadata = item.get("metadata", {})
        if query_lower in metadata.get("description", "").lower():
            return True
        
        # Check metadata action
        if query_lower in metadata.get("action", "").lower():
            return True
        
        # Check metadata scene
        if query_lower in metadata.get("scene", "").lower():
            return True
        
        return False
    
    def matches_filters(self, item: Dict[str, Any], filters: Dict[str, List[str]]) -> bool:
        """Check if item matches filters"""
        metadata = item.get('metadata', {})
        
        # Check category filter
        if filters.get("category"):
            category_filter = [c.lower() for c in filters["category"]]
            if item.get("category", "").lower() not in category_filter:
                return False
        
        # Check species filter
        if filters.get("species"):
            species_filter = [s.lower() for s in filters["species"]]
            item_species = (item.get("collection", "") or metadata.get("species", "")).lower()
            if not any(species.lower() in item_species for species in species_filter):
                return False
        
        # Check time filter
        if filters.get("time"):
            time_filter = [t.lower() for t in filters["time"]]
            time_info = metadata.get("time", "").lower()
            matched = False
            for time_val in time_filter:
                if time_val == "night" and ("night" in time_info or "dark" in time_info):
                    matched = True
                    break
                elif time_val == "day" and ("day" in time_info or "morning" in time_info or "afternoon" in time_info):
                    matched = True
                    break
                elif time_val == "dawn" and ("dawn" in time_info or "sunrise" in time_info):
                    matched = True
                    break
                elif time_val == "dusk" and ("dusk" in time_info or "sunset" in time_info):
                    matched = True
                    break
            if not matched:
                return False
        
        # Check season filter
        if filters.get("season"):
            season_filter = [s.lower() for s in filters["season"]]
            item_season = metadata.get("season", "").lower()
            if not any(season.lower() in item_season for season in season_filter):
                return False
        
        # Check action filter
        if filters.get("action"):
            action_filter = [a.lower() for a in filters["action"]]
            item_action = metadata.get("action", "").lower()
            if not any(action.lower() in item_action for action in action_filter):
                return False
        
        return True


class GenericDatasetAdapter(DatasetAdapter):
    """Generic adapter for custom dataset types with flexible schema"""
    
    def __init__(self, schema: DatasetSchema):
        super().__init__(schema)
        # Build dynamic filter extraction based on schema
        self.filter_field_map = {field: field for field in schema.filter_fields}
    
    def extract_filters(self, items: List[Dict[str, Any]]) -> FilterOptions:
        """Extract filters dynamically based on schema"""
        filter_values = {field: set() for field in self.schema.filter_fields}
        collections = set()
        
        for item in items:
            # Extract collection
            collection = item.get('collection') or item.get('category') or item.get('type', 'unknown')
            collections.add(collection)
            
            # Extract filter values based on schema
            for field in self.schema.filter_fields:
                value = item.get(field) or item.get('metadata', {}).get(field)
                if value:
                    if isinstance(value, list):
                        filter_values[field].update(str(v).lower() for v in value)
                    else:
                        filter_values[field].add(str(value).lower())
        
        # Convert to FilterOptions format
        # For generic datasets, we'll use a flexible structure
        # Map common fields to FilterOptions, others go to metadata
        result = FilterOptions(
            categories=sorted(list(filter_values.get('category', set()))),
            species=sorted(list(filter_values.get('species', set()) or collections)),
            times=sorted(list(filter_values.get('time', set()))),
            seasons=sorted(list(filter_values.get('season', set()))),
            actions=sorted(list(filter_values.get('action', set()))),
            plant_states=sorted(list(filter_values.get('plant_state', set()))),
            collections=sorted(list(collections))
        )
        
        return result
    
    def validate_item(self, item: Dict[str, Any]) -> bool:
        """Validate item against schema"""
        # Check required fields
        for field in self.schema.required_fields:
            if field not in item and field not in item.get('metadata', {}):
                return False
        return True
    
    def normalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize item to standard format"""
        normalized = item.copy()
        # Ensure basic structure
        if 'collection' not in normalized:
            normalized['collection'] = normalized.get('category') or normalized.get('type', 'unknown')
        return normalized
    
    def matches_query(self, item: Dict[str, Any], query: str) -> bool:
        """Generic query matching - search in all text fields"""
        query_lower = query.lower()
        
        # Search in all fields
        for key, value in item.items():
            if isinstance(value, str) and query_lower in value.lower():
                return True
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, str) and query_lower in sub_value.lower():
                        return True
        
        return False
    
    def matches_filters(self, item: Dict[str, Any], filters: Dict[str, List[str]]) -> bool:
        """Generic filter matching"""
        for filter_type, filter_values in filters.items():
            if not filter_values:
                continue
            
            # Check in top-level fields
            item_value = item.get(filter_type, "")
            if item_value:
                if not any(fv.lower() in str(item_value).lower() for fv in filter_values):
                    return False
            
            # Check in metadata
            metadata = item.get('metadata', {})
            item_value = metadata.get(filter_type, "")
            if item_value:
                if not any(fv.lower() in str(item_value).lower() for fv in filter_values):
                    return False
        
        return True


class DatasetAdapterRegistry:
    """Registry for dataset adapters"""
    
    def __init__(self):
        self.adapters: Dict[DatasetType, DatasetAdapter] = {}
        self._register_default_adapters()
    
    def _register_default_adapters(self):
        """Register default adapters"""
        # Register species observation adapter for wildlife, plants, and pests
        species_adapter = SpeciesObservationAdapter()
        self.adapters[DatasetType.WILDLIFE] = species_adapter
        self.adapters[DatasetType.PLANTS] = species_adapter
        self.adapters[DatasetType.PESTS] = species_adapter
    
    def register_adapter(self, dataset_type: DatasetType, adapter: DatasetAdapter):
        """Register a custom adapter for a dataset type"""
        self.adapters[dataset_type] = adapter
        print(f"📦 Registered adapter for dataset type: {dataset_type.value}")
    
    def get_adapter(self, dataset_type: DatasetType) -> DatasetAdapter:
        """Get adapter for a dataset type"""
        adapter = self.adapters.get(dataset_type)
        if not adapter:
            # Fallback to generic adapter
            schema = DatasetSchema(
                dataset_type=dataset_type,
                description=f"Generic adapter for {dataset_type.value} datasets"
            )
            adapter = GenericDatasetAdapter(schema)
            self.adapters[dataset_type] = adapter
        return adapter
    
    def create_custom_adapter(self, dataset_type: DatasetType, schema: DatasetSchema) -> DatasetAdapter:
        """Create and register a custom adapter with a specific schema"""
        adapter = GenericDatasetAdapter(schema)
        self.register_adapter(dataset_type, adapter)
        return adapter

