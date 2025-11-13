# Adding New Dataset Types to the MCP Server

This guide explains how to add support for new dataset types beyond species observation datasets.

## Overview

The MCP server uses an **adapter pattern** to support different dataset types. Each dataset type has its own adapter that handles:
- Filter extraction
- Item validation
- Query matching
- Filter matching
- Data normalization

## Architecture

### Components

1. **DatasetAdapter** (abstract base class) - Defines the interface for dataset adapters
2. **DatasetAdapterRegistry** - Manages registered adapters
3. **DatasetSchema** - Defines the schema for a dataset type
4. **DatasetRegistry** - Uses adapters to load and process datasets

### Current Adapters

- **SpeciesObservationAdapter** - For wildlife, plants, and pests (default)
- **GenericDatasetAdapter** - For custom dataset types with flexible schemas

## Method 1: Using Generic Adapter (Simplest)

For most custom datasets, you can use the generic adapter by specifying the dataset type in your MCP data file.

### Step 1: Create Your MCP Data File

Create a JSON file named `your_dataset_mcp_data.json` in `/opt/mcp-data-server/`:

```json
{
  "dataset_type": "custom",
  "description": "My custom dataset description",
  "schema": {
    "type": "custom",
    "filter_fields": ["category", "status", "location"],
    "metadata_fields": ["timestamp", "source", "notes"]
  },
  "images": [
    {
      "id": "item_001",
      "collection": "category_a",
      "category": "type1",
      "status": "active",
      "location": "site1",
      "metadata": {
        "timestamp": "2024-01-01T00:00:00Z",
        "source": "sensor1",
        "notes": "Additional information"
      }
    }
  ]
}
```

### Step 2: The System Auto-Detects

The system will:
1. Read `dataset_type` from your JSON file
2. Use the `GenericDatasetAdapter` for custom types
3. Extract filters from the fields you specify in `schema.filter_fields`
4. Enable search and filtering based on your schema

## Method 2: Create a Custom Adapter (Advanced)

For datasets with complex logic or specific requirements, create a custom adapter.

### Step 1: Define Your Dataset Type

Add your dataset type to `models.py`:

```python
class DatasetType(str, Enum):
    WILDLIFE = "wildlife"
    PLANTS = "plants"
    PESTS = "pests"
    CUSTOM = "custom"
    YOUR_NEW_TYPE = "your_new_type"  # Add this
```

### Step 2: Create Your Adapter Class

Create a new file `your_dataset_adapter.py`:

```python
from dataset_adapter import DatasetAdapter, DatasetSchema
from models import DatasetType, FilterOptions
from typing import Dict, List, Any

class YourDatasetAdapter(DatasetAdapter):
    """Adapter for your custom dataset type"""
    
    def __init__(self):
        schema = DatasetSchema(
            dataset_type=DatasetType.YOUR_NEW_TYPE,
            required_fields=['id', 'collection'],
            optional_fields=['category', 'metadata'],
            filter_fields=['category', 'status', 'location', 'collection'],
            metadata_fields=['timestamp', 'source', 'notes'],
            description="Your dataset description"
        )
        super().__init__(schema)
    
    def extract_filters(self, items: List[Dict[str, Any]]) -> FilterOptions:
        """Extract available filter options"""
        categories = set()
        statuses = set()
        locations = set()
        collections = set()
        
        for item in items:
            collections.add(item.get('collection', 'unknown'))
            if 'category' in item:
                categories.add(item['category'])
            if 'status' in item:
                statuses.add(item['status'])
            if 'location' in item:
                locations.add(item['location'])
        
        # Map to FilterOptions structure
        # Note: FilterOptions has fixed fields, so map your custom fields appropriately
        return FilterOptions(
            categories=sorted(list(categories)),
            species=sorted(list(locations)),  # Map location to species field
            times=sorted(list(statuses)),     # Map status to times field
            seasons=[],
            actions=[],
            plant_states=[],
            collections=sorted(list(collections))
        )
    
    def validate_item(self, item: Dict[str, Any]) -> bool:
        """Validate that an item conforms to schema"""
        return 'id' in item and 'collection' in item
    
    def normalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize item to standard format"""
        normalized = item.copy()
        if 'collection' not in normalized:
            normalized['collection'] = normalized.get('category', 'unknown')
        return normalized
    
    def matches_query(self, item: Dict[str, Any], query: str) -> bool:
        """Check if item matches search query"""
        query_lower = query.lower()
        
        # Search in relevant fields
        if query_lower in item.get("collection", "").lower():
            return True
        if query_lower in item.get("category", "").lower():
            return True
        if query_lower in item.get("status", "").lower():
            return True
        if query_lower in item.get("location", "").lower():
            return True
        
        # Search in metadata
        metadata = item.get("metadata", {})
        if query_lower in metadata.get("notes", "").lower():
            return True
        
        return False
    
    def matches_filters(self, item: Dict[str, Any], filters: Dict[str, List[str]]) -> bool:
        """Check if item matches the given filters"""
        # Check category filter
        if filters.get("category"):
            category_filter = [c.lower() for c in filters["category"]]
            if item.get("category", "").lower() not in category_filter:
                return False
        
        # Check status filter (mapped to time field in FilterOptions)
        if filters.get("time"):  # Using 'time' field for status
            status_filter = [s.lower() for s in filters["time"]]
            if item.get("status", "").lower() not in status_filter:
                return False
        
        # Check location filter (mapped to species field in FilterOptions)
        if filters.get("species"):  # Using 'species' field for location
            location_filter = [l.lower() for l in filters["species"]]
            if item.get("location", "").lower() not in location_filter:
                return False
        
        return True
```

### Step 3: Register Your Adapter

In your initialization code (or create a plugin file):

```python
from dataset_adapter import DatasetAdapterRegistry
from models import DatasetType
from your_dataset_adapter import YourDatasetAdapter

# Get the adapter registry (usually from DatasetRegistry)
adapter_registry = DatasetAdapterRegistry()

# Register your adapter
your_adapter = YourDatasetAdapter()
adapter_registry.register_adapter(DatasetType.YOUR_NEW_TYPE, your_adapter)
```

### Step 4: Create Your MCP Data File

```json
{
  "dataset_type": "your_new_type",
  "description": "Your dataset description",
  "images": [
    {
      "id": "item_001",
      "collection": "category_a",
      "category": "type1",
      "status": "active",
      "location": "site1",
      "metadata": {
        "timestamp": "2024-01-01T00:00:00Z",
        "source": "sensor1",
        "notes": "Additional information"
      }
    }
  ]
}
```

## Method 3: Extend FilterOptions (For Complex Cases)

If you need completely different filter structures, you can extend the `FilterOptions` model:

### Step 1: Extend FilterOptions

In `models.py`, create a new filter options class:

```python
@dataclass
class CustomFilterOptions(FilterOptions):
    """Extended filter options for custom datasets"""
    status: List[str] = field(default_factory=list)
    location: List[str] = field(default_factory=list)
    # Add your custom fields
```

### Step 2: Update Your Adapter

Modify your adapter to return `CustomFilterOptions` instead of `FilterOptions`.

## Integration with MCP Server

### Updating Search Logic

The MCP server's search logic in `mcp_core_server.py` uses adapters automatically. However, if you need custom search behavior, you can:

1. Override the search tool handler
2. Use the adapter's methods directly:

```python
adapter = dataset_registry.get_adapter_for_dataset(dataset_name)
if adapter:
    filtered_items = [item for item in items if adapter.matches_filters(item, filters)]
```

### Updating Web Interface

The web interface automatically uses the filters extracted by adapters. No changes needed unless you want custom UI elements.

## Best Practices

1. **Use Generic Adapter First**: Try the generic adapter before creating a custom one
2. **Specify Schema in JSON**: Always include `dataset_type` and `schema` in your MCP data files
3. **Normalize Data**: Ensure your data follows a consistent structure
4. **Document Fields**: Document what each field means in your dataset
5. **Test Validation**: Make sure your `validate_item` method catches invalid data

## Example: Adding a Weather Dataset

Here's a complete example for a weather observation dataset:

### 1. MCP Data File (`weather_mcp_data.json`)

```json
{
  "dataset_type": "custom",
  "description": "Weather observation dataset",
  "schema": {
    "type": "custom",
    "filter_fields": ["station", "condition", "temperature_range"],
    "metadata_fields": ["timestamp", "humidity", "pressure", "wind_speed"]
  },
  "images": [
    {
      "id": "weather_001",
      "collection": "station_a",
      "station": "station_a",
      "condition": "sunny",
      "temperature_range": "warm",
      "metadata": {
        "timestamp": "2024-01-01T12:00:00Z",
        "humidity": 65,
        "pressure": 1013.25,
        "wind_speed": 15.5
      }
    }
  ]
}
```

### 2. The System Handles It

The generic adapter will:
- Extract filters: `station`, `condition`, `temperature_range`
- Enable search across all fields
- Support filtering by the specified fields

## Troubleshooting

### Dataset Not Loading

- Check that your JSON file is named `*_mcp_data.json`
- Verify JSON syntax is valid
- Check that `dataset_type` is specified correctly

### Filters Not Appearing

- Ensure `schema.filter_fields` includes the fields you want to filter by
- Check that your data actually contains those fields
- Verify the adapter's `extract_filters` method is working

### Search Not Working

- Implement `matches_query` in your adapter
- Check that your data fields are being searched
- Verify query normalization

## Next Steps

- See `dataset_adapter.py` for the full adapter interface
- Check `dataset_registry.py` for how adapters are used
- Review `mcp_core_server.py` for search integration

