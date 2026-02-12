#!/usr/bin/env python3
"""
Add dataset provenance, citation, and documentation information to MCP JSON files.

This script adds provenance metadata to MCP format files so users know:
- Where images come from
- How to cite the dataset
- License information
- Links to original dataset sources
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import shared provenance data
try:
    from convert_to_mcp_format import DATASET_PROVENANCE, get_provenance_for_species, get_provenance_by_key
except ImportError:
    # Fallback if import fails
    DATASET_PROVENANCE = {
        "idaho_camera_traps": {
            "dataset_name": "Idaho Camera Traps",
            "dataset_url": "https://lila.science/datasets/idaho-camera-traps/",
            "source_organization": "LILA (Labeled Information Library of Alexandria)",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "citation": "Tabak, M. A., Norouzzadeh, M. S., Wolfson, D. W., Sweeney, S. J., Vercauteren, K. C., Snow, N. P., Halseth, J. M., Di Salvo, P. A., Lewis, J. S., White, M. D., Teton, B., Beasley, J. C., Schlichting, P. E., Boughton, R. K., Wight, B., Newkirk, E. S., Ivan, J. S., Odell, E. A., Brook, R. K., Lukacs, P. M., Moeller, A. K., Mandeville, E. G., Clune, J., & Miller, R. S. (2019). Machine learning to classify animal species in camera trap images: Applications in ecology. Methods in Ecology and Evolution, 10(4), 585-590. https://doi.org/10.1111/2041-210X.13120",
            "description": "Camera trap images from Idaho, USA, containing various wildlife species captured in natural settings. Images are captured using motion-activated trail cameras in forest and meadow environments.",
            "geographic_location": "Idaho, USA",
            "data_collection_period": "2014-2016",
            "image_types": ["Daylight", "Infrared/Night vision"],
            "species_included": [
                "bobcat", "coyote", "american_crow", "goat", "chicken", 
                "striped_skunk", "deer", "elk", "bear", "mountain_lion"
            ],
            "acknowledgment": "We acknowledge the use of images provided by the Idaho Department of Fish and Game"
        }
    }
    
    def get_provenance_for_species(species_name: str, dataset_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Determine which dataset provenance to use based on species name."""
        if dataset_key and dataset_key in DATASET_PROVENANCE:
            return DATASET_PROVENANCE[dataset_key]
        species_lower = species_name.lower().replace(" ", "_").replace("-", "_")
        for dataset_key, dataset_info in DATASET_PROVENANCE.items():
            species_list = dataset_info.get("species_included", [])
            if species_lower in [s.lower() for s in species_list]:
                return dataset_info
        return None
    
    def get_provenance_by_key(dataset_key: str) -> Optional[Dict[str, Any]]:
        """Get provenance information by explicit dataset key."""
        return DATASET_PROVENANCE.get(dataset_key)

def add_provenance_to_mcp_file(
    file_path: Path,
    provenance: Optional[Dict[str, Any]] = None,
    species_name: Optional[str] = None,
    dataset_key: Optional[str] = None,
    backup: bool = True
) -> bool:
    """
    Add provenance information to an MCP JSON file.
    
    Args:
        file_path: Path to MCP JSON file
        provenance: Provenance dictionary (if None, will try to infer from species or dataset_key)
        species_name: Species name to help determine provenance
        dataset_key: Explicit dataset key (e.g., "idaho_camera_traps") to use
        backup: Whether to create a backup before modifying
    """
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    print(f"📖 Reading MCP file: {file_path}")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return False
    
    # Determine provenance if not provided
    if not provenance:
        # Try explicit dataset key first
        if dataset_key:
            provenance = get_provenance_by_key(dataset_key)
            if provenance:
                print(f"📋 Using dataset: {provenance.get('dataset_name', dataset_key)}")
        else:
            if not species_name:
                # Try to infer from filename
                species_name = file_path.stem.replace('_mcp_data', '')
            provenance = get_provenance_for_species(species_name, dataset_key)
    
    if not provenance:
        print(f"⚠️  No provenance information found for species: {species_name}")
        if dataset_key:
            print(f"   Dataset key '{dataset_key}' not found in available datasets")
        print(f"   File will be updated but without provenance metadata")
        print(f"   Options:")
        print(f"     - Use --dataset-key to specify a dataset (run with --list-datasets to see options)")
        print(f"     - Use --provenance-file to load custom provenance from a JSON file")
        print(f"     - Add the dataset to DATASET_PROVENANCE in convert_to_mcp_format.py")
        # Still proceed to add structure (empty provenance)
        provenance = {}
    
    # Create backup if requested
    if backup and file_path.exists():
        backup_file = file_path.with_suffix('.json.backup')
        print(f"📁 Creating backup: {backup_file}")
        with open(backup_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    # Add provenance information at top level
    if 'provenance' not in data:
        data['provenance'] = {}
    
    # Update provenance fields
    if provenance:
        data['provenance'].update({
            "dataset_name": provenance.get("dataset_name", ""),
            "dataset_url": provenance.get("dataset_url", ""),
            "source_organization": provenance.get("source_organization", ""),
            "license": provenance.get("license", ""),
            "license_url": provenance.get("license_url", ""),
            "citation": provenance.get("citation", ""),
            "description": provenance.get("description", ""),
            "geographic_location": provenance.get("geographic_location", ""),
            "data_collection_period": provenance.get("data_collection_period", ""),
            "image_types": provenance.get("image_types", []),
            "acknowledgment": provenance.get("acknowledgment", ""),
        })
    
    # Add citation information (separate field for easy access)
    if provenance:
        data['citation'] = provenance.get("citation", "")
    
    # Add acknowledgment (separate field for easy access)
    if provenance:
        data['acknowledgment'] = provenance.get("acknowledgment", "")
    
    # Add dataset metadata
    if 'dataset_metadata' not in data:
        data['dataset_metadata'] = {}
    
    if provenance:
        data['dataset_metadata'].update({
            "source_url": provenance.get("dataset_url", ""),
            "license": provenance.get("license", ""),
            "license_url": provenance.get("license_url", ""),
            "geographic_location": provenance.get("geographic_location", ""),
            "collection_period": provenance.get("data_collection_period", ""),
        })
    
    # Save updated file
    print(f"💾 Saving updated file: {file_path}")
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Provenance information added successfully!")
    if provenance:
        print(f"   Dataset: {provenance.get('dataset_name', 'Unknown')}")
        print(f"   URL: {provenance.get('dataset_url', 'N/A')}")
        print(f"   License: {provenance.get('license', 'N/A')}")
    
    return True

def add_provenance_to_multiple_files(
    directory: Path,
    provenance: Optional[Dict[str, Any]] = None,
    pattern: str = "*_mcp_data.json",
    backup: bool = True,
    dataset_key: Optional[str] = None
) -> int:
    """
    Add provenance to multiple MCP files in a directory.
    
    Returns number of files successfully updated.
    """
    mcp_files = list(directory.glob(pattern))
    print(f"🔍 Found {len(mcp_files)} MCP files in {directory}")
    
    updated_count = 0
    for mcp_file in mcp_files:
        print(f"\n{'='*60}")
        print(f"Processing: {mcp_file.name}")
        print(f"{'='*60}")
        
        # Try to infer species from filename
        species_name = mcp_file.stem.replace('_mcp_data', '')
        
        if add_provenance_to_mcp_file(mcp_file, provenance, species_name, dataset_key, backup):
            updated_count += 1
    
    return updated_count

def load_provenance_from_file(provenance_file: Path) -> Dict[str, Any]:
    """Load provenance information from a JSON file."""
    try:
        with open(provenance_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading provenance file: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(
        description="Add dataset provenance information to MCP JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add provenance to a single file (auto-detect from species)
  python add_dataset_provenance.py bobcat_mcp_data.json
  
  # Add provenance to a single file with specific species
  python add_dataset_provenance.py coyote_mcp_data.json --species coyote
  
  # Add provenance to all files in a directory
  python add_dataset_provenance.py --directory mcp_json/
  
  # Use custom provenance from file
  python add_dataset_provenance.py dataset_mcp_data.json --provenance-file custom_provenance.json
        """
    )
    
    parser.add_argument("file", nargs="?", type=str, default=None,
                       help="Path to MCP JSON file (or use --directory for multiple files)")
    parser.add_argument("--directory", "-d", type=str, default=None,
                       help="Directory containing MCP JSON files to update")
    parser.add_argument("--species", "-s", type=str, default=None,
                       help="Species name (helps determine provenance)")
    parser.add_argument("--dataset-key", "-k", type=str, default=None,
                       help="Explicit dataset key (e.g., 'idaho_camera_traps'). Use --list-datasets to see available keys")
    parser.add_argument("--list-datasets", action="store_true",
                       help="List all available dataset keys and exit")
    parser.add_argument("--provenance-file", "-p", type=str, default=None,
                       help="Path to JSON file with custom provenance information")
    parser.add_argument("--no-backup", action="store_true",
                       help="Don't create backup files")
    parser.add_argument("--pattern", type=str, default="*_mcp_data.json",
                       help="File pattern to match (default: *_mcp_data.json)")
    
    args = parser.parse_args()
    
    # List available datasets if requested
    if args.list_datasets:
        print("Available datasets:")
        print("=" * 60)
        for key, info in DATASET_PROVENANCE.items():
            print(f"\nKey: {key}")
            print(f"  Name: {info.get('dataset_name', 'N/A')}")
            print(f"  URL: {info.get('dataset_url', 'N/A')}")
            species = info.get('species_included', [])
            if species:
                print(f"  Species: {', '.join(species[:5])}{'...' if len(species) > 5 else ''}")
        print("\n" + "=" * 60)
        return 0
    
    # Load custom provenance if provided
    provenance = None
    if args.provenance_file:
        provenance_file = Path(args.provenance_file)
        if provenance_file.exists():
            provenance = load_provenance_from_file(provenance_file)
            print(f"📖 Loaded provenance from: {provenance_file}")
        else:
            print(f"❌ Provenance file not found: {provenance_file}")
            return 1
    
    # Process files
    if args.directory:
        # Process directory
        directory = Path(args.directory)
        if not directory.exists():
            print(f"❌ Directory not found: {directory}")
            return 1
        
        updated_count = add_provenance_to_multiple_files(
            directory, provenance, args.pattern, not args.no_backup, args.dataset_key
        )
        print(f"\n{'='*60}")
        print(f"✅ Updated {updated_count} files")
        print(f"{'='*60}")
        
    elif args.file:
        # Process single file
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return 1
        
        success = add_provenance_to_mcp_file(
            file_path, provenance, args.species, args.dataset_key, not args.no_backup
        )
        return 0 if success else 1
    
    else:
        parser.print_help()
        print("\n❌ Please specify either a file or --directory")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

