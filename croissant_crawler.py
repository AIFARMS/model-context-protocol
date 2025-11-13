import requests
import json
import re
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class CroissantDataset:
    """Represents a discovered Croissant dataset"""
    name: str
    description: str
    url: str
    source_portal: str
    metadata: Dict[str, Any]
    fields: List[Dict[str, Any]]
    license: Optional[str] = None
    created_date: Optional[str] = None
    updated_date: Optional[str] = None
    download_urls: List[str] = None
    keywords: List[str] = None

class CroissantCrawler:
    """Crawls AI Institute portals for Croissant-formatted datasets"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AIFARMS-Croissant-Crawler/1.0'
        })
        self.logger = logging.getLogger(__name__)
        
        # Target portals
        self.portals = {
            'aifarms': 'https://data.aifarms.org',
            'cyverse': 'https://sierra.cyverse.org/datasets',
            'agaid_github': 'https://github.com/TrevorBuchanan/AgAIDResearch',
            'huggingface': 'https://huggingface.co/datasets'
        }
    
    async def crawl_all_portals(self) -> List[CroissantDataset]:
        """Crawl all configured portals for Croissant datasets"""
        datasets = []
        
        # Crawl each portal
        for portal_name, portal_url in self.portals.items():
            try:
                self.logger.info(f"🔍 Crawling {portal_name}: {portal_url}")
                portal_datasets = await self._crawl_portal(portal_name, portal_url)
                datasets.extend(portal_datasets)
                self.logger.info(f"✅ Found {len(portal_datasets)} datasets from {portal_name}")
            except Exception as e:
                self.logger.error(f"❌ Error crawling {portal_name}: {e}")
        
        return datasets
    
    async def _crawl_portal(self, portal_name: str, portal_url: str) -> List[CroissantDataset]:
        """Crawl a specific portal for Croissant datasets"""
        if portal_name == 'aifarms':
            return await self._crawl_aifarms_portal(portal_url)
        elif portal_name == 'cyverse':
            return await self._crawl_cyverse_portal(portal_url)
        elif portal_name == 'agaid_github':
            return await self._crawl_github_repo(portal_url)
        elif portal_name == 'huggingface':
            return await self._crawl_huggingface_portal(portal_url)
        else:
            return []
    
    async def _crawl_aifarms_portal(self, base_url: str) -> List[CroissantDataset]:
        """Crawl AIFARMS data portal for Croissant datasets"""
        datasets = []
        
        try:
            # Get the main page
            response = self.session.get(base_url)
            response.raise_for_status()
            
            # Look for Croissant files (typically .json files with croissant metadata)
            croissant_patterns = [
                r'href="([^"]*\.json[^"]*)"',
                r'href="([^"]*croissant[^"]*)"',
                r'href="([^"]*metadata[^"]*)"'
            ]
            
            for pattern in croissant_patterns:
                matches = re.findall(pattern, response.text, re.IGNORECASE)
                for match in matches:
                    croissant_url = urljoin(base_url, match)
                    dataset = await self._parse_croissant_file(croissant_url, 'aifarms')
                    if dataset:
                        datasets.append(dataset)
            
        except Exception as e:
            self.logger.error(f"Error crawling AIFARMS portal: {e}")
        
        return datasets
    
    async def _crawl_cyverse_portal(self, base_url: str) -> List[CroissantDataset]:
        """Crawl CyVerse Sierra portal for Croissant datasets"""
        datasets = []
        
        try:
            # Get the datasets page
            response = self.session.get(base_url)
            response.raise_for_status()
            
            # Look for dataset links
            dataset_patterns = [
                r'href="([^"]*dataset[^"]*)"',
                r'href="([^"]*data[^"]*)"'
            ]
            
            for pattern in dataset_patterns:
                matches = re.findall(pattern, response.text, re.IGNORECASE)
                for match in matches:
                    dataset_url = urljoin(base_url, match)
                    # Check if this dataset has Croissant metadata
                    croissant_url = await self._find_croissant_metadata(dataset_url)
                    if croissant_url:
                        dataset = await self._parse_croissant_file(croissant_url, 'cyverse')
                        if dataset:
                            datasets.append(dataset)
            
        except Exception as e:
            self.logger.error(f"Error crawling CyVerse portal: {e}")
        
        return datasets
    
    async def _crawl_github_repo(self, repo_url: str) -> List[CroissantDataset]:
        """Crawl GitHub repository for Croissant files"""
        datasets = []
        
        try:
            # Convert GitHub URL to API URL
            api_url = repo_url.replace('github.com', 'api.github.com/repos')
            api_url += '/contents'
            
            # Get repository contents
            response = self.session.get(api_url)
            response.raise_for_status()
            
            contents = response.json()
            
            # Look for Croissant files
            for item in contents:
                if item['type'] == 'file' and (
                    'croissant' in item['name'].lower() or
                    item['name'].endswith('.json') or
                    'metadata' in item['name'].lower()
                ):
                    croissant_url = item['download_url']
                    dataset = await self._parse_croissant_file(croissant_url, 'agaid_github')
                    if dataset:
                        datasets.append(dataset)
            
        except Exception as e:
            self.logger.error(f"Error crawling GitHub repo: {e}")
        
        return datasets
    
    async def _crawl_huggingface_portal(self, base_url: str) -> List[CroissantDataset]:
        """Crawl Hugging Face datasets portal for Croissant datasets"""
        datasets = []
        
        try:
            # For testing, let's start with the specific dataset you found
            test_datasets = [
                'UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled'
            ]
            
            for dataset_id in test_datasets:
                try:
                    dataset_url = f"https://huggingface.co/datasets/{dataset_id}"
                    self.logger.info(f"🔍 Checking Hugging Face dataset: {dataset_id}")
                    
                    # Check if the dataset has Croissant metadata
                    croissant_url = await self._find_huggingface_croissant(dataset_url, dataset_id)
                    if croissant_url:
                        dataset = await self._parse_croissant_file(croissant_url, 'huggingface')
                        if dataset:
                            # Update the URL to point to the actual dataset page
                            dataset.url = dataset_url
                            datasets.append(dataset)
                            self.logger.info(f"✅ Found Croissant dataset: {dataset.name}")
                    else:
                        self.logger.info(f"ℹ️  No Croissant metadata found for {dataset_id}")
                        
                except Exception as e:
                    self.logger.error(f"Error processing Hugging Face dataset {dataset_id}: {e}")
            
            # TODO: In the future, we could crawl the full Hugging Face datasets list
            # by using their API: https://huggingface.co/api/datasets
            
        except Exception as e:
            self.logger.error(f"Error crawling Hugging Face portal: {e}")
        
        return datasets
    
    async def _find_huggingface_croissant(self, dataset_url: str, dataset_id: str) -> Optional[str]:
        """Find Croissant metadata for a Hugging Face dataset"""
        try:
            # Try the standard Croissant metadata path
            croissant_paths = [
                f"https://huggingface.co/datasets/{dataset_id}/raw/main/croissant.json",
                f"https://huggingface.co/datasets/{dataset_id}/raw/main/metadata.json",
                f"https://huggingface.co/datasets/{dataset_id}/raw/main/dataset_infos.json"
            ]
            
            for croissant_url in croissant_paths:
                try:
                    response = self.session.head(croissant_url)
                    if response.status_code == 200:
                        self.logger.info(f"✅ Found Croissant metadata at: {croissant_url}")
                        return croissant_url
                except:
                    continue
            
            # If no direct Croissant file, check if dataset supports Croissant via tags
            api_url = f"https://huggingface.co/api/datasets/{dataset_id}"
            try:
                response = self.session.get(api_url)
                if response.status_code == 200:
                    dataset_info = response.json()
                    
                    # Check if dataset has Croissant-related tags
                    tags = dataset_info.get("tags", [])
                    croissant_tags = [tag for tag in tags if "croissant" in tag.lower()]
                    
                    if croissant_tags:
                        self.logger.info(f"✅ Found Croissant-enabled dataset: {dataset_id} (tags: {croissant_tags})")
                        # Create synthetic Croissant metadata URL (special marker)
                        return f"SYNTHETIC_CROISSANT:{dataset_id}"
                    else:
                        self.logger.info(f"ℹ️  Dataset {dataset_id} does not support Croissant format")
                        
            except Exception as e:
                self.logger.error(f"Error getting dataset info from HF API: {e}")
            
        except Exception as e:
            self.logger.error(f"Error finding Hugging Face Croissant metadata for {dataset_url}: {e}")
        
        return None
    
    async def _create_synthetic_croissant(self, dataset_info: Dict[str, Any], dataset_url: str) -> str:
        """Create synthetic Croissant metadata from Hugging Face dataset info"""
        try:
            # Create a temporary Croissant-like structure
            synthetic_metadata = {
                "@context": "https://schema.org/",
                "@type": "Dataset",
                "name": dataset_info.get("id", "Unknown Dataset"),
                "description": dataset_info.get("description", "No description available"),
                "url": dataset_url,
                "license": dataset_info.get("license", "Unknown"),
                "keywords": dataset_info.get("tags", []),
                "distribution": {
                    "@type": "DataDownload",
                    "contentUrl": dataset_url
                },
                "provider": {
                    "@type": "Organization",
                    "name": "Hugging Face"
                }
            }
            
            # For now, we'll return None since we can't create a temporary file
            # In a real implementation, you might want to create a temporary file
            # or modify the parser to handle dict objects directly
            return None
            
        except Exception as e:
            self.logger.error(f"Error creating synthetic Croissant metadata: {e}")
            return None
    
    async def _parse_huggingface_synthetic(self, dataset_id: str, source: str) -> Optional[CroissantDataset]:
        """Parse synthetic Croissant metadata from Hugging Face API"""
        try:
            # Get dataset info from HF API
            api_url = f"https://huggingface.co/api/datasets/{dataset_id}"
            response = self.session.get(api_url)
            response.raise_for_status()
            
            dataset_info = response.json()
            
            # Extract dataset information
            name = dataset_info.get('id', 'Unknown Dataset')
            description = dataset_info.get('description', 'No description available')
            
            # Extract tags as keywords
            keywords = []
            tags = dataset_info.get('tags', [])
            for tag in tags:
                # Clean up tags (remove prefixes like "license:", "size_categories:", etc.)
                if ':' in tag:
                    keyword = tag.split(':', 1)[1]
                else:
                    keyword = tag
                keywords.append(keyword)
            
            # Extract license
            license_info = None
            for tag in tags:
                if tag.startswith('license:'):
                    license_info = tag.split(':', 1)[1]
                    break
            
            # Extract download URLs (HF datasets are typically accessed via the datasets library)
            download_urls = [
                f"https://huggingface.co/datasets/{dataset_id}",
                f"huggingface://{dataset_id}"  # For datasets library
            ]
            
            # Create basic field information based on common patterns
            fields = []
            
            # Try to get more detailed info about the dataset structure
            try:
                # Check if there are any data files we can inspect
                tree_url = f"https://huggingface.co/api/datasets/{dataset_id}/tree/main"
                tree_response = self.session.get(tree_url)
                if tree_response.status_code == 200:
                    tree_data = tree_response.json()
                    for item in tree_data:
                        if item.get('type') == 'file' and item.get('path', '').endswith('.csv'):
                            # Add a field for CSV data
                            fields.append({
                                'name': 'data',
                                'description': 'Dataset content in CSV format',
                                'data_type': 'csv',
                                'source': {'file': item.get('path', '')}
                            })
                            break
            except Exception as e:
                self.logger.debug(f"Could not get detailed structure for {dataset_id}: {e}")
            
            # If no fields found, add a generic one
            if not fields:
                fields.append({
                    'name': 'data',
                    'description': 'Dataset content',
                    'data_type': 'text',
                    'source': {'url': f"https://huggingface.co/datasets/{dataset_id}"}
                })
            
            # Create CroissantDataset object
            dataset = CroissantDataset(
                name=name,
                description=description[:500] + "..." if len(description) > 500 else description,
                url=f"https://huggingface.co/datasets/{dataset_id}",
                source_portal=source,
                metadata=dataset_info,
                fields=fields,
                license=license_info,
                keywords=keywords,
                download_urls=download_urls,
                created_date=dataset_info.get('createdAt'),
                updated_date=dataset_info.get('lastModified')
            )
            
            self.logger.info(f"✅ Created synthetic Croissant dataset: {name}")
            return dataset
            
        except Exception as e:
            self.logger.error(f"Error parsing Hugging Face synthetic metadata for {dataset_id}: {e}")
            return None
    
    async def _find_croissant_metadata(self, dataset_url: str) -> Optional[str]:
        """Find Croissant metadata file for a dataset"""
        try:
            response = self.session.get(dataset_url)
            response.raise_for_status()
            
            # Look for Croissant metadata links
            croissant_patterns = [
                r'href="([^"]*croissant[^"]*\.json[^"]*)"',
                r'href="([^"]*metadata[^"]*\.json[^"]*)"',
                r'src="([^"]*croissant[^"]*\.json[^"]*)"'
            ]
            
            for pattern in croissant_patterns:
                matches = re.findall(pattern, response.text, re.IGNORECASE)
                for match in matches:
                    return urljoin(dataset_url, match)
            
        except Exception as e:
            self.logger.error(f"Error finding Croissant metadata for {dataset_url}: {e}")
        
        return None
    
    async def _parse_croissant_file(self, croissant_url: str, source: str) -> Optional[CroissantDataset]:
        """Parse a Croissant metadata file"""
        try:
            # Handle synthetic Croissant metadata from Hugging Face
            if croissant_url.startswith("SYNTHETIC_CROISSANT:"):
                dataset_id = croissant_url.replace("SYNTHETIC_CROISSANT:", "")
                return await self._parse_huggingface_synthetic(dataset_id, source)
            
            response = self.session.get(croissant_url)
            response.raise_for_status()
            
            # Parse JSON-LD
            metadata = response.json()
            
            # Extract dataset information
            name = metadata.get('name', 'Unknown Dataset')
            description = metadata.get('description', 'No description available')
            
            # Extract fields
            fields = []
            if 'recordSet' in metadata:
                for record_set in metadata['recordSet']:
                    if 'field' in record_set:
                        for field in record_set['field']:
                            fields.append({
                                'name': field.get('name', 'Unknown Field'),
                                'description': field.get('description', ''),
                                'data_type': field.get('dataType', 'Unknown'),
                                'source': field.get('source', {})
                            })
            
            # Extract download URLs
            download_urls = []
            if 'distribution' in metadata:
                for dist in metadata['distribution']:
                    if 'contentUrl' in dist:
                        download_urls.append(dist['contentUrl'])
            
            # Extract keywords
            keywords = []
            if 'keywords' in metadata:
                keywords = metadata['keywords']
            elif 'about' in metadata:
                keywords = [metadata['about']]
            
            return CroissantDataset(
                name=name,
                description=description,
                url=croissant_url,
                source_portal=source,
                metadata=metadata,
                fields=fields,
                license=metadata.get('license'),
                created_date=metadata.get('dateCreated'),
                updated_date=metadata.get('dateModified'),
                download_urls=download_urls,
                keywords=keywords
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing Croissant file {croissant_url}: {e}")
            return None

# Example usage
async def main():
    crawler = CroissantCrawler()
    datasets = await crawler.crawl_all_portals()
    
    print(f"🔍 Found {len(datasets)} Croissant datasets:")
    for dataset in datasets:
        print(f"  📊 {dataset.name} ({dataset.source_portal})")
        print(f"     Description: {dataset.description[:100]}...")
        print(f"     Fields: {len(dataset.fields)}")
        print(f"     Keywords: {', '.join(dataset.keywords[:3])}")
        print()

if __name__ == "__main__":
    asyncio.run(main())

