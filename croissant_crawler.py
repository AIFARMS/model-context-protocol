import requests
import requests.exceptions
import json
import re
import asyncio
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
from datetime import datetime
import logging

# Make aiohttp optional (not currently used but imported)
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    # Create a dummy aiohttp module if not available (not actually used in current code)
    class DummyAiohttp:
        pass
    aiohttp = DummyAiohttp()

# Try to import config, fall back to defaults if not available
try:
    from config import CROISSANT_CRAWLER_CONFIG
except (ImportError, FileNotFoundError, OSError) as e:
    # Handle cases where config import fails due to missing directories or other issues
    # This allows the crawler to work even if config.py has initialization issues
    CROISSANT_CRAWLER_CONFIG = {
        "huggingface_datasets": [
            "UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled",
            "AgMMU/AgMMU_v1"
        ],
        "auto_discover": True,
        "discovery_limit": 100,
        "filter_agriculture": True,
        "agriculture_keywords": [
            "agriculture", "agricultural", "farming", "farm", "crop", "plant",
            "livestock", "soil", "harvest", "agronomy", "agtech", "agrifood"
        ],
        "create_synthetic": True,
        "verify_ssl": True
    }

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
    
    def __init__(self, verify_ssl: bool = None, huggingface_datasets: List[str] = None):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AIFARMS-Croissant-Crawler/1.0'
        })
        # Handle SSL verification - can be disabled for problematic sites
        self.verify_ssl = verify_ssl if verify_ssl is not None else CROISSANT_CRAWLER_CONFIG.get("verify_ssl", True)
        if not self.verify_ssl:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self.logger = logging.getLogger(__name__)
        
        # Target portals
        # Note: CyVerse removed due to persistent SSL certificate issues
        self.portals = {
            'aifarms': 'https://data.aifarms.org',
            'agaid_github': 'https://github.com/TrevorBuchanan/AgAIDResearch',
            'huggingface': 'https://huggingface.co/datasets'
        }
        
        # Hugging Face datasets to crawl (configurable)
        self.huggingface_datasets = huggingface_datasets or CROISSANT_CRAWLER_CONFIG.get(
            "huggingface_datasets",
            ["UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled", "AgMMU/AgMMU_v1"]
        )
        self.create_synthetic = CROISSANT_CRAWLER_CONFIG.get("create_synthetic", True)
        self.auto_discover = CROISSANT_CRAWLER_CONFIG.get("auto_discover", True)
        self.discovery_limit = CROISSANT_CRAWLER_CONFIG.get("discovery_limit", 100)
        self.filter_agriculture = CROISSANT_CRAWLER_CONFIG.get("filter_agriculture", True)
        self.agriculture_keywords = [
            kw.strip().lower() for kw in CROISSANT_CRAWLER_CONFIG.get("agriculture_keywords", [
                "agriculture", "agricultural", "farming", "farm", "crop", "plant",
                "livestock", "soil", "harvest", "agronomy", "agtech", "agrifood"
            ])
        ]
    
    async def crawl_all_portals(self) -> List[CroissantDataset]:
        """Crawl all configured portals for Croissant datasets"""
        datasets = []
        
        # Crawl each portal
        for portal_name, portal_url in self.portals.items():
            try:
                print(f"🔍 Crawling {portal_name}: {portal_url}")
                self.logger.info(f"🔍 Crawling {portal_name}: {portal_url}")
                portal_datasets = await self._crawl_portal(portal_name, portal_url)
                print(f"✅ Found {len(portal_datasets)} datasets from {portal_name}")
                self.logger.info(f"✅ Found {len(portal_datasets)} datasets from {portal_name}")
                for ds in portal_datasets:
                    print(f"  - {ds.name}")
                datasets.extend(portal_datasets)
                print(f"📊 Total datasets so far: {len(datasets)}")
            except Exception as e:
                print(f"❌ Error crawling {portal_name}: {e}")
                self.logger.error(f"❌ Error crawling {portal_name}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"📊 Final total: {len(datasets)} datasets from all portals")
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
            response = self.session.get(base_url, verify=self.verify_ssl, timeout=10)
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
            # Note: CyVerse may have SSL certificate issues, so we try with verification first
            # and fall back to unverified if needed
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            try:
                response = self.session.get(base_url, verify=True, timeout=10)
                response.raise_for_status()
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                self.logger.warning(f"⚠️  SSL/connection error for CyVerse, trying without verification: {e}")
                try:
                    response = self.session.get(base_url, verify=False, timeout=10)
                    response.raise_for_status()
                except Exception as e2:
                    self.logger.warning(f"⚠️  Could not connect to CyVerse even without SSL verification: {e2}")
                    return datasets  # Return empty list if we can't connect
            
            # Look for dataset links (avoid matching the base URL itself)
            dataset_patterns = [
                r'href="([^"]*dataset[^"]*)"',
                r'href="([^"]*data[^"]*)"'
            ]
            
            seen_urls = set()  # Track URLs we've already processed
            
            for pattern in dataset_patterns:
                matches = re.findall(pattern, response.text, re.IGNORECASE)
                for match in matches:
                    try:
                        dataset_url = urljoin(base_url, match)
                        
                        # Skip if we've already processed this URL or if it's the base URL
                        if dataset_url in seen_urls or dataset_url == base_url:
                            continue
                        seen_urls.add(dataset_url)
                        
                        # Check if this dataset has Croissant metadata
                        croissant_url = await self._find_croissant_metadata(dataset_url)
                        if croissant_url:
                            dataset = await self._parse_croissant_file(croissant_url, 'cyverse')
                            if dataset:
                                datasets.append(dataset)
                    except Exception as e:
                        # Skip individual dataset errors and continue
                        self.logger.debug(f"Skipping dataset {match} due to error: {e}")
                        continue
            
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
            response = self.session.get(api_url, verify=self.verify_ssl, timeout=10)
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
    
    async def _discover_huggingface_croissant_datasets(self, limit: int = 100) -> List[str]:
        """Discover Hugging Face datasets with Croissant tags using the API"""
        discovered_datasets = []
        
        try:
            # Get datasets from Hugging Face API and filter by tags
            # Note: The API doesn't support direct filtering by tags, so we fetch and filter
            search_url = "https://huggingface.co/api/datasets"
            params = {
                "limit": min(limit * 2, 1000),  # Fetch more to account for filtering
                "sort": "downloads",  # Sort by popularity
                "direction": -1  # Descending order
            }
            
            print(f"🔍 Discovering Croissant datasets from Hugging Face API (checking up to {params['limit']} datasets)...")
            self.logger.info(f"🔍 Discovering Croissant datasets from Hugging Face API...")
            
            response = self.session.get(search_url, params=params, verify=self.verify_ssl, timeout=60)
            response.raise_for_status()
            
            api_results = response.json()
            
            # Check if API returned valid data
            if not isinstance(api_results, list):
                print(f"⚠️  Hugging Face API returned unexpected format: {type(api_results)}")
                self.logger.warning(f"Hugging Face API returned unexpected format: {type(api_results)}")
                return discovered_datasets
            
            print(f"📊 Received {len(api_results)} datasets from Hugging Face API")
            
            # Filter datasets by Croissant-related tags
            croissant_keywords = ['mlcroissant', 'croissant', 'library:mlcroissant']
            
            if isinstance(api_results, list):
                for item in api_results:
                    dataset_id = item.get('id')
                    if not dataset_id:
                        continue
                    
                    # Check tags for Croissant-related keywords
                    tags = item.get('tags', [])
                    has_croissant = False
                    if isinstance(tags, list):
                        tag_strings = [str(tag).lower() for tag in tags]
                        tag_text = ' '.join(tag_strings)
                        if any(keyword in tag_text for keyword in croissant_keywords):
                            has_croissant = True
                    
                    if not has_croissant:
                        continue
                    
                    # If agriculture filtering is enabled, check if dataset is agriculture-related
                    if self.filter_agriculture:
                        is_agriculture = False
                        agriculture_score = 0
                        
                        # Check tags (weight: 2 points)
                        if isinstance(tags, list):
                            tag_strings = [str(tag).lower() for tag in tags]
                            tag_text = ' '.join(tag_strings)
                            for keyword in self.agriculture_keywords:
                                if keyword in tag_text:
                                    agriculture_score += 2
                                    is_agriculture = True
                                    break
                        
                        # Check description (weight: 1 point, but requires agriculture to be prominent)
                        description = item.get('description', '').lower()
                        description_agriculture_count = sum(1 for keyword in self.agriculture_keywords if keyword in description)
                        if description_agriculture_count > 0:
                            # Require at least 2 agriculture keywords in description OR agriculture in title/name
                            if description_agriculture_count >= 2 or 'agriculture' in description[:500]:
                                agriculture_score += 1
                                is_agriculture = True
                        
                        # Check dataset name/ID (weight: 3 points - strong indicator)
                        dataset_id_lower = dataset_id.lower()
                        for keyword in self.agriculture_keywords:
                            if keyword in dataset_id_lower:
                                agriculture_score += 3
                                is_agriculture = True
                                break
                        
                        # Exclude datasets that are multi-discipline benchmarks where agriculture is just one of many subjects
                        # (e.g., MMMU has agriculture as one of 30 subjects but isn't agriculture-focused)
                        if is_agriculture:
                            # Check if it's a multi-discipline benchmark
                            multi_discipline_indicators = ['multi-discipline', 'multidisciplinary', 'multi-subject', 
                                                          'benchmark', 'mmmu', 'general', 'comprehensive', 'massive multi']
                            description_lower = description
                            is_multi_discipline = any(indicator in description_lower for indicator in multi_discipline_indicators)
                            
                            # Check for false positives (datasets that mention agriculture but aren't really about it)
                            false_positive_indicators = ['svhn', 'street view', 'house numbers', 'mnist', 'cifar']
                            is_false_positive = any(fp in description_lower or fp in dataset_id_lower for fp in false_positive_indicators)
                            
                            # If it's multi-discipline and agriculture score is low, exclude it
                            if is_multi_discipline and agriculture_score < 3:
                                print(f"  ⚠️  Skipping {dataset_id}: multi-discipline benchmark with low agriculture focus (score: {agriculture_score})")
                                continue
                            
                            # Exclude false positives
                            if is_false_positive:
                                print(f"  ⚠️  Skipping {dataset_id}: false positive (not actually agriculture-related)")
                                continue
                        
                        if not is_agriculture:
                            continue  # Skip non-agriculture datasets
                    
                    discovered_datasets.append(dataset_id)
                    if len(discovered_datasets) >= limit:
                        break
            
            filter_note = " (agriculture-filtered)" if self.filter_agriculture else ""
            print(f"✅ Discovered {len(discovered_datasets)} datasets with Croissant tags{filter_note} (from {len(api_results) if isinstance(api_results, list) else 0} checked)")
            self.logger.info(f"✅ Discovered {len(discovered_datasets)} datasets with Croissant tags{filter_note}")
            
        except Exception as e:
            print(f"⚠️  Error discovering datasets from Hugging Face API: {e}")
            self.logger.warning(f"Error discovering datasets from Hugging Face API: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to configured list if discovery fails
        
        return discovered_datasets
    
    async def _crawl_huggingface_portal(self, base_url: str) -> List[CroissantDataset]:
        """Crawl Hugging Face datasets portal for Croissant datasets"""
        datasets = []
        
        try:
            # Start with configured datasets
            datasets_to_crawl = list(self.huggingface_datasets)
            
            # If auto-discovery is enabled, discover additional datasets
            if self.auto_discover:
                discovered_datasets = await self._discover_huggingface_croissant_datasets(limit=self.discovery_limit)
                
                # Combine discovered datasets with configured datasets (remove duplicates)
                configured_set = set(self.huggingface_datasets)
                discovered_set = set(discovered_datasets)
                datasets_to_crawl = list(configured_set.union(discovered_set))
                
                print(f"🔍 Crawling {len(datasets_to_crawl)} Hugging Face datasets ({len(configured_set)} configured + {len(discovered_set)} discovered)")
                self.logger.info(f"🔍 Crawling {len(datasets_to_crawl)} Hugging Face datasets ({len(configured_set)} configured + {len(discovered_set)} discovered)")
            else:
                print(f"🔍 Crawling {len(datasets_to_crawl)} configured Hugging Face datasets (auto-discovery disabled)")
                self.logger.info(f"🔍 Crawling {len(datasets_to_crawl)} configured Hugging Face datasets")
            
            for dataset_id in datasets_to_crawl:
                try:
                    dataset_url = f"https://huggingface.co/datasets/{dataset_id}"
                    print(f"🔍 Checking Hugging Face dataset: {dataset_id}")
                    self.logger.info(f"🔍 Checking Hugging Face dataset: {dataset_id}")
                    
                    # Check if the dataset has Croissant metadata
                    croissant_url = await self._find_huggingface_croissant(dataset_url, dataset_id)
                    print(f"🔍 Croissant URL result for {dataset_id}: {croissant_url}")
                    dataset = None
                    
                    if croissant_url:
                        # Try to parse the Croissant file (real or synthetic)
                        print(f"🔍 Parsing Croissant file for {dataset_id}")
                        dataset = await self._parse_croissant_file(croissant_url, 'huggingface')
                        if dataset:
                            # Update the URL to point to the actual dataset page
                            dataset.url = dataset_url
                            datasets.append(dataset)
                            print(f"✅ Found Croissant dataset: {dataset.name}")
                            self.logger.info(f"✅ Found Croissant dataset: {dataset.name}")
                    
                    # If no dataset was found/created and synthetic creation is enabled, create a synthetic one
                    # This ensures we always have a dataset entry for datasets we're specifically looking for
                    if not dataset and self.create_synthetic:
                        print(f"ℹ️  Creating synthetic dataset for {dataset_id}")
                        self.logger.info(f"ℹ️  Creating synthetic dataset for {dataset_id}")
                        synthetic_dataset = await self._parse_huggingface_synthetic(dataset_id, 'huggingface')
                        if synthetic_dataset:
                            synthetic_dataset.url = dataset_url
                            datasets.append(synthetic_dataset)
                            print(f"✅ Created synthetic Croissant dataset: {synthetic_dataset.name}")
                            self.logger.info(f"✅ Created synthetic Croissant dataset: {synthetic_dataset.name}")
                        else:
                            print(f"❌ Failed to create synthetic dataset for {dataset_id}")
                            self.logger.error(f"❌ Failed to create synthetic dataset for {dataset_id}")
                        
                except Exception as e:
                    print(f"❌ Error processing Hugging Face dataset {dataset_id}: {e}")
                    self.logger.error(f"Error processing Hugging Face dataset {dataset_id}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # TODO: In the future, we could crawl the full Hugging Face datasets list
            # by using their API: https://huggingface.co/api/datasets
            
            print(f"📊 Hugging Face crawling complete: found {len(datasets)} datasets")
            self.logger.info(f"📊 Hugging Face crawling complete: found {len(datasets)} datasets")
            
            # If we didn't find any datasets but have configured ones, try to create synthetic datasets
            if len(datasets) == 0 and len(self.huggingface_datasets) > 0:
                print(f"⚠️  No datasets found, but {len(self.huggingface_datasets)} configured. Creating synthetic datasets...")
                for dataset_id in self.huggingface_datasets:
                    try:
                        synthetic_dataset = await self._parse_huggingface_synthetic(dataset_id, 'huggingface')
                        if synthetic_dataset:
                            synthetic_dataset.url = f"https://huggingface.co/datasets/{dataset_id}"
                            datasets.append(synthetic_dataset)
                            print(f"✅ Created synthetic dataset: {synthetic_dataset.name}")
                    except Exception as e:
                        print(f"❌ Failed to create synthetic dataset for {dataset_id}: {e}")
            
        except Exception as e:
            print(f"❌ Error crawling Hugging Face portal: {e}")
            self.logger.error(f"Error crawling Hugging Face portal: {e}")
            import traceback
            traceback.print_exc()
            # If error occurred but we have configured datasets, try to create at least those
            if len(datasets) == 0 and len(self.huggingface_datasets) > 0:
                print(f"⚠️  Error occurred, but trying to create synthetic datasets from configured list...")
                for dataset_id in self.huggingface_datasets:
                    try:
                        synthetic_dataset = await self._parse_huggingface_synthetic(dataset_id, 'huggingface')
                        if synthetic_dataset:
                            synthetic_dataset.url = f"https://huggingface.co/datasets/{dataset_id}"
                            datasets.append(synthetic_dataset)
                            print(f"✅ Created synthetic dataset: {synthetic_dataset.name}")
                    except Exception as e2:
                        print(f"❌ Failed to create synthetic dataset for {dataset_id}: {e2}")
        
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
                    response = self.session.head(croissant_url, verify=self.verify_ssl, timeout=10)
                    if response.status_code == 200:
                        self.logger.info(f"✅ Found Croissant metadata at: {croissant_url}")
                        return croissant_url
                except:
                    continue
            
            # If no direct Croissant file, check if dataset supports Croissant via tags
            api_url = f"https://huggingface.co/api/datasets/{dataset_id}"
            try:
                response = self.session.get(api_url, verify=self.verify_ssl, timeout=10)
                if response.status_code == 200:
                    dataset_info = response.json()
                    
                    # Check if dataset has Croissant-related tags
                    tags = dataset_info.get("tags", [])
                    croissant_tags = [tag for tag in tags if "croissant" in tag.lower()]
                    
                    if croissant_tags:
                        print(f"✅ Found Croissant-enabled dataset: {dataset_id} (tags: {croissant_tags})")
                        self.logger.info(f"✅ Found Croissant-enabled dataset: {dataset_id} (tags: {croissant_tags})")
                        # Create synthetic Croissant metadata URL (special marker)
                        return f"SYNTHETIC_CROISSANT:{dataset_id}"
                    else:
                        print(f"ℹ️  Dataset {dataset_id} does not have explicit Croissant tags, but will create synthetic dataset")
                        self.logger.info(f"ℹ️  Dataset {dataset_id} does not have explicit Croissant tags")
                        # Return None so we create a synthetic dataset anyway
                        return None
                        
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
            response = self.session.get(api_url, verify=self.verify_ssl, timeout=10)
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
                tree_response = self.session.get(tree_url, verify=self.verify_ssl, timeout=10)
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
            # Try with SSL verification first
            try:
                response = self.session.get(dataset_url, verify=True, timeout=10)
                response.raise_for_status()
            except requests.exceptions.SSLError:
                # Fall back to unverified SSL if verification fails
                self.logger.debug(f"SSL verification failed for {dataset_url}, trying without verification")
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                response = self.session.get(dataset_url, verify=False, timeout=10)
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
            self.logger.debug(f"Error finding Croissant metadata for {dataset_url}: {e}")
        
        return None
    
    async def _parse_croissant_file(self, croissant_url: str, source: str) -> Optional[CroissantDataset]:
        """Parse a Croissant metadata file"""
        try:
            # Handle synthetic Croissant metadata from Hugging Face
            if croissant_url.startswith("SYNTHETIC_CROISSANT:"):
                dataset_id = croissant_url.replace("SYNTHETIC_CROISSANT:", "")
                return await self._parse_huggingface_synthetic(dataset_id, source)
            
            response = self.session.get(croissant_url, verify=self.verify_ssl, timeout=10)
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

