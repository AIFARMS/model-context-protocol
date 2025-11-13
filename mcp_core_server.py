#!/usr/bin/env python3
"""
Core MCP Server
- Handles MCP protocol and tool registration
- Extensible tool system for adding new functionalities
- Core service that can be consumed by other applications
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# Import our modular components
from models import SearchRequest, SearchResponse, InferenceRequest, InferenceResult
from tool_registry import ToolRegistry
from dataset_registry import DatasetRegistry, Dataset
from model_registry import ModelRegistry
from config import MCP_CONFIG, MCP_PROTOCOL_CONFIG, BASE_DIR, LLM_CONFIG

# Optional imports
try:
    from llm_service import LLMService
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMService = None

try:
    from croissant_crawler import CroissantCrawler
    CROISSANT_CRAWLER_AVAILABLE = True
except ImportError:
    CROISSANT_CRAWLER_AVAILABLE = False
    CroissantCrawler = None

class MCPServer:
    """Core MCP Server that manages tools and provides MCP protocol endpoints"""
    
    def __init__(self):
        self.name = MCP_PROTOCOL_CONFIG["server_name"]
        self.version = MCP_PROTOCOL_CONFIG["server_version"]
        self.description = MCP_PROTOCOL_CONFIG["server_description"]
        
        print(f"🚀 Initializing {self.name} v{self.version}")
        print(f"📁 Base directory: {BASE_DIR}")
        print(f"📁 Looking for MCP data files in: {BASE_DIR}")
        
        # Check what MCP files exist
        mcp_files = list(BASE_DIR.glob("*_mcp_data.json"))
        print(f"🔍 Found {len(mcp_files)} MCP data files:")
        for f in mcp_files:
            print(f"   - {f.name}")
        
        # Initialize registries
        print("🔧 Initializing tool registry...")
        self.tool_registry = ToolRegistry()
        
        print("📁 Initializing dataset registry...")
        self.dataset_registry = DatasetRegistry()
        
        print("🤖 Initializing model registry...")
        self.model_registry = ModelRegistry()
        
        # Initialize LLM service
        print("🧠 Initializing LLM service...")
        if LLM_AVAILABLE and LLMService:
            self.llm_service = LLMService(
                api_key=LLM_CONFIG.get("api_key"),
                model=LLM_CONFIG.get("model")
            )
            print(f"🧠 LLM service: {'enabled' if self.llm_service.is_available() else 'disabled (fallback to rules)'}")
        else:
            self.llm_service = None
            print("🧠 LLM service: not available (module not found)")
        
        # Setup FastAPI app
        print("🌐 Setting up FastAPI app...")
        self.app = FastAPI(title=self.name, version=self.version)
        self._setup_middleware()
        self._setup_routes()
        self._register_default_tools()
        
        print(f"✅ Initialized {self.name} v{self.version}")
        print(f"📊 Summary:")
        print(f"   - Tools: {len(self.tool_registry.get_all_tools())}")
        print(f"   - Datasets: {len(self.dataset_registry.get_all_datasets())}")
        print(f"   - Models: {len(self.model_registry.get_all_models())}")
    
    def _setup_middleware(self):
        """Setup CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure as needed for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup MCP protocol routes and API endpoints"""
        
        # Health check
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "server": self.name,
                "version": self.version,
                "datasets_loaded": len(self.dataset_registry.datasets),
                "tools_available": len(self.tool_registry.get_all_tools()),
                "models_available": len(self.model_registry.get_all_models())
            }
        
        # MCP Protocol endpoints
        @self.app.get("/mcp")
        async def mcp_info():
            """MCP protocol information"""
            return {
                "name": self.name,
                "version": self.version,
                "description": self.description,
                "capabilities": {
                    "resources": self._get_resources(),
                    "tools": self._get_tools()
                },
                "endpoints": {
                    "mcp_base": "/mcp",
                    "health": "/health",
                    "tools": "/mcp/tools",
                    "resources": "/mcp/resources",
                    "datasets": "/api/datasets",
                    "models": "/api/models"
                }
            }
        
        # MCP Tools
        @self.app.get("/mcp/tools")
        async def list_tools():
            """List all available tools"""
            return {
                "tools": self.tool_registry.get_all_tools(),
                "total": len(self.tool_registry.get_all_tools())
            }
        
        @self.app.post("/mcp/tools/{tool_name}")
        async def execute_tool(tool_name: str, request: Request):
            """Execute a specific tool"""
            try:
                body = await request.json()
                result = await self.tool_registry.execute_tool(tool_name, body)
                return result
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        # MCP Resources
        @self.app.get("/mcp/resources")
        async def list_resources():
            """List all available resources"""
            return {
                "resources": self._get_resources(),
                "total": len(self._get_resources())
            }
        
        @self.app.get("/mcp/resources/{resource_name}")
        async def get_resource(resource_name: str):
            """Get a specific resource"""
            resources = self._get_resources()
            if resource_name not in resources:
                raise HTTPException(status_code=404, detail=f"Resource {resource_name} not found")
            return resources[resource_name]
        
        # Dataset API
        @self.app.get("/api/datasets")
        async def list_datasets():
            """List all available datasets"""
            return {
                "datasets": self.dataset_registry.get_all_datasets(),
                "total": len(self.dataset_registry.get_all_datasets())
            }
        
        @self.app.get("/api/datasets/{dataset_name}")
        async def get_dataset(dataset_name: str):
            """Get specific dataset information"""
            dataset = self.dataset_registry.get_dataset(dataset_name)
            if not dataset:
                raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
            return dataset
        
        @self.app.get("/api/datasets/{dataset_name}/images")
        async def get_dataset_images(dataset_name: str, limit: int = 100, offset: int = 0):
            """Get images from a specific dataset"""
            images = self.dataset_registry.get_images(dataset_name)
            if not images:
                raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found or has no images")
            
            total = len(images)
            paginated_images = images[offset:offset + limit]
            
            return {
                "dataset": dataset_name,
                "images": paginated_images,
                "total_count": total,
                "limit": limit,
                "offset": offset
            }
        
        # Model API
        @self.app.get("/api/models")
        async def list_models():
            """List all available models"""
            return {
                "models": self.model_registry.get_all_models(),
                "total": len(self.model_registry.get_all_models())
            }
        
        @self.app.get("/api/models/{model_name}")
        async def get_model(model_name: str):
            """Get specific model information"""
            model = self.model_registry.get_model(model_name)
            if not model:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
            return model
        
        # Search API
        @self.app.post("/api/search")
        async def search_images(request: Request):
            """Search across all datasets"""
            try:
                body = await request.json()
                search_request = SearchRequest(**body)
                
                # This will be implemented to search across all datasets
                # For now, return a placeholder
                return {
                    "message": "Search functionality will be implemented",
                    "request": body
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        # Datasets API
        @self.app.get("/api/datasets")
        async def get_datasets():
            """Get all available datasets"""
            try:
                datasets = []
                for dataset_name, dataset in self.dataset_registry.datasets.items():
                    # Get images from the registry instead of accessing dataset.images
                    images = self.dataset_registry.get_images(dataset_name)
                    dataset_info = {
                        "name": dataset_name,
                        "description": dataset.description,
                        "type": dataset.dataset_type.value,
                        "image_count": len(images),
                        "collections": list(dataset.collections.keys()),
                        "filters": self._extract_dataset_filters(dataset)
                    }
                    datasets.append(dataset_info)
                
                return {"datasets": datasets}
            except Exception as e:
                print(f"❌ Error getting datasets: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))
        
        # Inference API
        @self.app.post("/api/inference")
        async def run_inference(request: Request):
            """Run model inference"""
            try:
                body = await request.json()
                inference_request = InferenceRequest(**body)
                
                result = await self.model_registry.run_inference(inference_request)
                return result
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
    
    def _get_resources(self) -> Dict[str, Any]:
        """Get available resources"""
        return {
            "mcp://aifarms.org/resources/core": {
                "name": "Core Server Resources",
                "description": "Core MCP server resources and capabilities",
                "schema": {
                    "type": "object",
                    "properties": {
                        "server_info": {"type": "object"},
                        "available_tools": {"type": "array"},
                        "available_datasets": {"type": "array"},
                        "available_models": {"type": "array"}
                    }
                }
            },
            "mcp://aifarms.org/resources/datasets": {
                "name": "Available Datasets",
                "description": "List of all available datasets and their schemas",
                "schema": {
                    "type": "object",
                    "properties": {
                        "datasets": {"type": "array", "items": {"type": "object"}}
                    }
                }
            },
            "mcp://aifarms.org/resources/models": {
                "name": "Available Models",
                "description": "List of all available ML models and their capabilities",
                "schema": {
                    "type": "object",
                    "properties": {
                        "models": {"type": "array", "items": {"type": "object"}}
                    }
                }
            }
        }
    
    def _get_tools(self) -> Dict[str, Any]:
        """Get available tools from registry"""
        tools = {}
        for tool_name, tool_info in self.tool_registry.get_all_tools().items():
            tools[f"mcp://aifarms.org/tools/{tool_name}"] = tool_info
        return tools
    
    def _register_default_tools(self):
        """Register default tools with the server"""
        # Search tool
        self.tool_registry.register_tool(
            name="search_images",
            description="Search for images across all datasets using natural language queries and filters",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "dataset": {"type": "string", "description": "Specific dataset to search in"},
                    "filters": {"type": "object", "description": "Search filters"},
                    "limit": {"type": "integer", "default": 50},
                    "offset": {"type": "integer", "default": 0}
                }
            },
            handler=self._search_tool_handler,
            tags=["search", "images", "datasets"]
        )
        
        # Inference tool
        self.tool_registry.register_tool(
            name="run_inference",
            description="Run ML model inference on images",
            input_schema={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Dataset to run inference on"},
                    "model_name": {"type": "string", "description": "Model to use for inference"},
                    "image_ids": {"type": "array", "items": {"type": "string"}, "description": "Image IDs to process"},
                    "parameters": {"type": "object", "description": "Additional model parameters"}
                }
            },
            handler=self._inference_tool_handler,
            tags=["inference", "ml", "models"]
        )
        
        # LLM-powered search tool
        self.tool_registry.register_tool(
            name="llm_search",
            description="Intelligent search using LLM query understanding",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "dataset": {"type": "string", "description": "Specific dataset to search in"},
                    "limit": {"type": "integer", "default": 50},
                    "offset": {"type": "integer", "default": 0}
                }
            },
            handler=self._llm_search_handler,
            tags=["search", "llm", "intelligent", "semantic"]
        )
        
        # Dataset info tool
        self.tool_registry.register_tool(
            name="get_dataset_info",
            description="Get information about available datasets",
            input_schema={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Specific dataset name (optional)"}
                }
            },
            handler=self._dataset_info_tool_handler,
            tags=["datasets", "info"]
        )
        
        # Model info tool
        self.tool_registry.register_tool(
            name="get_model_info",
            description="Get information about available ML models",
            input_schema={
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Specific model name (optional)"}
                }
            },
            handler=self._model_info_tool_handler,
            tags=["models", "info"]
        )
        
        # Croissant dataset crawler tool
        if CROISSANT_CRAWLER_AVAILABLE:
            self.tool_registry.register_tool(
                name="crawl_croissant_datasets",
                description="Crawl AI Institute portals for Croissant-formatted datasets",
                input_schema={
                    "type": "object",
                    "properties": {}
                },
                handler=self._crawl_croissant_datasets_handler,
                tags=["crawler", "datasets", "croissant"]
            )
    
    # Tool handlers
    def _search_tool_handler(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for search tool"""
        query = input_data.get("query", "")
        dataset = input_data.get("dataset")
        filters = input_data.get("filters", {})
        limit = input_data.get("limit", 50)
        offset = input_data.get("offset", 0)
        
        print(f"🔍 Search request: query='{query}', dataset='{dataset}', filters={filters}")
        
        if dataset:
            # Search in specific dataset using adapter
            print(f"🔍 Searching in specific dataset: {dataset}")
            filtered_results = self.dataset_registry.search_dataset(dataset, query, filters)
            
            if filtered_results is None:
                return {
                    "dataset": dataset,
                    "query": query,
                    "results": [],
                    "total_count": 0,
                    "error": f"Dataset {dataset} not found or has no images"
                }
            
            return {
                "dataset": dataset,
                "query": query,
                "results": filtered_results[offset:offset + limit],
                "total_count": len(filtered_results)
            }
        else:
            # Search across all datasets using adapters
            print(f"🔍 Searching across all datasets")
            all_results = []
            
            for dataset_name in self.dataset_registry.datasets:
                print(f"🔍 Searching dataset: {dataset_name}")
                filtered_results = self.dataset_registry.search_dataset(dataset_name, query, filters)
                if filtered_results:
                    # Add dataset info to each result
                    for result in filtered_results:
                        result['dataset'] = dataset_name
                    all_results.extend(filtered_results)
            
            print(f"🔍 Total results found: {len(all_results)}")
            
            return {
                "query": query,
                "results": all_results[offset:offset + limit],
                "total_count": len(all_results),
                "searched_datasets": list(self.dataset_registry.datasets.keys())
            }
    
    def _apply_search_filters(self, images: List[Dict[str, Any]], query: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply search filters to images (legacy method - now uses adapters via search_dataset)"""
        # This method is kept for backward compatibility but search_dataset now uses adapters
        # The adapter-based search is handled in dataset_registry.search_dataset()
        filtered_images = images.copy()
        
        # Apply text search if query provided
        if query.strip():
            query_lower = query.lower()
            filtered_images = [
                img for img in filtered_images
                if self._image_matches_query(img, query_lower)
            ]
        
        # Apply category filter
        if filters.get("category"):
            category_filter = [c.lower() for c in filters["category"]]
            filtered_images = [
                img for img in filtered_images
                if img.get("category", "").lower() in category_filter
            ]
        
        # Apply species filter
        if filters.get("species"):
            species_filter = [s.lower() for s in filters["species"]]
            filtered_images = [
                img for img in filtered_images
                if img.get("collection", "").lower() in species_filter
            ]
        
        # Apply time filter
        if filters.get("time"):
            time_filter = [t.lower() for t in filters["time"]]
            filtered_images = [
                img for img in filtered_images
                if self._image_matches_time(img, time_filter)
            ]
        
        # Apply season filter
        if filters.get("season"):
            season_filter = [s.lower() for s in filters["season"]]
            filtered_images = [
                img for img in filtered_images
                if self._image_matches_season(img, season_filter)
            ]
        
        return filtered_images
    
    async def _llm_search_handler(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle LLM-powered search with intelligent query understanding"""
        query = input_data.get("query", "")
        dataset = input_data.get("dataset")
        limit = input_data.get("limit", 50)
        offset = input_data.get("offset", 0)
        
        print(f"🧠 LLM search request: query='{query}', dataset='{dataset}'")
        
        try:
            # Get available filters for LLM context
            available_filters = {}
            if dataset:
                if dataset in self.dataset_registry.datasets:
                    available_filters = self._extract_dataset_filters(self.dataset_registry.datasets[dataset])
            else:
                # Combine filters from all datasets
                for dataset_name, dataset_obj in self.dataset_registry.datasets.items():
                    dataset_filters = self._extract_dataset_filters(dataset_obj)
                    for key, values in dataset_filters.items():
                        if key not in available_filters:
                            available_filters[key] = []
                        available_filters[key].extend(values)
                
                # Remove duplicates
                for key in available_filters:
                    available_filters[key] = sorted(list(set(available_filters[key])))
            
            # Use LLM to understand the query
            print(f"🧠 Understanding query with LLM...")
            query_understanding = await self.llm_service.understand_query(query, available_filters)
            
            print(f"🧠 LLM Understanding:")
            print(f"   Intent: {query_understanding.intent}")
            print(f"   Entities: {query_understanding.entities}")
            print(f"   Filters: {query_understanding.filters}")
            print(f"   Confidence: {query_understanding.confidence}")
            print(f"   Reasoning: {query_understanding.reasoning}")
            
            # Perform search using the structured understanding and adapters
            if dataset:
                if dataset in self.dataset_registry.datasets:
                    # Use adapter-based search
                    filtered_results = self.dataset_registry.search_dataset(
                        dataset, query, query_understanding.filters
                    )
                    
                    if not filtered_results:
                        return {
                            "dataset": dataset,
                            "query": query,
                            "llm_understanding": query_understanding,
                            "results": [],
                            "total_count": 0,
                            "error": f"Dataset {dataset} has no matching results"
                        }
                    
                    # Add confidence scores to each result and sort by confidence
                    for result in filtered_results:
                        result['llm_confidence'] = query_understanding.confidence
                        result['llm_reasoning'] = query_understanding.reasoning
                        result['llm_intent'] = query_understanding.intent
                    
                    # Sort by confidence (highest first) and then by relevance
                    filtered_results.sort(key=lambda x: (x.get('llm_confidence', 0), x.get('id', '')), reverse=True)
                    
                    return {
                        "dataset": dataset,
                        "query": query,
                        "llm_understanding": query_understanding,
                        "results": filtered_results[offset:offset + limit],
                        "total_count": len(filtered_results)
                    }
                else:
                    return {
                        "dataset": dataset,
                        "query": query,
                        "llm_understanding": query_understanding,
                        "results": [],
                        "total_count": 0,
                        "error": f"Dataset {dataset} not found"
                    }
            else:
                # Search across all datasets using adapters
                all_results = []
                
                for dataset_name in self.dataset_registry.datasets:
                    print(f"🧠 Searching dataset: {dataset_name}")
                    filtered_results = self.dataset_registry.search_dataset(
                        dataset_name, query, query_understanding.filters
                    )
                    if filtered_results:
                        for result in filtered_results:
                            result['dataset'] = dataset_name
                            result['llm_confidence'] = query_understanding.confidence
                            result['llm_reasoning'] = query_understanding.reasoning
                            result['llm_intent'] = query_understanding.intent
                        all_results.extend(filtered_results)
                
                print(f"🧠 Total LLM-filtered results found: {len(all_results)}")
                
                # Sort by confidence (highest first) and then by dataset for consistency
                all_results.sort(key=lambda x: (x.get('llm_confidence', 0), x.get('dataset', ''), x.get('id', '')), reverse=True)
                
                return {
                    "query": query,
                    "llm_understanding": query_understanding,
                    "results": all_results[offset:offset + limit],
                    "total_count": len(all_results),
                    "searched_datasets": list(self.dataset_registry.datasets.keys())
                }
                
        except Exception as e:
            print(f"❌ LLM search error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "query": query,
                "error": f"LLM search failed: {str(e)}",
                "results": [],
                "total_count": 0
            }
    
    def _apply_llm_filters(self, images: List[Dict[str, Any]], llm_filters: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Apply LLM-understood filters to images"""
        filtered_images = []
        
        for image in images:
            if self._image_matches_llm_filters(image, llm_filters):
                filtered_images.append(image)
        
        return filtered_images
    
    def _image_matches_llm_filters(self, image: Dict[str, Any], llm_filters: Dict[str, List[str]]) -> bool:
        """Check if image matches LLM-understood filters"""
        # Check each filter category
        for filter_type, filter_values in llm_filters.items():
            if not filter_values:  # Skip empty filters
                continue
                
            if filter_type == "species":
                if not self._image_matches_species(image, filter_values):
                    return False
            elif filter_type == "time":
                if not self._image_matches_time(image, filter_values):
                    return False
            elif filter_type == "season":
                if not self._image_matches_season(image, filter_values):
                    return False
            elif filter_type == "action":
                if not self._image_matches_action(image, filter_values):
                    return False
            elif filter_type == "scene":
                if not self._image_matches_scene(image, filter_values):
                    return False
            elif filter_type == "weather":
                if not self._image_matches_weather(image, filter_values):
                    return False
            elif filter_type == "category":
                if not self._image_matches_category(image, filter_values):
                    return False
        
        return True
    
    def _image_matches_species(self, image: Dict[str, Any], species_filters: List[str]) -> bool:
        """Check if image matches species filters"""
        if not species_filters:
            return True
        
        # Look for species in metadata, not top-level
        image_species = image.get("metadata", {}).get("species", "").lower()
        if not image_species:
            return True
        
        return any(species.lower() in image_species for species in species_filters)
    
    def _image_matches_action(self, image: Dict[str, Any], action_filters: List[str]) -> bool:
        """Check if image matches action filters"""
        if not action_filters:
            return True
        
        image_action = image.get("metadata", {}).get("action", "").lower()
        if not image_action:
            return True
        
        return any(action.lower() in image_action for action in action_filters)
    
    def _image_matches_scene(self, image: Dict[str, Any], scene_filters: List[str]) -> bool:
        """Check if image matches scene filters"""
        if not scene_filters:
            return True
        
        image_scene = image.get("metadata", {}).get("scene", "").lower()
        if not image_scene:
            return True
        
        return any(scene.lower() in image_scene for scene in scene_filters)
    
    def _image_matches_weather(self, image: Dict[str, Any], weather_filters: List[str]) -> bool:
        """Check if image matches weather filters"""
        if not weather_filters:
            return True
        
        image_weather = image.get("metadata", {}).get("weather", "").lower()
        if not image_weather:
            return True
        
        return any(weather.lower() in image_weather for weather in weather_filters)
    
    def _image_matches_category(self, image: Dict[str, Any], category_filters: List[str]) -> bool:
        """Check if image matches category filters"""
        if not category_filters:
            return True
        
        image_category = image.get("category", "").lower()
        if not image_category:
            return True
        
        return any(category.lower() in image_category for category in category_filters)
    
    def _image_matches_query(self, image: Dict[str, Any], query: str) -> bool:
        """Check if image matches search query"""
        # Check collection name
        if query in image.get("collection", "").lower():
            return True
        
        # Check metadata description
        metadata = image.get("metadata", {})
        if query in metadata.get("description", "").lower():
            return True
        
        # Check metadata action
        if query in metadata.get("action", "").lower():
            return True
        
        # Check metadata scene
        if query in metadata.get("scene", "").lower():
            return True
        
        return False
    
    def _image_matches_time(self, image: Dict[str, Any], time_filters: List[str]) -> bool:
        """Check if image matches time filters"""
        metadata = image.get("metadata", {})
        time_info = metadata.get("time", "").lower()
        
        for time_filter in time_filters:
            if time_filter == "night" and ("night" in time_info or "dark" in time_info):
                return True
            elif time_filter == "day" and ("day" in time_info or "morning" in time_info or "afternoon" in time_info):
                return True
            elif time_filter == "dawn" and ("dawn" in time_info or "sunrise" in time_info):
                return True
            elif time_filter == "dusk" and ("dusk" in time_info or "sunset" in time_info):
                return True
        
        return False
    
    def _image_matches_season(self, image: Dict[str, Any], season_filters: List[str]) -> bool:
        """Check if image matches season filters"""
        if not season_filters:
            return True
        
        image_season = image.get("metadata", {}).get("season", "").lower()
        if not image_season:
            return True
        
        return any(season.lower() in image_season for season in season_filters)
    
    def _extract_dataset_filters(self, dataset: Dataset) -> Dict[str, List[str]]:
        """Extract available filters from a dataset"""
        # Get images from the registry instead of accessing dataset.images
        dataset_name = dataset.name
        images = self.dataset_registry.get_images(dataset_name)
        
        filters = {
            "categories": [],
            "species": [],
            "times": [],
            "seasons": [],
            "actions": [],
            "plant_states": []
        }
        
        for image in images:
            # Extract category
            if image.get("category"):
                filters["categories"].append(image["category"])
            
            # Extract species
            if image.get("species"):
                filters["species"].append(image["species"])
            
            # Extract time from metadata
            if image.get("metadata", {}).get("time"):
                filters["times"].append(image["metadata"]["time"])
            
            # Extract season from metadata
            if image.get("metadata", {}).get("season"):
                filters["seasons"].append(image["metadata"]["season"])
            
            # Extract action from metadata
            if image.get("metadata", {}).get("action"):
                filters["actions"].append(image["metadata"]["action"])
            
            # Extract plant state from metadata
            if image.get("metadata", {}).get("plant_state"):
                filters["plant_states"].append(image["metadata"]["plant_state"])
        
        # Remove duplicates and sort
        for key in filters:
            filters[key] = sorted(list(set(filters[key])))
        
        return filters
    
    def _inference_tool_handler(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for inference tool"""
        dataset_name = input_data["dataset_name"]
        model_name = input_data["model_name"]
        image_ids = input_data["image_ids"]
        parameters = input_data.get("parameters", {})
        
        # Create inference request
        inference_request = InferenceRequest(
            dataset_name=dataset_name,
            model_name=model_name,
            image_ids=image_ids,
            parameters=parameters
        )
        
        # Run inference (this would be async in practice)
        # For now, return a placeholder
        return {
            "message": "Inference tool handler - will be implemented with async support",
            "request": input_data
        }
    
    def _dataset_info_tool_handler(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for dataset info tool"""
        dataset_name = input_data.get("dataset_name")
        
        if dataset_name:
            dataset = self.dataset_registry.get_dataset(dataset_name)
            if not dataset:
                return {"error": f"Dataset {dataset_name} not found"}
            return {"dataset": dataset}
        else:
            return {
                "datasets": self.dataset_registry.get_all_datasets(),
                "total": len(self.dataset_registry.get_all_datasets())
            }
    
    def _model_info_tool_handler(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for model info tool"""
        model_name = input_data.get("model_name")
        
        if model_name:
            model = self.model_registry.get_model(model_name)
            if not model:
                return {"error": f"Model {model_name} not found"}
            return {"model": model}
        else:
            return {
                "models": self.model_registry.get_all_models(),
                "total": len(self.model_registry.get_all_models())
            }
    
    async def _crawl_croissant_datasets_handler(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Croissant dataset crawling"""
        if not CROISSANT_CRAWLER_AVAILABLE:
            return {
                "error": "Croissant crawler not available. Ensure croissant_crawler.py is in the same directory.",
                "datasets": [],
                "total_count": 0
            }
        
        try:
            print(f"🔍 Starting Croissant dataset crawling...")
            
            crawler = CroissantCrawler()
            datasets = await crawler.crawl_all_portals()
            
            # Convert to MCP format
            results = []
            for dataset in datasets:
                results.append({
                    'name': dataset.name,
                    'description': dataset.description,
                    'url': dataset.url,
                    'source': dataset.source_portal,
                    'fields': dataset.fields,
                    'keywords': dataset.keywords or [],
                    'license': dataset.license,
                    'download_urls': dataset.download_urls or [],
                    'created_date': dataset.created_date,
                    'updated_date': dataset.updated_date
                })
            
            print(f"✅ Found {len(results)} Croissant datasets")
            
            from datetime import datetime
            return {
                "datasets": results,
                "total_count": len(results),
                "sources": list(crawler.portals.keys()),
                "crawl_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Croissant crawling error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": f"Croissant crawling failed: {str(e)}",
                "datasets": [],
                "total_count": 0
            }
    
    def register_tool(self, name: str, description: str, input_schema: Dict[str, Any], 
                      handler: Callable, tags: List[str] = None):
        """Register a new tool with the server"""
        self.tool_registry.register_tool(name, description, input_schema, handler, tags)
    
    def get_tool_registry(self) -> ToolRegistry:
        """Get the tool registry for external access"""
        return self.tool_registry
    
    def get_dataset_registry(self) -> DatasetRegistry:
        """Get the dataset registry for external access"""
        return self.dataset_registry
    
    def get_model_registry(self) -> ModelRegistry:
        """Get the model registry for external access"""
        return self.model_registry
    
    def run(self, host: str = None, port: int = None):
        """Run the MCP server"""
        host = host or MCP_CONFIG.get("mcp_host", "0.0.0.0")
        port = port or MCP_CONFIG.get("mcp_port", 8188)
        
        print(f"🚀 Starting {self.name} on {host}:{port}")
        print(f"🔧 Available tools: {len(self.tool_registry.get_all_tools())}")
        print(f"📁 Available datasets: {len(self.dataset_registry.get_all_datasets())}")
        print(f"🤖 Available models: {len(self.model_registry.get_all_models())}")
        print(f"🔗 MCP Discovery: http://{host}:{port}/.well-known/mcp")
        print(f"🔧 Tools endpoint: http://{host}:{port}/mcp/tools")
        print(f"📁 Datasets endpoint: http://{host}:{port}/api/datasets")
        print(f"🤖 Models endpoint: http://{host}:{port}/api/models")
        print(f"💚 Health check: http://{host}:{port}/health")
        
        uvicorn.run(self.app, host=host, port=port)

# Global MCP server instance
mcp_server = MCPServer()

if __name__ == "__main__":
    # Run the core MCP server
    mcp_server.run()
