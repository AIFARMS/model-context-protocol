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


def _metadata_str(val: Any) -> str:
    """Normalize a metadata field value to a string (item metadata can have list values e.g. action)."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        return " ".join(str(x).strip() for x in val if x is not None and str(x).strip())
    return str(val)


from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# Import our modular components
from models import SearchRequest, SearchResponse, InferenceRequest, InferenceResult, DatasetType
from tool_registry import ToolRegistry
from dataset_registry import DatasetRegistry, Dataset
from model_registry import ModelRegistry
from config import MCP_CONFIG, MCP_PROTOCOL_CONFIG, BASE_DIR, LLM_CONFIG, IMAGES_DIR, IMAGES_TRY_SPECIES_FIRST

# Optional imports
try:
    from llm_service import LLMService
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMService = None

# IMPORTANT: This import happens at module load time
# Make sure croissant_crawler.py is in the same directory as this file
print("=" * 60)
print("🔍 CHECKING CROISSANT CRAWLER IMPORT...")
print("=" * 60)
try:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"📁 Current file directory: {current_dir}")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📁 Python path (first 3): {sys.path[:3]}")
    
    crawler_file = os.path.join(current_dir, "croissant_crawler.py")
    print(f"📄 Looking for: {crawler_file}")
    print(f"📄 File exists: {os.path.exists(crawler_file)}")
    
    if not os.path.exists(crawler_file):
        # Try current directory
        crawler_file = "croissant_crawler.py"
        print(f"📄 Trying current dir: {crawler_file}")
        print(f"📄 File exists: {os.path.exists(crawler_file)}")
    
    print(f"🔍 Attempting import...")
    from croissant_crawler import CroissantCrawler
    CROISSANT_CRAWLER_AVAILABLE = True
    print("=" * 60)
    print("✅ SUCCESS: Croissant crawler imported successfully!")
    print(f"   CroissantCrawler class: {CroissantCrawler}")
    print("=" * 60)
except ImportError as e:
    CROISSANT_CRAWLER_AVAILABLE = False
    CroissantCrawler = None
    print("=" * 60)
    print(f"❌ FAILED: Croissant crawler import error (ImportError)")
    print(f"   Error: {e}")
    print("=" * 60)
    import traceback
    traceback.print_exc()
    print("=" * 60)
except Exception as e:
    CROISSANT_CRAWLER_AVAILABLE = False
    CroissantCrawler = None
    print("=" * 60)
    print(f"❌ FAILED: Croissant crawler import error (Other)")
    print(f"   Error: {e}")
    print("=" * 60)
    import traceback
    traceback.print_exc()
    print("=" * 60)

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
        # Check environment variables (Azure takes precedence over OpenAI when set)
        import os
        google_key = os.getenv("GOOGLE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        print(f"🧠 Environment check:")
        print(f"   AZURE_OPENAI_ENDPOINT: {'SET' if azure_endpoint else 'NOT SET'}")
        print(f"   AZURE_OPENAI_API_KEY: {'SET' if azure_key else 'NOT SET'}")
        print(f"   OPENAI_API_KEY: {'SET' if openai_key else 'NOT SET'}")
        print(f"   GOOGLE_API_KEY: {'SET' if google_key else 'NOT SET'}")
        if google_key:
            print(f"   GOOGLE_API_KEY length: {len(google_key)}")
        
        if LLM_AVAILABLE and LLMService:
            self.llm_service = LLMService(
                api_key=LLM_CONFIG.get("api_key"),
                model=LLM_CONFIG.get("model"),
                azure_endpoint=LLM_CONFIG.get("azure_endpoint") or None,
                azure_api_key=LLM_CONFIG.get("azure_api_key") or None,
                azure_deployment=LLM_CONFIG.get("azure_deployment") or None,
                azure_api_version=LLM_CONFIG.get("azure_api_version") or None,
            )
            print(f"🧠 LLM service: {'enabled' if self.llm_service.is_available() else 'disabled (fallback to rules)'}")
            if self.llm_service:
                print(f"   OpenAI available: {self.llm_service.openai_available}")
                print(f"   Gemini available: {self.llm_service.gemini_available}")
                print(f"   Provider: {self.llm_service.provider}")
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
        
        @self.app.get("/mcp/tools/{tool_name}")
        async def execute_tool_get(tool_name: str, request: Request):
            """Execute a tool via GET (for tools that don't require input)"""
            # Check if tool exists
            if tool_name not in self.tool_registry.tools:
                available_tools = list(self.tool_registry.tools.keys())
                raise HTTPException(
                    status_code=404, 
                    detail=f"Tool '{tool_name}' not found. Available tools: {', '.join(available_tools)}. Use POST /mcp/tools/{tool_name} with JSON body for tools that require input."
                )
            
            # For GET requests, use empty body (only works for tools that don't require input)
            body = {}
            
            # Check if tool requires input by looking at its schema
            tool = self.tool_registry.get_tool(tool_name)
            if tool and tool.input_schema.get("properties"):
                # Tool has required properties - suggest using POST
                required = tool.input_schema.get("required", [])
                if required:
                    raise HTTPException(
                        status_code=405,
                        detail=f"Tool '{tool_name}' requires input parameters: {', '.join(required)}. Please use POST /mcp/tools/{tool_name} with a JSON body."
                    )
            
            print(f"🔧 Executing tool via GET: {tool_name}")
            print(f"   Input data: {body}")
            
            try:
                result = await self.tool_registry.execute_tool(tool_name, body)
                print(f"✅ Tool {tool_name} executed successfully")
                return result
            except Exception as e:
                print(f"❌ Tool execution error: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(
                    status_code=500, 
                    detail=f"Tool execution failed: {str(e)}. Try using POST /mcp/tools/{tool_name} with a JSON body."
                )
        
        @self.app.post("/mcp/tools/{tool_name}")
        async def execute_tool(tool_name: str, request: Request):
            """Execute a specific tool"""
            try:
                # Check if tool exists first
                if tool_name not in self.tool_registry.tools:
                    available_tools = list(self.tool_registry.tools.keys())
                    raise HTTPException(
                        status_code=404, 
                        detail=f"Tool '{tool_name}' not found. Available tools: {', '.join(available_tools)}"
                    )
                
                # Handle empty body or missing JSON
                try:
                    body = await request.json()
                except Exception as json_error:
                    # If no JSON body, use empty dict (some tools don't need input)
                    body = {}
                    print(f"⚠️  No JSON body provided, using empty dict: {json_error}")
                
                print(f"🔧 Executing tool: {tool_name}")
                print(f"   Input data: {body}")
                
                result = await self.tool_registry.execute_tool(tool_name, body)
                print(f"✅ Tool {tool_name} executed successfully")
                return result
            except HTTPException:
                # Re-raise HTTP exceptions as-is
                raise
            except ValueError as e:
                # Tool not found or validation error
                print(f"❌ Tool execution error (ValueError): {e}")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                # Other errors
                print(f"❌ Tool execution error: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")
        
        # OPTIMIZATION: Cache directory listing to avoid repeated iterations
        _image_dir_cache = {}
        _image_dir_cache_time = {}
        _image_dir_cache_ttl = 60  # Cache for 60 seconds
        
        def _get_cached_dir_listing(images_dir: Path) -> List[Path]:
            """Get cached directory listing to avoid repeated iterations"""
            import time
            current_time = time.time()
            
            dir_str = str(images_dir)
            if dir_str in _image_dir_cache:
                cache_time = _image_dir_cache_time.get(dir_str, 0)
                if current_time - cache_time < _image_dir_cache_ttl:
                    return _image_dir_cache[dir_str]
            
            # Cache miss - refresh cache
            try:
                listing = list(images_dir.iterdir())
                _image_dir_cache[dir_str] = listing
                _image_dir_cache_time[dir_str] = current_time
                return listing
            except Exception as e:
                print(f"⚠️  Error listing directory {images_dir}: {e}")
                return []
        
        # Image serving endpoint
        @self.app.get("/images/{filename:path}")
        async def serve_image(filename: str):
            """Serve images: order by IMAGES_TRY_SPECIES_FIRST (Taiga = subdirs only → species subdir first)."""
            try:
                from pathlib import Path
                from fastapi.responses import FileResponse
                
                images_dir = Path(IMAGES_DIR)
                filename_no_ext = Path(filename).stem
                image_path_flat = images_dir / filename
                try:
                    images_dir_resolved = images_dir.resolve()
                    image_path_flat_resolved = (images_dir_resolved / filename).resolve()
                except OSError:
                    image_path_flat_resolved = image_path_flat
                flat_to_try = image_path_flat_resolved if image_path_flat_resolved != image_path_flat else image_path_flat
                
                def _try_species():
                    if "_" not in filename_no_ext:
                        return None
                    subdir = filename_no_ext.split("_")[0]
                    species_dir = images_dir / subdir
                    sub_path = species_dir / filename
                    if sub_path.exists() and sub_path.is_file():
                        return FileResponse(str(sub_path))
                    for ext in ['.jpg', '.jpeg', '.png', '.gif', '.JPG', '.JPEG', '.PNG', '.GIF']:
                        sub_path = species_dir / f"{filename_no_ext}{ext}"
                        if sub_path.exists() and sub_path.is_file():
                            return FileResponse(str(sub_path))
                    return None
                
                print(f"🖼️  MCP Server: Looking for image: {filename}")
                if IMAGES_TRY_SPECIES_FIRST:
                    r = _try_species()
                    if r is not None:
                        return r
                    if flat_to_try.exists() and flat_to_try.is_file():
                        return FileResponse(str(flat_to_try))
                else:
                    if flat_to_try.exists() and flat_to_try.is_file():
                        return FileResponse(str(flat_to_try))
                    r = _try_species()
                    if r is not None:
                        return r
                filename_lower = filename.lower()
                dir_listing = _get_cached_dir_listing(images_dir)
                for item in dir_listing:
                    if item.is_file() and item.name.lower() == filename_lower:
                        return FileResponse(str(item))
                for ext in ['.jpg', '.jpeg', '.png', '.gif', '.JPG', '.JPEG', '.PNG', '.GIF']:
                    potential_file = images_dir / f"{filename_no_ext}{ext}"
                    if potential_file.exists() and potential_file.is_file():
                        return FileResponse(str(potential_file))
                
                print(f"❌ MCP Server: Image not found: {filename}")
                print(f"   Flat path resolved: {image_path_flat_resolved} (exists={image_path_flat_resolved.exists()})")
                prefix = filename_no_ext.split("_")[0] if "_" in filename_no_ext else filename_no_ext
                try:
                    same_prefix = [p.name for p in _get_cached_dir_listing(images_dir) if p.is_file() and p.name.lower().startswith(prefix.lower() + "_")]
                    if same_prefix:
                        print(f"   Files with prefix '{prefix}_': {same_prefix[:10]}{'...' if len(same_prefix) > 10 else ''}")
                    else:
                        print(f"   No files with prefix '{prefix}_' in images dir.")
                except Exception:
                    pass
                raise HTTPException(status_code=404, detail=f"Image {filename} not found")
            except HTTPException:
                raise
            except Exception as e:
                print(f"❌ MCP Server: Error serving image: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Error serving image: {str(e)}")
        
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
            # Get all models and convert to serializable format
            all_models = self.model_registry.get_all_models()
            models_dict = {}
            for name, model_info in all_models.items():
                models_dict[name] = {
                    "name": model_info.name,
                    "type": model_info.type.value if hasattr(model_info.type, 'value') else str(model_info.type),
                    "description": model_info.description,
                    "version": model_info.version,
                    "supported_datasets": model_info.supported_datasets,
                    "parameters": model_info.parameters,
                    "metadata": model_info.metadata
                }
            return {
                "models": models_dict,
                "total": len(models_dict)
            }
        
        @self.app.get("/api/models/{model_name}")
        async def get_model(model_name: str):
            """Get specific model information"""
            from models import ModelInfo
            
            model = self.model_registry.get_model(model_name)
            if not model:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
            
            # Convert Model to ModelInfo (removes non-serializable handler)
            model_info = ModelInfo(
                name=model.name,
                type=model.type,
                description=model.description,
                version=model.version,
                supported_datasets=model.supported_datasets,
                parameters=model.parameters,
                metadata=model.metadata
            )
            
            # Convert to dict for JSON serialization
            return {
                "name": model_info.name,
                "type": model_info.type.value if hasattr(model_info.type, 'value') else str(model_info.type),
                "description": model_info.description,
                "version": model_info.version,
                "supported_datasets": model_info.supported_datasets,
                "parameters": model_info.parameters,
                "metadata": model_info.metadata
            }
        
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
                    "offset": {"type": "integer", "default": 0},
                    "dataset_offset": {"type": "integer", "default": 0, "description": "Skip this many datasets to load next batch (e.g. 100 for next 100)"}
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
        print(f"🔧 Checking Croissant crawler availability: {CROISSANT_CRAWLER_AVAILABLE}")
        print(f"   CROISSANT_CRAWLER_AVAILABLE value: {CROISSANT_CRAWLER_AVAILABLE}")
        print(f"   CroissantCrawler class: {CroissantCrawler}")
        
        if CROISSANT_CRAWLER_AVAILABLE:
            try:
                print(f"🔧 Attempting to register crawl_croissant_datasets tool...")
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
                print("✅ Successfully registered crawl_croissant_datasets tool")
                
                # Verify it was actually registered
                all_tools = list(self.tool_registry.get_all_tools().keys())
                if "crawl_croissant_datasets" in all_tools:
                    print(f"✅ Verified: crawl_croissant_datasets is in registered tools list")
                else:
                    print(f"❌ WARNING: crawl_croissant_datasets NOT found in tools list!")
                    print(f"   Registered tools: {all_tools}")
            except Exception as e:
                print(f"❌ Failed to register crawl_croissant_datasets tool: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("⚠️  Croissant crawler tool NOT registered (CROISSANT_CRAWLER_AVAILABLE is False)")
            print("   This means the import failed. Check the import error messages above.")
    
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
            # Apply category pre-filtering for performance
            category_filter = filters.get("category", [])
            datasets_to_search = []
            
            if category_filter:
                # Filter datasets by category first (same logic as _llm_search_handler)
                print(f"🔍 Category pre-filtering: {category_filter}")
                for dataset_name, dataset_obj in self.dataset_registry.datasets.items():
                    dataset_category = dataset_obj.type.value.lower()
                    # Map category filter to dataset type
                    category_mapping = {
                        "pest": ["pests"],
                        "animal": ["wildlife"],
                        "wildlife": ["wildlife"],
                        "plant": ["plants"]
                    }
                    should_include = False
                    for cat in category_filter:
                        cat_lower = cat.lower()
                        if cat_lower in category_mapping:
                            if dataset_category in category_mapping[cat_lower]:
                                should_include = True
                                break
                        elif cat_lower == dataset_category:
                            should_include = True
                            break
                    
                    if should_include:
                        datasets_to_search.append(dataset_name)
            else:
                # No category filter - search all datasets
                datasets_to_search = list(self.dataset_registry.datasets.keys())
            
            print(f"🔍 Searching {len(datasets_to_search)} datasets (out of {len(self.dataset_registry.datasets)} total)")
            all_results = []
            
            for dataset_name in datasets_to_search:
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
                "searched_datasets": datasets_to_search
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
        dataset_offset = max(0, int(input_data.get("dataset_offset", 0)))
        
        print(f"🧠 LLM search request: query='{query}', dataset='{dataset}', dataset_offset={dataset_offset}")
        
        try:
            # Get available filters for LLM context
            available_filters = {}
            if dataset:
                if dataset in self.dataset_registry.datasets:
                    available_filters = self._extract_dataset_filters(self.dataset_registry.datasets[dataset])
                    # Also add collections to available_filters for species matching
                    dataset_obj = self.dataset_registry.datasets[dataset]
                    if dataset_obj.collections:
                        if "collections" not in available_filters:
                            available_filters["collections"] = []
                        available_filters["collections"].extend(list(dataset_obj.collections))
            else:
                # Combine filters from all datasets
                all_collections = set()
                for dataset_name, dataset_obj in self.dataset_registry.datasets.items():
                    dataset_filters = self._extract_dataset_filters(dataset_obj)
                    for key, values in dataset_filters.items():
                        if key not in available_filters:
                            available_filters[key] = []
                        available_filters[key].extend(values)
                    # Collect collections from all datasets
                    if dataset_obj.collections:
                        all_collections.update(dataset_obj.collections)
                
                # Add collections to available_filters
                if all_collections:
                    available_filters["collections"] = sorted(list(all_collections))
                
                # Treat dataset names as valid species so e.g. "raspberry" matches dataset "raspberry"
                for dataset_name in self.dataset_registry.datasets:
                    if dataset_name and dataset_name.strip():
                        available_filters.setdefault("species", []).append(dataset_name.strip())
                
                # Remove duplicates
                for key in available_filters:
                    available_filters[key] = sorted(list(set(available_filters[key])))
                # Add opossum/oppossum to species options when we have Virginia opossum dataset so LLM can match and return high confidence
                opossum_datasets = [d for d in self.dataset_registry.datasets if d and ("opossum" in d.lower() or "oppossum" in d.lower())]
                if opossum_datasets:
                    for syn in ("opossum", "oppossum", "opossums", "oppossums"):
                        if syn not in available_filters.get("species", []):
                            available_filters.setdefault("species", []).append(syn)
                    available_filters["species"] = sorted(list(set(available_filters["species"])))
            
            # Use LLM to understand the query - require LLM, no rule-based fallback
            if not self.llm_service or not self.llm_service.is_available():
                return {
                    "query": query,
                    "error": "LLM service is not available. Please set OPENAI_API_KEY environment variable.",
                    "results": [],
                    "total_count": 0,
                    "llm_understanding": None
                }
            
            print(f"🧠 Understanding query with LLM...")
            print(f"   LLM Service available: {self.llm_service.is_available() if self.llm_service else False}")
            if self.llm_service:
                print(f"   OpenAI available: {self.llm_service.openai_available}")
                print(f"   Gemini available: {self.llm_service.gemini_available}")
                print(f"   Provider: {self.llm_service.provider}")
            query_understanding = await self.llm_service.understand_query(query, available_filters)
            
            # Normalize filters so every value is a list of strings (LLM sometimes returns nested lists or non-strings)
            def _flatten_filter_val(v):
                if v is None:
                    return []
                if isinstance(v, str):
                    return [v.strip()] if v.strip() else []
                if isinstance(v, list):
                    out = []
                    for x in v:
                        if isinstance(x, list):
                            out.extend(_flatten_filter_val(x))
                        elif x is not None and str(x).strip():
                            out.append(str(x).strip())
                    return out
                return [str(v).strip()] if str(v).strip() else []
            if getattr(query_understanding, "filters", None):
                for key in list(query_understanding.filters.keys()):
                    query_understanding.filters[key] = _flatten_filter_val(query_understanding.filters[key])
                # Normalize species: LLMs sometimes return "(moth)" or "[moth]" — strip parens/brackets so we match "moth"
                if query_understanding.filters.get("species"):
                    normalized = []
                    for s in query_understanding.filters["species"]:
                        t = str(s).strip()
                        while t and t[0] in "([{\"" and t[-1] in ")]}\"":
                            t = t[1:-1].strip()
                        if t:
                            normalized.append(t)
                    if normalized:
                        query_understanding.filters["species"] = normalized
                # Map opossum/oppossum to canonical dataset name (e.g. virginia_opossum) so pre-filtering finds the dataset
                if query_understanding.filters.get("species") and self.dataset_registry.datasets:
                    species_list = query_understanding.filters["species"]
                    opossum_vals = {"opossum", "oppossum", "opossums", "oppossums"}
                    if any(str(s).lower() in opossum_vals for s in species_list):
                        canonical = [d for d in self.dataset_registry.datasets if d and ("opossum" in d.lower() or "oppossum" in d.lower())]
                        if canonical:
                            rest = [s for s in species_list if str(s).lower() not in opossum_vals]
                            query_understanding.filters["species"] = sorted(list(set(rest + canonical)))
            
            # Reject rule-based fallback results - require real LLM understanding
            if query_understanding.confidence < 0.7:
                print(f"⚠️  Low confidence ({query_understanding.confidence}) - this might be rule-based fallback")
                # Still proceed but log warning
            
            print(f"🧠 LLM Understanding:")
            print(f"   Intent: {query_understanding.intent}")
            print(f"   Entities: {query_understanding.entities}")
            print(f"   Filters: {query_understanding.filters}")
            print(f"   Confidence: {query_understanding.confidence}")
            print(f"   Reasoning: {query_understanding.reasoning}")
            
            # Ensure plant_state is set when query clearly asks for ripeness (e.g. "raspberry ripe")
            # LLM sometimes omits it; without it we return all images and strict filter never runs
            query_lower = query.lower()
            plant_fruit_species = {"raspberry", "raspberries", "strawberry", "strawberries", "blueberry", "blueberries", "blackberry", "blackberries"}
            species_in_query = [s for s in (query_understanding.filters.get("species") or []) if s]
            species_lower = {s.lower().strip().replace("_", "") for s in species_in_query}
            is_plant_query = bool(species_lower & {t.replace("_", "") for t in plant_fruit_species})
            if is_plant_query and "ripe" in query_lower and not query_understanding.filters.get("plant_state"):
                query_understanding.filters["plant_state"] = ["ripe"]
                print(f"   ✅ Injected plant_state: ['ripe'] from query (query implied ripe)")
            if is_plant_query and ("unripe" in query_lower or " green " in query_lower or "green fruit" in query_lower) and not query_understanding.filters.get("plant_state"):
                query_understanding.filters["plant_state"] = ["unripe"]
                print(f"   ✅ Injected plant_state: ['unripe'] from query")
            
            # If query clearly mentions a fruit/species but LLM left species empty, inject so we only search that dataset.
            query_species = query_understanding.filters.get("species") or []
            dataset_names_lower = {dn.lower(): dn for dn in self.dataset_registry.datasets}
            # Opossum / oppossum (common misspelling): inject first so we never search all datasets
            if not query_species and self.dataset_registry.datasets:
                if "opossum" in query_lower or "oppossum" in query_lower or "opossums" in query_lower or "oppossums" in query_lower:
                    opossum_datasets = [d for d in self.dataset_registry.datasets if d and ("opossum" in d.lower() or "oppossum" in d.lower())]
                    if opossum_datasets:
                        query_understanding.filters["species"] = sorted(opossum_datasets)
                        query_species = query_understanding.filters["species"]
                        print(f"   ✅ Injected species (opossum/oppossum → Virginia opossum): {query_understanding.filters['species']}")
                    else:
                        # Dataset not in collection — return clear message instead of searching all (which would show pest/irrelevant images)
                        return {
                            "query": query,
                            "llm_understanding": query_understanding,
                            "results": [],
                            "total_count": 0,
                            "error": "This species (opossum) is not available in the collection. Virginia opossum has not been added to this server.",
                            "searched_datasets": [],
                        }
            if not query_species and self.dataset_registry.datasets:
                fruit_to_try = []
                if "blueberry" in query_lower or "blueberries" in query_lower:
                    fruit_to_try.append("blueberry")
                if "raspberry" in query_lower or "raspberries" in query_lower:
                    fruit_to_try.append("raspberry")
                if "strawberry" in query_lower or "strawberries" in query_lower:
                    fruit_to_try.append("strawberry")
                if "blackberry" in query_lower or "blackberries" in query_lower:
                    fruit_to_try.append("blackberry")
                if "mango" in query_lower or "mangoes" in query_lower or "mangos" in query_lower:
                    fruit_to_try.append("mango")
                if "grape" in query_lower or "grapes" in query_lower:
                    fruit_to_try.append("grapes")
                if "apple" in query_lower or "apples" in query_lower:
                    fruit_to_try.append("apple")
                if "citrus" in query_lower or "orange" in query_lower or "oranges" in query_lower:
                    fruit_to_try.append("orange")
                for name in fruit_to_try:
                    norm = name.lower().strip().replace("_", "").replace("-", "")
                    matches = []
                    for d_lower, dataset_name in dataset_names_lower.items():
                        dn_norm = d_lower.replace("_", "").replace("-", "")
                        # Match exact (mango) or prefix with _/- (mango_1, mango_2); use d_lower for prefix so mango_1 matches
                        if (norm == dn_norm or
                                d_lower == norm or
                                d_lower.startswith(norm + "_") or
                                d_lower.startswith(norm + "-")):
                            matches.append(dataset_name)
                    if matches:
                        query_understanding.filters["species"] = sorted(matches)
                        print(f"   ✅ Injected species: {query_understanding.filters['species']} from query (matched '{name}')")
                        break
                    else:
                        # Query clearly asked for this fruit/species but no dataset exists — set species so we return "no dataset" instead of searching all
                        query_understanding.filters["species"] = [name]
                        print(f"   ⚠️  No dataset named '{name}' (or {name}_*) found; will return error instead of searching all datasets")
                        break
                # Fallback: if still no species, check if any query word exactly matches a dataset name (avoids searching all datasets for "mango")
                if not query_understanding.filters.get("species"):
                    words = [w.strip().lower() for w in query.split() if len(w.strip()) >= 3 and w.strip().isalpha()]
                    skip_words = {"the", "and", "for", "with", "images", "pictures", "photos", "species", "show", "find", "search", "ripe", "unripe", "green", "red"}
                    for word in words:
                        if word in skip_words:
                            continue
                        if word in dataset_names_lower:
                            query_understanding.filters["species"] = [dataset_names_lower[word]]
                            print(f"   ✅ Injected species: ['{dataset_names_lower[word]}'] from query word '{word}' (exact dataset name)")
                            break
                        # singular/plural: "mangoes" -> "mango"
                        word_singular = word[:-1] if word.endswith("s") and len(word) > 2 and not word.endswith("ss") else word
                        if word_singular in dataset_names_lower:
                            query_understanding.filters["species"] = [dataset_names_lower[word_singular]]
                            print(f"   ✅ Injected species: ['{dataset_names_lower[word_singular]}'] from query word '{word}'")
                            break
                # Rabbit = cottontail = white cottontail: if query mentions any and no species yet, use eastern_cottontail (not white_cottontail)
                if not query_understanding.filters.get("species") and self.dataset_registry.datasets:
                    if "rabbit" in query_lower or "cottontail" in query_lower or "rabbits" in query_lower or "cottontails" in query_lower or "white cottontail" in query_lower or "white cottontails" in query_lower:
                        eastern = [d for d in self.dataset_registry.datasets if d and d.lower() == "eastern_cottontail"]
                        cottontail_datasets = eastern if eastern else [d for d in self.dataset_registry.datasets if "cottontail" in d.lower()]
                        if cottontail_datasets:
                            query_understanding.filters["species"] = sorted(cottontail_datasets)
                            print(f"   ✅ Injected species (rabbit/cottontail/white cottontail → eastern_cottontail): {query_understanding.filters['species']}")
                        else:
                            return {
                                "query": query,
                                "llm_understanding": query_understanding,
                                "results": [],
                                "total_count": 0,
                                "error": "This species (rabbit / cottontail) is not available in the collection.",
                                "searched_datasets": [],
                            }
                # Pest type words: ensure "beetle", "butterfly", "wasp", "moth", "stink bug", etc. are in species filter when query mentions them
                # so search matches pest images via common_names (e.g. ["French Paper Wasp", "wasp"])
                _PEST_TYPE_WORDS = [
                    "beetle", "beetles", "butterfly", "butterflies", "moth", "moths",
                    "wasp", "wasps", "bee", "bees", "ant", "ants", "fly", "flies",
                    "grasshopper", "grasshoppers", "dragonfly", "dragonflies",
                    "spider", "spiders", "stink bug", "stink bugs", "true bug", "bugs", "insect", "insects",
                ]
                query_species_list = query_understanding.filters.get("species") or []
                species_set = {s.lower().strip() for s in query_species_list}
                for type_word in _PEST_TYPE_WORDS:
                    type_lower = type_word.lower()
                    type_singular = type_lower.rstrip("s") if type_lower.endswith("s") and len(type_lower) > 1 else type_lower
                    if type_lower in query_lower or type_singular in query_lower:
                        if type_singular not in species_set and type_lower not in species_set:
                            if not any(type_singular in s or type_lower in s for s in species_set):
                                query_species_list.append(type_singular)
                                species_set.add(type_singular)
                                print(f"   ✅ Injected species (pest type from query): '{type_singular}'")
                if query_species_list != (query_understanding.filters.get("species") or []):
                    query_understanding.filters["species"] = sorted(list(set(query_species_list)))

            # Validate filters were extracted correctly
            # Also check if query contains species words that aren't in available filters
            available_species = available_filters.get("species", []) + available_filters.get("collections", [])
            
            # Check if query contains common species words that aren't in available filters
            # This catches cases like "whale" where LLM might not extract it as a filter
            common_species_words = ["whale", "whales", "elephant", "elephants", "tiger", "tigers", "lion", "lions", 
                                   "bear", "bears", "eagle", "eagles", "shark", "sharks", "dolphin", "dolphins"]
            query_species_words = []
            for word in common_species_words:
                if word in query_lower:
                    # Check if this word matches any available species
                    word_matches_available = False
                    for avail in available_species:
                        avail_lower = avail.lower().strip()
                        word_normalized = word.replace("_", "").replace("-", "")
                        avail_normalized = avail_lower.replace("_", "").replace("-", "")
                        if (word == avail_lower or 
                            word_normalized == avail_normalized or
                            word in avail_lower or
                            avail_lower in word):
                            word_matches_available = True
                            break
                    if not word_matches_available:
                        query_species_words.append(word)
            
            # If query contains species words not in available filters, return error (no long species list)
            if query_species_words:
                return {
                    "query": query,
                    "error": f"Species '{', '.join(query_species_words)}' not found in our catalog. Try a different species or check the spelling.",
                    "results": [],
                    "total_count": 0,
                    "llm_understanding": query_understanding,
                }
            
            if query_understanding.filters.get("species"):
                print(f"   ✅ Species filter: {query_understanding.filters['species']}")
                # Check if species filter matches available species
                species_filter = query_understanding.filters["species"]
                unmatched_species = []
                for species in species_filter:
                    species_lower = species.lower().strip()
                    # Check if species matches any available species (with normalization)
                    matched = False
                    for avail in available_species:
                        avail_lower = avail.lower().strip()
                        # Normalize both for comparison
                        species_norm = species_lower.replace("_", "").replace("-", "")
                        avail_norm = avail_lower.replace("_", "").replace("-", "")
                        if (species_lower == avail_lower or 
                            species_norm == avail_norm or
                            species_lower in avail_lower or
                            avail_lower in species_lower):
                            matched = True
                            break
                    if not matched:
                        unmatched_species.append(species)
                
                # Treat as matched if a dataset name matches (e.g. dataset "raspberry" for species "raspberry")
                dataset_names_norm = {name.lower().strip().replace("_", "").replace("-", "") for name in self.dataset_registry.datasets}
                unmatched_species = [
                    s for s in unmatched_species
                    if (s.lower().strip().replace("_", "").replace("-", "")) not in dataset_names_norm
                ]
                
                # If species filter doesn't match any available species, return error (no long species list)
                if unmatched_species and len(unmatched_species) == len(species_filter):
                    return {
                        "query": query,
                        "error": f"Species '{', '.join(unmatched_species)}' not found in our catalog. Try a different species or check the spelling.",
                        "results": [],
                        "total_count": 0,
                        "llm_understanding": query_understanding,
                    }
            
            if query_understanding.filters.get("time"):
                print(f"   ✅ Time filter: {query_understanding.filters['time']}")
            if query_understanding.filters.get("plant_state"):
                print(f"   ✅ Plant state filter: {query_understanding.filters['plant_state']}")
            if query_understanding.filters.get("action"):
                print(f"   ✅ Action filter: {query_understanding.filters['action']}")
            
            # If no filters were extracted at all and query contains species words, return error
            # This prevents returning all images when query contains unknown species
            has_any_filters = any(query_understanding.filters.values())
            if not has_any_filters and query_species_words:
                return {
                    "query": query,
                    "error": f"Species '{', '.join(query_species_words)}' not found in our catalog. Try a different species or check the spelling.",
                    "results": [],
                    "total_count": 0,
                    "llm_understanding": query_understanding,
                }
            
            # OPTIMIZATION: Pre-filter datasets to reduce search space
            # 1) If species filter: only search datasets that contain that species (or matching dataset name)
            # 2) If category filter: only search datasets in that category
            # 3) Otherwise: search all datasets
            category_filter = query_understanding.filters.get("category", [])
            species_filter = query_understanding.filters.get("species", [])
            datasets_to_search = []

            def _species_match(species_filter_list: List[str], dataset_name: str, dataset_obj: Any) -> bool:
                """True if any requested species is in this dataset's species/collections or in dataset name."""
                if not species_filter_list:
                    return True
                requested = {s.lower().strip().replace(" ", "_") for s in species_filter_list if s}
                # Normalize singular/plural so "moths" matches dataset species "moth"
                def _normalize_plural(s: str) -> str:
                    s = s.lower().strip()
                    if len(s) > 1 and s.endswith("s") and not s.endswith("ss"):
                        return s[:-1]  # "moths" -> "moth"
                    return s
                requested_singular = {_normalize_plural(r) for r in requested}
                # Dataset name match (e.g. "raspberry" matches dataset "raspberry" or "raspberry_1")
                name_normalized = dataset_name.lower().replace(" ", "_")
                for r in requested:
                    if r == name_normalized or name_normalized.startswith(r + "_") or name_normalized.startswith(r + "-"):
                        return True
                    r_singular = _normalize_plural(r)
                    if r_singular == name_normalized or name_normalized.startswith(r_singular + "_") or name_normalized.startswith(r_singular + "-"):
                        return True
                # Species/collections from this dataset
                opts = dataset_obj.available_filters
                if opts:
                    for lst in (opts.species or [], opts.collections or []):
                        for val in lst or []:
                            v = str(val).lower().strip().replace(" ", "_")
                            v_singular = _normalize_plural(v)
                            if v in requested or v in requested_singular:
                                return True
                            if v_singular in requested or v_singular in requested_singular:
                                return True
                            if any(r in v or v in r or _normalize_plural(r) in v or v_singular in r for r in requested):
                                return True
                return False

            candidate_datasets = list(self.dataset_registry.datasets.items())
            dataset_names_set = set(self.dataset_registry.datasets.keys())

            if species_filter:
                # If exactly one species and it equals a dataset name (e.g. "carrot", "raspberry"), search only that dataset.
                dataset_names_lower = {name.lower(): name for name in dataset_names_set}
                if len(species_filter) == 1:
                    s = species_filter[0].lower().strip().replace(" ", "_")
                    if s in dataset_names_lower:
                        datasets_to_search = [dataset_names_lower[s]]
                        print(f"🧠 Species '{s}' matches dataset name → searching only dataset: {datasets_to_search[0]}")
                    else:
                        s_singular = s[:-1] if len(s) > 1 and s.endswith("s") and not s.endswith("ss") else s
                        if s_singular in dataset_names_lower:
                            datasets_to_search = [dataset_names_lower[s_singular]]
                            print(f"🧠 Species '{s}' (singular '{s_singular}') matches dataset name → searching only dataset: {datasets_to_search[0]}")
                if not datasets_to_search:
                    # Only datasets that have this species (or name match)
                    for dataset_name, dataset_obj in candidate_datasets:
                        if not _species_match(species_filter, dataset_name, dataset_obj):
                            continue
                        if category_filter:
                            dataset_category = dataset_obj.type.value.lower()
                            category_mapping = {
                                "pest": ["pests"], "animal": ["wildlife"], "wildlife": ["wildlife"], "plant": ["plants"]
                            }
                            if not any(
                                dataset_category in category_mapping.get(c.lower(), [c.lower()]) or c.lower() == dataset_category
                                for c in category_filter
                            ):
                                continue
                        datasets_to_search.append(dataset_name)
                # Species was requested but no dataset matches — return immediately instead of searching all
                if species_filter and not datasets_to_search:
                    available_list = sorted(list(dataset_names_set))[:15]
                    return {
                        "query": query,
                        "llm_understanding": query_understanding,
                        "results": [],
                        "total_count": 0,
                        "error": f"No dataset found for species '{', '.join(species_filter)}'. Available datasets include: {', '.join(available_list)}{'...' if len(self.dataset_registry.datasets) > 15 else ''}",
                        "searched_datasets": []
                    }
            elif category_filter:
                for dataset_name, dataset_obj in candidate_datasets:
                    dataset_category = dataset_obj.type.value.lower()
                    category_mapping = {
                        "pest": ["pests"], "animal": ["wildlife"], "wildlife": ["wildlife"], "plant": ["plants"]
                    }
                    should_include = any(
                        dataset_category in category_mapping.get(c.lower(), [c.lower()]) or c.lower() == dataset_category
                        for c in category_filter
                    )
                    if should_include:
                        datasets_to_search.append(dataset_name)
            else:
                datasets_to_search = list(self.dataset_registry.datasets.keys())

            # When species filter matched both wildlife and pest datasets (e.g. "fox" → red fox + pests with "fox" in name), prefer wildlife and show disambiguation message
            disambiguation_message = None
            if species_filter and len(datasets_to_search) > 1:
                wildlife_only = []
                pest_only = []
                for d in datasets_to_search:
                    obj = self.dataset_registry.datasets.get(d)
                    if not obj:
                        continue
                    if obj.type == DatasetType.WILDLIFE:
                        wildlife_only.append(d)
                    elif obj.type == DatasetType.PESTS:
                        pest_only.append(d)
                if wildlife_only and pest_only:
                    # Restrict to animal/wildlife results and add follow-up message
                    datasets_to_search = wildlife_only
                    term = (species_filter[0] or "this").replace("_", " ").strip()
                    disambiguation_message = f"Showing {term} (animal) images. Would you like to see pests that include '{term}' in their name? Try searching \"{term} pest\" to include them."

            print(f"🧠 Pre-filtering: searching {len(datasets_to_search)} datasets (out of {len(self.dataset_registry.datasets)} total)")
            
            # Cap datasets to search and support "load next 100" via dataset_offset
            MAX_DATASETS_TO_SEARCH = 100
            total_datasets_matching = len(datasets_to_search)
            search_capped = False
            if total_datasets_matching > MAX_DATASETS_TO_SEARCH or dataset_offset > 0:
                search_capped = True
            # Slice to current batch: skip dataset_offset datasets, take up to MAX_DATASETS_TO_SEARCH
            datasets_to_search = datasets_to_search[dataset_offset:dataset_offset + MAX_DATASETS_TO_SEARCH]
            if dataset_offset > 0:
                print(f"🧠 Loading next batch: datasets {dataset_offset + 1}–{dataset_offset + len(datasets_to_search)} of {total_datasets_matching}")
            elif len(datasets_to_search) < total_datasets_matching:
                print(f"🧠 Capped search to first {MAX_DATASETS_TO_SEARCH} datasets (of {total_datasets_matching}) for faster response")
            
            # Perform search using the structured understanding and adapters
            # IMPORTANT: Only use filters, not the query string, to avoid incorrect substring matches
            if dataset:
                if dataset in datasets_to_search and dataset in self.dataset_registry.datasets:
                    # Use adapter-based search with ONLY filters (no query string)
                    filtered_results = self.dataset_registry.search_dataset(
                        dataset, "", query_understanding.filters
                    )
                    # Apply strict plant_state filter when user asked for specific state (e.g. ripe only)
                    plant_state_filter = query_understanding.filters.get("plant_state") or []
                    if plant_state_filter:
                        before = len(filtered_results)
                        filtered_results = [r for r in filtered_results if self._passes_plant_state_strict(r, plant_state_filter)]
                        if before != len(filtered_results):
                            print(f"🧠 Plant-state strict filter (single dataset): kept {len(filtered_results)} of {before} results (requested: {plant_state_filter})")
                    
                    if not filtered_results:
                        # No results — return a clear, user-friendly message (do not list all species/catalogs)
                        species_filter = query_understanding.filters.get("species", [])
                        action_filter = query_understanding.filters.get("action", [])
                        if species_filter or action_filter:
                            parts = []
                            if species_filter:
                                parts.append(f"species '{', '.join(species_filter)}'")
                            if action_filter:
                                parts.append(f"action '{', '.join(action_filter)}'")
                            err = f"No images found matching {', '.join(parts)} in our catalog. Try a different species or action."
                        else:
                            err = f"Dataset '{dataset}' has no matching results for your query."
                        return {
                            "dataset": dataset,
                            "query": query,
                            "llm_understanding": query_understanding,
                            "results": [],
                            "total_count": 0,
                            "error": err,
                        }
                    
                    # Add confidence scores, image URLs, and top-level display fields to each result
                    for result in filtered_results:
                        result['llm_confidence'] = self._calculate_result_confidence(result, query_understanding, query)
                        result['llm_reasoning'] = query_understanding.reasoning
                        result['llm_intent'] = query_understanding.intent
                        result['image_url'] = self._construct_image_url(result)
                        meta = result.get('metadata') or {}
                        result['background'] = meta.get('background') or meta.get('scene')
                        result['scientific_name'] = meta.get('scientific_name')
                        cn = meta.get('common_names')
                        result['common_names'] = cn if isinstance(cn, list) else ([cn] if cn else None)
                    
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
                # Search across filtered datasets using adapters (optimized with category pre-filtering)
                all_results = []
                # First batch (dataset_offset=0): stop as soon as we have enough for the requested page (e.g. 20) for fast first response
                # Next batches (dataset_offset>0): collect up to 100 so "Load next 100" returns a full batch
                MAX_TOTAL_RESULTS = 100
                if dataset_offset == 0:
                    enough_results = offset + limit  # e.g. 20 — return as soon as we have one page
                else:
                    enough_results = min(offset + limit + 80, MAX_TOTAL_RESULTS)
                
                for dataset_name in datasets_to_search:
                    if len(all_results) >= enough_results:
                        print(f"🧠 Early termination: have {len(all_results)} results (need {offset + limit}), stopping dataset search")
                        break
                    print(f"🧠 Searching dataset: {dataset_name}")
                    # Use adapter-based search with ONLY filters (no query string)
                    filtered_results = self.dataset_registry.search_dataset(
                        dataset_name, "", query_understanding.filters
                    )
                    print(f"   Found {len(filtered_results)} matching results in {dataset_name}")
                    if filtered_results:
                        # Log first result for debugging
                        first_result = filtered_results[0]
                        print(f"   Sample result: collection={first_result.get('collection')}, species={first_result.get('metadata', {}).get('species')}")
                        remaining = enough_results - len(all_results)
                        to_add = filtered_results if remaining >= len(filtered_results) else filtered_results[:remaining]
                        for result in to_add:
                            result['dataset'] = dataset_name
                            result['llm_confidence'] = self._calculate_result_confidence(result, query_understanding, query)
                            result['llm_reasoning'] = query_understanding.reasoning
                            result['llm_intent'] = query_understanding.intent
                            result['image_url'] = self._construct_image_url(result)
                            # Top-level fields for UI (background, scientific_name, common_names)
                            meta = result.get('metadata') or {}
                            result['background'] = meta.get('background') or meta.get('scene')
                            result['scientific_name'] = meta.get('scientific_name')
                            cn = meta.get('common_names')
                            result['common_names'] = cn if isinstance(cn, list) else ([cn] if cn else None)
                        all_results.extend(to_add)
                        if len(all_results) >= enough_results:
                            break
                
                # When user asked for a specific plant_state (e.g. ripe), keep only items that pass strict check
                # (uses both metadata and description; excludes mixed/unripe so we return only ripe images)
                plant_state_filter = query_understanding.filters.get("plant_state") or []
                if plant_state_filter:
                    before = len(all_results)
                    all_results = [r for r in all_results if self._passes_plant_state_strict(r, plant_state_filter)]
                    if before != len(all_results):
                        print(f"🧠 Plant-state strict filter: kept {len(all_results)} of {before} results (requested: {plant_state_filter})")
                
                print(f"🧠 Total LLM-filtered results found: {len(all_results)}")
                
                # If no results, provide helpful error message
                if not all_results:
                    error_parts = []
                    if query_understanding.filters.get("species"):
                        species_filter = query_understanding.filters["species"]
                        error_parts.append(f"species '{', '.join(species_filter)}'")
                    if query_understanding.filters.get("time"):
                        time_filter = query_understanding.filters["time"]
                        error_parts.append(f"time '{', '.join(time_filter)}'")
                    if query_understanding.filters.get("action"):
                        action_filter = query_understanding.filters["action"]
                        error_parts.append(f"action '{', '.join(action_filter)}'")
                    if query_understanding.filters.get("scene"):
                        scene_filter = query_understanding.filters["scene"]
                        error_parts.append(f"scene '{', '.join(scene_filter)}'")
                    
                    if error_parts:
                        error_msg = f"No images found matching {', '.join(error_parts)} in our catalog."
                        # Short, user-friendly hint (do not list thousands of species/pest names)
                        if query_understanding.filters.get("plant_state"):
                            error_msg += " Try searching without a ripeness filter, or use a broader term."
                        elif query_understanding.filters.get("species") or query_understanding.filters.get("action"):
                            error_msg += " Try a different species or action."
                        
                        return {
                            "query": query,
                            "llm_understanding": query_understanding,
                            "results": [],
                            "total_count": 0,
                            "error": error_msg,
                            "searched_datasets": datasets_to_search
                        }
                    else:
                        return {
                            "query": query,
                            "llm_understanding": query_understanding,
                            "results": [],
                            "total_count": 0,
                            "error": "No images found matching your query",
                            "searched_datasets": datasets_to_search
                        }
                
                # Sort by confidence (highest first) and then by dataset for consistency
                all_results.sort(key=lambda x: (x.get('llm_confidence', 0), x.get('dataset', ''), x.get('id', '')), reverse=True)
                
                next_dataset_offset = dataset_offset + len(datasets_to_search)
                has_more_datasets = next_dataset_offset < total_datasets_matching
                out = {
                    "query": query,
                    "llm_understanding": query_understanding,
                    "results": all_results[offset:offset + limit],
                    "total_count": len(all_results),
                    "searched_datasets": datasets_to_search,
                    "dataset_offset": dataset_offset,
                    "next_dataset_offset": next_dataset_offset,
                    "total_datasets_matching": total_datasets_matching,
                    "has_more_datasets": has_more_datasets,
                }
                if search_capped:
                    out["search_capped"] = True
                if disambiguation_message is not None:
                    out["disambiguation_message"] = disambiguation_message
                return out
                
        except ValueError as e:
            # LLM service not available or failed
            print(f"❌ LLM search error: {e}")
            return {
                "query": query,
                "error": f"LLM service error: {str(e)}. Please ensure OPENAI_API_KEY is set and valid.",
                "results": [],
                "total_count": 0,
                "llm_understanding": None
            }
        except Exception as e:
            print(f"❌ LLM search error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "query": query,
                "error": f"LLM search failed: {str(e)}",
                "results": [],
                "total_count": 0,
                "llm_understanding": None
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
        """
        Check if image matches scene filters.
        
        IMPORTANT: Scene metadata is read ONLY from MCP metadata (metadata.scene).
        The system does NOT infer or set scene values - it only uses what's explicitly
        provided in the MCP data files. If scene values are incorrect, they need to be
        fixed in the source MCP data files.
        
        STRICT MODE: If scene filter is specified, image MUST have matching scene.
        Also validates against description to catch contradictions.
        """
        if not scene_filters:
            return True
        
        # Read scene directly from MCP metadata - no inference or defaults
        image_scene = image.get("metadata", {}).get("scene", "").lower().strip()
        image_description = image.get("metadata", {}).get("description", "").lower()
        
        # If no scene metadata, check description for scene keywords
        if not image_scene:
            scene_keyword_map = {
                "field": ["field", "meadow", "pasture", "open field", "grassland"],
                "forest": ["forest", "woodland", "woods", "trees"],
                "garden": ["garden", "garden area"],
                "farm": ["farm", "farmland", "farm area"],
                "indoor": ["indoor", "inside", "interior", "barn", "shed", "building"],
                "outdoor": ["outdoor", "outside", "exterior", "open air"]
            }
            
            for scene_filter in scene_filters:
                scene_lower = scene_filter.lower().strip()
                if scene_lower in scene_keyword_map:
                    keywords = scene_keyword_map[scene_lower]
                    if any(keyword in image_description for keyword in keywords):
                        return True
            # If no scene metadata and no description match, reject if scene filter is specified
            return False
        
        # Check if scene matches any filter
        for scene_filter in scene_filters:
            scene_lower = scene_filter.lower().strip()
            if scene_lower == image_scene or scene_lower in image_scene or image_scene in scene_lower:
                # VALIDATION: Check description for contradictions
                indoor_keywords = ["indoor", "inside", "interior", "barn", "shed", "building", "structure"]
                outdoor_keywords = ["outdoor", "outside", "field", "meadow", "pasture", "open", "exterior"]
                
                # If scene says "field" but description says "indoor", reject
                if scene_lower == "field" and any(keyword in image_description for keyword in indoor_keywords):
                    print(f"      ⚠️  Scene mismatch: scene='{image_scene}' but description indicates indoor - REJECTING")
                    continue  # Try next scene filter
                
                # If scene says "indoor" but description says outdoor keywords, reject
                if scene_lower in ["indoor", "inside"] and any(keyword in image_description for keyword in outdoor_keywords):
                    print(f"      ⚠️  Scene mismatch: scene='{image_scene}' but description indicates outdoor - REJECTING")
                    continue  # Try next scene filter
                
                return True
        
        # No match found
        return False
    
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
        if query in _metadata_str(metadata.get("description", "")).lower():
            return True
        
        # Check metadata action
        if query in _metadata_str(metadata.get("action", "")).lower():
            return True
        
        # Check metadata scene
        if query in _metadata_str(metadata.get("scene", "")).lower():
            return True
        
        return False
    
    def _image_matches_time(self, image: Dict[str, Any], time_filters: List[str]) -> bool:
        """Check if image matches time filters"""
        metadata = image.get("metadata", {})
        time_info = _metadata_str(metadata.get("time", "")).lower()
        
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
    
    def _construct_image_url(self, result: Dict[str, Any]) -> str:
        """Construct image URL from result data"""
        try:
            metadata = result.get("metadata", {})
            result_id = result.get("id", "")
            
            # Priority 1: Check if image_url is explicitly set
            if "image_url" in result:
                return result["image_url"]
            
            # Priority 2: Try id with common extensions (all images are expected under MCP names, e.g. grapes_300.jpg)
            if result_id:
                from pathlib import Path
                images_dir = Path(IMAGES_DIR)
                if images_dir.exists():
                    for ext in [".jpg", ".jpeg", ".png", ".gif", ".JPG", ".JPEG", ".PNG", ".GIF"]:
                        potential_filename = f"{result_id}{ext}"
                        potential_path = images_dir / potential_filename
                        if potential_path.exists() and potential_path.is_file():
                            return f"/images/{potential_filename}"
            
            # Priority 3: Try original_filename from metadata (when it equals MCP name)
            if "original_filename" in metadata:
                original_filename = metadata["original_filename"]
                filename = Path(original_filename).name
                from pathlib import Path
                images_dir = Path(IMAGES_DIR)
                if images_dir.exists():
                    potential_path = images_dir / filename
                    if potential_path.exists() and potential_path.is_file():
                        return f"/images/{filename}"
                    else:
                        # Try case-insensitive match
                        filename_lower = filename.lower()
                        for item in images_dir.iterdir():
                            if item.is_file() and item.name.lower() == filename_lower:
                                return f"/images/{item.name}"
                # If not found locally, return the filename anyway (might be on server)
                return f"/images/{filename}"
            
            # Priority 4: Fallback to id with .jpg
            if result_id:
                return f"/images/{result_id}.jpg"
            
            # If no image info, return placeholder
            return "/images/placeholder.jpg"
            
        except Exception as e:
            print(f"❌ Error constructing image URL for result {result.get('id', 'unknown')}: {e}")
            return "/images/placeholder.jpg"
    
    def _filter_strings(self, value: Any) -> List[str]:
        """Normalize a metadata value to a list of non-empty strings (handles str or list)."""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if v is not None and str(v).strip()]
        s = str(value).strip()
        return [s] if s else []

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
            
            # Extract species - prioritize metadata.species (where MCP stores it)
            # Also use common_name, common_names (pest list), and scientific_name so "moth"/"raspberry" etc. are available
            metadata = image.get("metadata", {})
            species_value = (
                metadata.get("species") or metadata.get("common_name") or metadata.get("scientific_name")
                or image.get("species") or image.get("collection")
            )
            for species_str in self._filter_strings(species_value):
                if not species_str:
                    continue
                species_base = species_str.split("_")[0].split("-")[0].strip().lower()
                if species_base:
                    filters["species"].append(species_base)
                if "_" in species_str:
                    parts = [p.strip().lower() for p in species_str.split("_") if p.strip()]
                    if len(parts) >= 2 and not parts[-1].isdigit():
                        compound = "_".join(parts)
                        if compound not in filters["species"]:
                            filters["species"].append(compound)
                        if len(parts) >= 3:
                            two_word_compound = "_".join(parts[:2])
                            if two_word_compound not in filters["species"]:
                                filters["species"].append(two_word_compound)
            # Add pest common_names so queries like "moth" match (common_names can be e.g. ["Raspberry crown moth", "moth"])
            for species_str in self._filter_strings(metadata.get("common_names")):
                if not species_str:
                    continue
                species_base = species_str.split("_")[0].split("-")[0].strip().lower()
                if species_base and species_base not in filters["species"]:
                    filters["species"].append(species_base)
                # Multi-word common name: add each word as a filter (e.g. "moth" from "raspberry crown moth")
                for word in species_str.replace("_", " ").replace("-", " ").split():
                    w = word.strip().lower()
                    if w and len(w) >= 2 and w not in filters["species"]:
                        filters["species"].append(w)
            # Pest type words: add "moth"/"beetle" etc. when they appear as a word in collection/species/scientific_name (e.g. cabbage_moth)
            _pest_type_words = {"beetle", "butterfly", "moth", "wasp", "bee", "ant", "fly", "grasshopper", "dragonfly", "spider", "bug", "insect"}
            for src in (image.get("collection") or "", metadata.get("species") or "", metadata.get("scientific_name") or ""):
                for w in (str(src).replace("_", " ").replace("-", " ").lower().split() or []):
                    w = w.strip()
                    if w in _pest_type_words and w not in filters["species"]:
                        filters["species"].append(w)
                    elif len(w) > 1 and w.endswith("s") and w[:-1] in _pest_type_words:
                        singular = w[:-1]
                        if singular not in filters["species"]:
                            filters["species"].append(singular)
            
            # Extract time, season, action, plant_state (each can be str or list)
            for t in self._filter_strings(metadata.get("time")):
                filters["times"].append(t)
            for s in self._filter_strings(metadata.get("season")):
                filters["seasons"].append(s)
            for a in self._filter_strings(metadata.get("action")):
                filters["actions"].append(a)
            for ps in self._filter_strings(metadata.get("plant_state")):
                filters["plant_states"].append(ps)
            
            # Also extract actions from description field
            description = image.get("metadata", {}).get("description", "").lower()
            if description:
                # Action keyword map to identify actions in descriptions
                # IMPORTANT: Map query keywords to canonical action names
                # This ensures "feeding" in description maps to "foraging" if that's the canonical name
                action_keyword_map = {
                    "foraging": ["feed", "feeding", "eating", "eat", "foraging", "forage"],  # Canonical: foraging
                    "sleeping": ["sleep", "sleeping", "rest", "resting"],  # Canonical: sleeping
                    "resting": ["rest", "resting"],  # Canonical: resting
                    "walking": ["walk", "walking", "moving"],  # Canonical: walking
                    "hunting": ["hunt", "hunting"],  # Canonical: hunting
                    "alert": ["alert", "alerts", "watch", "watching", "looking at camera", "looking at the camera", 
                             "staring at camera", "staring at the camera", "facing camera", "facing the camera",
                             "looking toward camera", "looking toward the camera", "staring toward camera", 
                             "staring toward the camera", "facing toward camera", "facing toward the camera",
                             "looking directly at camera", "looking directly at the camera", "staring directly at camera",
                             "staring directly at the camera", "facing directly at camera", "facing directly at the camera"],  # Canonical: alert
                    "moving": ["moving", "move"],  # Canonical: moving
                    "running": ["running", "run"],  # Canonical: running
                    "perching": ["perching", "perch", "sitting", "sit"],  # Canonical: perching
                    "flying": ["flying", "fly"],  # Canonical: flying
                    "blooming": ["blooming", "bloom", "flowering", "flower"],  # Canonical: blooming
                    "fruiting": ["fruiting", "fruit"],  # Canonical: fruiting
                    "growing": ["growing", "grow"],  # Canonical: growing
                    "mature": ["mature", "matured", "ripe"]  # Canonical: mature
                }
                
                # Check description for action keywords and map to canonical names
                for canonical_action, keywords in action_keyword_map.items():
                    if any(keyword in description for keyword in keywords):
                        # Only add canonical action name, not the keyword found
                        if canonical_action not in filters["actions"]:
                            filters["actions"].append(canonical_action)
        
        # Remove duplicates and sort (all values are now strings, so set() is safe)
        for key in filters:
            filters[key] = sorted(list(set(filters[key])))
        
        return filters
    
    def _passes_plant_state_strict(self, result: Dict[str, Any], plant_state_filter: List[str]) -> bool:
        """When user asked for a specific plant_state (e.g. ripe/unripe), keep items that match.
        For 'ripe': exclude mixed; for 'unripe': allow mixed if description mentions unripe/developing fruit.
        """
        if not plant_state_filter:
            return True
        meta = result.get("metadata", {})
        ps_val = meta.get("plant_state")
        if isinstance(ps_val, list):
            item_ps = " ".join(str(x).strip() for x in ps_val if x).strip().lower()
        else:
            item_ps = (str(ps_val).strip().lower() if ps_val else "")
        desc = (meta.get("description") or "").lower()
        for requested in plant_state_filter:
            r = requested.lower().strip()
            if r not in ["ripe", "ripening", "unripe", "mature", "green", "red", "blooming", "fruiting"]:
                continue
            # ---- Unripe: allow if plant_state is unripe OR description caption mentions unripe ----
            if r == "unripe":
                if item_ps == "unripe":
                    return True
                # If "unripe" appears anywhere in the description caption, include the image
                if "unripe" in desc:
                    return True
                if item_ps == "mixed":
                    unripe_phrases = [
                        "unripe berr", "unripe fruit", "unripe raspberr", "developing fruit",
                        "developing berr", "developing raspberr", "green fruit", "green berr",
                        "young raspberr", "young fruit", "young berr", "unripe berries",
                    ]
                    if any(p in desc for p in unripe_phrases):
                        return True
                    continue
                if item_ps == "fruiting":
                    if any(p in desc for p in ["developing fruit", "developing berr", "green fruit"]):
                        return True
                continue
            # ---- Ripe: exclude mixed (user wants ripe-only); allow ripe/ripening with care ----
            if item_ps == "mixed":
                return False
            # Ripening ≠ ripe: when user asks for "ripe", exclude items that are only "ripening"
            if r == "ripe" and item_ps == "ripening":
                return False
            if r == "ripe" and item_ps == "ripe":
                return True
            if r == "ripe":
                mixed_phrases = [
                    "varying stages", "various stages", "various ripening", "stages of ripeness", "varying ripeness",
                    "at various ripening", "at various stages", "displaying raspberries at various",
                    "mix of unripe and ripe", "unripe and ripe", "unripe raspberr", "unripe berry",
                    "unripe berries", "unripe fruit", "small, unripe", "developing raspberr",
                    "developing fruit", "developing berries", "different stages", "multiple stages",
                    "various stages of", "ripening stages",
                ]
                if any(p in desc for p in mixed_phrases):
                    return False
                if item_ps == "fruiting" and not any(
                    p in desc for p in ["ripe raspberr", "ripe berry", "ripe berries", "ripe fruit", "red ripe", "fully ripe", "mature berry"]
                ):
                    return False
            if item_ps == r:
                return True
            item_ps_words = (item_ps.split() if item_ps else [])
            if r in item_ps_words:
                return True
            if r == "ripe" and any(p in desc for p in ["ripe raspberr", "ripe berry", "ripe berries", "ripe fruit", "red ripe", "fully ripe", "mature berry"]):
                return True
        return False
    
    def _calculate_result_confidence(self, result: Dict[str, Any], query_understanding, query: Optional[str] = None) -> float:
        """Calculate per-image filter match score based on how well it matches the query filters.
        
        This is NOT an AI confidence score - it's a relevance/match score that measures
        how well the image's metadata matches the structured filters extracted from the query.
        The actual AI confidence (how well the LLM understood the query) is in query_understanding.confidence.
        When query is provided, results whose description contains the query or key phrases (e.g. "standing upright")
        get a boost so they rank above results that only match filters.
        """
        filters = query_understanding.filters
        metadata = result.get('metadata', {})
        
        # Count how many filters are specified in the query
        total_filters = sum(1 for key, values in filters.items() if values)
        if total_filters == 0:
            return query_understanding.confidence
        
        # Track match quality and count matches vs misses
        match_scores = []
        matched_filters = 0
        total_filter_checks = 0
        
        # Check species match quality (CRITICAL - highest weight)
        if filters.get("species"):
            total_filter_checks += 1
            item_species = _metadata_str(metadata.get("species", "")).lower().strip()
            item_collection = _metadata_str(result.get("collection", "")).lower().strip()
            best_match = 0.0
            for species_filter in filters["species"]:
                species_lower = species_filter.lower().strip()
                # Exact match in metadata.species is strongest
                if item_species == species_lower:
                    import hashlib
                    item_id = result.get('id', '')
                    id_hash = int(hashlib.md5(item_id.encode()).hexdigest()[:2], 16)
                    # Exact match: 0.98 to 1.0 (perfect match!)
                    best_match = max(best_match, 0.98 + (id_hash % 3) / 100.0)
                    matched_filters += 1
                elif item_collection == species_lower:
                    best_match = max(best_match, 0.94)
                    matched_filters += 1
                elif item_species.startswith(species_lower + "_") or item_collection.startswith(species_lower + "_"):
                    best_match = max(best_match, 0.90)
                    matched_filters += 1
                else:
                    # Normalized match (handles variations)
                    species_norm = species_lower.replace("_", "").replace("-", "")
                    item_species_norm = item_species.replace("_", "").replace("-", "")
                    item_collection_norm = item_collection.replace("_", "").replace("-", "")
                    if species_norm == item_species_norm or species_norm == item_collection_norm:
                        best_match = max(best_match, 0.87)
                        matched_filters += 1
            if best_match > 0:
                match_scores.append(best_match)
        
        # Check time match quality
        if filters.get("time"):
            total_filter_checks += 1
            item_time = _metadata_str(metadata.get("time", "")).lower().strip()
            best_match = 0.0
            for time_filter in filters["time"]:
                time_lower = time_filter.lower().strip()
                if item_time == time_lower:
                    import hashlib
                    item_id = result.get('id', '')
                    id_hash = int(hashlib.md5(item_id.encode()).hexdigest()[:2], 16)
                    best_match = max(best_match, 0.96 + (id_hash % 3) / 100.0)
                    matched_filters += 1
                elif time_lower in item_time or item_time in time_lower:
                    best_match = max(best_match, 0.91)
                    matched_filters += 1
            if best_match > 0:
                match_scores.append(best_match)
        
        # Check action match quality: check against action field AND against description (either can match)
        if filters.get("action"):
            total_filter_checks += 1
            item_action = _metadata_str(metadata.get("action", "")).lower().strip()
            item_description = _metadata_str(metadata.get("description", "")).lower()
            _action_keyword_map = {
                "sleeping": ["sleep", "sleeping", "rest", "resting"],
                "feeding": ["feed", "feeding", "eating", "eat", "foraging", "forage"],
                "foraging": ["feed", "feeding", "eating", "eat", "foraging", "forage"],
                "resting": ["rest", "resting", "sleep", "sleeping"],
                "walking": ["walk", "walking", "moving"],
                "hunting": ["hunt", "hunting"],
                "alert": ["alert", "alerts", "watch", "watching", "looking at camera", "looking at the camera"],
                "moving": ["move", "moving", "walk", "walking"],
                "running": ["run", "running", "moving"],
                "perching": ["perch", "perching", "sitting", "sit"],
                "flying": ["fly", "flying"],
            }
            best_match = 0.0
            for action_filter in filters["action"]:
                action_lower = action_filter.lower().strip()
                matched_this = False
                # Check against action field
                if item_action == action_lower:
                    import hashlib
                    item_id = result.get('id', '')
                    id_hash = int(hashlib.md5(item_id.encode()).hexdigest()[:2], 16)
                    best_match = max(best_match, 0.96 + (id_hash % 3) / 100.0)
                    matched_this = True
                elif action_lower in item_action:
                    best_match = max(best_match, 0.92)
                    matched_this = True
                # Check against description field (in addition to action field)
                desc_matched = False
                if action_lower in item_description:
                    best_match = max(best_match, 0.88)
                    desc_matched = True
                if not desc_matched:
                    for _keyword, _variations in _action_keyword_map.items():
                        if action_lower == _keyword or action_lower in _variations:
                            if any(v in item_description for v in _variations):
                                best_match = max(best_match, 0.88)
                                desc_matched = True
                                break
                if not desc_matched:
                    action_base = action_lower.rstrip('ing').rstrip('ed')
                    if action_base in item_description and len(action_base) >= 3:
                        best_match = max(best_match, 0.88)
                        desc_matched = True
                if desc_matched:
                    matched_this = True
                if matched_this:
                    matched_filters += 1
            if best_match > 0:
                match_scores.append(best_match)
        
        # Check season match quality
        if filters.get("season"):
            total_filter_checks += 1
            item_season = _metadata_str(metadata.get("season", "")).lower().strip()
            best_match = 0.0
            for season_filter in filters["season"]:
                season_lower = season_filter.lower().strip()
                if item_season == season_lower:
                    import hashlib
                    item_id = result.get('id', '')
                    id_hash = int(hashlib.md5(item_id.encode()).hexdigest()[:2], 16)
                    best_match = max(best_match, 0.96 + (id_hash % 3) / 100.0)
                    matched_filters += 1
                elif season_lower in item_season:
                    best_match = max(best_match, 0.91)
                    matched_filters += 1
            if best_match > 0:
                match_scores.append(best_match)
        
        # Check scene match quality (IMPORTANT: Scene mismatches should be heavily penalized)
        if filters.get("scene"):
            total_filter_checks += 1
            item_scene = _metadata_str(metadata.get("scene", "")).lower().strip()
            best_match = 0.0
            scene_matched = False
            for scene_filter in filters["scene"]:
                scene_lower = scene_filter.lower().strip()
                if item_scene == scene_lower:
                    import hashlib
                    item_id = result.get('id', '')
                    id_hash = int(hashlib.md5(item_id.encode()).hexdigest()[:2], 16)
                    best_match = max(best_match, 0.97 + (id_hash % 3) / 100.0)  # Higher score for exact match
                    matched_filters += 1
                    scene_matched = True
                elif scene_lower in item_scene or item_scene in scene_lower:
                    best_match = max(best_match, 0.93)  # Good partial match
                    matched_filters += 1
                    scene_matched = True
            
            if scene_matched:
                match_scores.append(best_match)
            else:
                # HEAVY PENALTY: Scene filter specified but doesn't match
                # This ensures "field" queries prioritize field images over indoor images
                match_scores.append(0.3)  # Low score for scene mismatch
                # Don't increment matched_filters - this counts as a miss
        
        # Check weather match quality
        if filters.get("weather"):
            total_filter_checks += 1
            item_weather = _metadata_str(metadata.get("weather", "")).lower().strip()
            best_match = 0.0
            for weather_filter in filters["weather"]:
                weather_lower = weather_filter.lower().strip()
                if item_weather == weather_lower:
                    import hashlib
                    item_id = result.get('id', '')
                    id_hash = int(hashlib.md5(item_id.encode()).hexdigest()[:2], 16)
                    best_match = max(best_match, 0.94 + (id_hash % 3) / 100.0)
                    matched_filters += 1
                elif weather_lower in item_weather:
                    best_match = max(best_match, 0.89)
                    matched_filters += 1
            if best_match > 0:
                match_scores.append(best_match)
        
        # Check plant_state match quality (important for ripeness/color queries)
        # Prioritize exact plant_state match (e.g. ripe) over mixed/fruiting when user asked for ripe
        if filters.get("plant_state"):
            total_filter_checks += 1
            item_plant_state = _metadata_str(metadata.get("plant_state", "")).lower().strip()
            item_description = _metadata_str(metadata.get("description", "")).lower()
            best_match = 0.0
            for plant_state_filter in filters["plant_state"]:
                plant_state_lower = plant_state_filter.lower().strip()
                if item_plant_state == plant_state_lower:
                    import hashlib
                    item_id = result.get('id', '')
                    id_hash = int(hashlib.md5(item_id.encode()).hexdigest()[:2], 16)
                    best_match = max(best_match, 0.97 + (id_hash % 3) / 100.0)
                    matched_filters += 1
                elif plant_state_lower in ("ripe", "unripe", "mature", "green", "red") and item_plant_state in ("mixed", "fruiting"):
                    # User asked for specific state but item is mixed/fruiting — deprioritize so exact matches sort first
                    best_match = max(best_match, 0.62)
                    matched_filters += 1
                elif plant_state_lower == "ripe" and item_plant_state == "ripening":
                    # Ripening ≠ ripe: do not treat ripening as a match for "ripe"
                    best_match = max(best_match, 0.55)
                    matched_filters += 1
                elif plant_state_lower in (item_plant_state.split() if item_plant_state else []):
                    # Whole-word match only (so "ripe" does not match "ripening")
                    best_match = max(best_match, 0.93)
                    matched_filters += 1
                elif plant_state_lower in item_description:
                    # If item has mixed/fruiting but user asked for specific state, don't over-reward description match
                    desc_score = 0.91
                    if item_plant_state in ("mixed", "fruiting") and plant_state_lower in ("ripe", "unripe", "mature", "green", "red"):
                        desc_score = 0.68  # so exact plant_state=ripe still sorts first
                    # For "ripe": require whole-word or explicit phrases so "ripening" does not match
                    if plant_state_lower == "ripe":
                        import re as _re
                        ripe_ok = (_re.search(r"\bripe\b", item_description) and "ripening" not in item_description) or any(
                            p in item_description for p in ["ripe raspberr", "ripe berry", "ripe berries", "ripe fruit", "red ripe", "fully ripe", "mature berry"]
                        )
                        if ripe_ok:
                            best_match = max(best_match, desc_score)
                            matched_filters += 1
                    else:
                        plant_state_keywords = {
                            "mature": ["mature", "ripe", "ready", "fully developed"],
                            "unripe": ["unripe", "green", "immature", "young", "developing"],
                            "green": ["green", "unripe", "immature"],
                        }
                        for key, variations in plant_state_keywords.items():
                            if plant_state_lower == key:
                                if any(v in item_description for v in variations):
                                    best_match = max(best_match, desc_score)
                                    matched_filters += 1
                                    break
                        if best_match < desc_score:
                            best_match = max(best_match, min(0.89, desc_score))
                            matched_filters += 1
            if best_match > 0:
                match_scores.append(best_match)
        
        # Calculate base confidence from match quality
        if match_scores:
            # Use weighted average (species gets more weight if present)
            if filters.get("species") and len(match_scores) > 0:
                # Give species match 40% weight, others share remaining 60%
                species_score = match_scores[0] if filters.get("species") else 0
                other_scores = match_scores[1:] if filters.get("species") else match_scores
                if other_scores:
                    avg_other = sum(other_scores) / len(other_scores)
                    avg_match_quality = 0.4 * species_score + 0.6 * avg_other
                else:
                    avg_match_quality = species_score
            else:
                avg_match_quality = sum(match_scores) / len(match_scores)
            
            # Calculate filter match ratio (how many filters matched)
            match_ratio = matched_filters / total_filter_checks if total_filter_checks > 0 else 1.0
            
            # Base confidence: start high for good matches, penalize for missing filters
            if match_ratio == 1.0:
                # All filters matched - high confidence
                base_confidence = 0.90 + (avg_match_quality - 0.90) * 0.3  # 90-96%
            elif match_ratio >= 0.75:
                # Most filters matched - good confidence
                base_confidence = 0.80 + (avg_match_quality - 0.85) * 0.4  # 75-88%
            elif match_ratio >= 0.5:
                # Half filters matched - moderate confidence
                base_confidence = 0.65 + (avg_match_quality - 0.80) * 0.3  # 60-75%
            else:
                # Few filters matched - lower confidence
                base_confidence = 0.50 + (avg_match_quality - 0.75) * 0.3  # 50-65%
        else:
            # No matches - low confidence
            base_confidence = 0.45
        
        # Calculate metadata completeness boost
        metadata_fields = ['species', 'time', 'season', 'action', 'scene', 'weather', 'description']
        populated_fields = sum(1 for field in metadata_fields if metadata.get(field))
        metadata_completeness = populated_fields / len(metadata_fields) if metadata_fields else 0.7
        completeness_boost = 1.0 + (metadata_completeness - 0.7) * 0.05  # Up to +1.5% boost
        
        # Description quality boost
        description = metadata.get("description", "")
        description_length = len(description) if description else 0
        description_quality = min(1.0, description_length / 100.0) if description_length > 0 else 0.6
        description_boost = 1.0 + (description_quality - 0.6) * 0.03  # Up to +1.2% boost
        
        adjusted_confidence = base_confidence * completeness_boost * description_boost
        
        # Add variation based on ID hash for differentiation (±1.5%)
        import hashlib
        item_id = result.get('id', '')
        hash_value = int(hashlib.md5(item_id.encode()).hexdigest()[:8], 16)
        variation = (hash_value % 31 - 15) / 1000.0  # Range: -0.015 to +0.015
        adjusted_confidence = adjusted_confidence + variation
        
        # Cap at 50% to 98% (wider range, higher max for perfect matches)
        adjusted_confidence = min(0.98, max(0.50, adjusted_confidence))
        
        # Prioritize results whose description explicitly mentions the query or key phrase (e.g. "standing upright")
        if query and isinstance(query, str) and query.strip():
            item_desc = _metadata_str(metadata.get("description", "")).lower()
            if item_desc:
                q = query.strip().lower()
                if q in item_desc:
                    adjusted_confidence = max(adjusted_confidence, 0.96)  # Full query in description → top rank
                else:
                    words = q.split()
                    if len(words) >= 2:
                        key_phrase = " ".join(words[1:])  # e.g. "standing upright" from "woodchuck standing upright"
                        if key_phrase in item_desc:
                            adjusted_confidence = max(adjusted_confidence, 0.94)  # Key phrase in description
                    if len(words) >= 3:
                        key_phrase_3 = " ".join(words[1:4])
                        if key_phrase_3 in item_desc:
                            adjusted_confidence = max(adjusted_confidence, 0.95)
        
        # Round to 1 decimal place
        return round(adjusted_confidence, 1)
    
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
        from models import ModelInfo
        
        model_name = input_data.get("model_name")
        
        if model_name:
            model = self.model_registry.get_model(model_name)
            if not model:
                return {"error": f"Model {model_name} not found"}
            
            # Convert Model to ModelInfo (removes non-serializable handler)
            model_info = ModelInfo(
                name=model.name,
                type=model.type,
                description=model.description,
                version=model.version,
                supported_datasets=model.supported_datasets,
                parameters=model.parameters,
                metadata=model.metadata
            )
            # Convert to dict for JSON serialization (handles Enum types)
            return {
                "model": {
                    "name": model_info.name,
                    "type": model_info.type.value if hasattr(model_info.type, 'value') else str(model_info.type),
                    "description": model_info.description,
                    "version": model_info.version,
                    "supported_datasets": model_info.supported_datasets,
                    "parameters": model_info.parameters,
                    "metadata": model_info.metadata
                }
            }
        else:
            # Get all models and convert to serializable format
            all_models = self.model_registry.get_all_models()
            models_dict = {}
            for name, model_info in all_models.items():
                models_dict[name] = {
                    "name": model_info.name,
                    "type": model_info.type.value if hasattr(model_info.type, 'value') else str(model_info.type),
                    "description": model_info.description,
                    "version": model_info.version,
                    "supported_datasets": model_info.supported_datasets,
                    "parameters": model_info.parameters,
                    "metadata": model_info.metadata
                }
            return {
                "models": models_dict,
                "total": len(models_dict)
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
            
            print(f"📊 Crawler returned {len(datasets)} datasets")
            for i, dataset in enumerate(datasets):
                print(f"  {i+1}. {dataset.name} (source: {dataset.source_portal})")
            
            # Convert to MCP format
            results = []
            for i, dataset in enumerate(datasets):
                try:
                    result = {
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
                    }
                    results.append(result)
                    print(f"  ✅ Converted dataset {i+1}: {dataset.name}")
                except Exception as e:
                    print(f"  ❌ Error converting dataset {i+1} ({dataset.name}): {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue with next dataset instead of failing completely
            
            print(f"✅ Converted {len(results)} datasets to MCP format")
            for i, result in enumerate(results):
                print(f"  {i+1}. {result['name']} (source: {result['source']})")
            
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
        all_tools = self.tool_registry.get_all_tools()
        tool_names = list(all_tools.keys())
        print(f"🔧 Available tools ({len(tool_names)}): {', '.join(tool_names)}")
        print(f"📁 Available datasets: {len(self.dataset_registry.get_all_datasets())}")
        print(f"🤖 Available models: {len(self.model_registry.get_all_models())}")
        print(f"🔗 MCP Discovery: http://{host}:{port}/.well-known/mcp")
        print(f"🔧 Tools endpoint: http://{host}:{port}/mcp/tools")
        print(f"📁 Datasets endpoint: http://{host}:{port}/api/datasets")
        print(f"🤖 Models endpoint: http://{host}:{port}/api/models")
        print(f"💚 Health check: http://{host}:{port}/health")
        
        # Check if croissant tool was registered
        if "crawl_croissant_datasets" not in tool_names:
            print(f"⚠️  WARNING: crawl_croissant_datasets tool is NOT registered!")
            print(f"   CROISSANT_CRAWLER_AVAILABLE was: {CROISSANT_CRAWLER_AVAILABLE}")
        else:
            print(f"✅ crawl_croissant_datasets tool is registered and available")
        
        uvicorn.run(self.app, host=host, port=port)

# Global MCP server instance
mcp_server = MCPServer()

if __name__ == "__main__":
    # Run the core MCP server
    mcp_server.run()
