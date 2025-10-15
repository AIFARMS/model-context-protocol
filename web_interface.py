#!/usr/bin/env python3
"""
FastAPI Web Interface
- Web UI that consumes the MCP server
- Handles web-specific concerns and user interface
- Communicates with MCP server via HTTP
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import httpx
import os

# Import configuration
from config import WEB_CONFIG, BASE_DIR

class WebInterface:
    """Web interface that consumes the MCP server"""
    
    def __init__(self):
        self.app = FastAPI(title="AIFARMS Web Interface", version="2.0.0")
        self.mcp_server_url = WEB_CONFIG["mcp_server_url"]
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        self._setup_templates()
        
        print(f"🌐 Initialized Web Interface")
        print(f"🔗 MCP Server URL: {self.mcp_server_url}")
    
    def _setup_middleware(self):
        """Setup CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup web interface routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def home(request: Request):
            """Home page with search interface"""
            try:
                # Get available filters from MCP server
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.mcp_server_url}/api/datasets")
                    datasets_data = response.json()
                    
                    # Extract filters from datasets
                    all_filters = self._extract_filters_from_datasets(datasets_data["datasets"])
                    
                return self.templates.TemplateResponse("search_interface.html", {
                    "request": request,
                    "filters": all_filters,
                    "datasets": datasets_data["datasets"],
                    "mcp_server_url": self.mcp_server_url
                })
            except Exception as e:
                print(f"Error loading home page: {e}")
                # Return basic interface if MCP server is unavailable
                return self.templates.TemplateResponse("search_interface.html", {
                    "request": request,
                    "filters": {},
                    "datasets": {},
                    "mcp_server_url": self.mcp_server_url,
                    "error": "MCP server unavailable"
                })
        
        @self.app.get("/search", response_class=HTMLResponse)
        async def search_page(request: Request):
            """Search results page"""
            return self.templates.TemplateResponse("search_results.html", {
                "request": request,
                "results": {},
                "query": "",
                "filters": {}
            })
        
        @self.app.post("/search", response_class=HTMLResponse)
        async def search_images(
            request: Request,
            query: str = Form(""),
            dataset: str = Form(""),
            category: List[str] = Form([]),
            species: List[str] = Form([]),
            time: List[str] = Form([]),
            season: List[str] = Form([]),
            action: List[str] = Form([]),
            plant_state: List[str] = Form([]),
            limit: int = Form(50),
            page: int = Form(1)
        ):
            """Handle search form submission"""
            try:
                print(f"🔍 Web interface search request: query='{query}', dataset='{dataset}'")
                
                # Prepare search request
                search_data = {
                    "query": query,
                    "dataset": dataset if dataset else None,
                    "filters": {
                        "category": category if category else [],
                        "species": species if species else [],
                        "time": time if time else [],
                        "season": season if season else [],
                        "action": action if action else [],
                        "plant_state": plant_state if plant_state else []
                    },
                    "limit": limit,
                    "offset": (page - 1) * limit
                }
                
                print(f"🔍 Calling MCP server with: {search_data}")
                
                # Call MCP server search tool
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.mcp_server_url}/mcp/tools/search_images",
                        json=search_data
                    )
                    mcp_response = response.json()
                    print(f"🔍 MCP server response: {mcp_response}")
                    
                    # Extract the actual results from the MCP response structure
                    search_results = {}
                    if "content" in mcp_response:
                        for content_item in mcp_response["content"]:
                            if content_item.get("type") == "result" and "data" in content_item:
                                search_results = content_item["data"]
                                break
                    
                    if not search_results:
                        search_results = {"error": "No results found in MCP response", "mcp_response": mcp_response}
                    
                    # Add image URLs to the results
                    if "results" in search_results and search_results["results"]:
                        for result in search_results["results"]:
                            result["image_url"] = self._construct_image_url(result)
                    
                    print(f"🔍 Extracted search results: {search_results}")
                
                # Get filters for the results page
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.mcp_server_url}/api/datasets")
                    datasets_data = response.json()
                    all_filters = self._extract_filters_from_datasets(datasets_data["datasets"])
                
                return self.templates.TemplateResponse("search_results.html", {
                    "request": request,
                    "results": search_results,
                    "query": query,
                    "filters": all_filters,
                    "datasets": datasets_data["datasets"],
                    "current_page": page,
                    "limit": limit
                })
                
            except Exception as e:
                print(f"❌ Search error: {e}")
                import traceback
                traceback.print_exc()
                return self.templates.TemplateResponse("search_results.html", {
                    "request": request,
                    "results": {"error": str(e)},
                    "query": query,
                    "filters": {},
                    "datasets": {},
                    "current_page": page,
                    "limit": limit
                })
        
        @self.app.post("/llm_search", response_class=HTMLResponse)
        async def llm_search_images(
            request: Request,
            query: str = Form(""),
            dataset: str = Form(""),
            limit: int = Form(50),
            page: int = Form(1)
        ):
            """Handle LLM-powered search with intelligent query understanding"""
            try:
                print(f"🧠 LLM search request: query='{query}', dataset='{dataset}'")
                
                # Prepare LLM search request
                search_data = {
                    "query": query,
                    "dataset": dataset if dataset else None,
                    "limit": limit,
                    "offset": (page - 1) * limit
                }
                
                print(f"🧠 Calling MCP server LLM search with: {search_data}")
                
                # Call MCP server LLM search tool
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.mcp_server_url}/mcp/tools/llm_search",
                        json=search_data
                    )
                    mcp_response = response.json()
                    print(f"🧠 MCP server LLM response: {mcp_response}")
                    
                    # Extract the actual results from the MCP response structure
                    search_results = {}
                    if "content" in mcp_response:
                        for content_item in mcp_response["content"]:
                            if content_item.get("type") == "result" and "data" in content_item:
                                search_results = content_item["data"]
                                break
                    
                    if not search_results:
                        search_results = {"error": "No results found in MCP LLM response", "mcp_response": mcp_response}
                    
                    # Add image URLs to the results and sort by confidence
                    if "results" in search_results and search_results["results"]:
                        for result in search_results["results"]:
                            result["image_url"] = self._construct_image_url(result)
                        
                        # Sort results by confidence (highest first) if available
                        if any('llm_confidence' in result for result in search_results["results"]):
                            search_results["results"].sort(key=lambda x: x.get('llm_confidence', 0), reverse=True)
                            print(f"🧠 Results sorted by confidence (highest first)")
                    
                    print(f"🧠 Extracted LLM search results: {search_results}")
                
                # Get filters for the results page
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.mcp_server_url}/api/datasets")
                    datasets_data = response.json()
                    all_filters = self._extract_filters_from_datasets(datasets_data["datasets"])
                
                return self.templates.TemplateResponse("llm_search_results.html", {
                    "request": request,
                    "results": search_results,
                    "query": query,
                    "filters": all_filters,
                    "datasets": datasets_data["datasets"],
                    "current_page": page,
                    "limit": limit
                })
                
            except Exception as e:
                print(f"❌ LLM search error: {e}")
                import traceback
                traceback.print_exc()
                return self.templates.TemplateResponse("llm_search_results.html", {
                    "request": request,
                    "results": {"error": str(e)},
                    "query": query,
                    "filters": {},
                    "datasets": {},
                    "current_page": page,
                    "limit": limit
                })
        
        @self.app.get("/api/search")
        async def api_search(
            query: str = "",
            dataset: str = "",
            category: List[str] = Query([]),
            species: List[str] = Query([]),
            time: List[str] = Query([]),
            season: List[str] = Query([]),
            action: List[str] = Query([]),
            plant_state: List[str] = Query([]),
            limit: int = Query(50),
            offset: int = Query(0)
        ):
            """API endpoint for search"""
            try:
                search_data = {
                    "query": query,
                    "dataset": dataset if dataset else None,
                    "filters": {
                        "category": category if category else [],
                        "species": species if species else [],
                        "time": time if time else [],
                        "season": season if season else [],
                        "action": action if action else [],
                        "plant_state": plant_state if plant_state else []
                    },
                    "limit": limit,
                    "offset": offset
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.mcp_server_url}/mcp/tools/search_images",
                        json=search_data
                    )
                    return response.json()
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/datasets")
        async def api_datasets():
            """Get available datasets"""
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.mcp_server_url}/api/datasets")
                    return response.json()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/models")
        async def api_models():
            """Get available models"""
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.mcp_server_url}/api/models")
                    return response.json()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/inference")
        async def api_inference(request: Request):
            """Run model inference"""
            try:
                body = await request.json()
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.mcp_server_url}/api/inference",
                        json=body
                    )
                    return response.json()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/image/{filename:path}")
        async def serve_image(filename: str):
            """Serve images from the MCP server"""
            try:
                # Redirect to MCP server for images
                image_url = f"{self.mcp_server_url}/image/{filename}"
                return JSONResponse({"image_url": image_url})
            except Exception as e:
                raise HTTPException(status_code=404, detail="Image not found")
        
        @self.app.get("/debug/search")
        async def debug_search():
            """Debug endpoint to test search functionality"""
            try:
                # Test search with a simple query
                search_data = {
                    "query": "bobcat",
                    "limit": 5
                }
                
                print(f"🔍 Debug search request: {search_data}")
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.mcp_server_url}/mcp/tools/search_images",
                        json=search_data
                    )
                    mcp_response = response.json()
                    print(f"🔍 Debug search response: {mcp_response}")
                    
                    # Extract the actual results from the MCP response structure
                    search_results = {}
                    if "content" in mcp_response:
                        for content_item in mcp_response["content"]:
                            if content_item.get("type") == "result" and "data" in content_item:
                                search_results = content_item["data"]
                                break
                    
                    if not search_results:
                        search_results = {"error": "No results found in MCP response", "mcp_response": mcp_response}
                    
                    # Add image URLs to the results
                    if "results" in search_results and search_results["results"]:
                        for result in search_results["results"]:
                            result["image_url"] = self._construct_image_url(result)
                    
                    print(f"🔍 Debug extracted results: {search_results}")
                
                return {
                    "debug_info": "Search test completed",
                    "request": search_data,
                    "response": mcp_response,
                    "extracted_results": search_results,
                    "mcp_server_url": self.mcp_server_url
                }
                
            except Exception as e:
                print(f"❌ Debug search error: {e}")
                import traceback
                traceback.print_exc()
                return {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
        
        @self.app.get("/debug/files")
        async def debug_files():
            """Debug endpoint to show file structure"""
            try:
                import os
                from pathlib import Path
                
                base_dir = Path("/opt/mcp-data-server")
                file_info = {
                    "base_directory": str(base_dir),
                    "base_exists": base_dir.exists(),
                    "base_contents": [],
                    "image_files": [],
                    "mcp_files": []
                }
                
                if base_dir.exists():
                    # List base directory contents
                    for item in base_dir.iterdir():
                        if item.is_dir():
                            file_info["base_contents"].append(f"📁 {item.name}/")
                            # Look for images in subdirectories
                            for subitem in item.iterdir():
                                if subitem.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                                    file_info["image_files"].append(f"{item.name}/{subitem.name}")
                        elif item.suffix == '.json':
                            file_info["mcp_files"].append(item.name)
                        elif item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                            file_info["image_files"].append(item.name)
                
                return file_info
                
            except Exception as e:
                return {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
        
        @self.app.get("/test/images")
        async def test_images():
            """Test endpoint to check if image files exist"""
            try:
                import os
                from pathlib import Path
                
                images_dir = Path("/opt/mcp-data-server/images")
                test_files = []
                
                if images_dir.exists():
                    # List first 10 image files
                    for item in images_dir.iterdir():
                        if item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                            test_files.append(item.name)
                            if len(test_files) >= 10:
                                break
                
                return {
                    "images_directory": str(images_dir),
                    "directory_exists": images_dir.exists(),
                    "sample_files": test_files,
                    "total_files": len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
                }
                
            except Exception as e:
                return {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
        
        @self.app.get("/images/{filename:path}")
        async def serve_image(filename: str):
            """Serve images from the flat images directory"""
            try:
                # For the flat images folder structure, all images are in /opt/mcp-data-server/images/
                image_path = f"/opt/mcp-data-server/images/{filename}"
                
                print(f"🖼️  Looking for image: {filename}")
                print(f"🖼️  Full path: {image_path}")
                
                if not os.path.exists(image_path):
                    print(f"❌ Image not found: {image_path}")
                    raise HTTPException(status_code=404, detail=f"Image not found: {filename}")
                
                print(f"✅ Found image at: {image_path}")
                
                # Return the image file
                from fastapi.responses import FileResponse
                return FileResponse(image_path)
                
            except Exception as e:
                print(f"❌ Error serving image {filename}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _setup_templates(self):
        """Setup Jinja2 templates"""
        templates_dir = BASE_DIR / "templates"
        templates_dir.mkdir(exist_ok=True)
        self.templates = Jinja2Templates(directory=str(templates_dir))
        
        # Create templates if they don't exist
        self._create_templates()
    
    def _create_templates(self):
        """Create HTML templates"""
        templates_dir = BASE_DIR / "templates"
        
        # Search interface template
        search_template = templates_dir / "search_interface.html"
        if not search_template.exists():
            with open(search_template, "w") as f:
                f.write(self._get_search_template())
        
        # Search results template
        results_template = templates_dir / "search_results.html"
        if not results_template.exists():
            with open(results_template, "w") as f:
                f.write(self._get_results_template())
        
        # LLM search results template
        llm_results_template = templates_dir / "llm_search_results.html"
        if not llm_results_template.exists():
            with open(llm_results_template, "w") as f:
                f.write(self._get_llm_results_template())
    
    def _extract_filters_from_datasets(self, datasets: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Extract available filters from datasets"""
        all_filters = {
            "categories": [],
            "species": [],
            "times": [],
            "seasons": [],
            "actions": [],
            "plant_states": []
        }
        
        for dataset in datasets:
            if "filters" in dataset:
                for filter_type, values in dataset["filters"].items():
                    if filter_type in all_filters and isinstance(values, list):
                        all_filters[filter_type].extend(values)
        
        # Remove duplicates and sort
        for filter_type in all_filters:
            all_filters[filter_type] = sorted(list(set(all_filters[filter_type])))
        
        return all_filters
    
    def _construct_image_url(self, result: Dict[str, Any]) -> str:
        """Construct image URL from result data"""
        try:
            # Use the MCP ID directly - it should match the actual filename
            if "id" in result:
                # The id should be like "bobcat_008" which matches the actual filename
                return f"/images/{result['id']}.jpg"
            
            # Fallback to any existing image_url
            elif "image_url" in result:
                return result["image_url"]
            
            # If no image info, return placeholder
            else:
                return "/images/placeholder.jpg"
                
        except Exception as e:
            print(f"❌ Error constructing image URL for result {result}: {e}")
            return "/images/placeholder.jpg"
    
    def _get_search_template(self) -> str:
        """Get the search interface HTML template"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AIFARMS Species Search</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .search-form { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .form-row { display: flex; gap: 20px; margin-bottom: 15px; }
        .form-group { flex: 1; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: bold; color: #333; }
        .form-group select, .form-group input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }
        .search-button { background: #007bff; color: white; padding: 12px 30px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        .search-button:hover { background: #0056b3; }
        .mcp-info { background: #d4edda; padding: 15px; border-radius: 8px; margin: 20px 0; }
        .error { background: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px; margin: 20px 0; }
        .dataset-selector { margin-bottom: 20px; }
        .llm-search-section { margin-top: 30px; padding: 20px; background: #f0f7ff; border-radius: 8px; }
        .llm-search-form { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .llm-search-button { background: #28a745; color: white; padding: 12px 30px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        .llm-search-button:hover { background: #218838; }
        .llm-examples { margin-top: 20px; padding-top: 15px; border-top: 1px solid #eee; }
        .llm-examples h4 { margin-bottom: 10px; }
        .llm-examples ul { list-style: none; padding: 0; margin: 0; }
        .llm-examples li { margin-bottom: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌿 AIFARMS Species Observation Search</h1>
            <p>Search across multiple datasets using natural language queries and advanced filters</p>
        </div>
        
        {% if error %}
        <div class="error">
            <strong>Warning:</strong> {{ error }}. Some features may be limited.
        </div>
        {% endif %}
        
        <div class="mcp-info">
            <h3>🤖 MCP Server Integration</h3>
            <p>This interface connects to the AIFARMS MCP Server at <code>{{ mcp_server_url }}</code></p>
            <p><strong>Available Datasets:</strong> {{ datasets|length }} datasets loaded</p>
            <p><strong>Features:</strong> Multi-dataset search, ML model inference, extensible tools</p>
        </div>
        
        <form method="POST" action="/search" class="search-form">
            <div class="dataset-selector">
                <label for="dataset">Dataset (Optional - leave empty to search all):</label>
                <select id="dataset" name="dataset">
                    <option value="">All Datasets</option>
                    {% for name, info in datasets.items() %}
                    <option value="{{ name }}">{{ name|title }} ({{ info.total_images }} images)</option>
                    {% endfor %}
                </select>
            </div>
            
            <div class="form-group">
                <label for="query">Search Query:</label>
                <input type="text" id="query" name="query" placeholder="e.g., bobcat at night, coyote hunting, blooming flowers" value="{{ request.query_params.get('query', '') }}">
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label for="category">Category:</label>
                    <select id="category" name="category" multiple>
                        {% for category in filters.categories %}
                        <option value="{{ category }}">{{ category|title }}</option>
                        {% endfor %}
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="species">Species/Type:</label>
                    <select id="species" name="species" multiple>
                        {% for species in filters.species %}
                        <option value="{{ species }}">{{ species|title }}</option>
                        {% endfor %}
                    </select>
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label for="time">Time of Day:</label>
                    <select id="time" name="time" multiple>
                        {% for time in filters.times %}
                        <option value="{{ time }}">{{ time|title }}</option>
                        {% endfor %}
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="season">Season:</label>
                    <select id="season" name="season" multiple>
                        {% for season in filters.seasons %}
                        <option value="{{ season }}">{{ season|title }}</option>
                        {% endfor %}
                    </select>
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label for="action">Actions (Animals):</label>
                    <select id="action" name="action" multiple>
                        {% for action in filters.actions %}
                        <option value="{{ action }}">{{ action|title }}</option>
                        {% endfor %}
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="plant_state">Plant States:</label>
                    <select id="plant_state" name="plant_state" multiple>
                        {% for state in filters.plant_states %}
                        <option value="{{ state }}">{{ state|title }}</option>
                        {% endfor %}
                    </select>
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label for="limit">Results per page:</label>
                    <select id="limit" name="limit">
                        <option value="10">10</option>
                        <option value="20" selected>20</option>
                        <option value="50">50</option>
                        <option value="100">100</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>&nbsp;</label>
                    <button type="submit" class="search-button">🔍 Search</button>
                </div>
            </div>
        </form>
            
        <!-- LLM-Powered Search Section -->
        <div class="llm-search-section">
            <h3>🧠 AI-Powered Search <span style="background: #28a745; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin-left: 10px;">CONFIDENCE SCORING</span></h3>
            <p>Use natural language to find exactly what you're looking for. Results are ranked by AI confidence scores for optimal relevance:</p>
            
            <form action="/llm_search" method="post" class="llm-search-form">
                <div class="form-row">
                    <div class="form-group">
                        <label for="llm_query">Natural Language Query:</label>
                        <input type="text" id="llm_query" name="query" placeholder="e.g., 'bobcats hunting at dawn in summer forest'" required>
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="llm_dataset">Dataset (optional):</label>
                        <select id="llm_dataset" name="dataset">
                            <option value="">All datasets</option>
                            {% for dataset in datasets %}
                            <option value="{{ dataset.name }}">{{ dataset.name|title }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="llm_limit">Results per page:</label>
                        <select id="llm_limit" name="limit">
                            <option value="10">10</option>
                            <option value="20" selected>20</option>
                            <option value="50">50</option>
                            <option value="100">100</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>&nbsp;</label>
                        <button type="submit" class="llm-search-button">🧠 AI Search</button>
                    </div>
                </div>
            </form>
            
            <div class="llm-examples">
                <h4>Example Queries:</h4>
                <ul>
                    <li><strong>"predators hunting at dawn"</strong> - Find hunting animals in early morning</li>
                    <li><strong>"animals in summer forest"</strong> - Wildlife in warm forest environments</li>
                    <li><strong>"coyotes walking in winter"</strong> - Coyotes moving in cold weather</li>
                    <li><strong>"plants growing in garden"</strong> - Vegetation in cultivated areas</li>
                </ul>
                
                <div style="margin-top: 15px; padding: 10px; background: #e8f5e8; border-radius: 6px; border-left: 4px solid #28a745;">
                    <h5 style="margin: 0 0 8px 0; color: #155724;">🎯 Confidence-Based Ranking</h5>
                    <p style="margin: 0; font-size: 14px; color: #155724;">
                        Results are automatically ranked by AI confidence scores. Higher confidence means better query understanding and more relevant results.
                    </p>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    def _get_results_template(self) -> str:
        """Get the search results HTML template"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Search Results - AIFARMS</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .back-link { margin-bottom: 20px; }
        .back-link a { color: #007bff; text-decoration: none; font-weight: bold; }
        .results-info { background: #e9ecef; padding: 15px; border-radius: 8px; margin: 20px 0; }
        .image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
        .image-card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: white; }
        .image-card img { width: 100%; height: 200px; object-fit: cover; border-radius: 4px; }
        .metadata { margin-top: 10px; font-size: 14px; }
        .no-results { text-align: center; padding: 40px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <div class="back-link">
            <a href="/">← Back to Search</a>
        </div>
        
        <h1>Search Results</h1>
        
        {% if results.error %}
        <div class="results-info">
            <p><strong>Error:</strong> {{ results.error }}</p>
        </div>
        {% elif results.results %}
        <div class="results-info">
            <p><strong>Query:</strong> {{ query or "All images" }}</p>
            <p><strong>Total results:</strong> {{ results.total_count or results.results|length }}</p>
            <p><strong>Page:</strong> {{ current_page }} of {{ (results.total_count / limit)|round(0, 'ceil')|int if results.total_count else 1 }}</p>
        </div>
        
        <div class="image-grid">
            {% for result in results.results %}
            <div class="image-card">
                <img src="{{ result.image_url }}" alt="{{ result.metadata.species|title if result.metadata and result.metadata.species else 'Image' }}">
                <div class="metadata">
                    {% if result.metadata %}
                        {% if result.metadata.species %}
                        <div><strong>Species:</strong> {{ result.metadata.species|title }}</div>
                        {% endif %}
                        {% if result.metadata.action %}
                        <div><strong>Action:</strong> {{ result.metadata.action|title }}</div>
                        {% endif %}
                        {% if result.metadata.time %}
                        <div><strong>Time:</strong> {{ result.metadata.time|title }}</div>
                        {% endif %}
                        {% if result.metadata.season %}
                        <div><strong>Season:</strong> {{ result.metadata.season|title }}</div>
                        {% endif %}
                        {% if result.metadata.scene %}
                        <div><strong>Scene:</strong> {{ result.metadata.scene|title }}</div>
                        {% endif %}
                        {% if result.metadata.weather %}
                        <div><strong>Weather:</strong> {{ result.metadata.weather|title }}</div>
                        {% endif %}
                        {% if result.metadata.date %}
                        <div><strong>Date:</strong> {{ result.metadata.date }}</div>
                        {% endif %}
                        {% if result.metadata.description %}
                        <div><strong>Description:</strong> {{ result.metadata.description[:100] }}{% if result.metadata.description|length > 100 %}...{% endif %}</div>
                        {% endif %}
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        </div>
        {% else %}
        <div class="no-results">
            <h2>No results found</h2>
            <p>Try adjusting your search criteria or filters.</p>
            <p><strong>Debug Info:</strong></p>
            <p>Query: {{ query }}</p>
            <p>Results structure: {{ results }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""
    
    def _get_llm_results_template(self) -> str:
        """Get the LLM search results HTML template"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>LLM Search Results - AIFARMS</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .back-link { margin-bottom: 20px; }
        .back-link a { color: #007bff; text-decoration: none; font-weight: bold; }
        .results-info { background: #e9ecef; padding: 15px; border-radius: 8px; margin: 20px 0; }
        .confidence-panel { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 12px; margin: 20px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
        .confidence-score { font-size: 24px; font-weight: bold; margin-bottom: 10px; }
        .confidence-bar { background: rgba(255,255,255,0.3); height: 8px; border-radius: 4px; margin: 10px 0; }
        .confidence-fill { background: #4CAF50; height: 100%; border-radius: 4px; transition: width 0.3s ease; }
        .confidence-details { font-size: 14px; opacity: 0.9; }
        .image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
        .image-card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: white; position: relative; }
        .image-card img { width: 100%; height: 200px; object-fit: cover; border-radius: 4px; }
        .confidence-badge { position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.8); color: white; padding: 5px 10px; border-radius: 20px; font-size: 12px; font-weight: bold; }
        .metadata { margin-top: 10px; font-size: 14px; }
        .no-results { text-align: center; padding: 40px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <div class="back-link">
            <a href="/">← Back to Search</a>
        </div>
        
        <h1>🧠 LLM Search Results</h1>
        
        {% if results.error %}
        <div class="results-info">
            <p><strong>Error:</strong> {{ results.error }}</p>
        </div>
        {% elif results.results %}
        <div class="results-info">
            <p><strong>Query:</strong> {{ query or "All images" }}</p>
            <p><strong>Total results:</strong> {{ results.total_count or results.results|length }}</p>
            <p><strong>Page:</strong> {{ current_page }} of {{ (results.total_count / limit)|round(0, 'ceil')|int if results.total_count else 1 }}</p>
        </div>
        
        {% if results.llm_understanding %}
        <div class="confidence-panel">
            <div class="confidence-score">
                🧠 AI Confidence: {{ "%.0f"|format(results.llm_understanding.confidence * 100) }}%
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {{ "%.0f"|format(results.llm_understanding.confidence * 100) }}%"></div>
            </div>
            <div class="confidence-details">
                <p><strong>Intent:</strong> {{ results.llm_understanding.intent }}</p>
                <p><strong>Reasoning:</strong> {{ results.llm_understanding.reasoning }}</p>
                <p><strong>Applied Filters:</strong> 
                    {% for filter_type, values in results.llm_understanding.filters.items() %}
                        {% if values %}
                            <strong>{{ filter_type|title }}:</strong> {{ values|join(", ") }}{% if not loop.last %} | {% endif %}
                        {% endif %}
                    {% endfor %}
                </p>
            </div>
        </div>
        {% endif %}
        
        <div class="image-grid">
            {% for result in results.results %}
            <div class="image-card">
                {% if result.llm_confidence %}
                <div class="confidence-badge">
                    {{ "%.0f"|format(result.llm_confidence * 100) }}%
                </div>
                {% endif %}
                <img src="{{ result.image_url }}" alt="{{ result.metadata.species|title if result.metadata and result.metadata.species else 'Image' }}">
                <div class="metadata">
                    {% if result.metadata %}
                        {% if result.metadata.species %}
                        <div><strong>Species:</strong> {{ result.metadata.species|title }}</div>
                        {% endif %}
                        {% if result.metadata.action %}
                        <div><strong>Action:</strong> {{ result.metadata.action|title }}</div>
                        {% endif %}
                        {% if result.metadata.time %}
                        <div><strong>Time:</strong> {{ result.metadata.time|title }}</div>
                        {% endif %}
                        {% if result.metadata.season %}
                        <div><strong>Season:</strong> {{ result.metadata.season|title }}</div>
                        {% endif %}
                        {% if result.metadata.scene %}
                        <div><strong>Scene:</strong> {{ result.metadata.scene|title }}</div>
                        {% endif %}
                        {% if result.metadata.weather %}
                        <div><strong>Weather:</strong> {{ result.metadata.weather|title }}</div>
                        {% endif %}
                        {% if result.metadata.date %}
                        <div><strong>Date:</strong> {{ result.metadata.date }}</div>
                        {% endif %}
                        {% if result.metadata.description %}
                        <div><strong>Description:</strong> {{ result.metadata.description[:100] }}{% if result.metadata.description|length > 100 %}...{% endif %}</div>
                        {% endif %}
                    {% endif %}
                    {% if result.llm_confidence %}
                    <div style="margin-top: 10px; padding: 5px; background: #f8f9fa; border-radius: 4px;">
                        <small><strong>AI Confidence:</strong> {{ "%.0f"|format(result.llm_confidence * 100) }}%</small>
                    </div>
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        </div>
        {% else %}
        <div class="no-results">
            <h2>No results found</h2>
            <p>Try adjusting your search criteria or filters.</p>
            <p><strong>Debug Info:</strong></p>
            <p>Query: {{ query }}</p>
            <p>Results structure: {{ results }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""
    
    def run(self, host: str = None, port: int = None):
        """Run the web interface"""
        host = host or WEB_CONFIG.get("web_host", "0.0.0.0")
        port = port or WEB_CONFIG.get("web_port", 8187)
        
        print(f"🌐 Starting Web Interface on {host}:{port}")
        print(f"🔗 MCP Server: {self.mcp_server_url}")
        print(f"📱 Web Interface: http://{host}:{port}")
        print(f"🔍 Search: http://{host}:{port}/")
        
        uvicorn.run(self.app, host=host, port=port)

# Global web interface instance
web_interface = WebInterface()

if __name__ == "__main__":
    # Run the web interface
    web_interface.run()
