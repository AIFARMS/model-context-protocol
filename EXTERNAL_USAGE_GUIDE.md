# MCP Server - External Usage Guide

This guide explains how others can use the AIFARMS MCP (Model Context Protocol) server programmatically and with LLMs.

## Overview

The MCP server provides a standardized interface for accessing datasets, running searches, and executing ML model inference. It can be used by:

- **Developers** - Via REST API calls from any programming language
- **LLMs and AI Agents** - Via MCP protocol or REST API with function calling
- **Web Applications** - Via HTTP endpoints
- **MCP-Compatible Clients** - Via the Model Context Protocol standard

## Table of Contents

1. [Programmatic Access](#programmatic-access)
2. [LLM Integration](#llm-integration)
3. [Available Endpoints](#available-endpoints)
4. [Usage Examples](#usage-examples)
5. [Deployment & Access](#deployment--access)
6. [Authentication & Security](#authentication--security)

---

## Programmatic Access

The MCP server exposes a REST API that can be called from any programming language or tool that supports HTTP requests.

### Base URL

```
http://your-server-address:8188
```

(Default: `http://localhost:8188` for local development)

### API Format

All tools are accessible via HTTP POST requests:

```
POST /mcp/tools/{tool_name}
Content-Type: application/json
```

### Quick Example

```python
import requests

# Server URL
MCP_SERVER = "http://your-server:8188"

# Execute a search tool
response = requests.post(
    f"{MCP_SERVER}/mcp/tools/llm_search",
    json={
        "query": "coyote looking at the camera",
        "dataset": "coyote",
        "limit": 20
    }
)

results = response.json()
print(f"Found {results['total_count']} results")
```

---

## LLM Integration

The MCP server is designed for seamless LLM integration in two ways:

### 1. MCP Protocol (Recommended for LLMs)

The server follows the [Model Context Protocol](https://github.com/modelcontextprotocol) standard, which allows LLMs to:

- **Auto-discover tools** by querying `/mcp/tools`
- **Understand tool schemas** automatically
- **Execute tools** with proper parameter validation
- **Get structured responses** in MCP-compliant format

#### How LLMs Connect

1. **Discovery**: LLM queries `GET /mcp` to get server capabilities
2. **Tool Listing**: LLM queries `GET /mcp/tools` to see available tools
3. **Tool Execution**: LLM calls `POST /mcp/tools/{tool_name}` with parameters
4. **Response Handling**: LLM receives structured JSON responses

#### Example: LLM Tool Discovery

```python
# LLM can discover available tools
response = requests.get(f"{MCP_SERVER}/mcp/tools")
tools = response.json()

# Each tool includes:
# - name: Tool identifier
# - description: What the tool does
# - input_schema: Required/optional parameters
# - tags: Categories for filtering
```

### 2. REST API with Function Calling

LLMs can also use the REST API directly with function calling capabilities:

```python
# Example: Using OpenAI with function calling
import openai

# Define the tool as a function
functions = [{
    "name": "search_images",
    "description": "Search for images using natural language queries",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "dataset": {"type": "string", "description": "Dataset name"},
            "limit": {"type": "integer", "description": "Result limit"}
        }
    }
}]

# LLM can call this function
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Find images of coyotes at night"}],
    functions=functions
)

# Then execute the tool call
if response.choices[0].message.get("function_call"):
    function_name = response.choices[0].message["function_call"]["name"]
    function_args = json.loads(response.choices[0].message["function_call"]["arguments"])
    
    # Call the MCP server
    tool_response = requests.post(
        f"{MCP_SERVER}/mcp/tools/{function_name}",
        json=function_args
    )
```

### 3. LLM-Powered Search Tool

The server includes a special `llm_search` tool that uses LLM query understanding:

- **Natural Language Processing**: Understands complex queries
- **Intent Recognition**: Extracts search intent from queries
- **Filter Extraction**: Automatically converts queries to structured filters
- **Confidence Scoring**: Returns results with confidence scores

```python
# LLM-powered search
response = requests.post(
    f"{MCP_SERVER}/mcp/tools/llm_search",
    json={
        "query": "wildlife animals hunting at night in winter",
        "limit": 50
    }
)

# Response includes:
# - llm_understanding: How the query was interpreted
# - results: Ranked by confidence
# - confidence scores: Per-result relevance scores
```

---

## Available Endpoints

### Protocol Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/mcp` | GET | Get MCP protocol information and capabilities |
| `/mcp/tools` | GET | List all available tools with schemas |
| `/mcp/tools/{tool_name}` | POST | Execute a specific tool |
| `/mcp/resources` | GET | List available resources |
| `/mcp/resources/{resource_name}` | GET | Get a specific resource |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/datasets` | GET | List all available datasets |
| `/api/datasets/{name}` | GET | Get specific dataset information |
| `/api/datasets/{name}/images` | GET | Get images from a dataset (paginated) |
| `/api/models` | GET | List all available ML models |
| `/api/models/{name}` | GET | Get specific model information |
| `/api/inference` | POST | Run ML model inference on images |
| `/api/search` | POST | Search across all datasets |

### Utility Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check endpoint |

---

## Usage Examples

### Example 1: List Available Datasets

```python
import requests

MCP_SERVER = "http://your-server:8188"

# Get all datasets
response = requests.get(f"{MCP_SERVER}/api/datasets")
datasets = response.json()

print(f"Available datasets: {len(datasets['datasets'])}")
for dataset in datasets['datasets']:
    print(f"  - {dataset['name']}: {dataset['image_count']} images")
```

### Example 2: Search for Images

```python
import requests

MCP_SERVER = "http://your-server:8188"

# Basic search
response = requests.post(
    f"{MCP_SERVER}/mcp/tools/search_images",
    json={
        "query": "coyote",
        "dataset": "coyote",
        "limit": 10
    }
)

results = response.json()
for result in results['results']:
    print(f"{result['id']}: {result.get('metadata', {}).get('description', '')}")
```

### Example 3: LLM-Powered Search

```python
import requests

MCP_SERVER = "http://your-server:8188"

# Intelligent search with LLM understanding
response = requests.post(
    f"{MCP_SERVER}/mcp/tools/llm_search",
    json={
        "query": "animals hunting at night in snowy conditions",
        "limit": 20
    }
)

results = response.json()

# Check LLM understanding
understanding = results.get('llm_understanding', {})
print(f"Intent: {understanding.get('intent')}")
print(f"Confidence: {understanding.get('confidence')}")
print(f"Reasoning: {understanding.get('reasoning')}")

# Process results
for result in results['results']:
    print(f"{result['id']} (confidence: {result.get('llm_confidence', 0)})")
```

### Example 4: Get Dataset Information

```python
import requests

MCP_SERVER = "http://your-server:8188"

# Get specific dataset info
response = requests.post(
    f"{MCP_SERVER}/mcp/tools/get_dataset_info",
    json={"dataset_name": "coyote"}
)

dataset = response.json()['dataset']
print(f"Dataset: {dataset['name']}")
print(f"Type: {dataset['type']}")
print(f"Images: {dataset['image_count']}")
print(f"Available filters: {list(dataset['filters'].keys())}")
```

### Example 5: JavaScript/Node.js Example

```javascript
const axios = require('axios');

const MCP_SERVER = 'http://your-server:8188';

// Search for images
async function searchImages(query, dataset = null) {
    try {
        const response = await axios.post(
            `${MCP_SERVER}/mcp/tools/llm_search`,
            {
                query: query,
                dataset: dataset,
                limit: 20
            }
        );
        
        return response.data;
    } catch (error) {
        console.error('Search error:', error.message);
        throw error;
    }
}

// Usage
searchImages('wildlife at night')
    .then(results => {
        console.log(`Found ${results.total_count} results`);
        results.results.forEach(result => {
            console.log(`- ${result.id}`);
        });
    });
```

### Example 6: cURL Examples

```bash
# Get server info
curl http://your-server:8188/mcp

# List all tools
curl http://your-server:8188/mcp/tools

# Search for images
curl -X POST http://your-server:8188/mcp/tools/search_images \
  -H "Content-Type: application/json" \
  -d '{
    "query": "coyote",
    "dataset": "coyote",
    "limit": 10
  }'

# LLM-powered search
curl -X POST http://your-server:8188/mcp/tools/llm_search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "animals hunting at night",
    "limit": 20
  }'

# Get dataset info
curl -X POST http://your-server:8188/mcp/tools/get_dataset_info \
  -H "Content-Type: application/json" \
  -d '{"dataset_name": "coyote"}'
```

---

## Available Tools

The server provides 6 main tools:

### 1. `search_images`
Search for images using natural language queries and filters.

**Parameters:**
- `query` (string, optional): Natural language search query
- `dataset` (string, optional): Specific dataset to search in
- `filters` (object, optional): Structured filters (species, time, season, etc.)
- `limit` (integer, default: 50): Maximum number of results
- `offset` (integer, default: 0): Pagination offset

### 2. `llm_search`
Intelligent search using LLM query understanding (recommended).

**Parameters:**
- `query` (string, required): Natural language search query
- `dataset` (string, optional): Specific dataset to search in
- `limit` (integer, default: 50): Maximum number of results
- `offset` (integer, default: 0): Pagination offset

**Response includes:**
- `llm_understanding`: Query interpretation with intent, entities, filters, confidence
- `results`: Images ranked by confidence score
- `total_count`: Total matching images

### 3. `get_dataset_info`
Get information about available datasets.

**Parameters:**
- `dataset_name` (string, optional): Specific dataset name (if omitted, returns all)

### 4. `get_model_info`
Get information about available ML models.

**Parameters:**
- `model_name` (string, optional): Specific model name (if omitted, returns all)

**Note:** Models are optional - the server works great for search without any models.

### 5. `run_inference`
Run ML model inference on images.

**Parameters:**
- `dataset_name` (string, required): Dataset to run inference on
- `model_name` (string, required): Model to use
- `image_ids` (array, optional): Specific image IDs to process
- `parameters` (object, optional): Additional model parameters

### 6. `crawl_croissant_datasets`
Crawl and discover Croissant datasets from Hugging Face.

**Parameters:**
- `portals` (array, optional): Portals to crawl (default: ["huggingface"])
- `limit` (integer, optional): Maximum number of datasets to discover

---

## Deployment & Access

### Making the Server Accessible

1. **Deploy the Server**
   ```bash
   # On your server
   python mcp_core_server.py
   # Or use a process manager (systemd, supervisor, etc.)
   ```

2. **Configure Network Access**
   - Set `MCP_HOST=0.0.0.0` to allow external connections
   - Set `MCP_PORT=8188` (or your preferred port)
   - Ensure firewall allows connections on the port

3. **Share the Server URL**
   - Provide the base URL: `http://your-server-address:8188`
   - Or use HTTPS: `https://your-server-address:8188`

### Environment Variables

```bash
# Server configuration
export MCP_HOST=0.0.0.0
export MCP_PORT=8188

# Optional: LLM configuration for enhanced search
export OPENAI_API_KEY=sk-...
export LLM_MODEL=gpt-4o-mini

# Or use Gemini
export GOOGLE_API_KEY=...
export GEMINI_MODEL=gemini-2.5-flash
```

### Docker Deployment (Optional)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8188
CMD ["python", "mcp_core_server.py"]
```

```bash
docker build -t mcp-server .
docker run -p 8188:8188 -e MCP_HOST=0.0.0.0 mcp-server
```

---

## Authentication & Security

### Current Configuration

The server currently has:
- **CORS enabled** for all origins (configured in `mcp_core_server.py`)
- **No authentication** by default (open access)

### Adding Authentication (Recommended for Production)

You can add authentication by:

1. **API Key Authentication**
   ```python
   # Add to mcp_core_server.py
   API_KEY = os.getenv("MCP_API_KEY")
   
   @self.app.middleware("http")
   async def verify_api_key(request: Request, call_next):
       if request.url.path.startswith("/mcp") or request.url.path.startswith("/api"):
           api_key = request.headers.get("X-API-Key")
           if api_key != API_KEY:
               return JSONResponse({"error": "Invalid API key"}, status_code=401)
       return await call_next(request)
   ```

2. **OAuth/JWT Authentication**
   - Integrate with your authentication provider
   - Validate tokens in middleware

3. **Rate Limiting**
   ```python
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)
   
   @self.app.post("/mcp/tools/{tool_name}")
   @limiter.limit("100/minute")
   async def execute_tool(...):
       ...
   ```

### Security Best Practices

1. **Use HTTPS** in production
2. **Restrict CORS** to specific origins
3. **Add rate limiting** to prevent abuse
4. **Validate all inputs** (already implemented)
5. **Monitor access logs** for suspicious activity
6. **Keep dependencies updated**

---

## Response Formats

### Success Response

```json
{
  "results": [...],
  "total_count": 150,
  "query": "coyote",
  "dataset": "coyote"
}
```

### Error Response

```json
{
  "error": "Dataset 'invalid' not found",
  "detail": "The requested dataset does not exist"
}
```

### LLM Search Response

```json
{
  "query": "animals hunting at night",
  "llm_understanding": {
    "intent": "search",
    "entities": ["animals", "hunting", "night"],
    "filters": {
      "action": ["hunting"],
      "time": ["night"]
    },
    "confidence": 0.95,
    "reasoning": "Query indicates search for images of animals performing hunting actions during nighttime"
  },
  "results": [
    {
      "id": "coyote_001",
      "collection": "coyote",
      "llm_confidence": 0.95,
      "metadata": {...}
    }
  ],
  "total_count": 45
}
```

---

## Integration with Popular LLMs

### OpenAI GPT-4 / GPT-3.5

```python
import openai
import requests

# Define MCP tools as OpenAI functions
mcp_tools = [
    {
        "type": "function",
        "function": {
            "name": "search_images",
            "description": "Search for images using natural language queries",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "dataset": {"type": "string"},
                    "limit": {"type": "integer"}
                }
            }
        }
    }
]

# Use with OpenAI
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Find images of coyotes"}],
    functions=mcp_tools
)

# Execute function call via MCP server
if response.choices[0].message.get("function_call"):
    # Call MCP server...
```

### Anthropic Claude

Claude can use the MCP protocol directly or via REST API with tool use.

### LangChain

```python
from langchain.tools import Tool
import requests

mcp_search = Tool(
    name="mcp_search_images",
    func=lambda query: requests.post(
        "http://your-server:8188/mcp/tools/search_images",
        json={"query": query}
    ).json(),
    description="Search for images using natural language"
)
```

---

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check server is running
   - Verify host and port settings
   - Check firewall rules

2. **CORS Errors**
   - Server has CORS enabled by default
   - If issues persist, check CORS configuration

3. **Tool Not Found**
   - Verify tool name matches exactly
   - Check `/mcp/tools` endpoint for available tools

4. **Empty Results**
   - Verify dataset exists
   - Check query parameters
   - Review dataset filters

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python mcp_core_server.py
```

---

## Support & Documentation

- **Full Tool Documentation**: See `MCP_TOOLS_GUIDE.md`
- **Setup Instructions**: See `README.md`
- **Adding Datasets**: See `ADDING_DATASET_TYPES.md`
- **Protocol Spec**: [Model Context Protocol](https://github.com/modelcontextprotocol)

---

## License

Apache License 2.0 - See `LICENSE` for details.
