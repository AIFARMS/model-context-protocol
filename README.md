# Model Context Protocol Server

This repository packages the AIFARMS implementation of the [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol) server stack.  
It exposes curated datasets, pluggable machine-learning tools, and an optional LLM-powered semantic search layer to downstream agents and web clients.

## Overview
- **Core MCP server** (`mcp_core_server.py`) provides protocol endpoints, registers tools, and exposes dataset/model metadata.
- **Dataset registry** dynamically discovers `*_mcp_data.json` files and surfaces their filterable attributes.
- **Model registry** keeps track of inference handlers and records execution history.
- **Tool registry** manages synchronous/async tools and wraps results in MCP-compliant envelopes.
- **FastAPI web interface** (`web_interface.py`) renders a search UI that calls the MCP server and highlights LLM-backed confidence scores.
- **Config module** centralizes base paths, MCP/HTTP settings, and optional LLM credentials.

## Repository Layout
- `config.py` – runtime configuration, environment variables, and directory bootstrap.
- `dataset_registry.py` – dataset discovery, filtering helpers, and caching with adapter support.
- `dataset_adapter.py` – extensible adapter system for different dataset types.
- `model_registry.py` – model definitions, registration API, and inference history.
- `tool_registry.py` – registration/execution wrapper producing MCP tool responses.
- `mcp_core_server.py` – FastAPI app exposing `/mcp`, `/api/datasets`, `/api/models`, `/api/inference`, etc.
- `web_interface.py` – FastAPI app that communicates with the MCP server using HTTPX; auto-generates Jinja templates under `/opt/mcp-data-server/templates`.
- `ADDING_DATASET_TYPES.md` – comprehensive guide for adding new dataset types.
- `example_custom_dataset.json` – example of a custom dataset format.
- `mcp_json/` – sample MCP JSON payloads for reference.
- `requirements.txt` – Python dependencies.
- `LICENSE` – Apache-2.0.

## Prerequisites
- Python 3.10+ recommended.
- Access to `/opt/mcp-data-server` (default `BASE_DIR`) with:
  - `images/` – flat image store used by the web interface.
  - `datasets/` and `plugins/` directories (created automatically).
  - One or more `<dataset>_mcp_data.json` files containing `images`, metadata, and optional filters.
- (Optional) OpenAI-compatible API key if you want to enable LLM-powered query understanding.

## Quick Start
1. **Clone**
   ```bash
   git clone https://github.com/AIFARMS/model-context-protocol.git
   cd model-context-protocol
   ```
2. **Create environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   The project memory suggests a Conda env `myenv`; alternatively:
   ```bash
   conda env create -f ../myenv.yml  # adjust path as needed
   conda activate myenv
   pip install -r requirements.txt
   ```
3. **Configure environment variables** (optional values shown).
   ```bash
   export OPENAI_API_KEY=sk-...
   export LLM_MODEL=gpt-4o-mini
   export MCP_HOST=0.0.0.0
   export MCP_PORT=8188
   export WEB_HOST=0.0.0.0
   export WEB_PORT=8187
   export MCP_BASE_URL=http://127.0.0.1:8188
   export MCP_SERVER_URL=http://127.0.0.1:8188
   ```
   Add additional overrides for data directories if `/opt/mcp-data-server` is not available.
4. **Seed datasets**: copy or generate `<name>_mcp_data.json` into the base directory and ensure referenced images exist in `images/`.

## Running the Services
Open two terminals (in the same virtual environment):

### 1. Core MCP Server
```bash
python mcp_core_server.py
```
The server logs discovered datasets, registered tools, and available models.  
It exposes:
- `GET /health` – service heartbeat.
- `GET /mcp` – protocol metadata.
- `GET /mcp/tools` & `POST /mcp/tools/{tool}` – tool enumeration/execution.
- `GET /mcp/resources` – resource listing.
- `GET /api/datasets`, `/api/datasets/{name}`, `/api/datasets/{name}/images`
- `GET /api/models`, `/api/models/{name}`
- `POST /api/inference`
- `POST /api/search` (placeholder returning incoming payload until implemented).

### 2. Web Interface
```bash
python web_interface.py
```
Default UI endpoints:
- `GET /` – dataset-aware search form with classic + AI search tabs.
- `POST /search` – synchronous tool-based search.
- `POST /llm_search` – LLM-assisted search ranked by confidence.
- `GET /api/search`, `/api/datasets`, `/api/models`, `/api/inference` – proxy to the MCP server.
- Debug helpers: `/debug/search`, `/debug/files`, `/test/images`.

If the LLM service is unavailable, the interface still functions with rule-based filtering (logs will show a fallback message).

## Dataset JSON Format
Each `*_mcp_data.json` file should contain an `images` array. Example entry:
```json
{
  "id": "bobcat_008",
  "collection": "bobcat",
  "category": "wildlife",
  "metadata": {
    "species": "lynx rufus",
    "time": "night",
    "season": "winter",
    "action": "hunting",
    "scene": "forest edge",
    "weather": "snow",
    "description": "Bobcat stalking prey near a snowy log."
  }
}
```
The dataset registry derives available filters (times, seasons, actions, etc.) and caches the image list in memory for search tools.

## Adding Your Own Extensions

### Datasets

The system supports **extensible dataset types** beyond species observation datasets:

1. **Simple Custom Datasets**: Drop a `<name>_mcp_data.json` with `dataset_type: "custom"` and a schema definition. The generic adapter will handle it automatically.

2. **Custom Dataset Adapters**: For complex datasets, create a custom adapter class implementing the `DatasetAdapter` interface.

See **[ADDING_DATASET_TYPES.md](ADDING_DATASET_TYPES.md)** for a complete guide with examples.

**Quick Example** - Custom dataset with schema:
```json
{
  "dataset_type": "custom",
  "description": "Weather observations",
  "schema": {
    "filter_fields": ["station", "condition", "temperature_range"]
  },
  "images": [
    {
      "id": "weather_001",
      "collection": "station_a",
      "station": "station_a",
      "condition": "sunny",
      "metadata": {"timestamp": "2024-01-01T12:00:00Z"}
    }
  ]
}
```

### Other Extensions
- **Models**: call `ModelRegistry.register_model` with a handler that returns prediction payloads. Modify `_register_default_models` or create a plugin.
- **Tools**: use `ToolRegistry.register_tool` inside `mcp_core_server.py` or from an external module; handlers can be async or sync.
- **Web templates**: override the generated HTML in `/opt/mcp-data-server/templates`.

## Local Development Tips
- Use `uvicorn mcp_core_server:mcp_server.app --reload` for hot reloads.
- Similarly `uvicorn web_interface:web_interface.app --reload`.
- Configure `.env` files or `.envaifarms` (project-specific naming) for persistent environment variables.
- Keep an eye on logs—many helpers print informative messages about dataset discovery and LLM resolution.

## License

Apache License 2.0 – see `LICENSE` for details.
