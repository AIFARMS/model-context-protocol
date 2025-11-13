# Croissant Dataset Crawler Implementation Instructions

## 📋 Overview
This package implements a Croissant dataset crawler that discovers datasets from AI Institute portals and integrates them with your existing MCP server and web interface.

## 📁 Files Included
- `croissant_crawler.py` - Main crawler module
- `mcp_server_updates.py` - Updates needed for mcp_core_server.py
- `web_interface_updates.py` - Updates needed for web_interface.py
- `croissant_datasets.html` - HTML template for dataset display
- `IMPLEMENTATION_INSTRUCTIONS.md` - This file

## 🚀 Implementation Steps

### Step 1: Add the Crawler Module
1. Copy `croissant_crawler.py` to your server: `/opt/mcp-data-server/`
2. Install required dependencies:
   ```bash
   pip install requests aiohttp
   ```

### Step 2: Update MCP Server
1. Open `mcp_core_server.py`
2. Add the import: `from croissant_crawler import CroissantCrawler`
3. Add the `_crawl_croissant_datasets_handler` method to the MCPServer class
4. Add the tool registration in the `__init__` method

### Step 3: Update Web Interface
1. Open `web_interface.py`
2. Add the `/croissant_datasets` endpoint
3. Add the `_get_croissant_datasets_template` method
4. Copy `croissant_datasets.html` to `templates/` directory

### Step 4: Test the Implementation
1. Restart your MCP server: `python3 mcp_core_server.py &`
2. Restart your web interface: `python3 web_interface.py &`
3. Visit `http://localhost:8187/croissant_datasets`

## 🔧 Configuration

### Target Portals
The crawler is configured to search:
- **AIFARMS Data Portal**: `https://data.aifarms.org`
- **CyVerse Sierra**: `https://sierra.cyverse.org/datasets`
- **AgAID GitHub**: `https://github.com/TrevorBuchanan/AgAIDResearch`

### Adding New Portals
To add new portals, update the `portals` dictionary in `CroissantCrawler.__init__()`:
```python
self.portals = {
    'aifarms': 'https://data.aifarms.org',
    'cyverse': 'https://sierra.cyverse.org/datasets',
    'agaid_github': 'https://github.com/TrevorBuchanan/AgAIDResearch',
    'new_portal': 'https://new-portal-url.com'
}
```

## 🎯 Features

### Dataset Discovery
- Automatically crawls AI Institute portals
- Parses Croissant metadata files
- Extracts dataset information, fields, and keywords

### Web Interface
- Beautiful dataset browsing interface
- Rich metadata display
- Source portal identification
- Direct links to original datasets

### MCP Integration
- New `crawl_croissant_datasets` tool
- Integrates with existing confidence-scoring search
- Asynchronous crawling for performance

## 🔍 Usage

### Via MCP Server
```bash
curl -X POST http://localhost:8188/mcp/tools/crawl_croissant_datasets \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Via Web Interface
Visit `http://localhost:8187/croissant_datasets` to browse discovered datasets.

## 🚨 Troubleshooting

### Common Issues
1. **Import errors**: Ensure `croissant_crawler.py` is in the same directory as `mcp_core_server.py`
2. **Template not found**: Ensure `croissant_datasets.html` is in the `templates/` directory
3. **Crawling errors**: Check network connectivity and portal availability

### Debug Mode
Enable debug logging by adding:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🎉 Expected Results

After implementation, you should have:
- **Automatic dataset discovery** from AI Institute portals
- **Rich metadata display** showing fields, keywords, and licensing
- **Beautiful web interface** for browsing datasets
- **Integration with your existing** confidence-scoring search system
- **Extensible architecture** for adding new portals

## 📞 Support

If you encounter issues:
1. Check the server logs for error messages
2. Verify all files are in the correct locations
3. Ensure all dependencies are installed
4. Test the crawler independently before integration

## 🚀 Future Enhancements

- **Scheduled crawling** for automatic updates
- **Dataset search** integration with confidence scoring
- **Metadata filtering** by source, license, or keywords
- **Download management** for discovered datasets
- **API endpoints** for programmatic access

---

**Happy crawling! 🚀🌾✨**

