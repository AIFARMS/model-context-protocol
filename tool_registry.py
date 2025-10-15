#!/usr/bin/env python3
"""
Extensible tool registry for the MCP server
"""

import asyncio
import time
from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass, field
from models import SearchRequest, SearchResponse, InferenceRequest, InferenceResult

@dataclass
class Tool:
    """Tool definition"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    async_handler: bool = False

class ToolRegistry:
    """Registry for managing and executing tools"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.execution_history: List[Dict[str, Any]] = []
    
    def register_tool(self, name: str, description: str, input_schema: Dict[str, Any], 
                      handler: Callable, tags: List[str] = None, version: str = "1.0.0"):
        """Register a new tool"""
        if name in self.tools:
            raise ValueError(f"Tool {name} already registered")
        
        # Check if handler is async
        async_handler = asyncio.iscoroutinefunction(handler)
        
        tool = Tool(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
            tags=tags or [],
            version=version,
            async_handler=async_handler
        )
        
        self.tools[name] = tool
        print(f"🔧 Registered tool: {name} (v{version})")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def get_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered tools with their schemas"""
        return {
            name: {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
                "tags": tool.tags,
                "version": tool.version
            }
            for name, tool in self.tools.items()
        }
    
    def get_tools_by_tag(self, tag: str) -> List[str]:
        """Get tools by tag"""
        return [name for name, tool in self.tools.items() if tag in tool.tags]
    
    async def execute_tool(self, name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name"""
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found")
        
        tool = self.tools[name]
        start_time = time.time()
        
        try:
            # Execute the tool
            if tool.async_handler:
                result = await tool.handler(input_data)
            else:
                result = tool.handler(input_data)
            
            execution_time = time.time() - start_time
            
            # Record execution
            self.execution_history.append({
                "tool": name,
                "input": input_data,
                "output": result,
                "execution_time": execution_time,
                "timestamp": time.time(),
                "success": True
            })
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Tool {name} executed successfully in {execution_time:.2f}s"
                    },
                    {
                        "type": "result",
                        "data": result
                    }
                ],
                "metadata": {
                    "tool": name,
                    "execution_time": execution_time,
                    "success": True
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Record failed execution
            self.execution_history.append({
                "tool": name,
                "input": input_data,
                "error": str(e),
                "execution_time": execution_time,
                "timestamp": time.time(),
                "success": False
            })
            
            raise e
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get tool execution history"""
        return self.execution_history[-limit:]
    
    def clear_history(self):
        """Clear execution history"""
        self.execution_history.clear()
