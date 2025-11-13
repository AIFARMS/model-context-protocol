#!/usr/bin/env python3
"""
LLM Service for intelligent query understanding
- Converts natural language queries to structured search criteria
- Provides semantic understanding of user intent
- Supports multiple LLM backends
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import httpx

@dataclass
class QueryUnderstanding:
    """Structured understanding of a natural language query"""
    intent: str  # What the user wants to find
    entities: List[str]  # Key entities mentioned
    filters: Dict[str, Any]  # Structured filters
    confidence: float  # Confidence in the understanding
    reasoning: str  # Why this understanding was chosen

class LLMService:
    """Service for LLM-powered query understanding"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        
        # Check if this is a project key or personal key
        if api_key and api_key.startswith("sk-proj-"):
            # Project API key - use different endpoint
            self.base_url = "https://api.openai.com/v1/chat/completions"
            self.is_project_key = True
            print(f"🧠 Using OpenAI Project API key (length: {len(api_key)})")
        elif api_key and api_key.startswith("sk-"):
            # Personal API key - use standard endpoint
            self.base_url = "https://api.openai.com/v1/chat/completions"
            self.is_project_key = False
            print(f"🧠 Using OpenAI Personal API key (length: {len(api_key)})")
        else:
            # No key or invalid format
            self.base_url = "https://api.openai.com/v1/chat/completions"
            self.is_project_key = False
            print(f"🧠 No valid OpenAI API key provided, will use rule-based fallback")
        
        # Define the search schema for the LLM
        self.search_schema = {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "What the user is looking for"
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key entities mentioned in the query"
                },
                "filters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Categories to search in (e.g., wildlife, plants, pests)"
                        },
                        "species": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific species mentioned"
                        },
                        "time": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Time of day (dawn, day, dusk, night)"
                        },
                        "season": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Season (spring, summer, fall, winter)"
                        },
                        "action": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Actions or behaviors (walking, eating, sleeping)"
                        },
                        "scene": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Environment or scene (forest, field, water)"
                        },
                        "weather": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Weather conditions"
                        }
                    }
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence in the understanding (0-1)"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of the understanding"
                }
            },
            "required": ["intent", "entities", "filters", "confidence", "reasoning"]
        }
    
    async def understand_query(self, query: str, available_filters: Dict[str, List[str]] = None) -> QueryUnderstanding:
        """Convert natural language query to structured understanding"""
        try:
            if not self.api_key:
                # Fallback to rule-based understanding if no API key
                return self._rule_based_understanding(query, available_filters)
            
            # Use OpenAI API for intelligent understanding
            return await self._openai_understanding(query, available_filters)
            
        except Exception as e:
            print(f"❌ LLM understanding error: {e}")
            # Fallback to rule-based
            return self._rule_based_understanding(query, available_filters)
    
    async def _openai_understanding(self, query: str, available_filters: Dict[str, List[str]] = None) -> QueryUnderstanding:
        """Use OpenAI API for query understanding"""
        system_prompt = f"""You are an expert at understanding natural language queries about agricultural and wildlife datasets.

Your task is to convert user queries into structured search criteria.

Available filter options:
{json.dumps(available_filters, indent=2) if available_filters else "All filters available"}

Return a JSON response matching this schema:
{json.dumps(self.search_schema, indent=2)}

Examples:
- "bobcat" → {{"intent": "find bobcat images", "entities": ["bobcat"], "filters": {{"species": ["bobcat"]}}, "confidence": 0.9, "reasoning": "Direct species mention"}}
- "animals in summer forest" → {{"intent": "find wildlife in summer forest environment", "entities": ["animals", "summer", "forest"], "filters": {{"category": ["wildlife"], "season": ["summer"], "scene": ["forest"]}}, "confidence": 0.8, "reasoning": "Combined environmental and seasonal criteria"}}
- "predators hunting at dawn" → {{"intent": "find hunting predators at dawn", "entities": ["predators", "hunting", "dawn"], "filters": {{"action": ["hunting"], "time": ["dawn"]}}, "confidence": 0.85, "reasoning": "Behavior and time specification"}}

Query: "{query}"
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        # Prepare headers based on key type
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.is_project_key:
            # Project API key uses different header format
            headers["Authorization"] = f"Bearer {self.api_key}"
            print(f"🧠 Using project API key authentication")
        else:
            # Personal API key
            headers["Authorization"] = f"Bearer {self.api_key}"
            print(f"🧠 Using personal API key authentication")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0.1,
                        "max_tokens": 500
                    },
                    timeout=30.0
                )
                
                print(f"🧠 OpenAI API response status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    try:
                        # Parse the JSON response
                        parsed = json.loads(content)
                        return QueryUnderstanding(**parsed)
                    except Exception as e:
                        print(f"❌ Failed to parse LLM response: {e}")
                        print(f"🧠 Raw response content: {content}")
                        return self._rule_based_understanding(query, available_filters)
                else:
                    print(f"❌ OpenAI API error: {response.status_code}")
                    print(f"🧠 Error response: {response.text}")
                    return self._rule_based_understanding(query, available_filters)
                    
        except Exception as e:
            print(f"❌ OpenAI API request failed: {e}")
            return self._rule_based_understanding(query, available_filters)
    
    def _rule_based_understanding(self, query: str, available_filters: Dict[str, List[str]] = None) -> QueryUnderstanding:
        """Fallback rule-based query understanding"""
        query_lower = query.lower()
        
        # Simple keyword matching
        entities = []
        filters = {
            "category": [],
            "species": [],
            "time": [],
            "season": [],
            "action": [],
            "scene": [],
            "weather": []
        }
        
        # Extract species
        species_keywords = ["bobcat", "coyote", "deer", "fox", "chicken", "pig", "goat", "carrot", "strawberry", "raspberry"]
        for species in species_keywords:
            if species in query_lower:
                entities.append(species)
                filters["species"].append(species)
        
        # Extract time
        time_keywords = ["dawn", "day", "dusk", "night", "morning", "afternoon", "evening"]
        for time in time_keywords:
            if time in query_lower:
                entities.append(time)
                filters["time"].append(time)
        
        # Extract season
        season_keywords = ["spring", "summer", "fall", "winter", "autumn"]
        for season in season_keywords:
            if season in query_lower:
                entities.append(season)
                filters["season"].append(season)
        
        # Extract actions
        action_keywords = ["walking", "hunting", "eating", "sleeping", "running", "standing"]
        for action in action_keywords:
            if action in query_lower:
                entities.append(action)
                filters["action"].append(action)
        
        # Extract scenes
        scene_keywords = ["forest", "field", "water", "mountain", "garden", "farm"]
        for scene in scene_keywords:
            if scene in query_lower:
                entities.append(scene)
                filters["scene"].append(scene)
        
        # Determine intent
        if filters["species"]:
            intent = f"find {', '.join(filters['species'])} images"
        elif filters["time"] or filters["season"]:
            intent = f"find images from {', '.join(filters['time'] + filters['season'])}"
        else:
            intent = f"search for {query}"
        
        return QueryUnderstanding(
            intent=intent,
            entities=entities,
            filters=filters,
            confidence=0.6 if entities else 0.3,
            reasoning="Rule-based keyword matching"
        )
    
    def get_available_models(self) -> List[str]:
        """Get list of available LLM models"""
        return [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
            "claude-3-sonnet",
            "claude-3-opus"
        ]
    
    def is_available(self) -> bool:
        """Check if LLM service is available"""
        return bool(self.api_key)

