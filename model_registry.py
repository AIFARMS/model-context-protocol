#!/usr/bin/env python3
"""
Model registry for managing ML models and inference
"""

import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from models import ModelInfo, ModelType, InferenceRequest, InferenceResult
from config import MODEL_CONFIG

@dataclass
class Model:
    """Model definition"""
    name: str
    type: ModelType
    description: str
    version: str
    supported_datasets: List[str]
    handler: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    async_handler: bool = False

class ModelRegistry:
    """Registry for managing ML models"""
    
    def __init__(self):
        self.models: Dict[str, Model] = {}
        self.inference_history: List[Dict[str, Any]] = []
        # Note: Default placeholder models are disabled - register real models when available
        # self._register_default_models()
    
    def _register_default_models(self):
        """Register default models"""
        # Baseline classifier
        self.register_model(
            name="baseline_classifier",
            type=ModelType.CLASSIFICATION,
            description="Baseline classification model for species identification",
            version="1.0.0",
            supported_datasets=["wildlife", "plants"],
            handler=self._baseline_classifier,
            parameters={
                "confidence_threshold": 0.5,
                "max_predictions": 5
            }
        )
        
        # Object detector
        self.register_model(
            name="object_detector",
            type=ModelType.DETECTION,
            description="Object detection model for wildlife and plants",
            version="1.0.0",
            supported_datasets=["wildlife", "plants"],
            handler=self._object_detector,
            parameters={
                "confidence_threshold": 0.3,
                "nms_threshold": 0.5
            }
        )
    
    def register_model(self, name: str, type: ModelType, description: str, version: str,
                      supported_datasets: List[str], handler: Callable, 
                      parameters: Dict[str, Any] = None, metadata: Dict[str, Any] = None):
        """Register a new model"""
        if name in self.models:
            raise ValueError(f"Model {name} already registered")
        
        # Check if handler is async
        import asyncio
        async_handler = asyncio.iscoroutinefunction(handler)
        
        model = Model(
            name=name,
            type=type,
            description=description,
            version=version,
            supported_datasets=supported_datasets,
            handler=handler,
            parameters=parameters or {},
            metadata=metadata or {},
            async_handler=async_handler
        )
        
        self.models[name] = model
        print(f"🤖 Registered model: {name} (v{version}) - {type.value}")
    
    def get_model(self, name: str) -> Optional[Model]:
        """Get a model by name"""
        return self.models.get(name)
    
    def get_all_models(self) -> Dict[str, ModelInfo]:
        """Get all registered models as ModelInfo objects"""
        return {
            name: ModelInfo(
                name=model.name,
                type=model.type,
                description=model.description,
                version=model.version,
                supported_datasets=model.supported_datasets,
                parameters=model.parameters,
                metadata=model.metadata
            )
            for name, model in self.models.items()
        }
    
    def get_models_by_type(self, model_type: ModelType) -> List[str]:
        """Get models by type"""
        return [name for name, model in self.models.items() if model.type == model_type]
    
    def get_models_for_dataset(self, dataset_name: str) -> List[str]:
        """Get models that support a specific dataset"""
        return [name for name, model in self.models.items() if dataset_name in model.supported_datasets]
    
    async def run_inference(self, request: InferenceRequest) -> InferenceResult:
        """Run inference using a specific model on a dataset"""
        if request.model_name not in self.models:
            raise ValueError(f"Model {request.model_name} not found")
        
        model = self.models[request.model_name]
        start_time = time.time()
        
        try:
            # Run inference
            if model.async_handler:
                results = await model.handler(request)
            else:
                results = model.handler(request)
            
            processing_time = time.time() - start_time
            
            # Record inference
            self.inference_history.append({
                "model": request.model_name,
                "dataset": request.dataset_name,
                "image_count": len(request.image_ids),
                "processing_time": processing_time,
                "timestamp": time.time(),
                "success": True
            })
            
            return InferenceResult(
                model_name=request.model_name,
                dataset_name=request.dataset_name,
                results=results,
                processing_time=processing_time,
                metadata={
                    "model_version": model.version,
                    "parameters": request.parameters
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Record failed inference
            self.inference_history.append({
                "model": request.model_name,
                "dataset": request.dataset_name,
                "image_count": len(request.image_ids),
                "error": str(e),
                "processing_time": processing_time,
                "timestamp": time.time(),
                "success": False
            })
            
            raise e
    
    def get_inference_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get inference history"""
        return self.inference_history[-limit:]
    
    # Default model implementations
    def _baseline_classifier(self, request: InferenceRequest) -> List[Dict[str, Any]]:
        """Baseline classifier implementation"""
        # This is a placeholder - you would implement your actual model here
        results = []
        for image_id in request.image_ids:
            results.append({
                "image_id": image_id,
                "predictions": [
                    {"class": "unknown", "confidence": 0.8, "bbox": None}
                ],
                "processing_time": 0.1
            })
        return results
    
    def _object_detector(self, request: InferenceRequest) -> List[Dict[str, Any]]:
        """Object detector implementation"""
        # This is a placeholder - you would implement your actual model here
        results = []
        for image_id in request.image_ids:
            results.append({
                "image_id": image_id,
                "detections": [
                    {"class": "object", "confidence": 0.7, "bbox": [100, 100, 200, 200]}
                ],
                "processing_time": 0.2
            })
        return results
