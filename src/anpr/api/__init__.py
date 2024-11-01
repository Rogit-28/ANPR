"""
ANPR API Module

This module provides the API service layer for the Automatic Number Plate Recognition system.
It includes endpoints for live stream processing, video processing, detection queries,
and system monitoring as specified in section 5 of the specification document.

The API follows REST principles and uses standardized request/response models
with proper error handling and response formatting.
"""

from .service import app
from .models import (
    SuccessResponse, ErrorResponse, StreamStartRequest, StreamStartResponse,
    StreamStatusResponse, StreamStopResponse, VideoProcessRequest,
    VideoProcessResponse, VideoStatusResponse, DetectionFilter,
    DetectionsQueryResponse, DetectionResponse, HealthResponse, StatsResponse
)

__all__ = [
    "app",
    "SuccessResponse",
    "ErrorResponse", 
    "StreamStartRequest",
    "StreamStartResponse",
    "StreamStatusResponse", 
    "StreamStopResponse",
    "VideoProcessRequest",
    "VideoProcessResponse",
    "VideoStatusResponse",
    "DetectionFilter",
    "DetectionsQueryResponse",
    "DetectionResponse",
    "HealthResponse",
    "StatsResponse"
]
