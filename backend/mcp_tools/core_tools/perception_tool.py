"""
Perception Tool

Provides perception and input processing capabilities including
vision, audio, and sensor data processing.
"""

from typing import Dict, List, Optional, Any, Union
import asyncio
import base64
import json
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import tempfile
import os


class InputType(Enum):
    """Types of input data"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    SENSOR = "sensor"
    DOCUMENT = "document"


@dataclass
class PerceptionResult:
    """Result of perception processing"""
    input_type: InputType
    analysis: Dict[str, Any]
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]
    timestamp: datetime


class PerceptionTool:
    """Tool for perception and input processing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.perception_history: List[PerceptionResult] = []
        self.supported_formats = {
            InputType.IMAGE: ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'],
            InputType.AUDIO: ['.mp3', '.wav', '.flac', '.ogg', '.m4a'],
            InputType.VIDEO: ['.mp4', '.avi', '.mov', '.wmv', '.flv'],
            InputType.DOCUMENT: ['.pdf', '.doc', '.docx', '.txt', '.rtf']
        }
    
    async def process_input(
        self,
        input_data: Any,
        input_type: str,
        analysis_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process input data and extract information
        
        Args:
            input_data: Input data to process
            input_type: Type of input ("text", "image", "audio", "video", "sensor", "document")
            analysis_options: Optional analysis configuration
            
        Returns:
            Perception analysis results
        """
        try:
            start_time = datetime.now()
            
            # Convert string to enum
            try:
                input_type_enum = InputType(input_type.lower())
            except ValueError:
                raise ValueError(f"Unsupported input type: {input_type}")
            
            # Process based on input type
            if input_type_enum == InputType.TEXT:
                analysis = await self._process_text(input_data, analysis_options or {})
            elif input_type_enum == InputType.IMAGE:
                analysis = await self._process_image(input_data, analysis_options or {})
            elif input_type_enum == InputType.AUDIO:
                analysis = await self._process_audio(input_data, analysis_options or {})
            elif input_type_enum == InputType.VIDEO:
                analysis = await self._process_video(input_data, analysis_options or {})
            elif input_type_enum == InputType.SENSOR:
                analysis = await self._process_sensor_data(input_data, analysis_options or {})
            elif input_type_enum == InputType.DOCUMENT:
                analysis = await self._process_document(input_data, analysis_options or {})
            else:
                raise ValueError(f"Processing not implemented for type: {input_type}")
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = PerceptionResult(
                input_type=input_type_enum,
                analysis=analysis,
                confidence=analysis.get('confidence', 0.8),
                processing_time=processing_time,
                metadata={
                    'analysis_options': analysis_options or {},
                    'input_size': len(str(input_data)) if isinstance(input_data, str) else 'unknown'
                },
                timestamp=datetime.now()
            )
            
            # Store in history
            self.perception_history.append(result)
            
            # Return formatted result
            return {
                "input_type": input_type,
                "analysis": analysis,
                "confidence": result.confidence,
                "processing_time_seconds": processing_time,
                "timestamp": result.timestamp.isoformat(),
                "metadata": result.metadata
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process input: {e}")
            raise
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment analysis results
        """
        try:
            # Simple sentiment analysis (in practice, you'd use a proper NLP library)
            positive_words = [
                'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied'
            ]
            
            negative_words = [
                'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike',
                'angry', 'frustrated', 'disappointed', 'sad', 'upset'
            ]
            
            text_lower = text.lower()
            words = text_lower.split()
            
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            # Calculate sentiment score
            if positive_count + negative_count == 0:
                sentiment_score = 0.0  # Neutral
                sentiment_label = "neutral"
            else:
                sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
                if sentiment_score > 0.2:
                    sentiment_label = "positive"
                elif sentiment_score < -0.2:
                    sentiment_label = "negative"
                else:
                    sentiment_label = "neutral"
            
            confidence = min(0.9, (positive_count + negative_count) / len(words) * 2)
            
            return {
                "sentiment_label": sentiment_label,
                "sentiment_score": sentiment_score,
                "confidence": confidence,
                "positive_words_found": positive_count,
                "negative_words_found": negative_count,
                "total_words": len(words)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze sentiment: {e}")
            raise
    
    async def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract entities from text
        
        Args:
            text: Text to process
            
        Returns:
            Entity extraction results
        """
        try:
            # Simple entity extraction (in practice, you'd use NER models)
            import re
            
            entities = {
                "emails": [],
                "urls": [],
                "phone_numbers": [],
                "dates": [],
                "numbers": [],
                "currencies": []
            }
            
            # Email extraction
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            entities["emails"] = re.findall(email_pattern, text)
            
            # URL extraction
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            entities["urls"] = re.findall(url_pattern, text)
            
            # Phone number extraction (simple pattern)
            phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
            entities["phone_numbers"] = re.findall(phone_pattern, text)
            
            # Date extraction (simple patterns)
            date_patterns = [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
                r'\b\d{4}-\d{2}-\d{2}\b',      # YYYY-MM-DD
                r'\b\d{1,2}-\d{1,2}-\d{4}\b'   # MM-DD-YYYY
            ]
            for pattern in date_patterns:
                entities["dates"].extend(re.findall(pattern, text))
            
            # Number extraction
            number_pattern = r'\b\d+\.?\d*\b'
            entities["numbers"] = re.findall(number_pattern, text)
            
            # Currency extraction
            currency_pattern = r'\$\d+\.?\d*|\d+\.?\d*\s*(?:USD|EUR|GBP|JPY|dollars?|euros?)'
            entities["currencies"] = re.findall(currency_pattern, text, re.IGNORECASE)
            
            # Count total entities
            total_entities = sum(len(entity_list) for entity_list in entities.values())
            
            return {
                "entities": entities,
                "total_entities": total_entities,
                "entity_types_found": [k for k, v in entities.items() if v],
                "confidence": 0.7  # Simple extraction has moderate confidence
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract entities: {e}")
            raise
    
    async def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect language of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Language detection results
        """
        try:
            # Simple language detection based on common words
            # In practice, you'd use a proper language detection library
            
            language_indicators = {
                'english': ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with'],
                'spanish': ['el', 'la', 'de', 'que', 'y', 'es', 'en', 'un', 'te', 'lo'],
                'french': ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir'],
                'german': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich'],
                'italian': ['il', 'di', 'che', 'è', 'e', 'la', 'per', 'una', 'in', 'del']
            }
            
            text_lower = text.lower()
            words = text_lower.split()
            
            language_scores = {}
            
            for language, indicators in language_indicators.items():
                score = sum(1 for word in words if word in indicators)
                language_scores[language] = score / len(words) if words else 0
            
            # Find most likely language
            if language_scores:
                detected_language = max(language_scores.items(), key=lambda x: x[1])
                confidence = detected_language[1]
                
                return {
                    "detected_language": detected_language[0],
                    "confidence": min(confidence * 5, 1.0),  # Scale up confidence
                    "language_scores": language_scores,
                    "text_length": len(text),
                    "word_count": len(words)
                }
            else:
                return {
                    "detected_language": "unknown",
                    "confidence": 0.0,
                    "language_scores": {},
                    "text_length": len(text),
                    "word_count": 0
                }
                
        except Exception as e:
            self.logger.error(f"Failed to detect language: {e}")
            raise
    
    async def _process_text(self, text: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process text input"""
        analysis = {
            "input_type": "text",
            "text_length": len(text),
            "word_count": len(text.split()),
            "character_count": len(text)
        }
        
        # Optional analyses based on options
        if options.get("sentiment_analysis", True):
            analysis["sentiment"] = await self.analyze_sentiment(text)
        
        if options.get("entity_extraction", True):
            analysis["entities"] = await self.extract_entities(text)
        
        if options.get("language_detection", True):
            analysis["language"] = await self.detect_language(text)
        
        # Text statistics
        sentences = text.split('.')
        analysis["sentence_count"] = len([s for s in sentences if s.strip()])
        analysis["average_word_length"] = sum(len(word) for word in text.split()) / len(text.split()) if text.split() else 0
        
        analysis["confidence"] = 0.8
        return analysis
    
    async def _process_image(self, image_data: Any, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process image input"""
        # Placeholder implementation
        # In practice, you'd use computer vision libraries like OpenCV, PIL, or ML models
        
        analysis = {
            "input_type": "image",
            "format": "unknown",
            "analysis_type": "basic_metadata"
        }
        
        # If image_data is base64, decode and analyze
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            try:
                # Extract format from data URL
                format_info = image_data.split(';')[0].split('/')[1]
                analysis["format"] = format_info
                
                # Decode base64 data
                base64_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(base64_data)
                analysis["size_bytes"] = len(image_bytes)
                
                # Placeholder analysis
                analysis["estimated_objects"] = ["unknown objects detected"]
                analysis["colors"] = ["various colors detected"]
                analysis["brightness"] = "medium"
                analysis["confidence"] = 0.6
                
            except Exception as e:
                analysis["error"] = str(e)
                analysis["confidence"] = 0.1
        else:
            analysis["error"] = "Unsupported image format or data"
            analysis["confidence"] = 0.1
        
        return analysis
    
    async def _process_audio(self, audio_data: Any, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process audio input"""
        # Placeholder implementation
        # In practice, you'd use audio processing libraries and speech recognition
        
        analysis = {
            "input_type": "audio",
            "format": "unknown",
            "analysis_type": "basic_metadata"
        }
        
        if isinstance(audio_data, bytes):
            analysis["size_bytes"] = len(audio_data)
            analysis["estimated_duration"] = "unknown"
            analysis["estimated_speech"] = "speech detected" if len(audio_data) > 1000 else "no speech detected"
            analysis["confidence"] = 0.5
        elif isinstance(audio_data, str):
            analysis["file_path"] = audio_data
            analysis["confidence"] = 0.3
        else:
            analysis["error"] = "Unsupported audio format"
            analysis["confidence"] = 0.1
        
        return analysis
    
    async def _process_video(self, video_data: Any, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process video input"""
        # Placeholder implementation
        # In practice, you'd use video processing libraries like OpenCV
        
        analysis = {
            "input_type": "video",
            "format": "unknown",
            "analysis_type": "basic_metadata"
        }
        
        if isinstance(video_data, bytes):
            analysis["size_bytes"] = len(video_data)
            analysis["estimated_duration"] = "unknown"
            analysis["estimated_frames"] = "unknown"
            analysis["confidence"] = 0.4
        elif isinstance(video_data, str):
            analysis["file_path"] = video_data
            analysis["confidence"] = 0.3
        else:
            analysis["error"] = "Unsupported video format"
            analysis["confidence"] = 0.1
        
        return analysis
    
    async def _process_sensor_data(self, sensor_data: Any, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process sensor data input"""
        analysis = {
            "input_type": "sensor",
            "data_type": type(sensor_data).__name__
        }
        
        if isinstance(sensor_data, dict):
            analysis["sensor_types"] = list(sensor_data.keys())
            analysis["data_points"] = len(sensor_data)
            
            # Analyze numeric data
            numeric_data = {}
            for key, value in sensor_data.items():
                if isinstance(value, (int, float)):
                    numeric_data[key] = value
                elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                    numeric_data[f"{key}_avg"] = sum(value) / len(value) if value else 0
                    numeric_data[f"{key}_min"] = min(value) if value else 0
                    numeric_data[f"{key}_max"] = max(value) if value else 0
            
            analysis["numeric_analysis"] = numeric_data
            analysis["confidence"] = 0.8
            
        elif isinstance(sensor_data, list):
            analysis["data_points"] = len(sensor_data)
            if sensor_data and all(isinstance(x, (int, float)) for x in sensor_data):
                analysis["statistics"] = {
                    "min": min(sensor_data),
                    "max": max(sensor_data),
                    "average": sum(sensor_data) / len(sensor_data),
                    "range": max(sensor_data) - min(sensor_data)
                }
                analysis["confidence"] = 0.9
            else:
                analysis["confidence"] = 0.6
        else:
            analysis["error"] = "Unsupported sensor data format"
            analysis["confidence"] = 0.1
        
        return analysis
    
    async def _process_document(self, document_data: Any, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process document input"""
        # Placeholder implementation
        # In practice, you'd use document processing libraries
        
        analysis = {
            "input_type": "document",
            "format": "unknown"
        }
        
        if isinstance(document_data, str):
            # Treat as text document
            analysis["format"] = "text"
            analysis["text_analysis"] = await self._process_text(document_data, options)
            analysis["confidence"] = 0.8
        elif isinstance(document_data, bytes):
            analysis["size_bytes"] = len(document_data)
            analysis["format"] = "binary"
            analysis["confidence"] = 0.4
        else:
            analysis["error"] = "Unsupported document format"
            analysis["confidence"] = 0.1
        
        return analysis
    
    async def get_perception_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent perception history"""
        try:
            recent_history = self.perception_history[-limit:]
            
            return [
                {
                    "input_type": result.input_type.value,
                    "confidence": result.confidence,
                    "processing_time": result.processing_time,
                    "timestamp": result.timestamp.isoformat(),
                    "analysis_summary": self._summarize_analysis(result.analysis)
                }
                for result in recent_history
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to get perception history: {e}")
            return []
    
    def _summarize_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of analysis results"""
        summary = {
            "input_type": analysis.get("input_type", "unknown")
        }
        
        # Add type-specific summary information
        if "sentiment" in analysis:
            summary["sentiment"] = analysis["sentiment"]["sentiment_label"]
        
        if "entities" in analysis:
            summary["entities_found"] = analysis["entities"]["total_entities"]
        
        if "language" in analysis:
            summary["language"] = analysis["language"]["detected_language"]
        
        if "confidence" in analysis:
            summary["confidence"] = analysis["confidence"]
        
        return summary
    
    async def clear_perception_history(self) -> None:
        """Clear perception history"""
        self.perception_history.clear()
        self.logger.info("Cleared perception history")
