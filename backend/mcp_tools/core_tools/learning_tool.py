"""
Learning Tool

Provides learning and adaptation capabilities including pattern recognition,
performance improvement, and knowledge acquisition.
"""

from typing import Dict, List, Optional, Any, Tuple
import asyncio
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics


@dataclass
class LearningEvent:
    """Represents a learning event"""
    id: str
    event_type: str  # 'success', 'failure', 'feedback', 'pattern'
    context: Dict[str, Any]
    outcome: Any
    feedback: Optional[str]
    timestamp: datetime
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class Pattern:
    """Represents a learned pattern"""
    id: str
    pattern_type: str
    description: str
    conditions: Dict[str, Any]
    outcomes: List[Any]
    confidence: float
    frequency: int
    last_seen: datetime
    examples: List[str]


class LearningTool:
    """Tool for learning and adaptation"""
    
    def __init__(self, max_events: int = 1000, pattern_threshold: int = 3):
        self.max_events = max_events
        self.pattern_threshold = pattern_threshold
        self.learning_events: deque[LearningEvent] = deque(maxlen=max_events)
        self.patterns: Dict[str, Pattern] = {}
        self.performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.knowledge_base: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
    
    async def record_event(
        self,
        event_type: str,
        context: Dict[str, Any],
        outcome: Any,
        feedback: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record a learning event
        
        Args:
            event_type: Type of event ('success', 'failure', 'feedback', 'pattern')
            context: Context in which the event occurred
            outcome: Outcome of the event
            feedback: Optional feedback about the event
            confidence: Confidence in the event data (0.0 to 1.0)
            metadata: Additional metadata
            
        Returns:
            Event ID
        """
        try:
            event_id = f"event_{len(self.learning_events)}_{datetime.now().timestamp()}"
            
            event = LearningEvent(
                id=event_id,
                event_type=event_type,
                context=context,
                outcome=outcome,
                feedback=feedback,
                timestamp=datetime.now(),
                confidence=max(0.0, min(1.0, confidence)),
                metadata=metadata or {}
            )
            
            self.learning_events.append(event)
            
            # Update performance metrics
            await self._update_performance_metrics(event)
            
            # Look for patterns
            await self._detect_patterns()
            
            # Update knowledge base
            await self._update_knowledge_base(event)
            
            self.logger.debug(f"Recorded learning event: {event_id}")
            return event_id
            
        except Exception as e:
            self.logger.error(f"Failed to record learning event: {e}")
            raise
    
    async def get_recommendations(
        self,
        context: Dict[str, Any],
        task_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations based on learned patterns
        
        Args:
            context: Current context
            task_type: Optional task type to filter recommendations
            
        Returns:
            List of recommendations
        """
        try:
            recommendations = []
            
            # Find relevant patterns
            relevant_patterns = await self._find_relevant_patterns(context, task_type)
            
            for pattern in relevant_patterns:
                # Generate recommendation based on pattern
                recommendation = {
                    "type": "pattern_based",
                    "pattern_id": pattern.id,
                    "description": pattern.description,
                    "suggested_actions": await self._generate_actions_from_pattern(pattern),
                    "confidence": pattern.confidence,
                    "frequency": pattern.frequency,
                    "last_seen": pattern.last_seen.isoformat()
                }
                recommendations.append(recommendation)
            
            # Add performance-based recommendations
            performance_recs = await self._get_performance_recommendations(context)
            recommendations.extend(performance_recs)
            
            # Sort by confidence and relevance
            recommendations.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            
            return recommendations[:10]  # Return top 10 recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to get recommendations: {e}")
            return []
    
    async def analyze_performance(
        self,
        metric_name: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Analyze performance trends
        
        Args:
            metric_name: Specific metric to analyze (analyzes all if None)
            time_window: Time window for analysis (default: last 30 days)
            
        Returns:
            Performance analysis results
        """
        try:
            if time_window is None:
                time_window = timedelta(days=30)
            
            cutoff_time = datetime.now() - time_window
            
            analysis = {
                "time_window": str(time_window),
                "cutoff_time": cutoff_time.isoformat(),
                "metrics": {}
            }
            
            # Analyze specific metric or all metrics
            metrics_to_analyze = [metric_name] if metric_name else list(self.performance_metrics.keys())
            
            for metric in metrics_to_analyze:
                if metric in self.performance_metrics:
                    metric_analysis = await self._analyze_metric(metric, cutoff_time)
                    analysis["metrics"][metric] = metric_analysis
            
            # Overall performance summary
            analysis["summary"] = await self._generate_performance_summary(analysis["metrics"])
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze performance: {e}")
            return {}
    
    async def update_knowledge(
        self,
        topic: str,
        information: Any,
        source: str = "user_input",
        confidence: float = 1.0
    ) -> bool:
        """
        Update knowledge base with new information
        
        Args:
            topic: Topic or key for the knowledge
            information: Information to store
            source: Source of the information
            confidence: Confidence in the information
            
        Returns:
            True if updated successfully
        """
        try:
            if topic not in self.knowledge_base:
                self.knowledge_base[topic] = {
                    "information": [],
                    "sources": [],
                    "confidences": [],
                    "timestamps": [],
                    "summary": None
                }
            
            knowledge_entry = self.knowledge_base[topic]
            knowledge_entry["information"].append(information)
            knowledge_entry["sources"].append(source)
            knowledge_entry["confidences"].append(confidence)
            knowledge_entry["timestamps"].append(datetime.now().isoformat())
            
            # Update summary
            await self._update_knowledge_summary(topic)
            
            self.logger.debug(f"Updated knowledge for topic: {topic}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update knowledge: {e}")
            return False
    
    async def query_knowledge(
        self,
        topic: str,
        min_confidence: float = 0.0
    ) -> Optional[Dict[str, Any]]:
        """
        Query knowledge base
        
        Args:
            topic: Topic to query
            min_confidence: Minimum confidence threshold
            
        Returns:
            Knowledge entry or None if not found
        """
        try:
            if topic not in self.knowledge_base:
                return None
            
            knowledge_entry = self.knowledge_base[topic].copy()
            
            # Filter by confidence if specified
            if min_confidence > 0.0:
                filtered_info = []
                filtered_sources = []
                filtered_confidences = []
                filtered_timestamps = []
                
                for i, conf in enumerate(knowledge_entry["confidences"]):
                    if conf >= min_confidence:
                        filtered_info.append(knowledge_entry["information"][i])
                        filtered_sources.append(knowledge_entry["sources"][i])
                        filtered_confidences.append(conf)
                        filtered_timestamps.append(knowledge_entry["timestamps"][i])
                
                knowledge_entry["information"] = filtered_info
                knowledge_entry["sources"] = filtered_sources
                knowledge_entry["confidences"] = filtered_confidences
                knowledge_entry["timestamps"] = filtered_timestamps
            
            return knowledge_entry
            
        except Exception as e:
            self.logger.error(f"Failed to query knowledge: {e}")
            return None
    
    async def get_learning_insights(self) -> Dict[str, Any]:
        """
        Get insights from learning history
        
        Returns:
            Learning insights and statistics
        """
        try:
            total_events = len(self.learning_events)
            if total_events == 0:
                return {"message": "No learning events recorded yet"}
            
            # Event type distribution
            event_types = defaultdict(int)
            success_rate = 0
            recent_events = []
            
            for event in self.learning_events:
                event_types[event.event_type] += 1
                if event.event_type == "success":
                    success_rate += 1
                
                # Collect recent events (last 10)
                if len(recent_events) < 10:
                    recent_events.append({
                        "id": event.id,
                        "type": event.event_type,
                        "timestamp": event.timestamp.isoformat(),
                        "confidence": event.confidence
                    })
            
            success_rate = (success_rate / total_events) * 100 if total_events > 0 else 0
            
            insights = {
                "total_events": total_events,
                "event_type_distribution": dict(event_types),
                "success_rate": success_rate,
                "total_patterns": len(self.patterns),
                "knowledge_topics": len(self.knowledge_base),
                "recent_events": recent_events,
                "top_patterns": await self._get_top_patterns(5),
                "performance_trends": await self._get_performance_trends()
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to get learning insights: {e}")
            return {}
    
    async def _update_performance_metrics(self, event: LearningEvent) -> None:
        """Update performance metrics based on event"""
        # Record success rate
        if event.event_type in ["success", "failure"]:
            self.performance_metrics["success_rate"].append(
                1.0 if event.event_type == "success" else 0.0
            )
        
        # Record confidence levels
        self.performance_metrics["confidence"].append(event.confidence)
        
        # Record task-specific metrics
        task_type = event.context.get("task_type")
        if task_type:
            metric_key = f"task_{task_type}_success"
            success = 1.0 if event.event_type == "success" else 0.0
            self.performance_metrics[metric_key].append(success)
    
    async def _detect_patterns(self) -> None:
        """Detect patterns in learning events"""
        if len(self.learning_events) < self.pattern_threshold:
            return
        
        # Group events by context similarity
        context_groups = defaultdict(list)
        
        for event in list(self.learning_events)[-50:]:  # Look at recent events
            # Create a simple context signature
            context_sig = self._create_context_signature(event.context)
            context_groups[context_sig].append(event)
        
        # Look for patterns in groups with sufficient events
        for context_sig, events in context_groups.items():
            if len(events) >= self.pattern_threshold:
                await self._create_or_update_pattern(context_sig, events)
    
    async def _create_or_update_pattern(self, context_sig: str, events: List[LearningEvent]) -> None:
        """Create or update a pattern based on events"""
        pattern_id = f"pattern_{hash(context_sig) % 10000}"
        
        # Analyze outcomes
        outcomes = [event.outcome for event in events]
        outcome_frequency = defaultdict(int)
        for outcome in outcomes:
            outcome_frequency[str(outcome)] += 1
        
        # Calculate confidence based on consistency
        most_common_outcome = max(outcome_frequency.items(), key=lambda x: x[1])
        confidence = most_common_outcome[1] / len(events)
        
        # Extract common conditions
        conditions = self._extract_common_conditions([event.context for event in events])
        
        if pattern_id in self.patterns:
            # Update existing pattern
            pattern = self.patterns[pattern_id]
            pattern.frequency += len(events)
            pattern.confidence = (pattern.confidence + confidence) / 2
            pattern.last_seen = datetime.now()
            pattern.examples.extend([event.id for event in events[-3:]])  # Keep recent examples
            pattern.examples = pattern.examples[-10:]  # Limit examples
        else:
            # Create new pattern
            self.patterns[pattern_id] = Pattern(
                id=pattern_id,
                pattern_type="context_outcome",
                description=f"Pattern for context: {context_sig}",
                conditions=conditions,
                outcomes=list(outcome_frequency.keys()),
                confidence=confidence,
                frequency=len(events),
                last_seen=datetime.now(),
                examples=[event.id for event in events[-3:]]
            )
    
    def _create_context_signature(self, context: Dict[str, Any]) -> str:
        """Create a signature for context to group similar contexts"""
        # Simple signature based on key-value pairs
        sorted_items = sorted(context.items())
        signature_parts = []
        
        for key, value in sorted_items:
            if isinstance(value, (str, int, float, bool)):
                signature_parts.append(f"{key}:{value}")
            elif isinstance(value, (list, tuple)):
                signature_parts.append(f"{key}:list_{len(value)}")
            elif isinstance(value, dict):
                signature_parts.append(f"{key}:dict_{len(value)}")
        
        return "|".join(signature_parts)
    
    def _extract_common_conditions(self, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract common conditions from multiple contexts"""
        if not contexts:
            return {}
        
        common_conditions = {}
        
        # Find keys that appear in all contexts
        all_keys = set(contexts[0].keys())
        for context in contexts[1:]:
            all_keys = all_keys.intersection(set(context.keys()))
        
        # For common keys, find common values
        for key in all_keys:
            values = [context[key] for context in contexts]
            unique_values = set(str(v) for v in values)
            
            if len(unique_values) == 1:
                # All values are the same
                common_conditions[key] = values[0]
            elif len(unique_values) <= len(values) * 0.8:
                # Most values are similar
                most_common = max(unique_values, key=lambda x: sum(1 for v in values if str(v) == x))
                common_conditions[f"{key}_common"] = most_common
        
        return common_conditions
    
    async def _find_relevant_patterns(
        self,
        context: Dict[str, Any],
        task_type: Optional[str] = None
    ) -> List[Pattern]:
        """Find patterns relevant to current context"""
        relevant_patterns = []
        context_sig = self._create_context_signature(context)
        
        for pattern in self.patterns.values():
            # Check context similarity
            similarity = self._calculate_context_similarity(context, pattern.conditions)
            
            if similarity > 0.5:  # Threshold for relevance
                relevant_patterns.append(pattern)
        
        # Sort by confidence and recency
        relevant_patterns.sort(
            key=lambda p: (p.confidence, p.last_seen.timestamp()),
            reverse=True
        )
        
        return relevant_patterns
    
    def _calculate_context_similarity(
        self,
        context1: Dict[str, Any],
        context2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two contexts"""
        if not context1 or not context2:
            return 0.0
        
        all_keys = set(context1.keys()).union(set(context2.keys()))
        if not all_keys:
            return 0.0
        
        matching_keys = 0
        for key in all_keys:
            if key in context1 and key in context2:
                if str(context1[key]) == str(context2[key]):
                    matching_keys += 1
        
        return matching_keys / len(all_keys)
    
    async def _generate_actions_from_pattern(self, pattern: Pattern) -> List[str]:
        """Generate suggested actions based on a pattern"""
        actions = []
        
        # Generic actions based on pattern type
        if pattern.confidence > 0.8:
            actions.append("Apply this pattern with high confidence")
        elif pattern.confidence > 0.6:
            actions.append("Consider applying this pattern with caution")
        
        # Actions based on outcomes
        for outcome in pattern.outcomes:
            if "success" in str(outcome).lower():
                actions.append(f"Expect positive outcome: {outcome}")
            elif "failure" in str(outcome).lower():
                actions.append(f"Be prepared for potential failure: {outcome}")
        
        return actions
    
    async def _get_performance_recommendations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get recommendations based on performance metrics"""
        recommendations = []
        
        # Analyze recent success rate
        if "success_rate" in self.performance_metrics:
            recent_success = list(self.performance_metrics["success_rate"])[-10:]
            if recent_success:
                avg_success = statistics.mean(recent_success)
                if avg_success < 0.5:
                    recommendations.append({
                        "type": "performance_improvement",
                        "description": "Recent success rate is low - consider reviewing approach",
                        "confidence": 0.8,
                        "metric": "success_rate",
                        "value": avg_success
                    })
        
        return recommendations
    
    async def _analyze_metric(self, metric_name: str, cutoff_time: datetime) -> Dict[str, Any]:
        """Analyze a specific performance metric"""
        values = list(self.performance_metrics[metric_name])
        
        if not values:
            return {"error": "No data available"}
        
        analysis = {
            "total_points": len(values),
            "average": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values)
        }
        
        if len(values) > 1:
            analysis["standard_deviation"] = statistics.stdev(values)
        
        # Trend analysis (simple)
        if len(values) >= 5:
            recent = values[-5:]
            older = values[:-5] if len(values) > 5 else values
            
            recent_avg = statistics.mean(recent)
            older_avg = statistics.mean(older)
            
            if recent_avg > older_avg * 1.1:
                analysis["trend"] = "improving"
            elif recent_avg < older_avg * 0.9:
                analysis["trend"] = "declining"
            else:
                analysis["trend"] = "stable"
        
        return analysis
    
    async def _generate_performance_summary(self, metrics_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall performance summary"""
        summary = {
            "total_metrics": len(metrics_analysis),
            "improving_metrics": 0,
            "declining_metrics": 0,
            "stable_metrics": 0
        }
        
        for metric_name, analysis in metrics_analysis.items():
            trend = analysis.get("trend", "unknown")
            if trend == "improving":
                summary["improving_metrics"] += 1
            elif trend == "declining":
                summary["declining_metrics"] += 1
            elif trend == "stable":
                summary["stable_metrics"] += 1
        
        # Overall assessment
        if summary["improving_metrics"] > summary["declining_metrics"]:
            summary["overall_trend"] = "positive"
        elif summary["declining_metrics"] > summary["improving_metrics"]:
            summary["overall_trend"] = "negative"
        else:
            summary["overall_trend"] = "neutral"
        
        return summary
    
    async def _update_knowledge_summary(self, topic: str) -> None:
        """Update summary for a knowledge topic"""
        if topic not in self.knowledge_base:
            return
        
        knowledge_entry = self.knowledge_base[topic]
        
        # Simple summary generation
        info_count = len(knowledge_entry["information"])
        avg_confidence = statistics.mean(knowledge_entry["confidences"]) if knowledge_entry["confidences"] else 0
        
        knowledge_entry["summary"] = {
            "total_entries": info_count,
            "average_confidence": avg_confidence,
            "last_updated": datetime.now().isoformat(),
            "sources": list(set(knowledge_entry["sources"]))
        }
    
    async def _get_top_patterns(self, limit: int) -> List[Dict[str, Any]]:
        """Get top patterns by confidence and frequency"""
        patterns = list(self.patterns.values())
        patterns.sort(key=lambda p: (p.confidence, p.frequency), reverse=True)
        
        return [
            {
                "id": p.id,
                "description": p.description,
                "confidence": p.confidence,
                "frequency": p.frequency,
                "last_seen": p.last_seen.isoformat()
            }
            for p in patterns[:limit]
        ]
    
    async def _get_performance_trends(self) -> Dict[str, str]:
        """Get performance trends for all metrics"""
        trends = {}
        
        for metric_name, values in self.performance_metrics.items():
            if len(list(values)) >= 5:
                recent = list(values)[-5:]
                older = list(values)[:-5] if len(list(values)) > 5 else list(values)
                
                recent_avg = statistics.mean(recent)
                older_avg = statistics.mean(older)
                
                if recent_avg > older_avg * 1.1:
                    trends[metric_name] = "improving"
                elif recent_avg < older_avg * 0.9:
                    trends[metric_name] = "declining"
                else:
                    trends[metric_name] = "stable"
            else:
                trends[metric_name] = "insufficient_data"
        
        return trends
