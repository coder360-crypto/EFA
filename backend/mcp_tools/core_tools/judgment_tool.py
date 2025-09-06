"""
Judgment Tool

Provides evaluation and feedback capabilities for assessing quality,
correctness, and appropriateness of responses and actions.
"""

from typing import Dict, List, Optional, Any, Union
import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class JudgmentCriteria(Enum):
    """Types of judgment criteria"""
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    EFFICIENCY = "efficiency"
    SAFETY = "safety"
    ETHICS = "ethics"


@dataclass
class JudgmentResult:
    """Result of a judgment evaluation"""
    criteria: JudgmentCriteria
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    reasoning: str
    evidence: List[str]
    timestamp: datetime


class JudgmentTool:
    """Tool for evaluation and judgment"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.judgment_history: List[Dict[str, Any]] = []
    
    async def evaluate_response(
        self,
        response: str,
        context: Dict[str, Any],
        criteria: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a response against multiple criteria
        
        Args:
            response: Response to evaluate
            context: Context information
            criteria: List of criteria to evaluate against
            
        Returns:
            Evaluation results
        """
        try:
            if criteria is None:
                criteria = ["accuracy", "relevance", "completeness", "clarity"]
            
            results = {}
            overall_score = 0.0
            
            for criterion in criteria:
                try:
                    criterion_enum = JudgmentCriteria(criterion.lower())
                    result = await self._evaluate_criterion(response, context, criterion_enum)
                    results[criterion] = {
                        "score": result.score,
                        "confidence": result.confidence,
                        "reasoning": result.reasoning,
                        "evidence": result.evidence
                    }
                    overall_score += result.score
                except ValueError:
                    self.logger.warning(f"Unknown judgment criterion: {criterion}")
                    continue
            
            overall_score /= len(results) if results else 1
            
            evaluation = {
                "response": response,
                "overall_score": overall_score,
                "criteria_results": results,
                "timestamp": datetime.now().isoformat(),
                "context": context
            }
            
            # Store in history
            self.judgment_history.append(evaluation)
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate response: {e}")
            raise
    
    async def judge_action(
        self,
        action: str,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Judge whether an action is appropriate and safe
        
        Args:
            action: Action to judge
            parameters: Action parameters
            context: Context information
            
        Returns:
            Judgment result
        """
        try:
            safety_score = await self._assess_safety(action, parameters, context)
            ethics_score = await self._assess_ethics(action, parameters, context)
            efficiency_score = await self._assess_efficiency(action, parameters, context)
            
            # Overall appropriateness score
            appropriateness = (safety_score + ethics_score + efficiency_score) / 3
            
            # Determine recommendation
            if appropriateness >= 0.8:
                recommendation = "approve"
            elif appropriateness >= 0.6:
                recommendation = "caution"
            else:
                recommendation = "reject"
            
            judgment = {
                "action": action,
                "parameters": parameters,
                "safety_score": safety_score,
                "ethics_score": ethics_score,
                "efficiency_score": efficiency_score,
                "overall_appropriateness": appropriateness,
                "recommendation": recommendation,
                "reasoning": await self._generate_action_reasoning(
                    action, parameters, safety_score, ethics_score, efficiency_score
                ),
                "timestamp": datetime.now().isoformat()
            }
            
            return judgment
            
        except Exception as e:
            self.logger.error(f"Failed to judge action: {e}")
            raise
    
    async def compare_options(
        self,
        options: List[Dict[str, Any]],
        criteria: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple options and rank them
        
        Args:
            options: List of options to compare
            criteria: Criteria to use for comparison
            
        Returns:
            Comparison results with rankings
        """
        try:
            if criteria is None:
                criteria = ["accuracy", "relevance", "efficiency"]
            
            option_scores = []
            
            for i, option in enumerate(options):
                option_id = option.get("id", f"option_{i}")
                content = option.get("content", "")
                context = option.get("context", {})
                
                # Evaluate option
                evaluation = await self.evaluate_response(content, context, criteria)
                
                option_scores.append({
                    "id": option_id,
                    "option": option,
                    "score": evaluation["overall_score"],
                    "evaluation": evaluation
                })
            
            # Sort by score (highest first)
            option_scores.sort(key=lambda x: x["score"], reverse=True)
            
            # Generate rankings
            for i, option_score in enumerate(option_scores):
                option_score["rank"] = i + 1
            
            comparison = {
                "options_count": len(options),
                "criteria_used": criteria,
                "rankings": option_scores,
                "best_option": option_scores[0] if option_scores else None,
                "timestamp": datetime.now().isoformat()
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Failed to compare options: {e}")
            raise
    
    async def _evaluate_criterion(
        self,
        response: str,
        context: Dict[str, Any],
        criteria: JudgmentCriteria
    ) -> JudgmentResult:
        """Evaluate response against a specific criterion"""
        
        if criteria == JudgmentCriteria.ACCURACY:
            return await self._assess_accuracy(response, context)
        elif criteria == JudgmentCriteria.RELEVANCE:
            return await self._assess_relevance(response, context)
        elif criteria == JudgmentCriteria.COMPLETENESS:
            return await self._assess_completeness(response, context)
        elif criteria == JudgmentCriteria.CLARITY:
            return await self._assess_clarity(response, context)
        elif criteria == JudgmentCriteria.EFFICIENCY:
            return await self._assess_response_efficiency(response, context)
        elif criteria == JudgmentCriteria.SAFETY:
            return await self._assess_response_safety(response, context)
        elif criteria == JudgmentCriteria.ETHICS:
            return await self._assess_response_ethics(response, context)
        else:
            # Default evaluation
            return JudgmentResult(
                criteria=criteria,
                score=0.5,
                confidence=0.1,
                reasoning="Unknown criteria",
                evidence=[],
                timestamp=datetime.now()
            )
    
    async def _assess_accuracy(self, response: str, context: Dict[str, Any]) -> JudgmentResult:
        """Assess accuracy of response"""
        # Simplified accuracy assessment
        score = 0.7  # Default moderate accuracy
        confidence = 0.6
        evidence = []
        reasoning = "Basic accuracy assessment"
        
        # Check for factual claims that can be verified
        if any(keyword in response.lower() for keyword in ["fact", "data", "study", "research"]):
            score += 0.1
            evidence.append("Contains reference to verifiable information")
        
        # Check for uncertainty indicators (good for accuracy)
        if any(phrase in response.lower() for phrase in ["might", "could", "possibly", "likely"]):
            score += 0.1
            evidence.append("Shows appropriate uncertainty")
        
        # Check for absolute claims (potentially less accurate)
        if any(word in response.lower() for word in ["always", "never", "all", "none"]):
            score -= 0.1
            evidence.append("Contains absolute statements")
        
        return JudgmentResult(
            criteria=JudgmentCriteria.ACCURACY,
            score=max(0.0, min(1.0, score)),
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            timestamp=datetime.now()
        )
    
    async def _assess_relevance(self, response: str, context: Dict[str, Any]) -> JudgmentResult:
        """Assess relevance of response to context"""
        query = context.get("query", "").lower()
        response_lower = response.lower()
        
        score = 0.5  # Default moderate relevance
        confidence = 0.7
        evidence = []
        
        if query:
            # Check for keyword overlap
            query_words = set(query.split())
            response_words = set(response_lower.split())
            overlap = len(query_words.intersection(response_words))
            
            if overlap > 0:
                score += min(0.4, overlap * 0.1)
                evidence.append(f"Found {overlap} matching keywords")
            
            # Check for semantic relevance (simplified)
            if any(word in response_lower for word in query_words):
                score += 0.1
                evidence.append("Contains query-related terms")
        
        reasoning = f"Relevance assessment based on query-response alignment"
        
        return JudgmentResult(
            criteria=JudgmentCriteria.RELEVANCE,
            score=max(0.0, min(1.0, score)),
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            timestamp=datetime.now()
        )
    
    async def _assess_completeness(self, response: str, context: Dict[str, Any]) -> JudgmentResult:
        """Assess completeness of response"""
        score = 0.6  # Default moderate completeness
        confidence = 0.5
        evidence = []
        
        # Length-based assessment (simple heuristic)
        word_count = len(response.split())
        if word_count < 10:
            score -= 0.2
            evidence.append("Response is quite short")
        elif word_count > 50:
            score += 0.2
            evidence.append("Response provides detailed information")
        
        # Structure indicators
        if any(indicator in response for indicator in ["first", "second", "finally", "in conclusion"]):
            score += 0.1
            evidence.append("Response has structured approach")
        
        # Question addressing
        query = context.get("query", "")
        if "?" in query:
            if any(phrase in response.lower() for phrase in ["because", "due to", "therefore", "as a result"]):
                score += 0.1
                evidence.append("Provides explanatory content")
        
        reasoning = "Completeness based on length, structure, and explanation depth"
        
        return JudgmentResult(
            criteria=JudgmentCriteria.COMPLETENESS,
            score=max(0.0, min(1.0, score)),
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            timestamp=datetime.now()
        )
    
    async def _assess_clarity(self, response: str, context: Dict[str, Any]) -> JudgmentResult:
        """Assess clarity of response"""
        score = 0.7  # Default good clarity
        confidence = 0.8
        evidence = []
        
        # Sentence length assessment
        sentences = response.split('. ')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        if avg_sentence_length > 25:
            score -= 0.1
            evidence.append("Sentences are quite long")
        elif avg_sentence_length < 5:
            score -= 0.1
            evidence.append("Sentences are very short")
        
        # Jargon check (simplified)
        complex_words = ["utilize", "implement", "facilitate", "optimize", "leverage"]
        jargon_count = sum(1 for word in complex_words if word in response.lower())
        if jargon_count > 2:
            score -= 0.1
            evidence.append("Contains technical jargon")
        
        # Clarity indicators
        if any(phrase in response.lower() for phrase in ["in other words", "simply put", "for example"]):
            score += 0.1
            evidence.append("Uses clarifying phrases")
        
        reasoning = "Clarity assessment based on sentence length and language complexity"
        
        return JudgmentResult(
            criteria=JudgmentCriteria.CLARITY,
            score=max(0.0, min(1.0, score)),
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            timestamp=datetime.now()
        )
    
    async def _assess_response_efficiency(self, response: str, context: Dict[str, Any]) -> JudgmentResult:
        """Assess efficiency of response"""
        score = 0.6
        confidence = 0.6
        evidence = []
        
        # Conciseness check
        word_count = len(response.split())
        query_length = len(context.get("query", "").split())
        
        if word_count <= query_length * 3:
            score += 0.2
            evidence.append("Response is concise")
        elif word_count > query_length * 10:
            score -= 0.2
            evidence.append("Response may be overly verbose")
        
        reasoning = "Efficiency based on response conciseness"
        
        return JudgmentResult(
            criteria=JudgmentCriteria.EFFICIENCY,
            score=max(0.0, min(1.0, score)),
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            timestamp=datetime.now()
        )
    
    async def _assess_response_safety(self, response: str, context: Dict[str, Any]) -> JudgmentResult:
        """Assess safety of response"""
        score = 0.9  # Default high safety
        confidence = 0.8
        evidence = []
        
        # Check for potentially harmful content
        harmful_keywords = ["violence", "harm", "illegal", "dangerous", "weapon"]
        for keyword in harmful_keywords:
            if keyword in response.lower():
                score -= 0.3
                evidence.append(f"Contains potentially harmful keyword: {keyword}")
        
        reasoning = "Safety assessment based on content analysis"
        
        return JudgmentResult(
            criteria=JudgmentCriteria.SAFETY,
            score=max(0.0, min(1.0, score)),
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            timestamp=datetime.now()
        )
    
    async def _assess_response_ethics(self, response: str, context: Dict[str, Any]) -> JudgmentResult:
        """Assess ethical implications of response"""
        score = 0.8  # Default good ethics
        confidence = 0.7
        evidence = []
        
        # Check for bias indicators
        bias_keywords = ["always", "never", "all people", "everyone"]
        for keyword in bias_keywords:
            if keyword in response.lower():
                score -= 0.1
                evidence.append(f"Potential bias indicator: {keyword}")
        
        # Check for inclusive language
        inclusive_phrases = ["some people", "many", "often", "generally"]
        for phrase in inclusive_phrases:
            if phrase in response.lower():
                score += 0.05
                evidence.append("Uses inclusive language")
        
        reasoning = "Ethics assessment based on bias and inclusivity analysis"
        
        return JudgmentResult(
            criteria=JudgmentCriteria.ETHICS,
            score=max(0.0, min(1.0, score)),
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            timestamp=datetime.now()
        )
    
    async def _assess_safety(self, action: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Assess safety of an action"""
        # Simplified safety assessment
        score = 0.8
        
        # Check for potentially dangerous actions
        dangerous_actions = ["delete", "remove", "destroy", "format", "shutdown"]
        if any(dangerous in action.lower() for dangerous in dangerous_actions):
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    async def _assess_ethics(self, action: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Assess ethical implications of an action"""
        # Simplified ethics assessment
        score = 0.8
        
        # Check for privacy concerns
        if "private" in str(parameters).lower() or "personal" in str(parameters).lower():
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    async def _assess_efficiency(self, action: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Assess efficiency of an action"""
        # Simplified efficiency assessment
        score = 0.7
        
        # Check for batch operations (more efficient)
        if "batch" in action.lower() or "bulk" in action.lower():
            score += 0.2
        
        return max(0.0, min(1.0, score))
    
    async def _generate_action_reasoning(
        self,
        action: str,
        parameters: Dict[str, Any],
        safety_score: float,
        ethics_score: float,
        efficiency_score: float
    ) -> str:
        """Generate reasoning for action judgment"""
        reasoning_parts = []
        
        if safety_score < 0.6:
            reasoning_parts.append("Safety concerns identified")
        elif safety_score > 0.8:
            reasoning_parts.append("Action appears safe")
        
        if ethics_score < 0.6:
            reasoning_parts.append("Ethical concerns present")
        elif ethics_score > 0.8:
            reasoning_parts.append("Action is ethically sound")
        
        if efficiency_score < 0.6:
            reasoning_parts.append("Efficiency could be improved")
        elif efficiency_score > 0.8:
            reasoning_parts.append("Action is efficient")
        
        if not reasoning_parts:
            reasoning_parts.append("Standard assessment completed")
        
        return "; ".join(reasoning_parts)
    
    async def get_judgment_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent judgment history"""
        return self.judgment_history[-limit:]
    
    async def clear_judgment_history(self) -> None:
        """Clear judgment history"""
        self.judgment_history.clear()
