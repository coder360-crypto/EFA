"""
Inference Engine

Orchestrates LLM calls and reasoning processes.
Handles complex multi-step reasoning, tool calling, and decision making.
"""

from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import logging
from datetime import datetime

from .llm_interface import LLMInterface, LLMMessage, LLMResponse
from .context_manager import ContextManager


class ReasoningStep(Enum):
    """Types of reasoning steps"""
    ANALYZE = "analyze"
    PLAN = "plan"
    EXECUTE = "execute"
    EVALUATE = "evaluate"
    REFLECT = "reflect"


@dataclass
class ReasoningTrace:
    """Trace of reasoning steps"""
    step_type: ReasoningStep
    input_data: Any
    output_data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceRequest:
    """Request for LLM inference"""
    prompt: str
    session_id: Optional[str] = None
    use_memory: bool = True
    use_reasoning: bool = False
    max_reasoning_steps: int = 5
    tools: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Result from LLM inference"""
    response: str
    reasoning_trace: List[ReasoningTrace] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class InferenceEngine:
    """Orchestrates LLM calls and reasoning"""
    
    def __init__(
        self,
        llm_adapter: LLMInterface,
        context_manager: ContextManager,
        available_tools: Optional[Dict[str, Callable]] = None
    ):
        self.llm_adapter = llm_adapter
        self.context_manager = context_manager
        self.available_tools = available_tools or {}
        self.logger = logging.getLogger(__name__)
    
    async def infer(self, request: InferenceRequest) -> InferenceResult:
        """
        Perform LLM inference with optional reasoning and tool calling
        
        Args:
            request: Inference request
            
        Returns:
            Inference result with response and trace
        """
        session_id = request.session_id or self.context_manager.create_context()
        
        # Build messages from context and request
        messages = self._build_messages(request, session_id)
        
        if request.use_reasoning:
            return await self._reasoning_inference(request, messages, session_id)
        else:
            return await self._simple_inference(request, messages, session_id)
    
    async def _simple_inference(
        self,
        request: InferenceRequest,
        messages: List[LLMMessage],
        session_id: str
    ) -> InferenceResult:
        """
        Perform simple LLM inference without reasoning
        
        Args:
            request: Inference request
            messages: Prepared messages
            session_id: Session ID
            
        Returns:
            Inference result
        """
        try:
            # Generate response
            response = await self.llm_adapter.generate(messages)
            
            # Add to context
            self.context_manager.add_message(
                session_id,
                LLMMessage(role="user", content=request.prompt)
            )
            self.context_manager.add_message(
                session_id,
                LLMMessage(role="assistant", content=response.content)
            )
            
            return InferenceResult(
                response=response.content,
                confidence=self._calculate_confidence(response),
                metadata={
                    'session_id': session_id,
                    'model_info': self.llm_adapter.get_model_info(),
                    'usage': response.usage
                }
            )
            
        except Exception as e:
            self.logger.error(f"Simple inference failed: {e}")
            return InferenceResult(
                response=f"Error: {str(e)}",
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    async def _reasoning_inference(
        self,
        request: InferenceRequest,
        messages: List[LLMMessage],
        session_id: str
    ) -> InferenceResult:
        """
        Perform inference with multi-step reasoning
        
        Args:
            request: Inference request
            messages: Prepared messages
            session_id: Session ID
            
        Returns:
            Inference result with reasoning trace
        """
        reasoning_trace = []
        current_state = {
            'problem': request.prompt,
            'solution': None,
            'confidence': 0.0
        }
        
        try:
            # Step 1: Analyze the problem
            analysis = await self._reasoning_step(
                ReasoningStep.ANALYZE,
                messages,
                "Analyze the given problem and identify key components, requirements, and constraints."
            )
            reasoning_trace.append(ReasoningTrace(
                step_type=ReasoningStep.ANALYZE,
                input_data=request.prompt,
                output_data=analysis.content
            ))
            current_state['analysis'] = analysis.content
            
            # Step 2: Plan the approach
            plan_messages = messages + [
                LLMMessage(role="assistant", content=analysis.content),
                LLMMessage(role="user", content="Based on this analysis, create a step-by-step plan to solve the problem.")
            ]
            plan = await self.llm_adapter.generate(plan_messages)
            reasoning_trace.append(ReasoningTrace(
                step_type=ReasoningStep.PLAN,
                input_data=current_state['analysis'],
                output_data=plan.content
            ))
            current_state['plan'] = plan.content
            
            # Step 3: Execute the plan
            execute_messages = plan_messages + [
                LLMMessage(role="assistant", content=plan.content),
                LLMMessage(role="user", content="Now execute this plan and provide the solution.")
            ]
            solution = await self.llm_adapter.generate(execute_messages)
            reasoning_trace.append(ReasoningTrace(
                step_type=ReasoningStep.EXECUTE,
                input_data=current_state['plan'],
                output_data=solution.content
            ))
            current_state['solution'] = solution.content
            
            # Step 4: Evaluate the solution
            evaluate_messages = execute_messages + [
                LLMMessage(role="assistant", content=solution.content),
                LLMMessage(role="user", content="Evaluate this solution for correctness and completeness. Rate confidence from 0-1.")
            ]
            evaluation = await self.llm_adapter.generate(evaluate_messages)
            reasoning_trace.append(ReasoningTrace(
                step_type=ReasoningStep.EVALUATE,
                input_data=current_state['solution'],
                output_data=evaluation.content
            ))
            
            # Extract confidence from evaluation
            confidence = self._extract_confidence(evaluation.content)
            current_state['confidence'] = confidence
            
            # Add to context
            self.context_manager.add_message(
                session_id,
                LLMMessage(role="user", content=request.prompt)
            )
            self.context_manager.add_message(
                session_id,
                LLMMessage(role="assistant", content=current_state['solution'])
            )
            
            # Store reasoning process as memory
            self.context_manager.add_memory(
                content=f"Reasoning process for: {request.prompt[:100]}...",
                importance=0.8,
                tags=['reasoning', 'problem_solving'],
                session_id=session_id,
                metadata={
                    'reasoning_trace': [trace.__dict__ for trace in reasoning_trace],
                    'confidence': confidence
                }
            )
            
            return InferenceResult(
                response=current_state['solution'],
                reasoning_trace=reasoning_trace,
                confidence=confidence,
                metadata={
                    'session_id': session_id,
                    'model_info': self.llm_adapter.get_model_info(),
                    'reasoning_steps': len(reasoning_trace)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Reasoning inference failed: {e}")
            return InferenceResult(
                response=f"Error during reasoning: {str(e)}",
                reasoning_trace=reasoning_trace,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    async def _reasoning_step(
        self,
        step_type: ReasoningStep,
        messages: List[LLMMessage],
        instruction: str
    ) -> LLMResponse:
        """
        Perform a single reasoning step
        
        Args:
            step_type: Type of reasoning step
            messages: Current messages
            instruction: Specific instruction for this step
            
        Returns:
            LLM response for this step
        """
        step_messages = messages + [
            LLMMessage(role="user", content=instruction)
        ]
        return await self.llm_adapter.generate(step_messages)
    
    def _build_messages(
        self,
        request: InferenceRequest,
        session_id: str
    ) -> List[LLMMessage]:
        """
        Build messages from context and request
        
        Args:
            request: Inference request
            session_id: Session ID
            
        Returns:
            List of messages for LLM
        """
        messages = []
        
        # Add system message
        system_prompt = self._build_system_prompt(request)
        messages.append(LLMMessage(role="system", content=system_prompt))
        
        # Add relevant memory if requested
        if request.use_memory:
            memories = self.context_manager.search_memory(
                query=request.prompt,
                min_importance=0.5,
                limit=5
            )
            if memories:
                memory_context = "\n".join([
                    f"Memory: {mem.content}" for mem in memories
                ])
                messages.append(LLMMessage(
                    role="system",
                    content=f"Relevant context from memory:\n{memory_context}"
                ))
        
        # Add recent conversation history
        recent_messages = self.context_manager.get_recent_messages(
            session_id, count=10
        )
        messages.extend(recent_messages)
        
        # Add current prompt
        messages.append(LLMMessage(role="user", content=request.prompt))
        
        return messages
    
    def _build_system_prompt(self, request: InferenceRequest) -> str:
        """
        Build system prompt based on request
        
        Args:
            request: Inference request
            
        Returns:
            System prompt string
        """
        base_prompt = "You are an intelligent assistant that provides helpful, accurate, and thoughtful responses."
        
        if request.use_reasoning:
            base_prompt += "\nUse step-by-step reasoning to analyze problems thoroughly before providing solutions."
        
        if request.tools:
            available_tools = [tool for tool in request.tools if tool in self.available_tools]
            if available_tools:
                base_prompt += f"\nYou have access to the following tools: {', '.join(available_tools)}"
        
        return base_prompt
    
    def _calculate_confidence(self, response: LLMResponse) -> float:
        """
        Calculate confidence score for a response
        
        Args:
            response: LLM response
            
        Returns:
            Confidence score between 0 and 1
        """
        # Simple heuristic based on response characteristics
        confidence = 0.5  # Base confidence
        
        if response.finish_reason == "stop":
            confidence += 0.2
        
        # Longer responses might indicate more thoughtful answers
        if len(response.content) > 100:
            confidence += 0.1
        
        # Check for uncertainty indicators
        uncertainty_phrases = [
            "i'm not sure", "maybe", "possibly", "i think",
            "might be", "could be", "perhaps", "uncertain"
        ]
        content_lower = response.content.lower()
        uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in content_lower)
        confidence -= uncertainty_count * 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _extract_confidence(self, evaluation_text: str) -> float:
        """
        Extract confidence score from evaluation text
        
        Args:
            evaluation_text: Text containing confidence evaluation
            
        Returns:
            Confidence score between 0 and 1
        """
        # Look for numerical confidence scores
        import re
        
        # Look for patterns like "confidence: 0.8" or "8/10" or "80%"
        patterns = [
            r'confidence[:\s]+([0-9]*\.?[0-9]+)',
            r'([0-9]*\.?[0-9]+)/10',
            r'([0-9]+)%',
            r'([0-9]*\.?[0-9]+)\s*out\s*of\s*10'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, evaluation_text.lower())
            if match:
                value = float(match.group(1))
                if pattern.endswith('%'):
                    return value / 100.0
                elif '/10' in pattern or 'out of 10' in pattern:
                    return value / 10.0
                else:
                    return min(1.0, value)  # Assume it's already 0-1 scale
        
        # Fallback to text analysis
        return self._analyze_confidence_text(evaluation_text)
    
    def _analyze_confidence_text(self, text: str) -> float:
        """
        Analyze text for confidence indicators
        
        Args:
            text: Text to analyze
            
        Returns:
            Confidence score between 0 and 1
        """
        text_lower = text.lower()
        
        high_confidence_words = [
            'excellent', 'perfect', 'accurate', 'correct', 'comprehensive',
            'complete', 'thorough', 'confident', 'certain', 'definitely'
        ]
        
        medium_confidence_words = [
            'good', 'adequate', 'reasonable', 'acceptable', 'satisfactory',
            'likely', 'probably', 'generally'
        ]
        
        low_confidence_words = [
            'poor', 'incomplete', 'uncertain', 'doubtful', 'questionable',
            'unclear', 'insufficient', 'maybe', 'possibly'
        ]
        
        high_score = sum(1 for word in high_confidence_words if word in text_lower)
        medium_score = sum(1 for word in medium_confidence_words if word in text_lower)
        low_score = sum(1 for word in low_confidence_words if word in text_lower)
        
        if high_score > low_score:
            return 0.8 + (high_score * 0.05)
        elif medium_score > low_score:
            return 0.6 + (medium_score * 0.05)
        else:
            return max(0.2, 0.5 - (low_score * 0.1))
    
    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Call a tool by name
        
        Args:
            tool_name: Name of the tool to call
            **kwargs: Arguments for the tool
            
        Returns:
            Tool result
        """
        if tool_name not in self.available_tools:
            raise ValueError(f"Tool '{tool_name}' not available")
        
        tool_func = self.available_tools[tool_name]
        
        try:
            if asyncio.iscoroutinefunction(tool_func):
                return await tool_func(**kwargs)
            else:
                return tool_func(**kwargs)
        except Exception as e:
            self.logger.error(f"Tool '{tool_name}' failed: {e}")
            raise
    
    def register_tool(self, name: str, func: Callable) -> None:
        """
        Register a new tool
        
        Args:
            name: Tool name
            func: Tool function
        """
        self.available_tools[name] = func
    
    def get_available_tools(self) -> List[str]:
        """
        Get list of available tool names
        
        Returns:
            List of tool names
        """
        return list(self.available_tools.keys())
