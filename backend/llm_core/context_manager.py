"""
Context Manager

Manages memory and prompt context for LLM interactions.
Handles conversation history, memory persistence, and context optimization.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid
from .llm_interface import LLMMessage


@dataclass
class MemoryEntry:
    """Individual memory entry"""
    id: str
    content: str
    timestamp: datetime
    importance: float = 0.5  # 0.0 to 1.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Context for a conversation session"""
    session_id: str
    messages: List[LLMMessage] = field(default_factory=list)
    memory_entries: List[MemoryEntry] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextManager:
    """Manages conversation context and memory"""
    
    def __init__(self, max_context_length: int = 8000, memory_size_limit: int = 1000):
        self.max_context_length = max_context_length
        self.memory_size_limit = memory_size_limit
        self.active_contexts: Dict[str, ConversationContext] = {}
        self.global_memory: List[MemoryEntry] = []
    
    def create_context(self, session_id: Optional[str] = None) -> str:
        """
        Create a new conversation context
        
        Args:
            session_id: Optional session ID, generates one if not provided
            
        Returns:
            Session ID for the created context
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        self.active_contexts[session_id] = ConversationContext(
            session_id=session_id
        )
        return session_id
    
    def add_message(self, session_id: str, message: LLMMessage) -> None:
        """
        Add a message to the conversation context
        
        Args:
            session_id: Session ID
            message: Message to add
        """
        if session_id not in self.active_contexts:
            self.create_context(session_id)
        
        context = self.active_contexts[session_id]
        context.messages.append(message)
        context.updated_at = datetime.now()
        
        # Optimize context if it exceeds length limit
        self._optimize_context(session_id)
    
    def add_memory(
        self,
        content: str,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a memory entry
        
        Args:
            content: Memory content
            importance: Importance score (0.0 to 1.0)
            tags: Optional tags for categorization
            session_id: Optional session to associate with
            metadata: Optional metadata
            
        Returns:
            Memory entry ID
        """
        memory_id = str(uuid.uuid4())
        memory_entry = MemoryEntry(
            id=memory_id,
            content=content,
            timestamp=datetime.now(),
            importance=importance,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        if session_id and session_id in self.active_contexts:
            self.active_contexts[session_id].memory_entries.append(memory_entry)
        else:
            self.global_memory.append(memory_entry)
        
        # Clean up memory if it exceeds size limit
        self._cleanup_memory()
        
        return memory_id
    
    def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """
        Get conversation context by session ID
        
        Args:
            session_id: Session ID
            
        Returns:
            ConversationContext if exists, None otherwise
        """
        return self.active_contexts.get(session_id)
    
    def get_recent_messages(
        self,
        session_id: str,
        count: int = 10
    ) -> List[LLMMessage]:
        """
        Get recent messages from a conversation
        
        Args:
            session_id: Session ID
            count: Number of recent messages to retrieve
            
        Returns:
            List of recent messages
        """
        context = self.get_context(session_id)
        if not context:
            return []
        
        return context.messages[-count:]
    
    def search_memory(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        min_importance: float = 0.0,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """
        Search memory entries
        
        Args:
            query: Search query
            tags: Optional tags to filter by
            min_importance: Minimum importance threshold
            limit: Maximum number of results
            
        Returns:
            List of matching memory entries
        """
        all_memories = self.global_memory.copy()
        
        # Add memories from all active contexts
        for context in self.active_contexts.values():
            all_memories.extend(context.memory_entries)
        
        # Filter by importance
        filtered_memories = [
            memory for memory in all_memories
            if memory.importance >= min_importance
        ]
        
        # Filter by tags if specified
        if tags:
            filtered_memories = [
                memory for memory in filtered_memories
                if any(tag in memory.tags for tag in tags)
            ]
        
        # Simple text search in content
        query_lower = query.lower()
        matching_memories = [
            memory for memory in filtered_memories
            if query_lower in memory.content.lower()
        ]
        
        # Sort by importance and timestamp
        matching_memories.sort(
            key=lambda x: (x.importance, x.timestamp),
            reverse=True
        )
        
        return matching_memories[:limit]
    
    def _optimize_context(self, session_id: str) -> None:
        """
        Optimize context by removing less important messages if needed
        
        Args:
            session_id: Session ID to optimize
        """
        context = self.active_contexts[session_id]
        
        # Calculate approximate token count (rough estimation)
        total_length = sum(len(msg.content) for msg in context.messages)
        
        if total_length > self.max_context_length:
            # Keep system messages and recent messages
            system_messages = [msg for msg in context.messages if msg.role == 'system']
            other_messages = [msg for msg in context.messages if msg.role != 'system']
            
            # Keep most recent messages that fit within limit
            recent_messages = []
            current_length = sum(len(msg.content) for msg in system_messages)
            
            for msg in reversed(other_messages):
                msg_length = len(msg.content)
                if current_length + msg_length <= self.max_context_length:
                    recent_messages.insert(0, msg)
                    current_length += msg_length
                else:
                    # Save removed messages as memories
                    self.add_memory(
                        content=f"[{msg.role}]: {msg.content}",
                        importance=0.3,
                        tags=['conversation_history'],
                        session_id=session_id
                    )
            
            context.messages = system_messages + recent_messages
    
    def _cleanup_memory(self) -> None:
        """Clean up memory entries if they exceed the size limit"""
        all_memories = self.global_memory.copy()
        
        for context in self.active_contexts.values():
            all_memories.extend(context.memory_entries)
        
        if len(all_memories) > self.memory_size_limit:
            # Sort by importance and keep only the most important ones
            all_memories.sort(key=lambda x: x.importance, reverse=True)
            
            # Update global memory
            important_global = [
                mem for mem in self.global_memory
                if mem in all_memories[:self.memory_size_limit]
            ]
            self.global_memory = important_global
            
            # Update context memories
            for context in self.active_contexts.values():
                important_context = [
                    mem for mem in context.memory_entries
                    if mem in all_memories[:self.memory_size_limit]
                ]
                context.memory_entries = important_context
    
    def save_context(self, session_id: str, filepath: str) -> None:
        """
        Save context to file
        
        Args:
            session_id: Session ID
            filepath: File path to save to
        """
        context = self.get_context(session_id)
        if not context:
            return
        
        # Convert to serializable format
        data = {
            'session_id': context.session_id,
            'messages': [
                {
                    'role': msg.role,
                    'content': msg.content,
                    'metadata': msg.metadata
                }
                for msg in context.messages
            ],
            'memory_entries': [
                {
                    'id': mem.id,
                    'content': mem.content,
                    'timestamp': mem.timestamp.isoformat(),
                    'importance': mem.importance,
                    'tags': mem.tags,
                    'metadata': mem.metadata
                }
                for mem in context.memory_entries
            ],
            'created_at': context.created_at.isoformat(),
            'updated_at': context.updated_at.isoformat(),
            'metadata': context.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_context(self, filepath: str) -> str:
        """
        Load context from file
        
        Args:
            filepath: File path to load from
            
        Returns:
            Session ID of loaded context
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Recreate context
        session_id = data['session_id']
        context = ConversationContext(
            session_id=session_id,
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            metadata=data['metadata']
        )
        
        # Restore messages
        for msg_data in data['messages']:
            context.messages.append(LLMMessage(
                role=msg_data['role'],
                content=msg_data['content'],
                metadata=msg_data.get('metadata')
            ))
        
        # Restore memory entries
        for mem_data in data['memory_entries']:
            context.memory_entries.append(MemoryEntry(
                id=mem_data['id'],
                content=mem_data['content'],
                timestamp=datetime.fromisoformat(mem_data['timestamp']),
                importance=mem_data['importance'],
                tags=mem_data['tags'],
                metadata=mem_data['metadata']
            ))
        
        self.active_contexts[session_id] = context
        return session_id
