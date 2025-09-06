"""
Memory Tool

Provides memory management capabilities including storage, retrieval,
and organization of information across sessions.
"""

from typing import Dict, List, Optional, Any, Union
import asyncio
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
import uuid
import os
import numpy as np
from pathlib import Path
import aiofiles
import hashlib
import faiss
import openai
from openai import AsyncOpenAI


@dataclass
class MemoryEntry:
    """Represents a memory entry"""
    id: str
    content: str
    memory_type: str  # 'episodic', 'semantic', 'procedural', 'short_term'
    importance: float  # 0.0 to 1.0
    tags: List[str]
    timestamp: datetime
    context: Dict[str, Any]
    associations: List[str]  # IDs of related memories
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None  # Vector embedding for semantic search
    content_hash: Optional[str] = None  # Hash for deduplication


@dataclass
class ShortTermMemory:
    """Represents short-term memory state"""
    content: str
    timestamp: datetime
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FAISSVectorDB:
    """FAISS-based vector database for efficient semantic search"""
    
    def __init__(self, dimension: int = 1536):  # OpenAI ada-002 embedding dimension
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity with normalized vectors
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        self.metadata: Dict[str, Dict] = {}
        self.next_idx = 0
    
    def add_vector(self, id: str, vector: List[float], metadata: Dict = None) -> None:
        """Add a vector to the database"""
        # Remove existing vector if it exists
        if id in self.id_to_idx:
            self.remove_vector(id)
        
        # Normalize vector for cosine similarity
        vector_np = np.array(vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vector_np)
        
        # Add to FAISS index
        self.index.add(vector_np)
        
        # Update mappings
        self.id_to_idx[id] = self.next_idx
        self.idx_to_id[self.next_idx] = id
        self.metadata[id] = metadata or {}
        self.next_idx += 1
    
    def search(self, query_vector: List[float], top_k: int = 10, threshold: float = 0.0) -> List[tuple]:
        """Search for similar vectors"""
        if self.index.ntotal == 0:
            return []
        
        # Normalize query vector
        query_vec = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vec)
        
        # Search with FAISS
        similarities, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))
        
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for empty slots
                break
            
            if similarity >= threshold:
                memory_id = self.idx_to_id[idx]
                results.append((memory_id, float(similarity), self.metadata[memory_id]))
        
        return results
    
    def remove_vector(self, id: str) -> None:
        """Remove a vector from the database"""
        if id in self.id_to_idx:
            # Note: FAISS doesn't support efficient removal, so we just remove from our mappings
            # The vector remains in FAISS but won't be accessible
            idx = self.id_to_idx[id]
            del self.id_to_idx[id]
            del self.idx_to_id[idx]
            del self.metadata[id]
    
    def rebuild_index(self) -> None:
        """Rebuild the index to remove deleted vectors (expensive operation)"""
        if not self.metadata:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.id_to_idx.clear()
            self.idx_to_id.clear()
            self.next_idx = 0
            return
        
        # This would require storing original vectors, which we don't do for memory efficiency
        # In practice, periodic full rebuilds would be needed for production use
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            "total_vectors": len(self.metadata),
            "faiss_index_size": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": "IndexFlatIP"
        }


class MemoryTool:
    """Tool for managing memory storage and retrieval with FAISS vector search and short-term memory"""
    
    def __init__(
        self, 
        max_memories: int = 10000, 
        persistence_path: Optional[str] = None,
        short_term_memory_file: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        use_openai_embeddings: bool = True
    ):
        self.max_memories = max_memories
        self.persistence_path = persistence_path
        self.short_term_memory_file = short_term_memory_file or "short_term_memory.txt"
        self.memories: Dict[str, MemoryEntry] = {}
        self.memory_index: Dict[str, List[str]] = {}  # Tag -> memory IDs
        self.vector_db = FAISSVectorDB()
        self.short_term_memory: Optional[ShortTermMemory] = None
        self.use_openai_embeddings = use_openai_embeddings
        self.embedding_model = embedding_model
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client if using OpenAI embeddings
        if self.use_openai_embeddings:
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                self.logger.warning("OpenAI API key not provided, falling back to simple embeddings")
                self.use_openai_embeddings = False
                self.openai_client = None
            else:
                self.openai_client = AsyncOpenAI(api_key=api_key)
        else:
            self.openai_client = None
        
        # Initialize short-term memory
        asyncio.create_task(self._load_short_term_memory())
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI or fallback method"""
        if self.use_openai_embeddings and self.openai_client:
            try:
                response = await self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                return response.data[0].embedding
            except Exception as e:
                self.logger.error(f"OpenAI embedding failed: {e}, falling back to simple embedding")
                return self._simple_embedding(text)
        else:
            return self._simple_embedding(text)
    
    def _simple_embedding(self, text: str) -> List[float]:
        """Simple text embedding using character frequencies (fallback)"""
        # This is a very basic embedding - expanded for better dimensionality
        import string
        import re
        
        # Create features from text
        features = []
        
        # Character frequency features
        text_clean = re.sub(r'[^\w\s]', '', text.lower())
        char_counts = {char: text_clean.count(char) / len(text_clean) if text_clean else 0 
                      for char in string.ascii_lowercase + string.digits + ' '}
        features.extend(char_counts.values())
        
        # Word length features
        words = text_clean.split()
        if words:
            avg_word_len = sum(len(word) for word in words) / len(words)
            features.extend([avg_word_len / 10, len(words) / 100])  # Normalized features
        else:
            features.extend([0.0, 0.0])
        
        # Text statistics
        features.extend([
            len(text) / 1000,  # Text length (normalized)
            text.count(' ') / len(text) if text else 0,  # Space ratio
            text.count('\n') / len(text) if text else 0,  # Newline ratio
        ])
        
        # Pad to reach target dimension (1536 for OpenAI compatibility)
        target_dim = 1536
        while len(features) < target_dim:
            features.extend([0.0] * min(100, target_dim - len(features)))
        
        # Truncate if too long
        features = features[:target_dim]
        
        # Normalize
        norm = sum(x**2 for x in features) ** 0.5
        if norm > 0:
            features = [x / norm for x in features]
        
        return features
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate hash for content deduplication"""
        return hashlib.md5(content.encode()).hexdigest()
    
    async def store_memory(
        self,
        content: str,
        memory_type: str = "episodic",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        associations: Optional[List[str]] = None,
        generate_embedding: bool = True
    ) -> str:
        """
        Store a new memory with optional vector embedding
        
        Args:
            content: Memory content
            memory_type: Type of memory ('episodic', 'semantic', 'procedural', 'short_term')
            importance: Importance score (0.0 to 1.0)
            tags: Tags for categorization
            context: Contextual information
            associations: IDs of related memories
            generate_embedding: Whether to generate vector embedding
            
        Returns:
            Memory ID
        """
        try:
            memory_id = str(uuid.uuid4())
            content_hash = self._calculate_content_hash(content)
            
            # Check for duplicate content
            existing_memory = await self._find_duplicate_memory(content_hash)
            if existing_memory:
                self.logger.debug(f"Duplicate content found, updating existing memory: {existing_memory.id}")
                # Update existing memory importance if new one is higher
                if importance > existing_memory.importance:
                    existing_memory.importance = importance
                existing_memory.access_count += 1
                existing_memory.last_accessed = datetime.now()
                return existing_memory.id
            
            # Generate embedding if requested
            embedding = None
            if generate_embedding:
                try:
                    embedding = await self._get_embedding(content)
                except Exception as e:
                    self.logger.warning(f"Failed to generate embedding: {e}")
            
            memory_entry = MemoryEntry(
                id=memory_id,
                content=content,
                memory_type=memory_type,
                importance=max(0.0, min(1.0, importance)),
                tags=tags or [],
                timestamp=datetime.now(),
                context=context or {},
                associations=associations or [],
                metadata={},
                embedding=embedding,
                content_hash=content_hash
            )
            
            # Store memory
            self.memories[memory_id] = memory_entry
            
            # Add to vector database if embedding exists
            if embedding:
                self.vector_db.add_vector(
                    memory_id, 
                    embedding, 
                    {"memory_type": memory_type, "importance": importance, "tags": tags or []}
                )
            
            # Update index
            await self._update_index(memory_entry)
            
            # Handle memory limit
            await self._enforce_memory_limit()
            
            # Save if persistence is enabled
            if self.persistence_path:
                await self._save_to_disk()
            
            self.logger.debug(f"Stored memory: {memory_id}")
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Failed to store memory: {e}")
            raise
    
    async def retrieve_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific memory by ID
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Memory data or None if not found
        """
        try:
            if memory_id not in self.memories:
                return None
            
            memory = self.memories[memory_id]
            
            # Update access statistics
            memory.access_count += 1
            memory.last_accessed = datetime.now()
            
            return asdict(memory)
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            return None
    
    async def search_memories(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
        max_results: int = 10,
        time_range: Optional[tuple[datetime, datetime]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search memories based on various criteria
        
        Args:
            query: Text query to search in content
            tags: Tags to filter by
            memory_type: Memory type to filter by
            min_importance: Minimum importance threshold
            max_results: Maximum number of results
            time_range: Tuple of (start_time, end_time) to filter by
            
        Returns:
            List of matching memories
        """
        try:
            matching_memories = []
            
            for memory in self.memories.values():
                # Apply filters
                if memory.importance < min_importance:
                    continue
                
                if memory_type and memory.memory_type != memory_type:
                    continue
                
                if time_range:
                    start_time, end_time = time_range
                    if not (start_time <= memory.timestamp <= end_time):
                        continue
                
                if tags:
                    if not any(tag in memory.tags for tag in tags):
                        continue
                
                if query:
                    query_lower = query.lower()
                    if query_lower not in memory.content.lower():
                        # Check tags as well
                        if not any(query_lower in tag.lower() for tag in memory.tags):
                            continue
                
                matching_memories.append(memory)
            
            # Sort by relevance (combination of importance and recency)
            matching_memories.sort(
                key=lambda m: (m.importance, m.timestamp.timestamp()),
                reverse=True
            )
            
            # Update access statistics for returned memories
            results = []
            for memory in matching_memories[:max_results]:
                memory.access_count += 1
                memory.last_accessed = datetime.now()
                results.append(asdict(memory))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search memories: {e}")
            return []
    
    async def semantic_search_memories(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.3,
        memory_types: Optional[List[str]] = None,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search memories using semantic similarity (vector search)
        
        Args:
            query: Text query to search for
            top_k: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            memory_types: Filter by specific memory types
            min_importance: Minimum importance threshold
            
        Returns:
            List of matching memories with similarity scores
        """
        try:
            # Generate embedding for query
            query_embedding = await self._get_embedding(query)
            
            # Search vector database
            vector_results = self.vector_db.search(
                query_embedding, 
                top_k=top_k * 2,  # Get more results to filter
                threshold=similarity_threshold
            )
            
            # Filter and format results
            results = []
            for memory_id, similarity, vector_metadata in vector_results:
                if memory_id not in self.memories:
                    continue
                
                memory = self.memories[memory_id]
                
                # Apply filters
                if memory.importance < min_importance:
                    continue
                
                if memory_types and memory.memory_type not in memory_types:
                    continue
                
                # Update access statistics
                memory.access_count += 1
                memory.last_accessed = datetime.now()
                
                # Add similarity score to result
                memory_dict = asdict(memory)
                memory_dict['similarity_score'] = similarity
                results.append(memory_dict)
                
                if len(results) >= top_k:
                    break
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to perform semantic search: {e}")
            return []
    
    async def hybrid_search_memories(
        self,
        query: str,
        top_k: int = 10,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining keyword and semantic search
        
        Args:
            query: Text query to search for
            top_k: Maximum number of results to return
            keyword_weight: Weight for keyword search (0.0 to 1.0)
            semantic_weight: Weight for semantic search (0.0 to 1.0)
            **kwargs: Additional arguments for search methods
            
        Returns:
            List of matching memories with combined scores
        """
        try:
            # Normalize weights
            total_weight = keyword_weight + semantic_weight
            if total_weight > 0:
                keyword_weight /= total_weight
                semantic_weight /= total_weight
            
            # Get results from both search methods
            keyword_results = await self.search_memories(query=query, max_results=top_k * 2, **kwargs)
            semantic_results = await self.semantic_search_memories(query, top_k=top_k * 2, **kwargs)
            
            # Combine results with weighted scores
            combined_scores = {}
            
            # Add keyword search scores
            for result in keyword_results:
                memory_id = result['id']
                combined_scores[memory_id] = {
                    'memory': result,
                    'keyword_score': 1.0,  # Simple scoring for now
                    'semantic_score': 0.0
                }
            
            # Add semantic search scores
            for result in semantic_results:
                memory_id = result['id']
                if memory_id in combined_scores:
                    combined_scores[memory_id]['semantic_score'] = result.get('similarity_score', 0.0)
                else:
                    combined_scores[memory_id] = {
                        'memory': result,
                        'keyword_score': 0.0,
                        'semantic_score': result.get('similarity_score', 0.0)
                    }
            
            # Calculate combined scores and sort
            final_results = []
            for memory_id, scores in combined_scores.items():
                combined_score = (
                    scores['keyword_score'] * keyword_weight +
                    scores['semantic_score'] * semantic_weight
                )
                
                memory_dict = scores['memory'].copy()
                memory_dict['combined_score'] = combined_score
                memory_dict['keyword_score'] = scores['keyword_score']
                memory_dict['semantic_score'] = scores['semantic_score']
                
                final_results.append(memory_dict)
            
            # Sort by combined score and return top results
            final_results.sort(key=lambda x: x['combined_score'], reverse=True)
            return final_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Failed to perform hybrid search: {e}")
            return []
    
    # Short-term memory methods
    async def set_short_term_memory(
        self,
        content: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Set the short-term memory content
        
        Args:
            content: Content to store in short-term memory
            session_id: Optional session identifier
            metadata: Optional metadata
            
        Returns:
            True if successful
        """
        try:
            self.short_term_memory = ShortTermMemory(
                content=content,
                timestamp=datetime.now(),
                session_id=session_id,
                metadata=metadata or {}
            )
            
            # Save to file
            await self._save_short_term_memory()
            
            self.logger.debug("Updated short-term memory")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set short-term memory: {e}")
            return False
    
    async def get_short_term_memory(self) -> Optional[Dict[str, Any]]:
        """
        Get the current short-term memory content
        
        Returns:
            Short-term memory data or None if empty
        """
        try:
            if not self.short_term_memory:
                return None
            
            return {
                'content': self.short_term_memory.content,
                'timestamp': self.short_term_memory.timestamp.isoformat(),
                'session_id': self.short_term_memory.session_id,
                'metadata': self.short_term_memory.metadata,
                'age_seconds': (datetime.now() - self.short_term_memory.timestamp).total_seconds()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get short-term memory: {e}")
            return None
    
    async def update_short_term_memory(self, additional_content: str) -> bool:
        """
        Append content to existing short-term memory
        
        Args:
            additional_content: Content to append
            
        Returns:
            True if successful
        """
        try:
            if not self.short_term_memory:
                return await self.set_short_term_memory(additional_content)
            
            # Append to existing content
            self.short_term_memory.content += f"\n{additional_content}"
            self.short_term_memory.timestamp = datetime.now()
            
            # Save to file
            await self._save_short_term_memory()
            
            self.logger.debug("Updated short-term memory with additional content")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update short-term memory: {e}")
            return False
    
    async def clear_short_term_memory(self, preserve_to_long_term: bool = False) -> bool:
        """
        Clear the short-term memory
        
        Args:
            preserve_to_long_term: If True, save content to long-term memory before clearing
            
        Returns:
            True if successful
        """
        try:
            if preserve_to_long_term and self.short_term_memory:
                # Save to long-term memory
                await self.store_memory(
                    content=f"[Short-term memory preserved]: {self.short_term_memory.content}",
                    memory_type="episodic",
                    importance=0.6,
                    tags=["preserved_short_term", "session_memory"],
                    context={"session_id": self.short_term_memory.session_id}
                )
            
            # Clear short-term memory
            self.short_term_memory = None
            
            # Clear the file
            await self._save_short_term_memory()
            
            self.logger.debug("Cleared short-term memory")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear short-term memory: {e}")
            return False
    
    async def _save_short_term_memory(self) -> None:
        """Save short-term memory to file"""
        try:
            if self.short_term_memory:
                content = f"[{self.short_term_memory.timestamp.isoformat()}]"
                if self.short_term_memory.session_id:
                    content += f" Session: {self.short_term_memory.session_id}"
                content += f"\n{self.short_term_memory.content}\n"
                
                if self.short_term_memory.metadata:
                    content += f"\nMetadata: {json.dumps(self.short_term_memory.metadata, indent=2)}\n"
            else:
                content = ""
            
            async with aiofiles.open(self.short_term_memory_file, 'w') as f:
                await f.write(content)
                
        except Exception as e:
            self.logger.error(f"Failed to save short-term memory to file: {e}")
    
    async def _load_short_term_memory(self) -> None:
        """Load short-term memory from file"""
        try:
            if not os.path.exists(self.short_term_memory_file):
                return
            
            async with aiofiles.open(self.short_term_memory_file, 'r') as f:
                content = await f.read()
            
            if not content.strip():
                return
            
            lines = content.strip().split('\n')
            if not lines:
                return
            
            # Parse timestamp and session from first line
            header = lines[0]
            timestamp_str = header.split(']')[0][1:]  # Remove [ and ]
            timestamp = datetime.fromisoformat(timestamp_str)
            
            session_id = None
            if 'Session:' in header:
                session_id = header.split('Session:')[1].strip()
            
            # Get content (everything except first line and metadata)
            content_lines = []
            metadata = {}
            
            i = 1
            while i < len(lines):
                if lines[i].startswith('Metadata:'):
                    # Parse metadata JSON
                    metadata_lines = lines[i+1:]
                    metadata_str = '\n'.join(metadata_lines)
                    try:
                        metadata = json.loads(metadata_str)
                    except:
                        pass
                    break
                else:
                    content_lines.append(lines[i])
                i += 1
            
            if content_lines:
                self.short_term_memory = ShortTermMemory(
                    content='\n'.join(content_lines),
                    timestamp=timestamp,
                    session_id=session_id,
                    metadata=metadata
                )
                
                self.logger.debug("Loaded short-term memory from file")
                
        except Exception as e:
            self.logger.error(f"Failed to load short-term memory from file: {e}")
    
    async def _find_duplicate_memory(self, content_hash: str) -> Optional[MemoryEntry]:
        """Find memory with same content hash"""
        for memory in self.memories.values():
            if memory.content_hash == content_hash:
                return memory
        return None
    
    async def associate_memories(self, memory_id1: str, memory_id2: str) -> bool:
        """
        Create an association between two memories
        
        Args:
            memory_id1: First memory ID
            memory_id2: Second memory ID
            
        Returns:
            True if association created successfully
        """
        try:
            if memory_id1 not in self.memories or memory_id2 not in self.memories:
                return False
            
            memory1 = self.memories[memory_id1]
            memory2 = self.memories[memory_id2]
            
            # Add bidirectional associations
            if memory_id2 not in memory1.associations:
                memory1.associations.append(memory_id2)
            
            if memory_id1 not in memory2.associations:
                memory2.associations.append(memory_id1)
            
            self.logger.debug(f"Associated memories: {memory_id1} <-> {memory_id2}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to associate memories: {e}")
            return False
    
    async def update_memory_importance(self, memory_id: str, new_importance: float) -> bool:
        """
        Update the importance of a memory
        
        Args:
            memory_id: Memory ID
            new_importance: New importance score (0.0 to 1.0)
            
        Returns:
            True if updated successfully
        """
        try:
            if memory_id not in self.memories:
                return False
            
            memory = self.memories[memory_id]
            memory.importance = max(0.0, min(1.0, new_importance))
            
            self.logger.debug(f"Updated importance for memory {memory_id}: {new_importance}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update memory importance: {e}")
            return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory
        
        Args:
            memory_id: Memory ID to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            if memory_id not in self.memories:
                return False
            
            memory = self.memories[memory_id]
            
            # Remove from index
            for tag in memory.tags:
                if tag in self.memory_index:
                    self.memory_index[tag] = [
                        mid for mid in self.memory_index[tag] if mid != memory_id
                    ]
                    if not self.memory_index[tag]:
                        del self.memory_index[tag]
            
            # Remove associations from other memories
            for other_memory in self.memories.values():
                if memory_id in other_memory.associations:
                    other_memory.associations.remove(memory_id)
            
            # Remove from vector database
            self.vector_db.remove_vector(memory_id)
            
            # Delete memory
            del self.memories[memory_id]
            
            self.logger.debug(f"Deleted memory: {memory_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete memory: {e}")
            return False
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored memories
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            if not self.memories:
                return {
                    "total_memories": 0,
                    "memory_types": {},
                    "tags": {},
                    "average_importance": 0.0,
                    "oldest_memory": None,
                    "newest_memory": None
                }
            
            # Calculate statistics
            total_memories = len(self.memories)
            memory_types = {}
            tags = {}
            importance_sum = 0.0
            oldest_time = None
            newest_time = None
            
            for memory in self.memories.values():
                # Memory types
                memory_types[memory.memory_type] = memory_types.get(memory.memory_type, 0) + 1
                
                # Tags
                for tag in memory.tags:
                    tags[tag] = tags.get(tag, 0) + 1
                
                # Importance
                importance_sum += memory.importance
                
                # Time range
                if oldest_time is None or memory.timestamp < oldest_time:
                    oldest_time = memory.timestamp
                if newest_time is None or memory.timestamp > newest_time:
                    newest_time = memory.timestamp
            
            # Vector database statistics
            vector_stats = self.vector_db.get_stats()
            
            # Short-term memory info
            short_term_info = None
            if self.short_term_memory:
                short_term_info = {
                    "has_content": True,
                    "age_seconds": (datetime.now() - self.short_term_memory.timestamp).total_seconds(),
                    "session_id": self.short_term_memory.session_id,
                    "content_length": len(self.short_term_memory.content)
                }
            else:
                short_term_info = {"has_content": False}
            
            return {
                "total_memories": total_memories,
                "memory_types": memory_types,
                "tags": dict(sorted(tags.items(), key=lambda x: x[1], reverse=True)),
                "average_importance": importance_sum / total_memories,
                "oldest_memory": oldest_time.isoformat() if oldest_time else None,
                "newest_memory": newest_time.isoformat() if newest_time else None,
                "memory_limit": self.max_memories,
                "usage_percentage": (total_memories / self.max_memories) * 100,
                "vector_database": vector_stats,
                "short_term_memory": short_term_info
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get memory statistics: {e}")
            return {}
    
    async def consolidate_memories(self, similarity_threshold: float = 0.8) -> int:
        """
        Consolidate similar memories to reduce redundancy
        
        Args:
            similarity_threshold: Threshold for considering memories similar
            
        Returns:
            Number of memories consolidated
        """
        try:
            consolidated_count = 0
            memories_to_remove = set()
            
            memory_list = list(self.memories.values())
            
            for i, memory1 in enumerate(memory_list):
                if memory1.id in memories_to_remove:
                    continue
                
                for j, memory2 in enumerate(memory_list[i+1:], i+1):
                    if memory2.id in memories_to_remove:
                        continue
                    
                    # Simple similarity check based on content and tags
                    similarity = await self._calculate_similarity(memory1, memory2)
                    
                    if similarity >= similarity_threshold:
                        # Merge memories (keep the more important one)
                        if memory1.importance >= memory2.importance:
                            primary, secondary = memory1, memory2
                        else:
                            primary, secondary = memory2, memory1
                        
                        # Update primary memory with secondary's information
                        primary.content += f" [Consolidated: {secondary.content}]"
                        primary.importance = max(primary.importance, secondary.importance)
                        primary.tags = list(set(primary.tags + secondary.tags))
                        primary.associations = list(set(primary.associations + secondary.associations))
                        
                        # Mark secondary for removal
                        memories_to_remove.add(secondary.id)
                        consolidated_count += 1
            
            # Remove consolidated memories
            for memory_id in memories_to_remove:
                await self.delete_memory(memory_id)
            
            self.logger.info(f"Consolidated {consolidated_count} memories")
            return consolidated_count
            
        except Exception as e:
            self.logger.error(f"Failed to consolidate memories: {e}")
            return 0
    
    async def _update_index(self, memory: MemoryEntry) -> None:
        """Update the memory index"""
        for tag in memory.tags:
            if tag not in self.memory_index:
                self.memory_index[tag] = []
            self.memory_index[tag].append(memory.id)
    
    async def _enforce_memory_limit(self) -> None:
        """Enforce memory limit by removing least important memories"""
        if len(self.memories) <= self.max_memories:
            return
        
        # Sort memories by importance and recency
        sorted_memories = sorted(
            self.memories.values(),
            key=lambda m: (m.importance, m.timestamp.timestamp())
        )
        
        # Remove least important memories
        memories_to_remove = len(self.memories) - self.max_memories
        for memory in sorted_memories[:memories_to_remove]:
            await self.delete_memory(memory.id)
    
    async def _calculate_similarity(self, memory1: MemoryEntry, memory2: MemoryEntry) -> float:
        """Calculate similarity between two memories (simplified)"""
        try:
            # Content similarity (basic word overlap)
            words1 = set(memory1.content.lower().split())
            words2 = set(memory2.content.lower().split())
            
            if not words1 and not words2:
                content_similarity = 1.0
            elif not words1 or not words2:
                content_similarity = 0.0
            else:
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                content_similarity = intersection / union if union > 0 else 0.0
            
            # Tag similarity
            tags1 = set(memory1.tags)
            tags2 = set(memory2.tags)
            
            if not tags1 and not tags2:
                tag_similarity = 1.0
            elif not tags1 or not tags2:
                tag_similarity = 0.0
            else:
                intersection = len(tags1.intersection(tags2))
                union = len(tags1.union(tags2))
                tag_similarity = intersection / union if union > 0 else 0.0
            
            # Type similarity
            type_similarity = 1.0 if memory1.memory_type == memory2.memory_type else 0.0
            
            # Overall similarity (weighted average)
            overall_similarity = (
                content_similarity * 0.5 +
                tag_similarity * 0.3 +
                type_similarity * 0.2
            )
            
            return overall_similarity
            
        except Exception:
            return 0.0
    
    async def _save_to_disk(self) -> None:
        """Save memories to disk for persistence"""
        if not self.persistence_path:
            return
        
        try:
            # Convert memories to serializable format
            serializable_memories = {}
            for memory_id, memory in self.memories.items():
                memory_dict = asdict(memory)
                memory_dict['timestamp'] = memory.timestamp.isoformat()
                if memory.last_accessed:
                    memory_dict['last_accessed'] = memory.last_accessed.isoformat()
                serializable_memories[memory_id] = memory_dict
            
            # Save memories to JSON file
            with open(self.persistence_path, 'w') as f:
                json.dump(serializable_memories, f, indent=2)
            
            # Save FAISS index separately (binary format)
            if self.vector_db.index.ntotal > 0:
                faiss_path = self.persistence_path.replace('.json', '_faiss.index')
                faiss.write_index(self.vector_db.index, faiss_path)
                
                # Save FAISS metadata
                metadata_path = self.persistence_path.replace('.json', '_faiss_metadata.json')
                faiss_metadata = {
                    'id_to_idx': self.vector_db.id_to_idx,
                    'idx_to_id': {str(k): v for k, v in self.vector_db.idx_to_id.items()},
                    'metadata': self.vector_db.metadata,
                    'next_idx': self.vector_db.next_idx
                }
                with open(metadata_path, 'w') as f:
                    json.dump(faiss_metadata, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to save memories to disk: {e}")
    
    async def load_from_disk(self) -> bool:
        """Load memories from disk"""
        if not self.persistence_path:
            return False
        
        try:
            # Load memories from JSON
            with open(self.persistence_path, 'r') as f:
                serializable_memories = json.load(f)
            
            # Convert back to MemoryEntry objects
            for memory_id, memory_dict in serializable_memories.items():
                memory_dict['timestamp'] = datetime.fromisoformat(memory_dict['timestamp'])
                if memory_dict.get('last_accessed'):
                    memory_dict['last_accessed'] = datetime.fromisoformat(memory_dict['last_accessed'])
                
                memory = MemoryEntry(**memory_dict)
                self.memories[memory_id] = memory
                await self._update_index(memory)
            
            # Load FAISS index if it exists
            faiss_path = self.persistence_path.replace('.json', '_faiss.index')
            metadata_path = self.persistence_path.replace('.json', '_faiss_metadata.json')
            
            if os.path.exists(faiss_path) and os.path.exists(metadata_path):
                try:
                    # Load FAISS index
                    self.vector_db.index = faiss.read_index(faiss_path)
                    
                    # Load FAISS metadata
                    with open(metadata_path, 'r') as f:
                        faiss_metadata = json.load(f)
                    
                    self.vector_db.id_to_idx = faiss_metadata['id_to_idx']
                    self.vector_db.idx_to_id = {int(k): v for k, v in faiss_metadata['idx_to_id'].items()}
                    self.vector_db.metadata = faiss_metadata['metadata']
                    self.vector_db.next_idx = faiss_metadata['next_idx']
                    
                    self.logger.info(f"Loaded FAISS index with {self.vector_db.index.ntotal} vectors")
                except Exception as e:
                    self.logger.warning(f"Failed to load FAISS index: {e}, will rebuild from memories")
                    # Rebuild vector database from loaded memories
                    await self._rebuild_vector_database()
            else:
                # Rebuild vector database from loaded memories if no FAISS files
                await self._rebuild_vector_database()
            
            self.logger.info(f"Loaded {len(self.memories)} memories from disk")
            return True
            
        except FileNotFoundError:
            self.logger.info("No persistence file found, starting with empty memory")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load memories from disk: {e}")
            return False
    
    async def _rebuild_vector_database(self) -> None:
        """Rebuild the vector database from existing memories"""
        try:
            self.logger.info("Rebuilding vector database from memories...")
            
            for memory_id, memory in self.memories.items():
                if memory.embedding:
                    self.vector_db.add_vector(
                        memory_id,
                        memory.embedding,
                        {"memory_type": memory.memory_type, "importance": memory.importance, "tags": memory.tags}
                    )
                elif memory.content:  # Generate embedding if missing
                    try:
                        embedding = await self._get_embedding(memory.content)
                        memory.embedding = embedding
                        self.vector_db.add_vector(
                            memory_id,
                            embedding,
                            {"memory_type": memory.memory_type, "importance": memory.importance, "tags": memory.tags}
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to generate embedding for memory {memory_id}: {e}")
            
            self.logger.info(f"Rebuilt vector database with {len(self.vector_db.metadata)} vectors")
            
        except Exception as e:
            self.logger.error(f"Failed to rebuild vector database: {e}")
