#!/usr/bin/env python3
"""
Memory Tool Demonstration

This script demonstrates the enhanced memory capabilities including:
- Vector-based semantic search
- Short-term memory management
- Memory persistence
- Hybrid search (keyword + semantic)
"""

import asyncio
import logging
from datetime import datetime
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mcp_tools.core_tools.memory_tool import MemoryTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_memory_features():
    """Demonstrate various memory features"""
    
    # Initialize memory tool with persistence and OpenAI embeddings
    memory_tool = MemoryTool(
        max_memories=1000,
        persistence_path="demo_memories.json",
        short_term_memory_file="demo_short_term.txt",
        use_openai_embeddings=True,  # Set to False to use fallback embeddings
        embedding_model="text-embedding-3-small"
    )
    
    logger.info("=== EFA Memory Tool Demo ===")
    
    # 1. Store some sample memories
    logger.info("\n1. Storing sample memories...")
    
    memories = [
        ("The user prefers Python for backend development", "semantic", 0.8, ["preferences", "programming"]),
        ("Meeting scheduled for tomorrow at 2 PM", "episodic", 0.9, ["meetings", "schedule"]),
        ("Use async/await for better performance", "procedural", 0.7, ["programming", "best_practices"]),
        ("User's favorite color is blue", "semantic", 0.6, ["preferences", "personal"]),
        ("FastAPI is good for building APIs", "semantic", 0.8, ["programming", "frameworks"]),
    ]
    
    memory_ids = []
    for content, mem_type, importance, tags in memories:
        memory_id = await memory_tool.store_memory(
            content=content,
            memory_type=mem_type,
            importance=importance,
            tags=tags
        )
        memory_ids.append(memory_id)
        logger.info(f"Stored memory: {content[:50]}... (ID: {memory_id[:8]})")
    
    # 2. Demonstrate keyword search
    logger.info("\n2. Keyword search for 'Python'...")
    results = await memory_tool.search_memories(query="Python", max_results=3)
    for result in results:
        logger.info(f"Found: {result['content']} (Importance: {result['importance']})")
    
    # 3. Demonstrate semantic search
    logger.info("\n3. Semantic search for 'coding practices'...")
    results = await memory_tool.semantic_search_memories(
        query="coding practices",
        top_k=3,
        similarity_threshold=0.1
    )
    for result in results:
        logger.info(f"Found: {result['content']} (Similarity: {result.get('similarity_score', 0):.3f})")
    
    # 4. Demonstrate hybrid search
    logger.info("\n4. Hybrid search for 'API development'...")
    results = await memory_tool.hybrid_search_memories(
        query="API development",
        top_k=3,
        keyword_weight=0.4,
        semantic_weight=0.6
    )
    for result in results:
        logger.info(f"Found: {result['content']} (Combined: {result.get('combined_score', 0):.3f})")
    
    # 5. Demonstrate short-term memory
    logger.info("\n5. Testing short-term memory...")
    
    await memory_tool.set_short_term_memory(
        "User is currently working on implementing vector search",
        session_id="demo_session_001"
    )
    logger.info("Set short-term memory")
    
    short_term = await memory_tool.get_short_term_memory()
    if short_term:
        logger.info(f"Current short-term memory: {short_term['content']}")
        logger.info(f"Age: {short_term['age_seconds']:.1f} seconds")
    
    # Update short-term memory
    await memory_tool.update_short_term_memory("Added memory persistence functionality")
    
    updated_short_term = await memory_tool.get_short_term_memory()
    if updated_short_term:
        logger.info(f"Updated short-term memory: {updated_short_term['content']}")
    
    # 6. Memory statistics
    logger.info("\n6. Memory statistics...")
    stats = await memory_tool.get_memory_statistics()
    logger.info(f"Total memories: {stats.get('total_memories', 0)}")
    logger.info(f"Memory types: {stats.get('memory_types', {})}")
    logger.info(f"Vector database: {stats.get('vector_database', {})}")
    logger.info(f"Short-term memory: {stats.get('short_term_memory', {})}")
    
    # 7. Memory associations
    logger.info("\n7. Creating memory associations...")
    if len(memory_ids) >= 2:
        success = await memory_tool.associate_memories(memory_ids[0], memory_ids[2])
        if success:
            logger.info("Successfully associated Python preference with async/await knowledge")
    
    # 8. Clear short-term memory with preservation
    logger.info("\n8. Clearing short-term memory (preserving to long-term)...")
    await memory_tool.clear_short_term_memory(preserve_to_long_term=True)
    
    cleared_short_term = await memory_tool.get_short_term_memory()
    logger.info(f"Short-term memory cleared: {cleared_short_term is None}")
    
    # 9. Show final statistics
    logger.info("\n9. Final statistics...")
    final_stats = await memory_tool.get_memory_statistics()
    logger.info(f"Final total memories: {final_stats.get('total_memories', 0)}")
    
    logger.info("\n=== Demo Complete ===")
    logger.info("Check the following files created:")
    logger.info("- demo_memories.json (persistent memory storage)")
    logger.info("- demo_memories_faiss.index (FAISS vector index)")
    logger.info("- demo_memories_faiss_metadata.json (FAISS metadata)")
    logger.info("- demo_short_term.txt (short-term memory file)")
    logger.info("\nNote: Set OPENAI_API_KEY environment variable to use OpenAI embeddings")


if __name__ == "__main__":
    asyncio.run(demo_memory_features())
