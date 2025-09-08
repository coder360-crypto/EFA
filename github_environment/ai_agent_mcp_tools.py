# enhanced_fastmcp_ai_agent_tools.py
"""
Enhanced FastMCP server with comprehensive AI agent tools
Includes detailed descriptions and extensive file manipulation capabilities
"""

import asyncio
import httpx
import json
import os
import re
from typing import Dict, Any, List, Optional
from fastmcp import FastMCP
from datetime import datetime
import uuid

# Initialize FastMCP server
mcp = FastMCP(
    name="Enhanced AI Agent Tools MCP Server",
    dependencies=["httpx", "requests"]
)

# Configuration
AI_AGENT_BASE_URL = "http://localhost:8001"
DEFAULT_SESSION_ID = "default_mcp_session"

class MCPSessionManager:
    """Simple session manager for MCP tools"""
    def __init__(self):
        self.sessions = {}
    
    async def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        if not session_id:
            session_id = DEFAULT_SESSION_ID
        
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "id": session_id,
                "created_at": datetime.now().isoformat(),
                "active": True
            }
        
        return session_id

session_manager = MCPSessionManager()

# =============================================================================
# BASIC FILE SYSTEM TOOLS
# =============================================================================

@mcp.tool(description="List directory contents with optional detailed view and recursive listing")
async def list_directory(
    path: str = "/workspace/repo",
    detailed: bool = False,
    recursive: bool = False,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """List directory contents (equivalent to ls command)
    
    Shows files and directories in the specified path. Use detailed=True for file permissions,
    sizes, and timestamps. Use recursive=True to list all subdirectories.
    
    Args:
        path: Directory path to list
        detailed: Include file details (like ls -la)
        recursive: List recursively (like ls -R)
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{AI_AGENT_BASE_URL}/sessions/{session_id}/agent/ls",
            params={"path": path, "detailed": detailed, "recursive": recursive},
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()

@mcp.tool(description="Get the current working directory path")
async def get_working_directory(session_id: Optional[str] = None) -> Dict[str, Any]:
    """Get current working directory (equivalent to pwd command)
    
    Returns the absolute path of the current working directory in the sandbox.
    
    Args:
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{AI_AGENT_BASE_URL}/sessions/{session_id}/agent/pwd",
            timeout=10.0
        )
        response.raise_for_status()
        return response.json()

@mcp.tool(description="Search for files and directories using patterns and filters")
async def find_files(
    pattern: str = "*",
    file_type: str = "f",
    path: str = "/workspace/repo",
    max_results: int = 100,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Find files and directories (equivalent to find command)
    
    Search for files using glob patterns. Supports wildcards and regular expressions.
    
    Args:
        pattern: File pattern to search for (e.g., "*.py", "test*", "main.js")
        file_type: Type - f (file), d (directory), l (link)
        path: Path to search in
        max_results: Maximum number of results
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{AI_AGENT_BASE_URL}/sessions/{session_id}/agent/find",
            params={"pattern": pattern, "file_type": file_type, "path": path, "max_results": max_results},
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()

# =============================================================================
# FILE CONTENT OPERATIONS
# =============================================================================

@mcp.tool(description="Read complete file contents or specific lines from start/end")
async def read_file(
    file_path: str,
    lines: Optional[int] = None,
    tail: Optional[int] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Read file contents (equivalent to cat/head/tail commands)
    
    Read entire file or specific portions. Use lines parameter for head-style reading,
    tail parameter for reading from end.
    
    Args:
        file_path: Path to the file to read
        lines: Number of lines from start (head -n)
        tail: Number of lines from end (tail -n)
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    params = {"file_path": file_path}
    if lines is not None:
        params["lines"] = lines
    if tail is not None:
        params["tail"] = tail
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{AI_AGENT_BASE_URL}/sessions/{session_id}/agent/cat",
            params=params,
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()

@mcp.tool(description="Write content to a file with overwrite or append modes")
async def write_file(
    file_path: str,
    content: str,
    mode: str = "overwrite",
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Write content to a file
    
    Create new file or update existing file with specified content. Supports both
    overwrite and append modes.
    
    Args:
        file_path: Path where to write the file
        content: Content to write to the file
        mode: Write mode - "overwrite" or "append"
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{AI_AGENT_BASE_URL}/sessions/{session_id}/agent/write",
            params={"file_path": file_path, "content": content, "mode": mode},
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()

@mcp.tool(description="Update file by inserting code at specific markers or line numbers")
async def update_file_at_points(
    file_path: str,
    insertions: List[Dict[str, Any]],
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Update a file by inserting code at specific points.
    
    Insert code at specific markers or line numbers. Supports multiple insertion points
    with before, after, or replace modes.

    Args:
        file_path: Path to the file to update (relative to /workspace/repo).
        insertions: List of dicts, each with:
            - 'insertion_point': str or int (marker text or line number)
            - 'code': str (code to insert)
            - 'mode': str (either 'after', 'before', or 'replace'; default 'after')
        session_id: Optional session context.

    Returns:
        Dict with update status and details.
    """
    session_id = await session_manager.get_or_create_session(session_id)
    abs_path = os.path.join("/workspace/repo", file_path)
    if not os.path.isfile(abs_path):
        return {"success": False, "error": f"File not found: {file_path}"}

    # Read file
    with open(abs_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Process insertions
    def get_line_index(ins):
        if isinstance(ins["insertion_point"], int):
            return ins["insertion_point"]
        # If marker, find first occurrence
        for idx, line in enumerate(lines):
            if ins["insertion_point"] in line:
                return idx
        return -1

    # Compute all insertion indices
    for ins in insertions:
        ins["index"] = get_line_index(ins)

    # Sort insertions by index descending so earlier insertions don't affect later ones
    insertions_sorted = sorted(insertions, key=lambda x: x["index"], reverse=True)

    updated_lines = lines[:]
    for ins in insertions_sorted:
        idx = ins["index"]
        code = ins["code"]
        mode = ins.get("mode", "after")
        if idx == -1:
            continue  # Skip if marker not found

        if mode == "after":
            updated_lines = updated_lines[: idx + 1] + [code + "\n"] + updated_lines[idx + 1 :]
        elif mode == "before":
            updated_lines = updated_lines[:idx] + [code + "\n"] + updated_lines[idx:]
        elif mode == "replace":
            updated_lines = updated_lines[:idx] + [code + "\n"] + updated_lines[idx + 1 :]

    # Write back to file
    with open(abs_path, "w", encoding="utf-8") as f:
        f.writelines(updated_lines)

    return {
        "success": True,
        "file_path": file_path,
        "insertions_applied": [
            {"insertion_point": ins["insertion_point"], "mode": ins.get("mode", "after"), "applied_at_index": ins["index"]}
            for ins in insertions_sorted if ins["index"] != -1
        ],
        "skipped_insertions": [
            {"insertion_point": ins["insertion_point"], "reason": "marker not found"}
            for ins in insertions_sorted if ins["index"] == -1
        ],
        "session_id": session_id,
    }

# =============================================================================
# ADVANCED FILE EDITING TOOLS
# =============================================================================

@mcp.tool(description="Search and replace text in files with regex support")
async def search_replace_in_file(
    file_path: str,
    search_pattern: str,
    replace_with: str,
    use_regex: bool = False,
    global_replace: bool = True,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Search and replace text in a file
    
    Replace text or patterns in files. Supports both simple text replacement and
    regular expression patterns.
    
    Args:
        file_path: Path to the file to modify
        search_pattern: Text or regex pattern to search for
        replace_with: Replacement text
        use_regex: Whether to treat search_pattern as regex
        global_replace: Replace all occurrences (True) or just first (False)
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    abs_path = os.path.join("/workspace/repo", file_path)
    
    if not os.path.isfile(abs_path):
        return {"success": False, "error": f"File not found: {file_path}"}
    
    with open(abs_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    original_content = content
    replacements_count = 0
    
    try:
        if use_regex:
            if global_replace:
                content, replacements_count = re.subn(search_pattern, replace_with, content)
            else:
                content, replacements_count = re.subn(search_pattern, replace_with, content, count=1)
        else:
            if global_replace:
                replacements_count = content.count(search_pattern)
                content = content.replace(search_pattern, replace_with)
            else:
                if search_pattern in content:
                    content = content.replace(search_pattern, replace_with, 1)
                    replacements_count = 1
        
        # Write back to file
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        return {
            "success": True,
            "file_path": file_path,
            "replacements_made": replacements_count,
            "search_pattern": search_pattern,
            "replace_with": replace_with,
            "use_regex": use_regex,
            "global_replace": global_replace,
            "session_id": session_id
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Search/replace failed: {str(e)}",
            "file_path": file_path
        }

@mcp.tool(description="Delete specific lines from a file by line numbers or patterns")
async def delete_lines_from_file(
    file_path: str,
    line_numbers: Optional[List[int]] = None,
    pattern: Optional[str] = None,
    use_regex: bool = False,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Delete lines from a file
    
    Remove specific lines by line numbers or by matching patterns.
    
    Args:
        file_path: Path to the file to modify
        line_numbers: List of line numbers to delete (1-based)
        pattern: Pattern to match lines for deletion
        use_regex: Whether pattern is regex
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    abs_path = os.path.join("/workspace/repo", file_path)
    
    if not os.path.isfile(abs_path):
        return {"success": False, "error": f"File not found: {file_path}"}
    
    with open(abs_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    lines_to_delete = set()
    deleted_count = 0
    
    # Collect line numbers to delete
    if line_numbers:
        for line_num in line_numbers:
            if 1 <= line_num <= len(lines):
                lines_to_delete.add(line_num - 1)  # Convert to 0-based
    
    if pattern:
        for i, line in enumerate(lines):
            if use_regex:
                if re.search(pattern, line):
                    lines_to_delete.add(i)
            else:
                if pattern in line:
                    lines_to_delete.add(i)
    
    # Keep lines that are not in the delete set
    filtered_lines = [line for i, line in enumerate(lines) if i not in lines_to_delete]
    deleted_count = len(lines) - len(filtered_lines)
    
    # Write back to file
    with open(abs_path, "w", encoding="utf-8") as f:
        f.writelines(filtered_lines)
    
    return {
        "success": True,
        "file_path": file_path,
        "lines_deleted": deleted_count,
        "original_line_count": len(lines),
        "new_line_count": len(filtered_lines),
        "session_id": session_id
    }

@mcp.tool(description="Insert lines at specific positions in a file")
async def insert_lines_in_file(
    file_path: str,
    insertions: List[Dict[str, Any]],
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Insert lines at specific positions in a file
    
    Add new lines at specified line numbers or after/before matching patterns.
    
    Args:
        file_path: Path to the file to modify
        insertions: List of dicts with:
            - 'line_number': int (1-based line number) OR 'pattern': str
            - 'content': str (content to insert)
            - 'position': str ('before', 'after') - default 'after'
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    abs_path = os.path.join("/workspace/repo", file_path)
    
    if not os.path.isfile(abs_path):
        return {"success": False, "error": f"File not found: {file_path}"}
    
    with open(abs_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Process insertions in reverse order to maintain line numbers
    insertions_processed = []
    for insertion in sorted(insertions, key=lambda x: x.get('line_number', 0), reverse=True):
        line_num = insertion.get('line_number')
        pattern = insertion.get('pattern')
        content = insertion['content']
        position = insertion.get('position', 'after')
        
        if line_num:
            # Insert at specific line number
            if 1 <= line_num <= len(lines) + 1:
                insert_idx = line_num - 1 if position == 'before' else line_num
                lines.insert(insert_idx, content + '\n')
                insertions_processed.append({"line_number": line_num, "position": position})
        
        elif pattern:
            # Find pattern and insert
            for i, line in enumerate(lines):
                if pattern in line:
                    insert_idx = i if position == 'before' else i + 1
                    lines.insert(insert_idx, content + '\n')
                    insertions_processed.append({"pattern": pattern, "position": position, "found_at_line": i + 1})
                    break
    
    # Write back to file
    with open(abs_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    return {
        "success": True,
        "file_path": file_path,
        "insertions_applied": insertions_processed,
        "new_line_count": len(lines),
        "session_id": session_id
    }

# =============================================================================
# FILE MANAGEMENT TOOLS
# =============================================================================

@mcp.tool(description="Copy files or directories to new locations")
async def copy_file_or_directory(
    source_path: str,
    destination_path: str,
    overwrite: bool = False,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Copy files or directories
    
    Copy files or entire directories to new locations with optional overwrite.
    
    Args:
        source_path: Source file or directory path
        destination_path: Destination path
        overwrite: Whether to overwrite existing files
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    command = f"cp -r {'--force' if overwrite else ''} '{source_path}' '{destination_path}'"
    
    return await execute_shell_command(command, "/workspace/repo", 30, session_id)

@mcp.tool(description="Move or rename files and directories")
async def move_file_or_directory(
    source_path: str,
    destination_path: str,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Move or rename files and directories
    
    Move files/directories to new locations or rename them.
    
    Args:
        source_path: Source file or directory path
        destination_path: Destination path
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    command = f"mv '{source_path}' '{destination_path}'"
    
    return await execute_shell_command(command, "/workspace/repo", 30, session_id)

@mcp.tool(description="Delete files or directories")
async def delete_file_or_directory(
    path: str,
    recursive: bool = False,
    force: bool = False,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Delete files or directories
    
    Remove files or directories with optional recursive and force options.
    
    Args:
        path: Path to file or directory to delete
        recursive: Delete directories recursively
        force: Force deletion without prompts
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    flags = []
    if recursive:
        flags.append("-r")
    if force:
        flags.append("-f")
    
    command = f"rm {' '.join(flags)} '{path}'"
    
    return await execute_shell_command(command, "/workspace/repo", 30, session_id)

@mcp.tool(description="Create new directories with optional parent directory creation")
async def create_directory(
    path: str,
    parents: bool = True,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create directories
    
    Create new directories with optional parent directory creation.
    
    Args:
        path: Directory path to create
        parents: Create parent directories if they don't exist
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    command = f"mkdir {'-p' if parents else ''} '{path}'"
    
    return await execute_shell_command(command, "/workspace/repo", 30, session_id)

# =============================================================================
# SEARCH AND ANALYSIS TOOLS
# =============================================================================

@mcp.tool(description="Search for patterns in files with powerful grep functionality")
async def search_in_files(
    pattern: str,
    path: str = "/workspace/repo",
    file_pattern: str = "*",
    ignore_case: bool = False,
    line_numbers: bool = True,
    max_results: int = 50,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Search for patterns in files (equivalent to grep command)
    
    Search for text patterns across multiple files with advanced options.
    
    Args:
        pattern: Search pattern/regex
        path: Path to search in
        file_pattern: File pattern to include (e.g., "*.py")
        ignore_case: Ignore case in search
        line_numbers: Show line numbers in results
        max_results: Maximum number of results
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{AI_AGENT_BASE_URL}/sessions/{session_id}/agent/grep",
            params={
                "pattern": pattern, "path": path, "file_pattern": file_pattern,
                "ignore_case": ignore_case, "line_numbers": line_numbers, "max_results": max_results
            },
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()

@mcp.tool(description="Show directory structure in tree format")
async def show_directory_tree(
    path: str = "/workspace/repo",
    max_depth: int = 3,
    show_files: bool = True,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Show directory tree structure (equivalent to tree command)
    
    Display directory structure in a tree format with configurable depth and file visibility.
    
    Args:
        path: Root path for tree view
        max_depth: Maximum depth to show
        show_files: Include files in tree view
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{AI_AGENT_BASE_URL}/sessions/{session_id}/agent/tree",
            params={"path": path, "max_depth": max_depth, "show_files": show_files},
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()

@mcp.tool(description="Get comprehensive file information including size, permissions, and timestamps")
async def get_file_info(
    file_path: str,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get detailed file information (equivalent to stat command)
    
    Get comprehensive information about files including size, permissions, timestamps,
    and file type.
    
    Args:
        file_path: File path to get information about
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{AI_AGENT_BASE_URL}/sessions/{session_id}/agent/file-info",
            params={"file_path": file_path},
            timeout=15.0
        )
        response.raise_for_status()
        return response.json()

# =============================================================================
# COMMAND EXECUTION TOOLS
# =============================================================================

@mcp.tool(description="Execute any shell command in the sandbox environment")
async def execute_shell_command(
    command: str,
    working_dir: str = "/workspace/repo",
    timeout: int = 30,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Execute any shell command in the sandbox
    
    Run arbitrary shell commands with configurable working directory and timeout.
    
    Args:
        command: Shell command to execute
        working_dir: Working directory for command execution
        timeout: Command timeout in seconds
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{AI_AGENT_BASE_URL}/sessions/{session_id}/agent/command",
            params={"command": command, "working_dir": working_dir, "timeout": timeout},
            timeout=float(timeout + 10)
        )
        response.raise_for_status()
        return response.json()

@mcp.tool(description="Execute multiple shell commands in sequence with error handling")
async def execute_multiple_commands(
    commands: List[str],
    working_dir: str = "/workspace/repo",
    stop_on_error: bool = True,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Execute multiple shell commands sequentially
    
    Run a series of commands in order with configurable error handling.
    
    Args:
        commands: List of commands to execute in order
        working_dir: Working directory for command execution
        stop_on_error: Stop execution if any command fails
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{AI_AGENT_BASE_URL}/sessions/{session_id}/agent/multi-command",
            params={"commands": commands, "working_dir": working_dir, "stop_on_error": stop_on_error},
            timeout=120.0
        )
        response.raise_for_status()
        return response.json()

# =============================================================================
# GIT OPERATIONS TOOLS
# =============================================================================

@mcp.tool(description="Get git repository status showing modified, added, and deleted files")
async def git_status(
    path: str = "/workspace/repo",
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get git repository status
    
    Show the current status of the git repository including staged, modified, and untracked files.
    
    Args:
        path: Repository path
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    command = "git status --porcelain"
    return await execute_shell_command(command, path, 30, session_id)

@mcp.tool(description="Show git commit history with configurable number of commits")
async def git_log(
    max_commits: int = 10,
    oneline: bool = True,
    path: str = "/workspace/repo",
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Show git commit history
    
    Display recent commit history with configurable format and number of commits.
    
    Args:
        max_commits: Maximum number of commits to show
        oneline: Show commits in one-line format
        path: Repository path
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    command = f"git log {'--oneline' if oneline else ''} -n {max_commits}"
    return await execute_shell_command(command, path, 30, session_id)

@mcp.tool(description="Show differences between files or commits")
async def git_diff(
    file_path: Optional[str] = None,
    staged: bool = False,
    path: str = "/workspace/repo",
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Show git diff
    
    Display differences between working directory and index, or between commits.
    
    Args:
        file_path: Specific file to diff (optional)
        staged: Show staged changes instead of working directory
        path: Repository path
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    command = f"git diff {'--staged' if staged else ''} {file_path or ''}"
    return await execute_shell_command(command, path, 30, session_id)

@mcp.tool(description="Stage files for commit")
async def git_add(
    files: List[str],
    path: str = "/workspace/repo",
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Stage files for commit
    
    Add files to the staging area for the next commit.
    
    Args:
        files: List of file paths to stage
        path: Repository path
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    command = f"git add {' '.join(files)}"
    return await execute_shell_command(command, path, 30, session_id)

@mcp.tool(description="Create a git commit with message")
async def git_commit(
    message: str,
    path: str = "/workspace/repo",
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create a git commit
    
    Commit staged changes with the provided commit message.
    
    Args:
        message: Commit message
        path: Repository path
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    command = f"git commit -m '{message}'"
    return await execute_shell_command(command, path, 30, session_id)

# =============================================================================
# CODE ANALYSIS TOOLS
# =============================================================================

@mcp.tool(description="Count lines of code in files by extension")
async def count_lines_of_code(
    path: str = "/workspace/repo",
    extensions: List[str] = ["py", "js", "ts", "java", "cpp", "c"],
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Count lines of code by file type
    
    Count total lines, code lines, and comment lines for different file types.
    
    Args:
        path: Directory to analyze
        extensions: File extensions to include
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    results = {}
    for ext in extensions:
        command = f"find {path} -name '*.{ext}' -exec wc -l {{}} + 2>/dev/null || echo '0 total'"
        result = await execute_shell_command(command, "/", 30, session_id)
        results[ext] = result
    
    return {
        "success": True,
        "path": path,
        "extensions_analyzed": extensions,
        "results": results,
        "session_id": session_id
    }

@mcp.tool(description="Find TODO, FIXME, and other code comments")
async def find_code_comments(
    path: str = "/workspace/repo",
    keywords: List[str] = ["TODO", "FIXME", "HACK", "NOTE", "BUG"],
    file_pattern: str = "*",
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Find code comments with specific keywords
    
    Search for TODO, FIXME, and other special comments in code files.
    
    Args:
        path: Directory to search
        keywords: Keywords to search for in comments
        file_pattern: File pattern to search in
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    results = {}
    for keyword in keywords:
        search_result = await search_in_files(
            pattern=keyword,
            path=path,
            file_pattern=file_pattern,
            ignore_case=True,
            max_results=100,
            session_id=session_id
        )
        results[keyword] = search_result
    
    return {
        "success": True,
        "path": path,
        "keywords_searched": keywords,
        "results": results,
        "session_id": session_id
    }

@mcp.tool(description="Analyze file sizes and find large files")
async def analyze_file_sizes(
    path: str = "/workspace/repo",
    min_size: str = "1M",
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze file sizes in repository
    
    Find large files and analyze disk usage patterns.
    
    Args:
        path: Directory to analyze
        min_size: Minimum file size to report (e.g., "1M", "500K")
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    commands = [
        f"find {path} -type f -size +{min_size} -exec ls -lh {{}} + | sort -k5 -hr",
        f"du -h {path} | sort -hr | head -20"
    ]
    
    result = await execute_multiple_commands(commands, "/", False, session_id)
    
    return {
        "success": True,
        "path": path,
        "min_size": min_size,
        "analysis": result,
        "session_id": session_id
    }

# =============================================================================
# ENVIRONMENT AND SYSTEM TOOLS
# =============================================================================

@mcp.tool(description="Get comprehensive environment and system information")
async def get_environment_info(session_id: Optional[str] = None) -> Dict[str, Any]:
    """Get comprehensive environment information
    
    Retrieve detailed information about the system environment, installed tools,
    and available resources.
    
    Args:
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{AI_AGENT_BASE_URL}/sessions/{session_id}/agent/environment",
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()

@mcp.tool(description="Get GitHub issue context and repository information")
async def get_issue_context(session_id: Optional[str] = None) -> Dict[str, Any]:
    """Get complete context about the GitHub issue and repository
    
    Retrieve comprehensive context about the current GitHub issue being worked on
    and repository structure.
    
    Args:
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{AI_AGENT_BASE_URL}/sessions/{session_id}/agent/issue-context",
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()

# =============================================================================
# COMPOSITE WORKFLOW TOOLS
# =============================================================================

@mcp.tool(description="Perform comprehensive repository structure analysis")
async def analyze_repository_structure(
    path: str = "/workspace/repo",
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze repository structure and provide comprehensive overview
    
    Get a complete analysis of repository structure including file types, sizes,
    dependencies, and organization patterns.
    
    Args:
        path: Repository root path
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    results = {}
    
    try:
        # Get directory tree
        results["tree_structure"] = await show_directory_tree(path, max_depth=2, show_files=True, session_id=session_id)
        
        # Find key files
        results["readme_files"] = await find_files("README*", "f", path, 5, session_id)
        results["python_files"] = await find_files("*.py", "f", path, 20, session_id)
        results["config_files"] = await find_files("*.json", "f", path, 10, session_id)
        
        # Get environment info
        results["environment"] = await get_environment_info(session_id)
        
        # Count lines of code
        results["code_stats"] = await count_lines_of_code(path, ["py", "js", "ts"], session_id)
        
        return {
            "analysis_complete": True,
            "repository_path": path,
            "analysis_results": results,
            "session_id": session_id
        }
    
    except Exception as e:
        return {
            "analysis_complete": False,
            "error": str(e),
            "partial_results": results
        }

@mcp.tool(description="Perform quick multi-pattern code search across file types")
async def quick_code_scan(
    search_terms: List[str],
    file_patterns: List[str] = ["*.py", "*.js", "*.ts"],
    path: str = "/workspace/repo",
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Perform quick code scan for specific terms across multiple file types
    
    Search for multiple patterns across different file types simultaneously.
    
    Args:
        search_terms: List of terms/patterns to search for
        file_patterns: List of file patterns to search in
        path: Path to search in
        session_id: Session identifier for context
    """
    session_id = await session_manager.get_or_create_session(session_id)
    
    scan_results = {}
    
    for term in search_terms:
        term_results = {}
        for pattern in file_patterns:
            try:
                search_result = await search_in_files(
                    pattern=term, path=path, file_pattern=pattern,
                    ignore_case=True, max_results=20, session_id=session_id
                )
                term_results[pattern] = search_result
            except Exception as e:
                term_results[pattern] = {"error": str(e)}
        scan_results[term] = term_results
    
    return {
        "scan_complete": True,
        "search_terms": search_terms,
        "file_patterns": file_patterns,
        "results": scan_results,
        "session_id": session_id
    }

# =============================================================================
# SESSION MANAGEMENT TOOLS
# =============================================================================

@mcp.tool(description="Create a new isolated session for operations")
async def create_new_session() -> Dict[str, Any]:
    """Create a new session for isolated operations
    
    Create a fresh session for isolated work without interfering with other sessions.
    """
    new_session_id = f"mcp_session_{uuid.uuid4().hex[:8]}"
    await session_manager.get_or_create_session(new_session_id)
    
    return {
        "session_id": new_session_id,
        "created_at": datetime.now().isoformat(),
        "status": "created"
    }

@mcp.tool(description="List all active MCP sessions")
async def list_active_sessions() -> Dict[str, Any]:
    """List all active MCP sessions
    
    Show all currently active sessions with their details and activity status.
    """
    return {
        "active_sessions": list(session_manager.sessions.keys()),
        "total_sessions": len(session_manager.sessions),
        "sessions_detail": session_manager.sessions
    }

# =============================================================================
# RESOURCES (Read-only data)
# =============================================================================

@mcp.resource("session://{session_id}/context")
async def get_session_context(session_id: str) -> str:
    """Get session context information"""
    try:
        context = await get_issue_context(session_id)
        return json.dumps(context, indent=2)
    except:
        return f"No context available for session {session_id}"

@mcp.resource("workspace://file-tree")
async def get_workspace_tree() -> str:
    """Get workspace file tree"""
    try:
        tree = await show_directory_tree("/workspace/repo", max_depth=3, show_files=True)
        return tree.get("tree_output", "No tree available")
    except:
        return "Unable to generate file tree"

@mcp.resource("environment://info")
async def get_env_resource() -> str:
    """Get environment information as resource"""
    try:
        env = await get_environment_info()
        return json.dumps(env, indent=2)
    except:
        return "Environment information not available"
    
    
@mcp.custom_route("/list-tools", methods=["GET"])
async def list_all_tools():
    """Direct HTTP endpoint to list all tools"""
    tools = []
    for tool in mcp._tool_manager.list_tools():
        tools.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters
        })
    
    return {
        "total_tools": len(tools),
        "tools": tools
    }

# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("🚀 Starting Enhanced AI Agent Tools MCP Server...")
    print(f"📡 Connecting to AI Agent endpoints at: {AI_AGENT_BASE_URL}")
    print("🔧 Available tools (30 total):")
    
    tools = [
        # File System Operations (7)
        "list_directory", "get_working_directory", "find_files", "show_directory_tree",
        "get_file_info", "create_directory", "analyze_file_sizes",
        
        # File Content Operations (8)
        "read_file", "write_file", "update_file_at_points", "search_replace_in_file",
        "delete_lines_from_file", "insert_lines_in_file", "search_in_files", "copy_file_or_directory",
        
        # File Management (3)
        "move_file_or_directory", "delete_file_or_directory", "count_lines_of_code",
        
        # Command Execution (2)
        "execute_shell_command", "execute_multiple_commands",
        
        # Git Operations (5)
        "git_status", "git_log", "git_diff", "git_add", "git_commit",
        
        # Code Analysis (2)
        "find_code_comments", "quick_code_scan",
        
        # Environment & Context (2)
        "get_environment_info", "get_issue_context",
        
        # Composite Workflows (1)
        "analyze_repository_structure",
        
        # Session Management (2)
        "create_new_session", "list_active_sessions"
    ]
    
    for i, tool in enumerate(tools, 1):
        print(f"   {i:2d}. {tool}")
    
    print(f"\n📊 Total Tools: {len(tools)}")
    print("\n🎯 Tool Categories:")
    print("   • File System Operations: 7 tools")
    print("   • File Content Operations: 8 tools")
    print("   • File Management: 3 tools")
    print("   • Command Execution: 2 tools")
    print("   • Git Operations: 5 tools")
    print("   • Code Analysis: 2 tools")
    print("   • Environment & Context: 2 tools")
    print("   • Workflow Automation: 1 tool")
    print("   • Session Management: 2 tools")
    
    print("\n📖 To install in Claude Desktop:")
    print("   fastmcp install enhanced_fastmcp_ai_agent_tools.py")
    print("\n⚙️  Make sure your AI agent endpoints are running on localhost:8001")
    
    # Run the server
    mcp.run(transport="http", port=8002, host="0.0.0.0")