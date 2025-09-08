# ai_agent_endpoints.py
"""
Standalone AI Agent Tool Endpoints FastAPI Application
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import logging
import os
import stat
from pathlib import Path

from sandbox_manager import sandbox_manager, SessionContext

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Agent Tools API",
    description="AI Agent tool endpoints for GitHub issue solving",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session context dependency
async def get_session(session_id: str) -> SessionContext:
    """Get session context for AI agent operations"""
    if session_id not in sandbox_manager.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sandbox_manager.sessions[session_id]

# =============================================================================
# AI AGENT TOOL ENDPOINTS
# =============================================================================

@app.get("/sessions/{session_id}/agent/ls")
async def agent_list_directory(
    session_id: str,
    path: str = Query("/workspace/repo", description="Directory path to list"),
    detailed: bool = Query(False, description="Include file details (like ls -la)"),
    recursive: bool = Query(False, description="List recursively (like ls -R)"),
    session_context: SessionContext = Depends(get_session)
):
    """
    AI Agent Tool: List directory contents (like ls command)
    Designed for AI agents to explore file structure programmatically
    """
    
    try:
        if recursive:
            command = f"find {path} -type f -o -type d | head -500"  # Limit to prevent overwhelming output
        elif detailed:
            command = f"ls -la {path}"
        else:
            command = f"ls {path}"
        
        request = ExecuteCodeRequest(
            command=command,
            working_dir="/workspace",
            user_id="ai_agent"
        )
        
        result = await execute_code(session_id, request, session_context)
        
        # Parse output for structured response
        if result.exit_code == 0:
            lines = result.stdout.strip().split('\n')
            
            if recursive:
                # For find command, organize by directory
                files = []
                directories = []
                for line in lines:
                    if line.strip():
                        # Check if it's a directory
                        check_cmd = f"test -d '{line}' && echo 'dir' || echo 'file'"
                        check_result = sandbox_manager.execute_code(session_id, check_cmd, "/workspace", "ai_agent")
                        
                        item_type = check_result.get('stdout', '').strip()
                        if item_type == 'dir':
                            directories.append(line)
                        else:
                            files.append(line)
                
                return {
                    "command": command,
                    "success": True,
                    "path": path,
                    "directories": directories,
                    "files": files,
                    "total_items": len(lines),
                    "raw_output": result.stdout
                }
            
            elif detailed:
                # Parse ls -la output
                items = []
                for line in lines[1:]:  # Skip first line (total)
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 9:
                            items.append({
                                "permissions": parts[0],
                                "links": parts[1],
                                "owner": parts[2],
                                "group": parts[3],
                                "size": parts[4],
                                "date": f"{parts[5]} {parts[6]} {parts[7]}",
                                "name": " ".join(parts[8:]),
                                "type": "directory" if parts[0].startswith('d') else "file"
                            })
                
                return {
                    "command": command,
                    "success": True,
                    "path": path,
                    "items": items,
                    "total_items": len(items),
                    "raw_output": result.stdout
                }
            
            else:
                # Simple ls output
                items = [item.strip() for item in lines if item.strip()]
                return {
                    "command": command,
                    "success": True,
                    "path": path,
                    "items": items,
                    "total_items": len(items),
                    "raw_output": result.stdout
                }
        
        else:
            return {
                "command": command,
                "success": False,
                "error": result.stderr,
                "path": path
            }
    
    except Exception as e:
        logger.error(f"Agent ls command failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list directory: {str(e)}")


@app.get("/sessions/{session_id}/agent/pwd")
async def agent_get_working_directory(
    session_id: str,
    session_context: SessionContext = Depends(get_session)
):
    """AI Agent Tool: Get current working directory"""
    
    try:
        request = ExecuteCodeRequest(
            command="pwd",
            working_dir="/workspace/repo",
            user_id="ai_agent"
        )
        
        result = await execute_code(session_id, request, session_context)
        
        return {
            "current_directory": result.stdout.strip(),
            "success": result.exit_code == 0,
            "command": "pwd"
        }
    
    except Exception as e:
        logger.error(f"Agent pwd command failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get working directory: {str(e)}")


@app.get("/sessions/{session_id}/agent/find")
async def agent_find_files(
    session_id: str,
    pattern: str = Query("*", description="File pattern to search for"),
    file_type: str = Query("f", description="Type: f (file), d (directory), l (link)"),
    path: str = Query("/workspace/repo", description="Path to search in"),
    max_results: int = Query(100, description="Maximum number of results"),
    session_context: SessionContext = Depends(get_session)
):
    """AI Agent Tool: Find files and directories (like find command)"""
    
    try:
        command = f"find {path} -type {file_type} -name '{pattern}' | head -{max_results}"
        
        request = ExecuteCodeRequest(
            command=command,
            working_dir="/workspace",
            user_id="ai_agent"
        )
        
        result = await execute_code(session_id, request, session_context)
        
        if result.exit_code == 0:
            found_items = [item.strip() for item in result.stdout.split('\n') if item.strip()]
            
            return {
                "command": command,
                "success": True,
                "pattern": pattern,
                "file_type": file_type,
                "search_path": path,
                "found_items": found_items,
                "count": len(found_items),
                "truncated": len(found_items) == max_results
            }
        else:
            return {
                "command": command,
                "success": False,
                "error": result.stderr,
                "pattern": pattern,
                "found_items": []
            }
    
    except Exception as e:
        logger.error(f"Agent find command failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to find files: {str(e)}")


@app.post("/sessions/{session_id}/agent/command")
async def agent_execute_command(
    session_id: str,
    command: str = Query(..., description="Command to execute"),
    working_dir: str = Query("/workspace/repo", description="Working directory"),
    timeout: int = Query(30, description="Command timeout in seconds"),
    session_context: SessionContext = Depends(get_session)
):
    """
    AI Agent Tool: Execute any shell command
    Enhanced version of execute endpoint specifically for AI agents
    """
    
    try:
        # Log that this is an AI agent action
        logger.info(f"AI Agent executing command: {command} in {working_dir}")
        
        result = sandbox_manager.execute_code(
            session_id=session_id,
            command=command,
            working_dir=working_dir,
            user_id="ai_agent"
        )
        
        return {
            "command": command,
            "working_directory": working_dir,
            "exit_code": result["exit_code"],
            "success": result["exit_code"] == 0,
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "execution_time": result["execution_time"],
            "timestamp": result["timestamp"]
        }
    
    except Exception as e:
        logger.error(f"Agent command execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to execute command: {str(e)}")


@app.get("/sessions/{session_id}/agent/cat")
async def agent_read_file(
    session_id: str,
    file_path: str = Query(..., description="File path to read"),
    lines: Optional[int] = Query(None, description="Number of lines to read (head -n)"),
    tail: Optional[int] = Query(None, description="Read from end (tail -n)"),
    session_context: SessionContext = Depends(get_session)
):
    """AI Agent Tool: Read file contents (like cat/head/tail commands)"""
    
    try:
        if lines:
            command = f"head -n {lines} '{file_path}'"
        elif tail:
            command = f"tail -n {tail} '{file_path}'"
        else:
            command = f"cat '{file_path}'"
        
        request = ExecuteCodeRequest(
            command=command,
            working_dir="/workspace",
            user_id="ai_agent"
        )
        
        result = await execute_code(session_id, request, session_context)
        
        if result.exit_code == 0:
            # Get file info
            stat_command = f"stat -c '%s %Y' '{file_path}'"
            stat_request = ExecuteCodeRequest(command=stat_command, working_dir="/workspace", user_id="ai_agent")
            stat_result = await execute_code(session_id, stat_request, session_context)
            
            size, modified = 0, 0
            if stat_result.exit_code == 0:
                try:
                    size, modified = map(int, stat_result.stdout.strip().split())
                except:
                    pass
            
            return {
                "file_path": file_path,
                "content": result.stdout,
                "success": True,
                "size_bytes": size,
                "modified_timestamp": modified,
                "lines_read": lines or tail,
                "command_used": command
            }
        else:
            return {
                "file_path": file_path,
                "success": False,
                "error": result.stderr,
                "command_used": command
            }
    
    except Exception as e:
        logger.error(f"Agent cat command failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")


@app.post("/sessions/{session_id}/agent/write")
async def agent_write_file(
    session_id: str,
    file_path: str = Query(..., description="File path to write to"),
    content: str = Query(..., description="Content to write"),
    mode: str = Query("overwrite", description="Mode: overwrite, append"),
    session_context: SessionContext = Depends(get_session)
):
    """AI Agent Tool: Write content to file"""
    
    try:
        # Escape content for shell
        escaped_content = content.replace("'", "'\"'\"'")
        
        if mode == "append":
            command = f"echo '{escaped_content}' >> '{file_path}'"
        else:
            command = f"echo '{escaped_content}' > '{file_path}'"
        
        request = ExecuteCodeRequest(
            command=command,
            working_dir="/workspace",
            user_id="ai_agent"
        )
        
        result = await execute_code(session_id, request, session_context)
        
        return {
            "file_path": file_path,
            "success": result.exit_code == 0,
            "mode": mode,
            "content_length": len(content),
            "error": result.stderr if result.exit_code != 0 else None
        }
    
    except Exception as e:
        logger.error(f"Agent write command failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to write file: {str(e)}")


@app.get("/sessions/{session_id}/agent/grep")
async def agent_grep_search(
    session_id: str,
    pattern: str = Query(..., description="Search pattern"),
    path: str = Query("/workspace/repo", description="Path to search in"),
    file_pattern: str = Query("*", description="File pattern to include"),
    ignore_case: bool = Query(False, description="Ignore case in search"),
    line_numbers: bool = Query(True, description="Show line numbers"),
    max_results: int = Query(50, description="Maximum results to return"),
    session_context: SessionContext = Depends(get_session)
):
    """AI Agent Tool: Search for patterns in files (grep command)"""
    
    try:
        # Build grep command
        grep_options = []
        if ignore_case:
            grep_options.append("-i")
        if line_numbers:
            grep_options.append("-n")
        
        grep_opts = " ".join(grep_options)
        command = f"grep -r {grep_opts} '{pattern}' --include='{file_pattern}' {path} | head -{max_results}"
        
        request = ExecuteCodeRequest(
            command=command,
            working_dir="/workspace",
            user_id="ai_agent"
        )
        
        result = await execute_code(session_id, request, session_context)
        
        matches = []
        if result.exit_code == 0 and result.stdout:
            for line in result.stdout.strip().split('\n'):
                if ':' in line and line.strip():
                    parts = line.split(':', 2)
                    if len(parts) >= 2:
                        file_path = parts[0].replace(f"{path}/", "")
                        if line_numbers and len(parts) >= 3:
                            line_num = parts[1]
                            content = parts[2]
                        else:
                            line_num = None
                            content = parts[1]
                        
                        matches.append({
                            "file": file_path,
                            "line_number": line_num,
                            "content": content.strip(),
                            "full_line": line
                        })
        
        return {
            "pattern": pattern,
            "search_path": path,
            "file_pattern": file_pattern,
            "matches": matches,
            "total_matches": len(matches),
            "truncated": len(matches) == max_results,
            "command": command,
            "success": result.exit_code == 0
        }
    
    except Exception as e:
        logger.error(f"Agent grep command failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to search files: {str(e)}")


@app.get("/sessions/{session_id}/agent/tree")
async def agent_tree_view(
    session_id: str,
    path: str = Query("/workspace/repo", description="Root path for tree view"),
    max_depth: int = Query(3, description="Maximum depth to show"),
    show_files: bool = Query(True, description="Show files in tree"),
    session_context: SessionContext = Depends(get_session)
):
    """AI Agent Tool: Show directory tree structure (like tree command)"""
    
    try:
        # Build tree command (install tree if not available, fallback to find)
        tree_cmd = f"tree -L {max_depth} {'-a' if show_files else '-d'} {path}"
        fallback_cmd = f"find {path} -maxdepth {max_depth} {'.' if show_files else '-type d'} | sort"
        
        # Try tree command first
        request = ExecuteCodeRequest(
            command=f"{tree_cmd} 2>/dev/null || {fallback_cmd}",
            working_dir="/workspace",
            user_id="ai_agent"
        )
        
        result = await execute_code(session_id, request, session_context)
        
        if result.exit_code == 0:
            return {
                "path": path,
                "max_depth": max_depth,
                "show_files": show_files,
                "tree_output": result.stdout,
                "success": True
            }
        else:
            return {
                "path": path,
                "success": False,
                "error": result.stderr,
                "tree_output": ""
            }
    
    except Exception as e:
        logger.error(f"Agent tree command failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate tree view: {str(e)}")


@app.get("/sessions/{session_id}/agent/file-info")
async def agent_file_info(
    session_id: str,
    file_path: str = Query(..., description="File path to get info for"),
    session_context: SessionContext = Depends(get_session)
):
    """AI Agent Tool: Get detailed file information (like stat command)"""
    
    try:
        # Get comprehensive file information
        command = f"stat '{file_path}' && file '{file_path}' && wc -l '{file_path}'"
        
        request = ExecuteCodeRequest(
            command=command,
            working_dir="/workspace",
            user_id="ai_agent"
        )
        
        result = await execute_code(session_id, request, session_context)
        
        if result.exit_code == 0:
            lines = result.stdout.strip().split('\n')
            
            # Parse stat output (first several lines)
            stat_info = {}
            file_type = ""
            line_count = 0
            
            for line in lines:
                if 'Size:' in line:
                    parts = line.split()
                    stat_info['size_bytes'] = int(parts[1])
                elif 'Access:' in line and 'Uid:' not in line:
                    stat_info['permissions'] = line.split('(')[1].split('/')[0]
                elif 'Modify:' in line:
                    stat_info['modified'] = line.split('Modify: ')[1]
                elif ': ' in line and ('text' in line or 'binary' in line or 'data' in line):
                    file_type = line.split(': ', 1)[1]
                elif line.strip().isdigit():
                    line_count = int(line.strip())
            
            return {
                "file_path": file_path,
                "success": True,
                "file_type": file_type,
                "size_bytes": stat_info.get('size_bytes', 0),
                "permissions": stat_info.get('permissions', ''),
                "modified": stat_info.get('modified', ''),
                "line_count": line_count,
                "raw_output": result.stdout
            }
        else:
            return {
                "file_path": file_path,
                "success": False,
                "error": result.stderr,
                "exists": False
            }
    
    except Exception as e:
        logger.error(f"Agent file-info command failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get file info: {str(e)}")


@app.post("/sessions/{session_id}/agent/multi-command")
async def agent_execute_multiple_commands(
    session_id: str,
    commands: List[str] = Query(..., description="List of commands to execute sequentially"),
    working_dir: str = Query("/workspace/repo", description="Working directory"),
    stop_on_error: bool = Query(True, description="Stop execution if any command fails"),
    session_context: SessionContext = Depends(get_session)
):
    """AI Agent Tool: Execute multiple commands sequentially"""
    
    try:
        results = []
        
        for i, command in enumerate(commands):
            logger.info(f"AI Agent executing command {i+1}/{len(commands)}: {command}")
            
            result = sandbox_manager.execute_code(
                session_id=session_id,
                command=command,
                working_dir=working_dir,
                user_id="ai_agent"
            )
            
            cmd_result = {
                "command": command,
                "sequence": i + 1,
                "exit_code": result["exit_code"],
                "success": result["exit_code"] == 0,
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "execution_time": result["execution_time"]
            }
            
            results.append(cmd_result)
            
            # Stop on error if requested
            if stop_on_error and result["exit_code"] != 0:
                break
        
        overall_success = all(r["success"] for r in results)
        
        return {
            "commands": commands,
            "results": results,
            "total_commands": len(commands),
            "executed_commands": len(results),
            "overall_success": overall_success,
            "stopped_on_error": len(results) < len(commands) and stop_on_error
        }
    
    except Exception as e:
        logger.error(f"Agent multi-command execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to execute commands: {str(e)}")


@app.get("/sessions/{session_id}/agent/environment")
async def agent_get_environment_info(
    session_id: str,
    session_context: SessionContext = Depends(get_session)
):
    """AI Agent Tool: Get comprehensive environment information"""
    
    try:
        # Get various environment details
        commands = {
            "working_directory": "pwd",
            "system_info": "uname -a",
            "python_version": "python3 --version",
            "node_version": "node --version || echo 'Node.js not available'",
            "git_status": "git status --porcelain",
            "git_branch": "git branch --show-current",
            "environment_variables": "env | grep -E '^(PATH|HOME|USER|SHELL)=' | head -10",
            "disk_usage": "df -h /workspace",
            "available_commands": "which python3 node npm git pip"
        }
        
        env_info = {}
        
        for key, command in commands.items():
            request = ExecuteCodeRequest(
                command=command,
                working_dir="/workspace/repo",
                user_id="ai_agent"
            )
            
            result = await execute_code(session_id, request, session_context)
            env_info[key] = {
                "output": result.stdout.strip(),
                "success": result.exit_code == 0
            }
        
        return {
            "session_id": session_id,
            "environment_info": env_info,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Agent environment info failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get environment info: {str(e)}")


# Add this endpoint for AI agents to understand the repository structure and issue context
@app.get("/sessions/{session_id}/agent/issue-context")
async def agent_get_issue_context(
    session_id: str,
    session_context: SessionContext = Depends(get_session)
):
    """AI Agent Tool: Get complete context about the GitHub issue and repository"""
    
    try:
        # Read the session context that was saved during setup
        context_file_path = "/workspace/session_context.json"
        
        with session_context.sandbox.open(context_file_path, "r") as f:
            context_data = json.loads(f.read())
        
        # Get repository structure
        file_tree = sandbox_manager.get_file_tree(session_id, "/workspace/repo")
        
        # Get repository information
        repo_info_commands = {
            "readme_files": "find /workspace/repo -maxdepth 2 -iname 'readme*' -type f",
            "python_files": "find /workspace/repo -name '*.py' | head -20",
            "config_files": "find /workspace/repo -maxdepth 2 -name '*.json' -o -name '*.yaml' -o -name '*.yml' -o -name '*.toml' -o -name '*.cfg' | head -10",
            "test_files": "find /workspace/repo -name '*test*.py' -o -name 'test_*.py' | head -10"
        }
        
        repo_structure = {}
        for key, command in repo_info_commands.items():
            result = sandbox_manager.execute_code(session_id, command, "/workspace", "ai_agent")
            if result["exit_code"] == 0 and result["stdout"].strip():
                repo_structure[key] = result["stdout"].strip().split('\n')
            else:
                repo_structure[key] = []
        
        return {
            "session_context": context_data,
            "repository_structure": repo_structure,
            "file_tree": file_tree,
            "base_commit": context_data.get("base_commit"),
            "issue_info": context_data.get("issue_data", {}),
            "working_directory": "/workspace/repo"
        }
    
    except Exception as e:
        logger.error(f"Agent issue context failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get issue context: {str(e)}")