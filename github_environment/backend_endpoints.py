# backend_endpoints.py
"""
FastAPI Backend Endpoints for Human-AI Collaborative Code Environment
Handles all API endpoints for the web interface
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import logging

from sandbox_manager import sandbox_manager, SessionContext

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GitHub Issue Solver API",
    description="Backend API for human-AI collaborative GitHub issue solving platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response schemas
class CreateSessionRequest(BaseModel):
    github_issue_url: HttpUrl
    user_id: Optional[str] = None

class CreateSessionResponse(BaseModel):
    session_id: str
    github_issue_url: str
    repo_info: Dict[str, Any]
    commits: Dict[str, str]
    created_at: str
    message: str

class FileTreeResponse(BaseModel):
    name: str
    path: str
    type: str
    children: Optional[List[Dict]] = None

class FileContentRequest(BaseModel):
    file_path: str
    content: str
    user_id: Optional[str] = None

class FileContentResponse(BaseModel):
    file_path: str
    content: str
    size: int
    modified: int
    extension: str

class ExecuteCodeRequest(BaseModel):
    command: str
    working_dir: Optional[str] = "/workspace/repo"
    user_id: Optional[str] = None

class ExecuteCodeResponse(BaseModel):
    command: str
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    timestamp: str

class ActivityLogEntry(BaseModel):
    session_id: str
    user_id: Optional[str]
    event_id: str
    timestamp: str
    event_type: str
    data: Dict[str, Any]

class SessionInfo(BaseModel):
    session_id: str
    github_issue_url: str
    repo_info: Dict[str, Any]
    commits: Dict[str, str]
    created_at: str
    activity_count: int

class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: str


# Dependency to validate session
def get_session(session_id: str) -> SessionContext:
    """Dependency to validate and get session context"""
    if session_id not in sandbox_manager.active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return sandbox_manager.active_sessions[session_id]


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "active_sessions": len(sandbox_manager.active_sessions)
    }


# Session Management Endpoints
@app.post("/sessions/create", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest, background_tasks: BackgroundTasks):
    """Create a new coding session with GitHub issue"""
    
    try:
        logger.info(f"Creating session for issue: {request.github_issue_url}")
        
        # Create session (this might take time due to sandbox creation)
        session_context = sandbox_manager.create_session(
            str(request.github_issue_url),
            request.user_id
        )
        
        logger.info(f"Session created successfully: {session_context.session_id}")
        
        return CreateSessionResponse(
            session_id=session_context.session_id,
            github_issue_url=session_context.github_issue_url,
            repo_info={
                "owner": session_context.repo_owner,
                "name": session_context.repo_name,
                "issue_number": session_context.issue_number
            },
            commits={
                "base_commit": session_context.base_commit,
                "solution_commit": session_context.solution_commit or ""
            },
            created_at=session_context.created_at.isoformat(),
            message="Session created successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to create session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@app.get("/sessions", response_model=List[Dict])
async def list_sessions():
    """List all active sessions"""
    
    try:
        sessions = sandbox_manager.list_active_sessions()
        return sessions
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str, session_context: SessionContext = Depends(get_session)):
    """Get information about a specific session"""
    
    try:
        session_info = sandbox_manager.get_session_info(session_id)
        return SessionInfo(**session_info)
        
    except Exception as e:
        logger.error(f"Failed to get session info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get session info: {str(e)}")


@app.delete("/sessions/{session_id}")
async def cleanup_session(session_id: str, background_tasks: BackgroundTasks):
    """Clean up a session and its resources"""
    
    try:
        # Schedule cleanup in background to return response quickly
        background_tasks.add_task(sandbox_manager.cleanup_session, session_id)
        
        return {
            "message": f"Session {session_id} cleanup initiated",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup session: {str(e)}")


# File System Endpoints
@app.get("/sessions/{session_id}/files", response_model=Dict)
async def get_file_tree(
    session_id: str,
    path: str = Query("/workspace/repo", description="Path to get file tree for"),
    session_context: SessionContext = Depends(get_session)
):
    """Get file tree structure for the repository"""
    
    try:
        logger.info(f"Getting file tree for session {session_id}, path: {path}")
        
        file_tree = sandbox_manager.get_file_tree(session_id, path)
        
        return file_tree
        
    except Exception as e:
        logger.error(f"Failed to get file tree: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get file tree: {str(e)}")


@app.get("/sessions/{session_id}/files/content", response_model=FileContentResponse)
async def get_file_content(
    session_id: str,
    file_path: str = Query(..., description="Path to the file to read"),
    session_context: SessionContext = Depends(get_session)
):
    """Get content of a specific file"""
    
    try:
        logger.info(f"Reading file: {file_path} in session {session_id}")
        
        file_data = sandbox_manager.get_file_content(session_id, file_path)
        
        return FileContentResponse(**file_data)
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to read file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")


@app.post("/sessions/{session_id}/files/content")
async def save_file_content(
    session_id: str,
    request: FileContentRequest,
    session_context: SessionContext = Depends(get_session)
):
    """Save content to a file"""
    
    try:
        logger.info(f"Saving file: {request.file_path} in session {session_id}")
        
        result = sandbox_manager.save_file_content(
            session_id,
            request.file_path,
            request.content,
            request.user_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to save file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")


# Code Execution Endpoints
@app.post("/sessions/{session_id}/execute", response_model=ExecuteCodeResponse)
async def execute_code(
    session_id: str,
    request: ExecuteCodeRequest,
    session_context: SessionContext = Depends(get_session)
):
    """Execute code/command in the sandbox"""
    
    try:
        logger.info(f"Executing command in session {session_id}: {request.command}")
        
        result = sandbox_manager.execute_code(
            session_id,
            request.command,
            request.working_dir,
            request.user_id
        )
        
        return ExecuteCodeResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to execute code: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to execute code: {str(e)}")


@app.post("/sessions/{session_id}/execute/python")
async def execute_python_code(
    session_id: str,
    request: ExecuteCodeRequest,
    session_context: SessionContext = Depends(get_session)
):
    """Execute Python code in the sandbox"""
    
    try:
        # Wrap command in python execution
        escaped_command = request.command.replace('"', '\\"')
        python_command = f"python3 -c \"{escaped_command}\""
        
        modified_request = ExecuteCodeRequest(
            command=python_command,
            working_dir=request.working_dir,
            user_id=request.user_id
        )
        
        return await execute_code(session_id, modified_request, session_context)
        
    except Exception as e:
        logger.error(f"Failed to execute Python code: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to execute Python code: {str(e)}")


@app.post("/sessions/{session_id}/execute/file")
async def execute_file(
    session_id: str,
    file_path: str = Query(..., description="Path to file to execute"),
    args: str = Query("", description="Command line arguments"),
    user_id: Optional[str] = Query(None),
    session_context: SessionContext = Depends(get_session)
):
    """Execute a specific file in the sandbox"""
    
    try:
        # Determine execution method based on file extension
        if file_path.endswith('.py'):
            command = f"python3 {file_path} {args}"
        elif file_path.endswith('.js'):
            command = f"node {file_path} {args}"
        elif file_path.endswith('.sh'):
            command = f"bash {file_path} {args}"
        else:
            command = f"{file_path} {args}"
        
        request = ExecuteCodeRequest(
            command=command,
            working_dir="/workspace/repo",
            user_id=user_id
        )
        
        return await execute_code(session_id, request, session_context)
        
    except Exception as e:
        logger.error(f"Failed to execute file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to execute file: {str(e)}")


# Git Operations Endpoints
@app.get("/sessions/{session_id}/git/status")
async def git_status(
    session_id: str,
    session_context: SessionContext = Depends(get_session)
):
    """Get git status for the repository"""
    
    try:
        request = ExecuteCodeRequest(
            command="git status --porcelain",
            working_dir="/workspace/repo"
        )
        
        result = await execute_code(session_id, request, session_context)
        
        # Parse git status output
        status_lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        files = []
        for line in status_lines:
            if len(line) >= 3:
                status = line[:2]
                filename = line[3:]
                files.append({
                    "status": status,
                    "filename": filename
                })
        
        return {
            "modified_files": files,
            "raw_output": result.stdout
        }
        
    except Exception as e:
        logger.error(f"Failed to get git status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get git status: {str(e)}")


@app.get("/sessions/{session_id}/git/diff")
async def git_diff(
    session_id: str,
    file_path: Optional[str] = Query(None, description="Specific file to diff"),
    session_context: SessionContext = Depends(get_session)
):
    """Get git diff for changes"""
    
    try:
        if file_path:
            command = f"git diff {file_path}"
        else:
            command = "git diff"
        
        request = ExecuteCodeRequest(
            command=command,
            working_dir="/workspace/repo"
        )
        
        result = await execute_code(session_id, request, session_context)
        
        return {
            "diff": result.stdout,
            "file_path": file_path
        }
        
    except Exception as e:
        logger.error(f"Failed to get git diff: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get git diff: {str(e)}")


@app.post("/sessions/{session_id}/git/commit")
async def git_commit(
    session_id: str,
    commit_message: str = Query(..., description="Commit message"),
    user_id: Optional[str] = Query(None),
    session_context: SessionContext = Depends(get_session)
):
    """Commit changes to git"""
    
    try:
        # Add all changes and commit
        escaped_commit_message = commit_message.replace('"', '\\"')
        commands = [
            "git add -A",
            f"git commit -m \"{escaped_commit_message}\""
        ]
        
        results = []
        for command in commands:
            request = ExecuteCodeRequest(
                command=command,
                working_dir="/workspace/repo",
                user_id=user_id
            )
            
            result = await execute_code(session_id, request, session_context)
            results.append(result)
        
        return {
            "message": "Changes committed successfully",
            "commit_message": commit_message,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Failed to commit changes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to commit changes: {str(e)}")


# Activity Tracking Endpoints
@app.get("/sessions/{session_id}/activity", response_model=List[ActivityLogEntry])
async def get_activity_log(
    session_id: str,
    limit: int = Query(100, description="Maximum number of entries to return"),
    session_context: SessionContext = Depends(get_session)
):
    """Get activity log for a session"""
    
    try:
        activity_log = sandbox_manager.get_activity_log(session_id)
        
        # Return most recent entries up to limit
        recent_activity = activity_log[-limit:] if len(activity_log) > limit else activity_log
        
        return [ActivityLogEntry(**entry) for entry in recent_activity]
        
    except Exception as e:
        logger.error(f"Failed to get activity log: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get activity log: {str(e)}")


@app.get("/sessions/{session_id}/activity/summary")
async def get_activity_summary(
    session_id: str,
    session_context: SessionContext = Depends(get_session)
):
    """Get activity summary for a session"""
    
    try:
        activity_log = sandbox_manager.get_activity_log(session_id)
        
        # Analyze activity patterns
        event_counts = {}
        total_execution_time = 0
        files_modified = set()
        
        for entry in activity_log:
            event_type = entry.get('event_type', 'unknown')
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            # Track execution time
            if event_type == 'code_executed':
                exec_time = entry.get('data', {}).get('execution_time_seconds', 0)
                total_execution_time += exec_time
            
            # Track modified files
            if event_type == 'file_saved':
                file_path = entry.get('data', {}).get('file_path')
                if file_path:
                    files_modified.add(file_path)
        
        return {
            "session_id": session_id,
            "total_activities": len(activity_log),
            "event_counts": event_counts,
            "total_execution_time_seconds": total_execution_time,
            "files_modified_count": len(files_modified),
            "files_modified": list(files_modified)
        }
        
    except Exception as e:
        logger.error(f"Failed to get activity summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get activity summary: {str(e)}")


# Terminal/Console Endpoints
@app.post("/sessions/{session_id}/terminal")
async def terminal_command(
    session_id: str,
    request: ExecuteCodeRequest,
    session_context: SessionContext = Depends(get_session)
):
    """Execute terminal command with enhanced logging"""
    
    try:
        # This is essentially the same as execute_code but with terminal-specific logging
        result = sandbox_manager.execute_code(
            session_id,
            request.command,
            request.working_dir,
            request.user_id
        )
        
        # Add terminal-specific formatting
        return {
            **result,
            "prompt": f"{request.working_dir}$ {request.command}",
            "formatted_output": f"{result['stdout']}{result['stderr']}"
        }
        
    except Exception as e:
        logger.error(f"Failed to execute terminal command: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to execute terminal command: {str(e)}")


# Search and Navigation Endpoints
@app.get("/sessions/{session_id}/search")
async def search_files(
    session_id: str,
    query: str = Query(..., description="Search query"),
    file_pattern: str = Query("*", description="File pattern to search in"),
    session_context: SessionContext = Depends(get_session)
):
    """Search for text within files"""
    
    try:
        # Use grep to search through files
        search_command = f"grep -r -n '{query}' --include='{file_pattern}' /workspace/repo/ || true"
        
        request = ExecuteCodeRequest(
            command=search_command,
            working_dir="/workspace/repo"
        )
        
        result = await execute_code(session_id, request, session_context)
        
        # Parse grep output
        matches = []
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if ':' in line and line.strip():
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        matches.append({
                            "file": parts[0].replace('/workspace/repo/', ''),
                            "line_number": parts[1],
                            "content": parts[2]
                        })
        
        return {
            "query": query,
            "matches": matches,
            "total_matches": len(matches)
        }
        
    except Exception as e:
        logger.error(f"Failed to search files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to search files: {str(e)}")


# Error handler for general exceptions
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message=str(exc),
            timestamp=datetime.utcnow().isoformat()
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend_endpoints:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )