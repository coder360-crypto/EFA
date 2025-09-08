# sandbox_manager.py
"""
Modal Sandbox Manager for GitHub Issue Code Environment
Handles sandbox creation, GitHub repo loading, and file operations
"""

import modal
import requests
import json
import os
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import uuid


@dataclass
class SessionContext:
    session_id: str
    github_issue_url: str
    repo_owner: str
    repo_name: str
    issue_number: int
    base_commit: str
    solution_commit: Optional[str]
    sandbox: Optional[modal.Sandbox] = None
    created_at: datetime = None
    activity_log: List[Dict] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.activity_log is None:
            self.activity_log = []


class GitHubAPIClient:
    """Handle GitHub API interactions"""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token
        self.headers = {}
        if token:
            self.headers["Authorization"] = f"token {token}"
    
    def parse_issue_url(self, issue_url: str) -> tuple:
        """Parse GitHub issue URL to extract owner, repo, issue_number"""
        parts = issue_url.rstrip('/').split('/')
        if len(parts) < 7 or parts[2] != 'github.com':
            raise ValueError("Invalid GitHub issue URL")
        
        owner = parts[3]
        repo = parts[4] 
        issue_number = int(parts[6])
        
        return owner, repo, issue_number
    
    def get_issue_data(self, owner: str, repo: str, issue_number: int) -> Dict:
        """Fetch issue details from GitHub API"""
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch issue: {response.status_code}")
        
        return response.json()
    
    def find_base_commit(self, owner: str, repo: str, issue_created_date: str) -> str:
        """Find commit closest to when issue was created"""
        url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        params = {"until": issue_created_date, "per_page": 1}
        
        response = requests.get(url, params=params, headers=self.headers)
        commits = response.json()
        
        return commits[0]["sha"] if commits else "HEAD"
    
    def find_solution_commit(self, owner: str, repo: str, issue_number: int) -> Optional[str]:
        """Find the commit that fixed the issue"""
        # Check issue events for closing commit
        events_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/events"
        response = requests.get(events_url, headers=self.headers)
        
        if response.status_code == 200:
            events = response.json()
            for event in events:
                if event.get("event") == "closed" and event.get("commit_id"):
                    return event["commit_id"]
        
        # Fallback: search for commits referencing this issue
        search_url = "https://api.github.com/search/commits"
        params = {
            "q": f"repo:{owner}/{repo} #{issue_number}",
            "sort": "committer-date",
            "per_page": 1
        }
        response = requests.get(search_url, params=params, headers=self.headers)
        
        if response.status_code == 200:
            commits = response.json().get("items", [])
            return commits[0]["sha"] if commits else None
        
        return None


class ActivityLogger:
    """Handle activity logging for user sessions"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.activity_log = []
    
    def log_activity(self, event_type: str, data: Dict, user_id: str = None):
        """Log user activity"""
        entry = {
            "session_id": self.session_id,
            "user_id": user_id,
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "data": data
        }
        
        self.activity_log.append(entry)
        return entry
    
    def get_activity_log(self) -> List[Dict]:
        return self.activity_log
    
    def save_to_file(self, filepath: str):
        """Save activity log to file"""
        with open(filepath, 'w') as f:
            json.dump(self.activity_log, f, indent=2)


class ModalSandboxManager:
    """Manage Modal sandboxes for code environment sessions"""
    
    def __init__(self, app_name: str = "github-issue-solver"):
        self.app_name = app_name
        self.app = modal.App.lookup(app_name, create_if_missing=True)
        self.github_client = GitHubAPIClient()
        self.active_sessions: Dict[str, SessionContext] = {}
    
    def create_session(self, github_issue_url: str, user_id: str = None) -> SessionContext:
        """Create new coding session with sandbox"""
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Parse GitHub issue URL
        owner, repo, issue_number = self.github_client.parse_issue_url(github_issue_url)
        
        # Get issue data
        issue_data = self.github_client.get_issue_data(owner, repo, issue_number)
        
        # Find relevant commits
        base_commit = self.github_client.find_base_commit(
            owner, repo, issue_data['created_at']
        )
        solution_commit = self.github_client.find_solution_commit(
            owner, repo, issue_number
        )
        
        # Create session context
        session_context = SessionContext(
            session_id=session_id,
            github_issue_url=github_issue_url,
            repo_owner=owner,
            repo_name=repo,
            issue_number=issue_number,
            base_commit=base_commit,
            solution_commit=solution_commit
        )
        
        # Create sandbox
        sandbox = self._create_sandbox_environment(session_context)
        session_context.sandbox = sandbox
        
        # Store session
        self.active_sessions[session_id] = session_context
        
        # Initialize activity logger
        activity_logger = ActivityLogger(session_id)
        activity_logger.log_activity("session_created", {
            "github_issue_url": github_issue_url,
            "base_commit": base_commit,
            "solution_commit": solution_commit,
            "issue_data": issue_data
        }, user_id)
        
        return session_context
    
    def _create_sandbox_environment(self, session_context: SessionContext) -> modal.Sandbox:
        """Create Modal sandbox with repository loaded"""
        
        # Create development image with necessary tools
        image = (
            modal.Image.debian_slim()
            .apt_install(
                "git", "curl", "wget", "build-essential", "vim", "nano",
                "python3", "python3-pip", "nodejs", "npm"
            )
            .pip_install(
                "requests", "python-dotenv", "fastapi", "uvicorn",
                "pytest", "black", "flake8"
            )
        )
        
        # Create sandbox
        sandbox = modal.Sandbox.create(
            image=image,
            app=self.app,
            timeout=7200,  # 2 hours
            workdir="/workspace"
        )
        
        # Setup repository
        self._setup_repository(sandbox, session_context)
        
        return sandbox
    
    def _setup_repository(self, sandbox: modal.Sandbox, session_context: SessionContext):
        """Clone and setup repository in sandbox"""
        
        repo_url = f"https://github.com/{session_context.repo_owner}/{session_context.repo_name}"
        
        # Setup commands
        setup_commands = [
            f"git clone {repo_url} /workspace/repo",
            "cd /workspace/repo",
            f"git checkout {session_context.base_commit}",
            "git checkout -b solving-session",
        ]
        
        # Execute setup commands
        for cmd in setup_commands:
            result = sandbox.exec("bash", "-c", cmd, timeout=120)
            if result.returncode != 0:
                error_msg = result.stderr.read() if result.stderr else "Unknown error"
                raise Exception(f"Setup failed at: {cmd}. Error: {error_msg}")
        
        # Install project dependencies
        self._install_dependencies(sandbox)
        
        # Save session context to sandbox
        context_data = {
            "session_id": session_context.session_id,
            "github_issue_url": session_context.github_issue_url,
            "base_commit": session_context.base_commit,
            "solution_commit": session_context.solution_commit,
            "repo_info": {
                "owner": session_context.repo_owner,
                "name": session_context.repo_name,
                "issue_number": session_context.issue_number
            }
        }
        
        with sandbox.open("/workspace/session_context.json", "w") as f:
            f.write(json.dumps(context_data, indent=2))
    
    def _install_dependencies(self, sandbox: modal.Sandbox):
        """Auto-detect and install project dependencies"""
        
        dependency_commands = []
        
        # Check for Python projects
        python_files = [
            "/workspace/repo/requirements.txt",
            "/workspace/repo/pyproject.toml", 
            "/workspace/repo/setup.py"
        ]
        
        for file_path in python_files:
            check_result = sandbox.exec("test", "-f", file_path)
            if check_result.returncode == 0:
                if "requirements.txt" in file_path:
                    dependency_commands.append("cd /workspace/repo && pip install -r requirements.txt")
                elif "pyproject.toml" in file_path:
                    dependency_commands.append("cd /workspace/repo && pip install -e .")
                elif "setup.py" in file_path:
                    dependency_commands.append("cd /workspace/repo && pip install -e .")
        
        # Check for Node.js projects
        package_json_check = sandbox.exec("test", "-f", "/workspace/repo/package.json")
        if package_json_check.returncode == 0:
            dependency_commands.append("cd /workspace/repo && npm install")
        
        # Install dependencies
        for cmd in dependency_commands:
            try:
                sandbox.exec("bash", "-c", cmd, timeout=300)
            except Exception as e:
                print(f"Warning: Dependency installation failed: {e}")
    
    def get_file_tree(self, session_id: str, path: str = "/workspace/repo") -> Dict:
        """Get file tree structure for the repository"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        sandbox = self.active_sessions[session_id].sandbox
        
        # Get file tree using find command
        result = sandbox.exec("find", path, "-type", "f", "-o", "-type", "d", timeout=30)
        
        if result.returncode != 0:
            raise Exception("Failed to get file tree")
        
        file_paths = result.stdout.read().strip().split('\n')
        
        # Build tree structure
        tree = self._build_tree_structure(file_paths, path)
        
        return tree
    
    def _build_tree_structure(self, file_paths: List[str], root_path: str) -> Dict:
        """Build nested tree structure from file paths"""
        
        tree = {
            "name": os.path.basename(root_path) or "repo",
            "path": root_path,
            "type": "directory",
            "children": []
        }
        
        # Sort paths to ensure directories come before files
        file_paths.sort()
        
        for file_path in file_paths:
            if not file_path or file_path == root_path:
                continue
                
            # Get relative path
            rel_path = os.path.relpath(file_path, root_path)
            path_parts = rel_path.split('/')
            
            current_node = tree
            current_path = root_path
            
            for i, part in enumerate(path_parts):
                current_path = os.path.join(current_path, part)
                
                # Check if this part already exists in children
                existing_child = None
                for child in current_node["children"]:
                    if child["name"] == part:
                        existing_child = child
                        break
                
                if existing_child:
                    current_node = existing_child
                else:
                    # Determine if it's a file or directory
                    is_file = i == len(path_parts) - 1 and os.path.isfile(file_path)
                    
                    new_node = {
                        "name": part,
                        "path": current_path,
                        "type": "file" if is_file else "directory",
                        "children": [] if not is_file else None
                    }
                    
                    current_node["children"].append(new_node)
                    current_node = new_node
        
        return tree
    
    def get_file_content(self, session_id: str, file_path: str) -> Dict:
        """Get content of a specific file"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        sandbox = self.active_sessions[session_id].sandbox
        
        # Check if file exists
        check_result = sandbox.exec("test", "-f", file_path)
        if check_result.returncode != 0:
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file content
        with sandbox.open(file_path, "r") as f:
            content = f.read()
        
        # Get file info
        stat_result = sandbox.exec("stat", "-c", "%s %Y", file_path)
        if stat_result.returncode == 0:
            size, modified = stat_result.stdout.read().strip().split()
            size = int(size)
            modified = int(modified)
        else:
            size, modified = 0, 0
        
        return {
            "file_path": file_path,
            "content": content,
            "size": size,
            "modified": modified,
            "extension": os.path.splitext(file_path)[1]
        }
    
    def save_file_content(self, session_id: str, file_path: str, content: str, user_id: str = None) -> Dict:
        """Save content to a file and log the activity"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        sandbox = self.active_sessions[session_id].sandbox
        session_context = self.active_sessions[session_id]
        
        # Get original content for diff
        original_content = ""
        try:
            with sandbox.open(file_path, "r") as f:
                original_content = f.read()
        except:
            pass  # New file
        
        # Save new content
        with sandbox.open(file_path, "w") as f:
            f.write(content)
        
        # Log activity
        activity_logger = ActivityLogger(session_id)
        activity_logger.log_activity("file_saved", {
            "file_path": file_path,
            "content_length": len(content),
            "original_content_length": len(original_content),
            "is_new_file": len(original_content) == 0
        }, user_id)
        
        session_context.activity_log.extend(activity_logger.get_activity_log())
        
        return {
            "success": True,
            "file_path": file_path,
            "size": len(content),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def execute_code(self, session_id: str, command: str, working_dir: str = "/workspace/repo", user_id: str = None) -> Dict:
        """Execute code/command in the sandbox"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        sandbox = self.active_sessions[session_id].sandbox
        session_context = self.active_sessions[session_id]
        
        # Execute command
        start_time = datetime.utcnow()
        
        result = sandbox.exec(
            "bash", "-c", f"cd {working_dir} && {command}",
            timeout=60
        )
        
        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds()
        
        # Get output
        stdout = result.stdout.read() if result.stdout else ""
        stderr = result.stderr.read() if result.stderr else ""
        
        # Log activity
        activity_logger = ActivityLogger(session_id)
        activity_logger.log_activity("code_executed", {
            "command": command,
            "working_dir": working_dir,
            "exit_code": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "execution_time_seconds": execution_time
        }, user_id)
        
        session_context.activity_log.extend(activity_logger.get_activity_log())
        
        return {
            "command": command,
            "exit_code": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "execution_time": execution_time,
            "timestamp": end_time.isoformat()
        }
    
    def get_session_info(self, session_id: str) -> Dict:
        """Get information about a session"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session_context = self.active_sessions[session_id]
        
        return {
            "session_id": session_context.session_id,
            "github_issue_url": session_context.github_issue_url,
            "repo_info": {
                "owner": session_context.repo_owner,
                "name": session_context.repo_name,
                "issue_number": session_context.issue_number
            },
            "commits": {
                "base_commit": session_context.base_commit,
                "solution_commit": session_context.solution_commit
            },
            "created_at": session_context.created_at.isoformat(),
            "activity_count": len(session_context.activity_log)
        }
    
    def get_activity_log(self, session_id: str) -> List[Dict]:
        """Get activity log for a session"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        return self.active_sessions[session_id].activity_log
    
    def cleanup_session(self, session_id: str):
        """Clean up session resources"""
        
        if session_id in self.active_sessions:
            session_context = self.active_sessions[session_id]
            
            # Terminate sandbox
            if session_context.sandbox:
                try:
                    session_context.sandbox.terminate()
                except:
                    pass  # Ignore errors during cleanup
            
            # Remove from active sessions
            del self.active_sessions[session_id]
    
    def list_active_sessions(self) -> List[Dict]:
        """List all active sessions"""
        
        sessions = []
        for session_id, session_context in self.active_sessions.items():
            sessions.append({
                "session_id": session_id,
                "github_issue_url": session_context.github_issue_url,
                "repo_name": f"{session_context.repo_owner}/{session_context.repo_name}",
                "issue_number": session_context.issue_number,
                "created_at": session_context.created_at.isoformat(),
                "activity_count": len(session_context.activity_log)
            })
        
        return sessions


# Global sandbox manager instance
sandbox_manager = ModalSandboxManager()