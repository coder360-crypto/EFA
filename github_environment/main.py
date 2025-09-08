import subprocess
import sys
import time
import os
from dotenv import load_dotenv

load_dotenv()

def run_server(file_path, port):
    """Run a FastAPI server on specified port"""
    return subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        f"{file_path.replace('.py', '')}:app", 
        "--host", "0.0.0.0", 
        "--port", str(port),
        "--reload"
    ])

def main():
    print("Starting GitHub Issue Solver Services...")
    
    # Start backend endpoints server
    backend_port = int(os.getenv('PORT_BACKEND', 8000))
    print(f"Starting backend server on port {backend_port}")
    backend_process = run_server('backend_endpoints', backend_port)
    
    # Start AI agent endpoints server  
    agent_port = int(os.getenv('PORT_AI_AGENT', 8001))
    print(f"Starting AI agent server on port {agent_port}")
    agent_process = run_server('ai_agent_endpoints', agent_port)
    
    # Start FastMCP server
    mcp_port = 8002
    print(f"Starting FastMCP server on port {mcp_port}")
    mcp_process = subprocess.Popen([
        sys.executable, "ai_agent_mcp_tools.py"
    ])
    
    print("\n🚀 All services started!")
    print(f"📋 Backend API: http://localhost:{backend_port}/docs")
    print(f"🤖 AI Agent API: http://localhost:{agent_port}/docs") 
    print(f"🔧 MCP Server: Running with FastMCP")
    print("\nPress Ctrl+C to stop all services")
    
    try:
        backend_process.wait()
        agent_process.wait()
        mcp_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        backend_process.terminate()
        agent_process.terminate() 
        mcp_process.terminate()
        print("✅ All services stopped")

if __name__ == "__main__":
    main()