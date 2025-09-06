# EFA Backend

Enhanced Function Agent Backend - A sophisticated MCP (Model Context Protocol) server with LLM integration, environment management, and advanced tooling capabilities.

## Overview

The EFA Backend provides a comprehensive platform for building intelligent agents that can:

- Integrate with multiple LLM models through OpenRouter (unified access to OpenAI, Anthropic, Meta, Google, and more)
- Manage different execution environments (Nextcloud, Custom environments)
- Utilize core tools for memory, judgment, learning, planning, and perception
- Expose capabilities through the Model Context Protocol (MCP)

## Architecture

```
/backend
├── /llm_core/              # LLM integration and management
├── /mcp_servers/           # MCP server implementations
├── /environments/          # Environment management
├── /mcp_tools/            # Core and environment-specific tools
├── /server/               # HTTP/WebSocket server
├── /configs/              # Configuration files
└── main.py               # Application entry point
```

### Core Components

1. **LLM Core**: OpenRouter integration for unified access to multiple LLM models
2. **MCP Servers**: Model Context Protocol server implementations
3. **Environments**: Pluggable environments for different contexts
4. **MCP Tools**: Core tools for memory, judgment, learning, planning, and perception
5. **Server**: HTTP and WebSocket server for client communication

## Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd EFA/backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure credentials**:
   ```bash
   cp configs/credentials.yaml configs/credentials.local.yaml
   # Edit credentials.local.yaml with your OpenRouter API key
   # Get your API key from https://openrouter.ai/keys
   ```

5. **Start the server**:
   ```bash
   python main.py
   ```

The server will be available at:
- HTTP: http://localhost:8000
- WebSocket: ws://localhost:8000/ws
- Health: http://localhost:8000/health
- Status: http://localhost:8000/status

## Configuration

### Main Configuration (`configs/config.yaml`)

The main configuration file controls all aspects of the backend:

- **Server settings**: Host, port, CORS, WebSocket settings
- **LLM configuration**: OpenRouter settings and model selection
- **Environment settings**: Nextcloud, custom environments
- **Tool configurations**: Memory, judgment, learning, planning, perception
- **Security settings**: Authentication, rate limiting, CORS
- **Performance settings**: Timeouts, limits, monitoring

### Credentials (`configs/credentials.yaml`)

Store sensitive information like API keys and passwords:

- OpenRouter API key (get yours at https://openrouter.ai/keys)
- Environment credentials (Nextcloud, external services)
- Database credentials
- Security keys and tokens

**Important**: Never commit credentials to version control. Use environment variables in production.

## LLM Models via OpenRouter

OpenRouter provides unified access to models from multiple providers with automatic routing, fallbacks, and cost optimization. See [available models](https://openrouter.ai/models).

### Basic Configuration
```yaml
llm:
  provider: "openrouter"
  openrouter:
    model: "meta-llama/llama-3.1-8b-instruct:free"  # Default model (free tier)
    temperature: 0.7
    max_tokens: 2048
```

### Model Selection Examples
```yaml
llm:
  openrouter:
    models:
      fast: "meta-llama/llama-3.1-8b-instruct:free"     # Fast, free model
      smart: "meta-llama/llama-3.1-70b-instruct:free"   # Higher quality free model
      creative: "meta-llama/llama-3.1-8b-instruct:free" # Creative tasks
      coding: "meta-llama/llama-3.1-8b-instruct:free"   # Code generation
      free: "meta-llama/llama-3.1-8b-instruct:free"     # Free tier
```

### Provider Routing
```yaml
llm:
  openrouter:
    provider: "openai"      # Prefer specific provider
    route: "fallback"       # Enable automatic fallbacks
```

### Rate Limits and Credits

OpenRouter handles rate limiting automatically based on your account:
- **Free tier**: 20 requests/minute for `:free` models
- **Paid accounts**: Higher limits based on credit balance
- **Credit monitoring**: Built-in account balance checking

Check your usage at: https://openrouter.ai/activity

## Environments

### Nextcloud Environment

Provides file management capabilities with Nextcloud:

```yaml
environments:
  nextcloud:
    enabled: true
    timeout: 30
    verify_ssl: true
```

Required credentials:
```yaml
environments:
  nextcloud:
    username: "your_username"
    password: "your_password"
    base_url: "https://your-nextcloud.com"
```

### Custom Environment

Template for creating custom environments:

```yaml
environments:
  custom:
    enabled: true
    debug_mode: false
    timeout: 30
```

## Core Tools

### Memory Tool
- Store and retrieve information across sessions
- Support for different memory types (episodic, semantic, procedural)
- Search and categorization with tags
- Automatic consolidation and cleanup

### Judgment Tool
- Evaluate responses and actions against multiple criteria
- Assess accuracy, relevance, completeness, clarity, safety, ethics
- Compare options and provide rankings
- Historical judgment tracking

### Learning Tool
- Record learning events and outcomes
- Pattern detection and recognition
- Performance analysis and trends
- Recommendation generation based on learned patterns

### Planning Tool
- Create and manage DAG-based execution plans
- Task dependencies and scheduling
- Progress tracking and status management
- Critical path analysis

### Perception Tool
- Process various input types (text, image, audio, video, sensor data)
- Sentiment analysis and entity extraction
- Language detection and content analysis
- Multi-modal input processing

## API Usage

### HTTP API

The server exposes a JSON-RPC API at `/rpc`:

```bash
curl -X POST http://localhost:8000/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/list",
    "params": {},
    "id": 1
  }'
```

### WebSocket API

Connect to `ws://localhost:8000/ws` for real-time communication:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.send(JSON.stringify({
  "jsonrpc": "2.0",
  "method": "memory/store",
  "params": {
    "content": "Important information",
    "tags": ["important", "note"]
  },
  "id": 1
}));
```

### Available Methods

- `initialize`: Initialize MCP connection
- `tools/list`: List available tools
- `tools/call`: Execute a tool
- `resources/list`: List available resources
- `resources/read`: Read a resource
- `memory/store`: Store memory
- `memory/search`: Search memories
- `judgment/evaluate`: Evaluate responses
- `learning/record`: Record learning events
- `planning/create`: Create execution plans
- `perception/process`: Process input data

## Development

### Project Structure

```
backend/
├── llm_core/
│   ├── __init__.py
│   ├── llm_interface.py           # Abstract LLM interface
│   ├── context_manager.py         # Memory and context management
│   ├── inference_engine.py        # LLM orchestration
│   └── adapters/                  # LLM provider adapters
├── mcp_servers/
│   ├── base_adapter.py           # MCP adapter interface
│   ├── local_mcp_server.py       # Local MCP implementation
│   ├── http_mcp_adapter.py       # HTTP MCP client
│   └── performance_monitor.py     # Performance monitoring
├── environments/
│   ├── base_environment/          # Base environment classes
│   ├── nextcloud_environment/     # Nextcloud integration
│   ├── custom_environment/        # Custom environment template
│   └── environment_manager.py     # Environment coordination
├── mcp_tools/
│   ├── core_tools/               # Core functionality tools
│   └── environment_specific_tools/ # Environment-specific tools
├── server/
│   ├── mcp_server.py             # Main MCP server
│   └── request_router.py         # Request routing
└── configs/                      # Configuration files
```

### Adding New LLM Providers

1. Create adapter in `llm_core/adapters/`
2. Implement `LLMInterface` methods
3. Add configuration in `config.yaml`
4. Register in `environment_manager.py`

### Adding New Environments

1. Create environment in `environments/`
2. Inherit from `Environment` base class
3. Implement required methods
4. Add configuration support
5. Register with `EnvironmentManager`

### Adding New Tools

1. Create tool in `mcp_tools/core_tools/`
2. Implement tool functionality
3. Register with tools registry
4. Add to MCP server handlers

## Performance and Monitoring

### Performance Monitoring

The backend includes comprehensive performance monitoring:

- Request/response times
- Error rates and success rates
- Memory usage and tool statistics
- Environment health checks

### Metrics Endpoints

- `/health`: Health check endpoint
- `/status`: Server status and statistics
- `/metrics`: Performance metrics (if enabled)

### Logging

Configurable logging with multiple levels and outputs:

```yaml
logging:
  level: "INFO"
  file_logging: true
  log_file: "logs/efa_backend.log"
  max_log_size: 10485760  # 10MB
  backup_count: 5
```

## Security

### Authentication

Multiple authentication methods supported:

```yaml
security:
  authentication:
    enabled: true
    method: "api_key"  # api_key, jwt, oauth
```

### Rate Limiting

Configurable rate limiting:

```yaml
security:
  rate_limiting:
    enabled: true
    default_limit: 60  # requests per minute
    burst_limit: 10
```

### CORS

Cross-origin resource sharing configuration:

```yaml
security:
  cors:
    allowed_origins: ["*"]
    allowed_methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allowed_headers: ["*"]
```

## Deployment

### Production Deployment

1. **Environment Variables**: Use environment variables for credentials
2. **HTTPS**: Enable SSL/TLS in production
3. **Reverse Proxy**: Use nginx or similar for load balancing
4. **Process Management**: Use systemd, supervisor, or container orchestration
5. **Monitoring**: Set up monitoring and alerting
6. **Backup**: Regular backup of data and configuration

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

### Environment Variables

Set these environment variables in production:

```bash
OPENROUTER_API_KEY=your_openrouter_key
NEXTCLOUD_USERNAME=your_username
NEXTCLOUD_PASSWORD=your_password
JWT_SECRET_KEY=your_secret_key
```

## Troubleshooting

### Common Issues

1. **Server won't start**: Check configuration files and credentials
2. **OpenRouter API errors**: Verify API key and check credit balance at https://openrouter.ai/activity
3. **Environment failures**: Check environment-specific credentials
4. **Memory issues**: Adjust memory limits in configuration
5. **Performance issues**: Enable monitoring and check metrics

### Debug Mode

Enable debug mode for detailed logging:

```yaml
development:
  debug: true
logging:
  level: "DEBUG"
```

### Health Checks

Use the health endpoint to diagnose issues:

```bash
curl http://localhost:8000/health
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

[License information here]

## Support

For support and questions:
- Documentation: [Link to docs]
- Issues: [Link to issue tracker]
- Community: [Link to community]
