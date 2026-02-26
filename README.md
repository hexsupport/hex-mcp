# HexagonML ModelManager MCP Server

A modular, well-organized MCP server for interacting with the HexagonML ModelManager API with maximum tool discoverability and clean architecture.

## 🏗️ Architecture

The server has been refactored into a clean, modular architecture:

```
server/
├── config.py              # Environment configuration and server setup
├── clients.py              # ModelManager API client factory and context
├── utils.py                # Common utilities (response handling, validation)
├── validators.py           # Payload validation functions
├── base.py                 # Base classes and common patterns
├── model_tools.py          # Model management tools
├── usecase_tools.py        # Usecase management tools
├── modelcard_tools.py      # Model card management tools
├── forecasting_tools.py    # Forecasting tools
├── main.py                 # Main entry point
└── server_simple.py        # FastMCP CLI entry point
```

## 🚀 Features

### **Maximum MCP Discoverability**
- **Individual parameters** instead of dict parameters
- **Auto-completion** support in MCP tool playgrounds
- **Type safety** with explicit parameter types
- **Clear documentation** for each parameter

### **Modular Design**
- **Separation of concerns** - Each module has a single responsibility
- **Reusable components** - Common patterns extracted into base classes
- **Easy maintenance** - Clear structure makes updates simple
- **Testable components** - Each module can be tested independently

### **Robust Error Handling**
- **Standardized error responses** across all tools
- **Progress reporting** with user feedback
- **Validation** with helpful error messages
- **Graceful degradation** when API calls fail

## 📋 Available Tools

### **Model Management**
- `add_model` - Create new ML models
- `update_model` - Update existing models
- `delete_model` - Delete models
- `get_latest_metrics` - Retrieve model performance metrics

### **Usecase Management**
- `add_usecase` - Create new usecases (supports forecasting)
- `update_usecase` - Update existing usecases
- `delete_usecase` - Delete usecases
- `get_usecase_data` - List all usecases

### **Model Cards**
- `create_modelcard` - Create individual model cards
- `create_modelcard_bulk` - Create bulk model cards
- `get_modelcard_data` - Retrieve model card data

### **Forecasting**
- `get_forecast` - Retrieve forecasts for usecases

## 🔧 Configuration

### Environment Variables
```bash
SECRET_KEY=your_secret_key_here
MM_API_BASE_URL=http://localhost:8000
OUTPUT_DIR=/path/to/output/directory
HOST=0.0.0.0
PORT=9000
```

### Setup
1. Copy `.env.example` to `.env` (or create `.env`)
2. Fill in your configuration values
3. Run the server

## 🏃‍♂️ Usage

### Start the Server

#### Option 1: FastMCP CLI (Recommended)
```bash
fastmcp run server/server_simple.py --transport http --host 127.0.0.1 --port 8080
```

#### Option 2: Standalone
```bash
python server/main.py
```

#### Option 3: Development Mode
```bash
fastmcp dev server/server_simple.py
```

### Example Tool Usage
```python
# Create a new model
add_model(
    name="Fraud Detection Model",
    description="Model for detecting fraudulent transactions",
    project="123",
    transformer_type="Classification",
    training_dataset="/path/to/train.csv",
    target_column="is_fraud"
)

# Create a forecasting usecase
add_usecase(
    name="Sales Forecast",
    usecase_type="Forecasting",
    description="Forecast future sales",
    forecasting_template="two_conditions",
    notification_emails=["user@example.com"],
    result_tab=True,
    series_tab=True
)

# Get model metrics
get_latest_metrics(
    model_id="123",
    metric_type="accuracy"
)

# Retrieve forecast
get_forecast(
    usecase_name="Sales Forecast",
    series="sales_data",
    prediction_period=30
)
```

## 🔌 MCP Integration Configuration

### For IDEs (VSCode, Windsurf, Claude Desktop)

#### Local Configuration
```json
{
  "mcpServers": {
    "hex-mm-mcp": {
      "command": "fastmcp",
      "args": [
        "run",
        "hex-mm-mcp/server/server_simple.py",
        "--transport",
        "http",
        "--host",
        "127.0.0.1",
        "--port",
        "8080"
      ]
    }
  }
}
```

#### Docker Configuration
```json
{
  "mcpServers": {
    "hex-mm-mcp-docker": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-i",
        "--network=host",
        "-e",
        "SECRET_KEY",
        "-e",
        "MM_API_BASE_URL",
        "-e",
        "OUTPUT_DIR",
        "modelmanager-mcp:latest"
      ],
      "env": {
        "SECRET_KEY": "your-secret-key",
        "MM_API_BASE_URL": "http://host-ip:8000",
        "OUTPUT_DIR": "/tmp/mm-output"
      }
    }
  }
}
```

### Docker Setup

#### Build Image
```bash
docker build --platform=linux/amd64 -t modelmanager-mcp:latest .
```

#### Run Container
```bash
docker run --platform=linux/amd64 -d --name mm-mcp \
  -p 8080:8080 \
  --env-file .env \
  modelmanager-mcp:latest
```

## 🧪 Development

### Adding New Tools
1. Create a new module file (e.g., `new_tools.py`)
2. Import `mcp` from `config`
3. Use the `@mcp.tool` decorator
4. Import the module in `server_simple.py` or `main.py`

### Code Structure Template
```python
# new_tools.py
from fastmcp import Context
from config import mcp
from clients import get_mm_client
from utils import safe_response_to_dict, create_error_response
import asyncio

@mcp.tool(
    name="new_tool",
    description="Description of the tool",
    tags={"category", "tool"},
    meta={"version": "1.0", "author": "HexagonML"}
)
async def new_tool(ctx: Context, required_param: str, optional_param: str = None) -> dict:
    """Tool description.
    
    Args:
        ctx: MCP server context.
        required_param: Description of required parameter.
        optional_param: Description of optional parameter.
        
    Returns:
        dict: Response data.
    """
    # Validation
    if not required_param:
        return create_error_response(
            message="Required parameter is missing",
            error_type="ValidationError"
        )
    
    # Progress reporting
    await ctx.info("Starting operation")
    await ctx.report_progress(progress=20, total=100)
    
    try:
        client = get_mm_client(ctx, 'client_type')
        await ctx.report_progress(progress=40, total=100)
        
        response = await asyncio.to_thread(client.api_method, required_param)
        await ctx.report_progress(progress=80, total=100)
        
        result = safe_response_to_dict(response)
        await ctx.info("Operation completed successfully")
        await ctx.report_progress(progress=100, total=100)
        
        return result
        
    except Exception as e:
        await ctx.error(f"Operation failed: {str(e)}")
        return create_error_response(
            message=f"Operation failed: {str(e)}",
            error_type=type(e).__name__
        )
```

## 🔧 Troubleshooting

### Common Issues

#### Port in Use
```bash
# Kill process using port 8080
kill -9 $(lsof -t -i:8080)

# Or use a different port
fastmcp run server/server_simple.py --port 8081
```

#### Missing Environment Variables
The server will exit with a clear message listing missing required variables:
- `SECRET_KEY` - Required for API authentication
- `MM_API_BASE_URL` - Required for API endpoint
- `OUTPUT_DIR` - Required for file outputs

#### Connection Issues
- Verify ModelManager API is running at `MM_API_BASE_URL`
- Check network connectivity between MCP server and ModelManager
- Ensure SECRET_KEY is valid and has proper permissions

## 🔄 Migration from Legacy

The legacy monolithic `mm_mcp_server.py` has been replaced with the modular architecture:

### What Changed
1. **No breaking changes** - All tools work the same way
2. **Better discoverability** - Tools now use individual parameters instead of dicts
3. **Improved error handling** - More consistent and helpful error messages
4. **Progress reporting** - Users get feedback on long-running operations
5. **Clean architecture** - Modular, maintainable codebase

### Migration Steps
1. Update your MCP configuration to use `server/server_simple.py`
2. Update environment variables if needed
3. Test your existing tool calls - they should work without changes

## 📝 Benefits

### **For Developers**
- **Clean code** - Easy to read and maintain
- **Modular testing** - Test each component independently
- **Reusable patterns** - Base classes reduce duplication
- **Type safety** - Better IDE support and fewer bugs

### **For Users**
- **Better discoverability** - See all available parameters in tool playgrounds
- **Auto-completion** - IDEs suggest parameter names and types
- **Clear errors** - Helpful validation messages
- **Progress feedback** - Know when operations are running

### **For Operations**
- **Easier debugging** - Clear separation of concerns
- **Better monitoring** - Standardized logging and error reporting
- **Scalable architecture** - Easy to add new features
- **Maintainable codebase** - Reduced technical debt

## 📚 Additional Resources

- **FastMCP Documentation**: https://gofastmcp.com
- **ModelManager API**: Available at your `MM_API_BASE_URL`
- **Docker Hub**: ModelManager container images
- **GitHub Project**: https://github.com/hexagonml/modelmanager-mcp

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your new tools following the modular pattern
4. Include tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
