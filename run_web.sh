#!/bin/bash
# Startup script of the Action Recognition Web Visualization System

echo "======================================"
echo "Action Recognition Web Visualization System"
echo "Action Recognition Web Visualization"
echo "======================================"
echo ""

# Check Python
PYTHON_CMD="python"
if ! command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_CMD="python3"
fi

echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version
echo ""

# Check the dependencies
echo "Checking dependencies..."
$PYTHON_CMD -c "from flask import Flask; print('✓ Flask is installed')" 2>/dev/null || {
    echo "❌ Flask is not installed, installing..."
    pip install flask flask-cors
}
$PYTHON_CMD -c "from PIL import Image; print('✓ Pillow is installed')" 2>/dev/null || {
    echo "❌ Pillow is not installed, installing..."
    pip install Pillow
}
echo ""

# Get the host IP
HOST_IP="0.0.0.0"
PORT=5000

echo "======================================"
echo "Starting the web server..."
echo "======================================"
echo ""
echo "📡 Access address:"
echo "   Local: http://localhost:$PORT"
echo "   Remote: http://<server IP>:$PORT"
echo ""
echo "💡 Notes:"
echo "   - If this is a remote server, make sure port $PORT is open"
echo "   - You can use an SSH tunnel:"
echo "     ssh -L $PORT:localhost:$PORT user@server"
echo "   - Then open http://localhost:$PORT in your local browser"
echo ""
echo "======================================"
echo ""

# Start the service
$PYTHON_CMD web_app.py --host $HOST_IP --port $PORT --debug "$@"
