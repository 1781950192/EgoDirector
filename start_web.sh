#!/bin/bash
# Web service daemon script

LOG_FILE="/mnt/data/xgl/mycode/action_agent_vllm/web_run.log"

echo "======================================"
echo "Web service daemon"
echo "======================================"
echo ""

cd /mnt/data/xgl/mycode/action_agent_vllm

# Check for and kill the old process
pkill -9 -f "python.*web_app.py" 2>/dev/null
sleep 2

# Start a new process (use python3 and capture both stdout and stderr)
nohup python3 -u web_app.py --host 0.0.0.0 --port 5000 > web.log 2>&1 &
WEB_PID=$!

echo "Web service started (PID: $WEB_PID)"
echo ""
echo "Access address:"
echo "  Local: http://localhost:5000"
echo "  Remote: http://10.1.20.231:5000"
echo ""
echo "Log file: web.log"
echo ""

# Wait 10 seconds and check whether it started successfully
sleep 10

if ps -p $WEB_PID > /dev/null; then
    echo "✅ Web service is running"
else
    echo "❌ Web service failed to start, check the log:"
    tail -50 web.log
fi
