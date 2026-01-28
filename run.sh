#!/bin/bash
cd "$(dirname "$0")"

# Start server in background
.venv/bin/python server.py &
SERVER_PID=$!

# Wait for server to be ready
sleep 1

# Open in Windows Chrome
cmd.exe /c start http://127.0.0.1:8765 2>/dev/null

echo "Server running at http://127.0.0.1:8765 (PID: $SERVER_PID)"
echo "Press Ctrl+C to stop"

# Wait for server
wait $SERVER_PID
