#!/bin/bash
set -e

# Export standard paths for non-interactive AppleScript environment
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

# Change directory to script location
cd "$(dirname "$0")"

echo "⚡ Starting AetherClaw System..."

# Check if ai_env exists, if not, create it
if [ ! -d "ai_env" ]; then
    echo "Creating virtual environment 'ai_env'..."
    python3 -m venv ai_env
fi

# Activate environment
source ai_env/bin/activate

# Install/verify requirements silently
echo "Checking dependencies..."
pip install -r requirements.txt -q

# Start server if not already running on port 8000
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "Port 8000 already in use. Assuming server is running."
else
    echo "Starting AetherClaw API Server..."
    python3 server.py &
    SERVER_PID=$!
    # Wait a moment for server to bind
    sleep 3
fi

# Open browser
echo "Opening Dashboard in browser..."
if which xdg-open > /dev/null; then
  xdg-open http://localhost:8000
elif which open > /dev/null; then
  open http://localhost:8000
else
  echo "Could not detect the web browser launcher. Please navigate to http://localhost:8000 manually."
fi

echo "AetherClaw is now running. Press Ctrl+C in this terminal to exit the server."

# Wait for process if started by this script
if [ -n "$SERVER_PID" ]; then
    wait $SERVER_PID
fi
