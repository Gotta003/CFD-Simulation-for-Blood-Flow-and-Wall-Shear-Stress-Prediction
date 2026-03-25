#!/bin/bash
APP_NAME="patients_management.py"
PYTHON_EXEC="./.evar_env/bin/python3"
DISPLAY_NAME="M3DITECH"

cd "$(dirname "$0")"
if pgrep -f $APP_NAME > /dev/null; then
    echo "DISPLAY_NAME is already running."
    if command -v notify-send > /dev/null; then
        notify-send "M3DITECH" "Application is already running."
    fi
    exit 1
else
    echo "Starting $DISPLAY_NAME..."
    nohup "$PYTHON_EXEC" "$APP_NAME" > /dev/null 2>&1 &
    sleep 1
    exit 0
fi