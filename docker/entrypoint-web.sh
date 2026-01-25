#!/bin/sh
set -e

if [ ! -d "node_modules" ] || [ -z "$(ls -A node_modules 2>/dev/null)" ] || [ ! -d "node_modules/react-markdown" ]; then
  echo "Installing web dependencies..."
  npm install
fi

exec npm run dev