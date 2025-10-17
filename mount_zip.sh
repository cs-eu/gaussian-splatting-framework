#!/bin/bash

set -e  # Exit on error

ZIP_FILE="$1"
MOUNT_POINT="$2"

# Check arguments
if [[ -z "$ZIP_FILE" || -z "$MOUNT_POINT" ]]; then
  echo "Usage: $0 path/to/your.zip /mount/point"
  exit 1
fi

# Ensure the mount point exists
mkdir -p "$MOUNT_POINT"

# Install fuse-zip if missing
if ! command -v fuse-zip &> /dev/null; then
  echo "Installing fuse-zip..."
  sudo apt update
  sudo apt install -y fuse-zip
fi

# Mount the zip file
echo "Mounting $ZIP_FILE to $MOUNT_POINT..."
fuse-zip "$ZIP_FILE" "$MOUNT_POINT"

echo "Mounted successfully."
