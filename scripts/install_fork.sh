#!/bin/bash
# VectorBT Fork Installation Script
#
# This script downloads VectorBT v0.26.2 source from PyPI into lib/vectorbt
# and installs it in editable mode.
#
# Note: VectorBT v0.26.2 is not tagged in the Git repo, so we download
# the source distribution from PyPI to ensure version pinning.
#
# Usage: ./scripts/install_fork.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LIB_DIR="$PROJECT_ROOT/lib"
VECTORBT_DIR="$LIB_DIR/vectorbt"
VENV_DIR="$PROJECT_ROOT/.venv"
VECTORBT_VERSION="0.26.2"

echo "Installing VectorBT v${VECTORBT_VERSION} fork..."

# Use project virtual environment if it exists
if [ -d "$VENV_DIR" ]; then
    echo "Using project virtual environment: $VENV_DIR"
    source "$VENV_DIR/bin/activate"
fi

# Check if lib directory exists
if [ ! -d "$LIB_DIR" ]; then
    mkdir -p "$LIB_DIR"
fi

# Check if vectorbt already exists
if [ -d "$VECTORBT_DIR" ]; then
    echo "VectorBT fork already exists at $VECTORBT_DIR"
    echo "To reinstall, remove the directory first: rm -rf $VECTORBT_DIR"
    exit 0
fi

# Create temp directory for download
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Download VectorBT source from PyPI
echo "Downloading VectorBT v${VECTORBT_VERSION} from PyPI..."
pip download vectorbt==${VECTORBT_VERSION} --no-deps --no-binary :all: -d "$TEMP_DIR"

# Extract the source tarball
echo "Extracting source..."
TARBALL=$(ls "$TEMP_DIR"/vectorbt-*.tar.gz 2>/dev/null | head -1)
if [ -z "$TARBALL" ]; then
    echo "Error: Could not find vectorbt source tarball"
    exit 1
fi

tar -xzf "$TARBALL" -C "$TEMP_DIR"

# Move extracted directory to lib/vectorbt
EXTRACTED_DIR=$(ls -d "$TEMP_DIR"/vectorbt-* 2>/dev/null | grep -v ".tar.gz" | head -1)
if [ -z "$EXTRACTED_DIR" ]; then
    echo "Error: Could not find extracted vectorbt directory"
    exit 1
fi

mv "$EXTRACTED_DIR" "$VECTORBT_DIR"

# Initialize as a git repo to track our fork
echo "Initializing git repository for fork tracking..."
cd "$VECTORBT_DIR"
git init -q
git add -A
git commit -q -m "VectorBT v${VECTORBT_VERSION} - initial fork from PyPI source"

# Install VectorBT in editable mode
echo "Installing VectorBT in editable mode..."
pip install -e .

echo ""
echo "VectorBT v${VECTORBT_VERSION} fork installation complete!"
echo "Location: $VECTORBT_DIR"
echo ""
echo "Verification:"
PYTHONPATH="" python3 -c "import vectorbt as vbt; print(f'  VectorBT version: {vbt.__version__}')"

echo ""
echo "Note: If you have another 'vectorbt' directory in your home folder or PYTHONPATH,"
echo "you may need to clear PYTHONPATH or remove the conflicting directory to avoid import conflicts."
