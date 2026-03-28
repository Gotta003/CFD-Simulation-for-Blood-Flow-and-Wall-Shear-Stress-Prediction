#!/usr/bin/env bash
# Run:
#   chmod +x setup.sh
#   ./setup.sh
# After the script finishes, activate environment
#  source .venv/bin/activate

set -e
# Color setup
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

info() {
    echo -e "${CYAN}[INFO]${RESET} $*";
}
success() {
    echo -e "${GREEN}[OK]${RESET} $*";
}
warn() {
    echo -e "${YELLOW}[WARN]${RESET} $*";
}
error() {
    echo -e "${RED}[ERROR]${RESET} $*";
}

echo -e "Setup Starting...\n";

# 0) Check python is installed
#PYTHON3
info "Checking python3 is installed...";
command -v python3 >/dev/null 2>&1 || { error "Python3 is not installed. Please install Python3 and try again."; exit 1; }
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')");
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1);
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2);
if [[ $PYTHON_MAJOR -lt 3 || ($PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 12) ]]; then
    error "Python 3.12+ required (found $PYTHON_VERSION)."
fi
success "Python3 is installed (version $PYTHON_VERSION).";
#PIP3
command -v pip3 >/dev/null 2>&1 || error "pip3 not found"
success "pip3 is installed."

# 1) Create virtual environment
VENV_DIR=".evar_env"
if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment directory '$VENV_DIR' already exists. Skipping creation."
    warn "To rebuild from scratch: rm -rf $VENV_DIR && ./setup.sh"
else
    info "Creating virtual environment in './$VENV_DIR'..."
    python3 -m venv $VENV_DIR
    success "Virtual environment created"
fi

source "$VENV_DIR/bin/activate"
success "Virtual environment activated"

# 2) Install Packets
info "Upgrading pip, setuptools, wheel"
pip install --upgrade pip setuptools wheel -q
success "pip upgraded"
#PyTorch Installation
info "Installing PyTorch..."
if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[\d.]+")
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    info "NVIDIA GPU detected — CUDA $CUDA_VER"

    if [ "$CUDA_MAJOR" -ge 12 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        PYG_CUDA="cu121"
    elif [ "$CUDA_MAJOR" -ge 11 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        PYG_CUDA="cu118"
    else
        warn "Older CUDA ($CUDA_VER). Falling back to CPU build."
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        PYG_CUDA="cpu"
    fi
else
    info "No GPU detected — installing CPU build of PyTorch"
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    PYG_CUDA="cpu"
fi
pip install torch --index-url $TORCH_INDEX -q
success "PyTorch installed"
#Pytorch Geometric Installation (GNN Framework)
TORCH=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
PYG_BASE="https://data.pyg.org/whl/torch-${TORCH}+${PYG_CUDA}.html"
pip install torch-geometric -f $PYG_BASE -q
success "PyTorch Geometric installed"

#Other Packets Installation
info "Installing packages from requirements.txt..."
pip install -r requirements.txt -q
success "Packages installed"

# 3) Check Installations
info "Checking package installations..."
python3 packages_checker.py 
success "All packages are installed correctly"

# 4) Create structure
info "Creating project folder structure..."
mkdir -p data/{dataset.vtp_files,reports,labels,pointclouds}
mkdir -p src/{extraction,models,explainability,visualization}
mkdir -p outputs/{dataset,features,models,shap_plots,mesh_heatmaps,splits}
mkdir -p outputs/models/{xgboost,gnn_pinn}
mkdir -p scripts/
mkdir -p notebooks
touch src/__init__.py
touch src/extraction/__init__.py
touch src/models/__init__.py
touch src/explainability/__init__.py
touch src/visualization/__init__.py
success "Project structure created"