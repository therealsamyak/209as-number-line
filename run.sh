#!/bin/bash

# run.sh - Setup and run the particle system simulation

echo "================================"
echo "Particle System Simulation Setup"
echo "================================"
echo ""

# Check if uv is installed
if command -v uv &> /dev/null
then
    echo "✓ UV detected - using UV for fast package management"
    echo ""
    
    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment with UV..."
        uv venv
        echo "✓ Virtual environment created"
    else
        echo "✓ Virtual environment already exists"
    fi
    
    echo ""
    
    # Install dependencies using UV
    echo "Installing dependencies from requirements.txt..."
    uv pip install -r requirements.txt
    echo "✓ Dependencies installed"
    
    echo ""
    echo "================================"
    echo "Running Particle System Simulation"
    echo "================================"
    echo ""
    
    # Run the main script using UV
    uv run main2.py
else
    echo "UV not detected - using standard Python/pip"
    echo ""
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
        echo "✓ Virtual environment created"
    else
        echo "✓ Virtual environment already exists"
    fi
    
    echo ""
    
    # Activate virtual environment
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "✓ Virtual environment activated"
    
    echo ""
    
    # Install dependencies
    echo "Installing dependencies from requirements.txt..."
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    echo "✓ Dependencies installed"
    
    echo ""
    echo "================================"
    echo "Running Particle System Simulation"
    echo "================================"
    echo ""
    
    # Run the main script
    python main2.py
    
    # Deactivate virtual environment
    deactivate
fi 