#!/bin/bash

# Remove existing virtual environment if it exists
rm -rf venv

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

export HF=your-hf-token
echo "Installation complete!" 

