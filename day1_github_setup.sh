#!/bin/bash
# ============================================================
# DAY 1 — GitHub Setup Script
# Run this script ONCE on your local machine
# Author: Chalani Dinitha (20211032)
# ============================================================
#
# HOW TO USE:
#   1. Copy your entire project folder to your Mac/PC
#   2. Open Terminal and navigate to the project folder
#   3. Run: bash day1_github_setup.sh
#
# BEFORE RUNNING:
#   - Create a GitHub account at https://github.com
#   - Create a NEW repository called: hallucination-detection-llm
#     (Go to https://github.com/new)
#   - Set it to PUBLIC
#   - Do NOT add README or .gitignore (we have our own)
# ============================================================

echo "=============================================="
echo "  Day 1 — GitHub Repository Setup"
echo "  Hallucination Detection System"
echo "=============================================="
echo ""

# Step 1: Configure your Git identity
echo "Step 1: Configuring Git identity..."
git config --global user.name "Chalani Dinitha"
git config --global user.email "chalani.20211032@iit.ac.lk"
echo "✓ Git identity set"
echo ""

# Step 2: Initialize the repository
echo "Step 2: Initializing Git repository..."
git init
echo "✓ Git repository initialized"
echo ""

# Step 3: Create and switch to setup branch
echo "Step 3: Creating setup branch..."
git checkout -b setup/project-structure
echo "✓ Branch 'setup/project-structure' created"
echo ""

# Step 4: Stage all files
echo "Step 4: Staging all files..."
git add .
echo "✓ All files staged"
echo ""

# Step 5: First commit
echo "Step 5: Creating first commit..."
git commit -m "chore(setup): initial project structure, config, and documentation

- Create complete folder structure for all 4 detection modules
- Add requirements.txt with all Python dependencies
- Add .gitignore (excludes model weights, .env, __pycache__)
- Add .env.example with all configuration options
- Add README.md with project overview and setup instructions
- Add backend/config.py with centralized configuration
- Add docker-compose.yml for containerized deployment
- Add docs/architecture.md with module design overview

Addresses: Gap 1 (real-time detection) and Gap 2 (token-level localization)
Research: Fine-Grained Hallucination Detection Using Hidden States"

echo "✓ First commit created"
echo ""

# Step 6: Add remote origin
echo "Step 6: Adding GitHub remote..."
git remote add origin https://github.com/dinitha2004/hallucination-detection-llm.git
echo "✓ Remote origin added"
echo ""

# Step 7: Push to GitHub
echo "Step 7: Pushing to GitHub..."
echo "(You may be asked for your GitHub username and password/token)"
git push -u origin setup/project-structure
echo ""

echo "=============================================="
echo "  ✅ Day 1 COMPLETE!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Go to https://github.com/dinitha2004/hallucination-detection-llm"
echo "  2. You should see your 'setup/project-structure' branch"
echo "  3. Create a Pull Request to merge into main"
echo "  4. Come back to start Day 2 (LLM Model Loading)"
echo ""
echo "Git status:"
git log --oneline
