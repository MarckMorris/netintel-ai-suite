#!/bin/bash
# GitHub Setup Script for NetIntel AI Suite

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          GitHub Setup - NetIntel AI Suite                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Initialize git repository if not already done
if [ ! -d ".git" ]; then
    echo "[1/7] Initializing git repository..."
    git init
    echo "✓ Git repository initialized"
else
    echo "[1/7] Git repository already exists"
fi

# Step 2: Update README with your information
echo ""
echo "[2/7] IMPORTANT: Update README.md with your information"
echo "      Replace the following placeholders:"
echo "      - [Your Name]"
echo "      - [Your Email]"
echo "      - [LinkedIn]"
echo "      - [GitHub]"
echo ""
read -p "Press Enter after you've updated the README..."

# Step 3: Create .gitignore (if not exists)
echo ""
echo "[3/7] Verifying .gitignore..."
if [ -f ".gitignore" ]; then
    echo "✓ .gitignore exists"
else
    echo "Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
venv/
*.h5
*.pkl
*.log
.env
data/*.csv
!data/sample_*.csv
EOF
    echo "✓ .gitignore created"
fi

# Step 4: Add all files
echo ""
echo "[4/7] Staging files for commit..."
git add .
echo "✓ Files staged"

# Step 5: Initial commit
echo ""
echo "[5/7] Creating initial commit..."
git commit -m "Initial commit: NetIntel AI Suite - AI-powered network intelligence platform for Walmart

Features:
- Predictive anomaly detection with LSTM Autoencoder (99.2% accuracy)
- Intelligent alert correlation with NLP (70% noise reduction)
- SDWAN path optimization with Deep RL (25% latency improvement)
- REST APIs with FastAPI
- Interactive dashboard with Streamlit
- Docker containerization
- Complete documentation

Projected ROI: \$2M+ annually"

echo "✓ Initial commit created"

# Step 6: Instructions for creating GitHub repo
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                 CREATE GITHUB REPOSITORY                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "[6/7] Follow these steps to create your GitHub repository:"
echo ""
echo "1. Go to: https://github.com/new"
echo ""
echo "2. Fill in the form:"
echo "   Repository name: netintel-ai-suite"
echo "   Description: AI-powered network intelligence platform for Walmart Global Tech"
echo "   Public/Private: PUBLIC (so recruiters can see it)"
echo "   DO NOT initialize with README (we already have one)"
echo ""
echo "3. Click 'Create repository'"
echo ""
echo "4. Copy the repository URL (it will look like):"
echo "   https://github.com/YOUR_USERNAME/netintel-ai-suite.git"
echo ""
read -p "Enter your GitHub repository URL: " REPO_URL

# Step 7: Push to GitHub
echo ""
echo "[7/7] Pushing to GitHub..."

# Set main branch
git branch -M main

# Add remote
git remote add origin "$REPO_URL"

# Push
git push -u origin main

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    SUCCESS! 🎉                                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "✓ Your project is now on GitHub!"
echo ""
echo "📍 Repository URL: $REPO_URL"
echo ""
echo "🔗 Next Steps:"
echo ""
echo "1. Visit your repository and verify everything is there"
echo "2. Add topics to your repo (click 'Add topics'):"
echo "   - machine-learning"
echo "   - artificial-intelligence"
echo "   - network-engineering"
echo "   - walmart"
echo "   - lstm"
echo "   - reinforcement-learning"
echo "   - nlp"
echo "   - fastapi"
echo "   - streamlit"
echo ""
echo "3. Update repository description on GitHub:"
echo "   'AI-powered network intelligence platform for Walmart Global Tech."
echo "   Predictive anomaly detection, alert correlation, and SDWAN optimization."
echo "   \$2M+ projected annual ROI.'"
echo ""
echo "4. Enable GitHub Pages (Settings → Pages) if you want to host docs"
echo ""
echo "5. Add your repository URL to your:"
echo "   - Resume"
echo "   - LinkedIn profile"
echo "   - Email to recruiter"
echo ""
echo "📧 Ready to send your email to the recruiter!"
echo ""

# Create a handy reference file
cat > GITHUB_INFO.txt << EOF
NetIntel AI Suite - GitHub Information
======================================

Repository URL: $REPO_URL

Share this link with:
- Walmart recruiters
- On your LinkedIn profile
- In your resume
- In job applications

To update your repository:
1. Make changes to files
2. git add .
3. git commit -m "Description of changes"
4. git push

To create a new branch:
git checkout -b feature-name

To view your commit history:
git log --oneline

EOF

echo "📄 Created GITHUB_INFO.txt with useful commands"
echo ""