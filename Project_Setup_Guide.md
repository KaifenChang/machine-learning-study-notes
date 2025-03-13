# Simple Python Project Setup Guide

***
## Starting from GitHub

| Step | Instructions |
|------|-------------|
| **1. Clone Repository** | `git clone https://github.com/yourusername/your-repo.git`<br>`cd your-repo` |
| **2. Set Up Virtual Environment** | `python -m venv venv` |
| **3. Activate Environment** | Windows: `venv\Scripts\activate`<br>Mac/Linux: `source venv/bin/activate` |
| **4. Install Dependencies** | Install packages:<br>`pip install package_name`<br><br>Save dependencies:<br>`pip freeze > requirements.txt`<br><br>Install from requirements:<br>`pip install -r requirements.txt` |
| **5. Create .gitignore** | `echo "venv/\n__pycache__/\n*.py[cod]\n*$py.class\n.env\n.DS_Store" > .gitignore` |

***
## Starting Locally

| Step | Instructions |
|------|-------------|
| **1. Create Project Folder** | `mkdir my_project`<br>`cd my_project` |
| **2. Initialize Git** | `git init` |
| **3. Set Up Virtual Environment** | `python -m venv venv` |
| **4. Activate Environment** | Windows: `venv\Scripts\activate`<br>Mac/Linux: `source venv/bin/activate` |
| **5. Install Dependencies** | Install packages:<br>`pip install package_name`<br><br>Save dependencies:<br>`pip freeze > requirements.txt` |
| **6. Create .gitignore** | `echo "venv/\n__pycache__/\n*.py[cod]\n*$py.class\n.env\n.DS_Store" > .gitignore` |
| **7. Connect to GitHub** | Create GitHub repo via website first, then:<br>`git remote add origin https://github.com/yourusername/your-repo.git`<br>`git add .`<br>`git commit -m "Initial commit"`<br>`git push -u origin main` |

***
## Common .gitignore Contents

```
# Virtual Environment
venv/
env/
ENV/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment Variables
.env
.env.local

# OS specific
.DS_Store
Thumbs.db

# IDE specific
.idea/
.vscode/
*.swp
*.swo
```


