# KaggleSlayer

This repository contains code and resources for building an autonomous agent system to participate in Kaggle tabular competitions. The goal is to fully automate the pipeline — from data ingestion to submission.

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate  # On Windows
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Set Up Kaggle API
- Create a Kaggle account if you don't have one.
- Go to your Kaggle account settings and create a new API token. This will download a `kaggle.json` file.
- Place the `kaggle.json` file in the following directory:
  - On Linux/Mac: `~/.kaggle/kaggle.json`
  - On Windows: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
- Ensure the file has the correct permissions:
```bash
chmod 600 ~/.kaggle/kaggle.json
```
- Test the setup by running:
```bash
kaggle competitions list
```

### 5. Run Streamlit App
```bash
streamlit run app.py
```
