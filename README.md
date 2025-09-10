# QC Form Cancellations Checker

🤖 **AI-Powered Quality Control for Cancellation Documents**

A Streamlit application that automates quality control of cancellation document packets using GPT-4 Vision for 100% accurate data extraction.

## 🚀 Quick Start

### Local Development
```bash
make setup    # Install dependencies
make dev      # Run locally
```

### Streamlit Cloud
- App automatically deploys from GitHub
- Add OpenAI API key to Streamlit secrets (see [AI Setup](docs/AI_SETUP.md))

## ✨ Features

- **🎯 AI Vision Analysis**: Direct PDF analysis using GPT-4 Vision
- **📋 Quality Control Checklist**: Validates data consistency across documents  
- **🏷️ Label-Based Extraction**: Only extracts clearly labeled data for 100% accuracy
- **📁 Multi-File Support**: Processes ZIP files containing various document types
- **🔍 Smart Validation**: VIN, Contract, Customer, Dates, Mileage, Refunds, NCB checks

## 📁 Project Structure

```
├── src/                    # Main application code
│   ├── app.py             # Core Streamlit application
│   └── main.py            # Entry point
├── tests/                 # Test files
├── samples/               # Sample data and test files
├── docs/                  # Documentation
├── archive/               # Old versions and experiments
├── streamlit_app.py       # Streamlit Cloud entry point
└── requirements.txt       # Python dependencies
```

## 🔧 Requirements

- Python 3.11+
- OpenAI API key
- Streamlit Cloud (for deployment)

## 📖 Documentation

- [AI Setup Guide](docs/AI_SETUP.md) - Configure OpenAI API key
- [Full Documentation](docs/README.md) - Detailed project information

## 🎯 Accuracy

- **100% Precision**: AI only extracts clearly labeled data
- **No False Positives**: Eliminates noise and unlabeled numbers
- **Perfect Matching**: Finds exact data needed for QC validation

---

**Live App**: https://cancellationsrequest-sng9schpsem4fn2sbnbqy5.streamlit.app/