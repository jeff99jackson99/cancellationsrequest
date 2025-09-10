# QC Form Cancellations Checker

A Streamlit application that automates quality control of cancellation document packets using AI vision analysis.

## Features

- **AI-Powered Extraction**: Uses GPT-4 Vision to analyze PDFs directly
- **Quality Control Checklist**: Validates data consistency across documents
- **Label-Based Analysis**: Only extracts clearly labeled data for 100% accuracy
- **Multiple File Support**: Processes ZIP files containing various document types

## Quick Start

1. **Local Development**:
   ```bash
   make setup
   make dev
   ```

2. **Streamlit Cloud**: 
   - App automatically deploys from GitHub
   - Add OpenAI API key to Streamlit secrets

## Project Structure

```
├── src/                    # Main application code
│   ├── app.py             # Core Streamlit application
│   └── main.py            # Entry point
├── tests/                 # Test files
├── samples/               # Sample data and test files
├── docs/                  # Documentation
├── archive/               # Old versions and experiments
└── streamlit_app.py       # Streamlit Cloud entry point
```

## AI Setup

See [AI_SETUP.md](AI_SETUP.md) for OpenAI API key configuration.

## Requirements

- Python 3.11+
- OpenAI API key
- Streamlit Cloud (for deployment)

## License

Private project for cancellation document quality control.