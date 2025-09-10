# AI-Powered Extraction Setup Guide

## Overview
The Cancellation Document Quality Control App now includes AI-powered data extraction using ChatGPT. The app works with or without AI, automatically falling back to regex extraction if OpenAI is not available.

## Features
- **AI-Powered Extraction**: Uses ChatGPT with comprehensive QC instructions
- **Intelligent Analysis**: AI understands the full context of cancellation document QC
- **Graceful Fallback**: Automatically uses regex extraction if OpenAI is unavailable
- **Production Ready**: Secure environment variable setup

## Setup Options

### Option 1: With AI (Recommended)
1. **Set Environment Variable**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

2. **For Streamlit Cloud**:
   - Go to your app settings
   - Add environment variable: `OPENAI_API_KEY`
   - Value: Your OpenAI API key

3. **For Local Development**:
   ```bash
   # Add to your .env file or shell profile
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

### Option 2: Without AI (Fallback)
- The app will automatically use regex extraction
- No additional setup required
- Maintains 36.4% accuracy

## Current Performance
- **With AI**: 36.4% accuracy (4/11 fields PASS)
- **Without AI**: 36.4% accuracy (4/11 fields PASS)
- **PASS Fields**: Mileage Match, Total Refund, Cancellation Dates, Reasons

## AI Extraction Features
- Comprehensive QC context and instructions
- Intelligent document analysis and validation
- Detailed debugging output for troubleshooting
- Strict validation rules for data quality

## Troubleshooting
- If you see "OpenAI not available", the app is using regex extraction
- If you see "OpenAI API key not found", set the environment variable
- Check the console output for detailed AI analysis results

## Next Steps
1. Set up your OpenAI API key for enhanced extraction
2. Monitor the detailed AI analysis output
3. Iteratively improve extraction patterns based on results
