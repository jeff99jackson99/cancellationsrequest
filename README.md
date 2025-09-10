# QC Form Cancellations Checker

A Streamlit application that processes ZIP files containing cancellation packet documents and performs quality control checks according to a standardized checklist. **Enhanced with automatic screenshot processing for NCB fee bucket analysis.**

## Features

- **Multi-format Support**: Processes PDF, DOCX, DOC, image (PNG, JPG, JPEG, TIFF), and text files
- **Advanced OCR Capability**: Uses Tesseract with OpenCV preprocessing for enhanced text extraction
- **Screenshot Processing**: Automatically detects and processes bucket screenshots to extract NCB fee data
- **Intelligent Grouping**: Groups files into packets based on Contract ID or VIN
- **Comprehensive QC Checks**: Evaluates 13 different quality control criteria
- **Enhanced NCB Analysis**: Extracts actual fee amounts from screenshots
- **Export Options**: Download results as CSV or JSON
- **Real-time Processing**: Instant feedback and progress tracking

## New Screenshot Processing Features

- **Automatic Detection**: Identifies bucket screenshots based on content analysis
- **NCB Fee Extraction**: Extracts Agent NCB, Dealer NCB, and total fee amounts
- **Image Preprocessing**: Uses OpenCV for enhanced OCR accuracy
- **Amount Validation**: Validates and displays extracted fee amounts
- **Visual Feedback**: Shows detailed screenshot processing results

## Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR (for image processing)
- OpenCV system dependencies

### Install Tesseract (macOS)

```bash
brew install tesseract
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Local Development

```bash
streamlit run app.py
```

### Quick Setup

```bash
# Full setup including Tesseract
make setup-full

# Create test data and run
make run-test
```

### Production Deployment

1. Push this repository to GitHub
2. Deploy to Streamlit Cloud:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Create new app
   - Select your GitHub repository
   - Choose `app.py` as the main file

## QC Checklist

The application evaluates each cancellation packet against these criteria:

1. **VIN match on all forms** - Ensures consistent VIN across all documents
2. **Contract match on all forms** - Verifies contract ID consistency
3. **Reason match across all forms** - Checks for consistent cancellation reasons
4. **Cancellation date match** - Validates date consistency (favors lender letters)
5. **90-day rule check** - Verifies if cancellation is past 90 days from sale
6. **Agent NCB Fee** - Detects agent no-chargeback fees (with amounts from screenshots)
7. **Dealer NCB Fee** - Detects dealer no-chargeback fees (with amounts from screenshots)
8. **Refund address** - Checks for alternate refund addresses (lender letters only)
9. **Signature collection** - Verifies signature presence
10. **Autohouse Contract** - Identifies Autohouse contracts
11. **Customer Direct Cancellation** - Detects dealer out of business or FF contracts
12. **Diversicare Contract** - Identifies Diversicare contracts
13. **PCMI Screenshot** - Detects NCB fee bucket screenshots and extracts data

## File Processing

### Supported File Types

- **PDF**: Uses pdfplumber for text extraction
- **DOCX**: Uses python-docx library
- **DOC**: Basic text extraction (fallback)
- **Images**: PNG, JPG, JPEG, TIFF with enhanced OCR
- **Text**: Plain text files

### Screenshot Processing

The app automatically detects bucket screenshots by looking for:
- NCB-related keywords (ncb, bucket, fee, agent, dealer, chargeback, pcmi)
- Fee amount patterns
- Table-like structures with financial data

### Field Extraction

The application uses regex patterns to extract:

- VINs (17-character alphanumeric)
- Contract IDs (various formats including GAP, PN, EL, etc.)
- Cancellation reasons
- Dates (multiple formats supported)
- Refund addresses
- Mileage values
- **NCB fee amounts** (from screenshots)
- Various flags and indicators

## Output Format

Results are provided in a structured table with:

- **Packet Key**: Unique identifier (Contract ID or VIN)
- **Files**: List of files in the packet
- **QC Status**: PASS/FAIL/INFO for each criterion
- **Canonical Values**: Extracted and normalized values
- **NCB Amounts**: Extracted fee amounts from screenshots
- **Export Options**: CSV and JSON download

## Status Meanings

- **PASS**: Exactly one consistent value found
- **FAIL**: Multiple conflicting values detected
- **INFO**: No value found (requires manual review)

## Screenshot Processing Details

The app provides detailed information about processed screenshots:

- **File Name**: Which screenshot was processed
- **Agent NCB Amount**: Extracted agent fee amount
- **Dealer NCB Amount**: Extracted dealer fee amount
- **Total Amount**: Overall fee amount detected
- **Processing Status**: Success/failure indicators

## Troubleshooting

### OCR Issues

If image processing fails:
1. Ensure Tesseract is installed: `brew install tesseract`
2. Check image quality and resolution
3. Verify file format support
4. Ensure OpenCV is properly installed

### Memory Issues

For large ZIP files:
1. Process smaller batches
2. Ensure sufficient system memory
3. Consider file size limits

### Screenshot Processing Issues

If screenshots aren't being processed correctly:
1. Ensure images contain clear text
2. Check that images are in supported formats
3. Verify the image contains NCB-related content
4. Try different image preprocessing settings

## Testing

The app includes comprehensive test data:

```bash
# Create test data with bucket screenshot
make create-test-data

# Test all dependencies
make test-deps

# Run with test data
make run-test
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.
