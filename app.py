import streamlit as st
import pandas as pd
import zipfile
import tempfile
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
import pdfplumber

# Configure page
st.set_page_config(
    page_title="QC Form Cancellations Checker",
    layout="wide"
)

class SimpleCancellationProcessor:
    def __init__(self):
        self.files_data = []
        
    def extract_text_from_pdf(self, file_path):
        """Simple PDF text extraction"""
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        except Exception as e:
            print(f"PDF extraction failed: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path):
        """Extract text from DOCX files"""
        try:
            from docx import Document
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"DOCX extraction failed: {e}")
            return ""
    
    def extract_text_from_image(self, file_path):
        """Extract text from images using OCR"""
        try:
            from PIL import Image
            import pytesseract
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            print(f"Image OCR failed: {e}")
            return ""
    
    def extract_text_from_file(self, file_path):
        """Extract text from any supported file type"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
            return self.extract_text_from_image(file_path)
        elif file_ext == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                return ""
        else:
            return ""
    
    def extract_fields(self, text, filename):
        """Extract key fields from text"""
        fields = {
            'vin': [],
            'contract': [],
            'customer_name': [],
            'cancellation_date': [],
            'sale_date': [],
            'reason': [],
            'mileage': [],
            'total_refund': [],
            'dealer_ncb': [],
            'no_chargeback': []
        }
        
        # VIN patterns
        vin_patterns = [
            r'\b([A-HJ-NPR-Z0-9]{17})\b',
            r'VIN[:\s]*([A-HJ-NPR-Z0-9]{17})',
            r'Vehicle[:\s]*ID[:\s]*([A-HJ-NPR-Z0-9]{17})'
        ]
        for pattern in vin_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            fields['vin'].extend(matches)
        
        # Contract patterns
        contract_patterns = [
            r'Contract[:\s]*#?\s*:?\s*([A-Z0-9]{8,15})(?:\s|$|\n)',
            r'Contract[:\s]*Number[:\s]*([A-Z0-9]{8,15})(?:\s|$|\n)',
            r'PN[:\s]*([A-Z0-9]{8,15})(?:\s|$|\n)',
            r'DL[:\s]*([A-Z0-9]{8,15})(?:\s|$|\n)',
            r'#\s*([A-Z0-9]{8,15})(?:\s|$|\n)'
        ]
        for pattern in contract_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) >= 8 and match.isalnum():
                    fields['contract'].append(match)
        
        # Customer name patterns
        customer_patterns = [
            r'Customer[:\s]*Name[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?:\s|$|\n)',
            r'Name[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?:\s|$|\n)',
            r'Buyer[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?:\s|$|\n)'
        ]
        for pattern in customer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if 2 <= len(match.split()) <= 4:
                    fields['customer_name'].append(match)
        
        # Date patterns
        date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
            r'([A-Za-z]+ \d{1,2},? \d{4})'
        ]
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            fields['cancellation_date'].extend(matches)
            fields['sale_date'].extend(matches)
        
        # Reason patterns
        reason_patterns = [
            r'Reason[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})(?:\s|$|\n)',
            r'Cancellation[:\s]*Reason[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})(?:\s|$|\n)',
            r'Customer[:\s]*Request[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})(?:\s|$|\n)'
        ]
        for pattern in reason_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if 1 <= len(match.split()) <= 3:
                    fields['reason'].append(match)
        
        # Mileage patterns
        mileage_patterns = [
            r'(\d{3,8})\s*miles?',
            r'Mileage[:\s]*(\d{3,8})',
            r'(\d{1,3}(?:,\d{3})*)\s*miles?'
        ]
        for pattern in mileage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean up the match
                clean_match = re.sub(r'[^\d]', '', match)
                if len(clean_match) >= 3 and len(clean_match) <= 8:
                    fields['mileage'].append(clean_match)
        
        # Financial patterns
        money_patterns = [
            r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*dollars?'
        ]
        for pattern in money_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            fields['total_refund'].extend(matches)
            fields['dealer_ncb'].extend(matches)
        
        # NCB patterns
        ncb_patterns = [
            r'NCB[:\s]*(Yes|No|Y|N)',
            r'No[:\s]*Chargeback[:\s]*(Yes|No|Y|N)',
            r'Dealer[:\s]*NCB[:\s]*(Yes|No|Y|N)'
        ]
        for pattern in ncb_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            fields['dealer_ncb'].extend(matches)
            fields['no_chargeback'].extend(matches)
        
        return fields
    
    def process_zip(self, zip_file):
        """Process uploaded ZIP file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract ZIP
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Process each file
            all_fields = {
                'vin': [],
                'contract': [],
                'customer_name': [],
                'cancellation_date': [],
                'sale_date': [],
                'reason': [],
                'mileage': [],
                'total_refund': [],
                'dealer_ncb': [],
                'no_chargeback': []
            }
            
            files_processed = []
            
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    filename = os.path.basename(file_path)
                    
                    # Extract text
                    text = self.extract_text_from_file(file_path)
                    if text:
                        # Extract fields
                        fields = self.extract_fields(text, filename)
                        
                        # Add to combined results
                        for key, values in fields.items():
                            all_fields[key].extend(values)
                        
                        files_processed.append({
                            'filename': filename,
                            'fields': fields,
                            'text_length': len(text)
                        })
            
            return all_fields, files_processed
    
    def check_matches(self, all_fields):
        """Check for matches and conflicts"""
        results = {}
        
        # VIN match
        unique_vins = list(set(all_fields['vin']))
        results['vin_match'] = 'PASS' if len(unique_vins) == 1 else 'FAIL' if len(unique_vins) > 1 else 'INFO'
        results['vin_values'] = unique_vins
        
        # Contract match
        unique_contracts = list(set(all_fields['contract']))
        results['contract_match'] = 'PASS' if len(unique_contracts) == 1 else 'FAIL' if len(unique_contracts) > 1 else 'INFO'
        results['contract_values'] = unique_contracts
        
        # Customer name match
        unique_customers = list(set(all_fields['customer_name']))
        results['customer_match'] = 'PASS' if len(unique_customers) == 1 else 'FAIL' if len(unique_customers) > 1 else 'INFO'
        results['customer_values'] = unique_customers
        
        # Mileage match
        unique_mileages = list(set(all_fields['mileage']))
        results['mileage_match'] = 'PASS' if len(unique_mileages) == 1 else 'FAIL' if len(unique_mileages) > 1 else 'INFO'
        results['mileage_values'] = unique_mileages
        
        # Other fields
        results['cancellation_dates'] = list(set(all_fields['cancellation_date']))
        results['sale_dates'] = list(set(all_fields['sale_date']))
        results['reasons'] = list(set(all_fields['reason']))
        results['total_refunds'] = list(set(all_fields['total_refund']))
        results['dealer_ncb'] = list(set(all_fields['dealer_ncb']))
        results['no_chargeback'] = list(set(all_fields['no_chargeback']))
        
        return results

def main():
    st.title("📋 QC Form Cancellations Checker")
    st.write("Upload a ZIP file containing cancellation packet files to perform quality control checks.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload Cancellation Packet ZIP",
        type=['zip'],
        help="Upload a ZIP file containing cancellation packet files"
    )
    
    if uploaded_file is not None:
        processor = SimpleCancellationProcessor()
        
        with st.spinner("Processing files..."):
            # Process ZIP
            all_fields, files_processed = processor.process_zip(uploaded_file)
            
            # Check matches
            results = processor.check_matches(all_fields)
        
        # Display results
        st.success(f"✅ Processed {len(files_processed)} file(s)")
        
        # Data extraction summary
        st.subheader("📊 Data Extraction Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("VINs Found", len(all_fields['vin']))
            st.metric("Contracts Found", len(all_fields['contract']))
        
        with col2:
            st.metric("Customer Names", len(all_fields['customer_name']))
            st.metric("Reasons Found", len(all_fields['reason']))
        
        with col3:
            st.metric("Cancellation Dates", len(all_fields['cancellation_date']))
            st.metric("Sale Dates Found", len(all_fields['sale_date']))
        
        with col4:
            st.metric("Mileages Found", len(all_fields['mileage']))
            st.metric("Files Processed", len(files_processed))
        
        # QC Results
        st.subheader("📋 QC Checklist Results")
        
        # Create results DataFrame
        qc_data = []
        
        # VIN
        status_color = {'PASS': '🟢', 'FAIL': '🔴', 'INFO': '🟡'}
        qc_data.append({
            'Field': 'VIN Match',
            'Status': f"{status_color[results['vin_match']]} {results['vin_match']}",
            'Values': ', '.join(results['vin_values']) if results['vin_values'] else 'Not found'
        })
        
        # Contract
        qc_data.append({
            'Field': 'Contract Match',
            'Status': f"{status_color[results['contract_match']]} {results['contract_match']}",
            'Values': ', '.join(results['contract_values']) if results['contract_values'] else 'Not found'
        })
        
        # Customer Name
        qc_data.append({
            'Field': 'Customer Name',
            'Status': f"{status_color[results['customer_match']]} {results['customer_match']}",
            'Values': ', '.join(results['customer_values']) if results['customer_values'] else 'Not found'
        })
        
        # Mileage
        qc_data.append({
            'Field': 'Mileage Match',
            'Status': f"{status_color[results['mileage_match']]} {results['mileage_match']}",
            'Values': ', '.join(results['mileage_values']) if results['mileage_values'] else 'Not found'
        })
        
        # Other fields
        qc_data.append({
            'Field': 'Cancellation Dates',
            'Status': 'INFO',
            'Values': ', '.join(results['cancellation_dates']) if results['cancellation_dates'] else 'Not found'
        })
        
        qc_data.append({
            'Field': 'Sale Dates',
            'Status': 'INFO',
            'Values': ', '.join(results['sale_dates']) if results['sale_dates'] else 'Not found'
        })
        
        qc_data.append({
            'Field': 'Reasons',
            'Status': 'INFO',
            'Values': ', '.join(results['reasons']) if results['reasons'] else 'Not found'
        })
        
        qc_data.append({
            'Field': 'Total Refund',
            'Status': 'INFO',
            'Values': ', '.join(results['total_refunds']) if results['total_refunds'] else 'Not found'
        })
        
        qc_data.append({
            'Field': 'Dealer NCB',
            'Status': 'INFO',
            'Values': ', '.join(results['dealer_ncb']) if results['dealer_ncb'] else 'Not found'
        })
        
        qc_data.append({
            'Field': 'No Chargeback',
            'Status': 'INFO',
            'Values': ', '.join(results['no_chargeback']) if results['no_chargeback'] else 'Not found'
        })
        
        # Display results table
        df = pd.DataFrame(qc_data)
        st.dataframe(df, use_container_width=True)
        
        # File downloads
        st.subheader("📁 Files - Click to Download")
        cols = st.columns(4)
        for idx, file_data in enumerate(files_processed):
            col = cols[idx % 4]
            with col:
                filename = file_data['filename']
                st.download_button(
                    label=f"📄 {filename}",
                    data=open(os.path.join(temp_dir, filename), 'rb').read() if os.path.exists(os.path.join(temp_dir, filename)) else b"",
                    file_name=filename,
                    mime="application/octet-stream",
                    key=f"download_{idx}"
                )

if __name__ == "__main__":
    main()
