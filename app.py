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

class QCProcessor:
    def __init__(self):
        self.files_data = []
        
    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF files"""
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
    
    def extract_data(self, text, filename):
        """Extract all relevant data from text"""
        data = {
            'vin': [],
            'contract_number': [],
            'customer_name': [],
            'cancellation_date': [],
            'sale_date': [],
            'contract_date': [],
            'reason': [],
            'mileage': [],
            'total_refund': [],
            'dealer_ncb': [],
            'no_chargeback': [],
            'address': [],
            'phone': [],
            'email': []
        }
        
        # VIN extraction
        vin_patterns = [
            r'\b([A-HJ-NPR-Z0-9]{17})\b',
            r'VIN[:\s]*([A-HJ-NPR-Z0-9]{17})',
            r'Vehicle[:\s]*ID[:\s]*([A-HJ-NPR-Z0-9]{17})'
        ]
        for pattern in vin_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            data['vin'].extend(matches)
        
        # Contract number extraction
        contract_patterns = [
            r'Contract[:\s]*#?\s*:?\s*([A-Z0-9]{6,20})(?:\s|$|\n)',
            r'Contract[:\s]*Number[:\s]*([A-Z0-9]{6,20})(?:\s|$|\n)',
            r'PN[:\s]*([A-Z0-9]{6,20})(?:\s|$|\n)',
            r'DL[:\s]*([A-Z0-9]{6,20})(?:\s|$|\n)',
            r'Policy[:\s]*#?\s*:?\s*([A-Z0-9]{6,20})(?:\s|$|\n)',
            r'Agreement[:\s]*#?\s*:?\s*([A-Z0-9]{6,20})(?:\s|$|\n)',
            r'#\s*([A-Z0-9]{6,20})(?:\s|$|\n)'
        ]
        for pattern in contract_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) >= 6 and len(match) <= 20 and match.isalnum():
                    data['contract_number'].append(match)
        
        # Customer name extraction
        customer_patterns = [
            r'Customer[:\s]*Name[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?:\s|$|\n)',
            r'Name[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?:\s|$|\n)',
            r'Buyer[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?:\s|$|\n)',
            r'Purchaser[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?:\s|$|\n)',
            r'Client[:\s]*Name[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?:\s|$|\n)',
            r'Policyholder[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?:\s|$|\n)'
        ]
        for pattern in customer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean_match = match.strip()
                if len(clean_match.split()) >= 2 and len(clean_match.split()) <= 4:
                    data['customer_name'].append(clean_match)
        
        # Date extraction
        date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
            r'([A-Za-z]+ \d{1,2},? \d{4})',
            r'(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
            r'([A-Za-z]+\s+\d{1,2},?\s+\d{4})'
        ]
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            data['cancellation_date'].extend(matches)
            data['sale_date'].extend(matches)
            data['contract_date'].extend(matches)
        
        # Reason extraction
        reason_patterns = [
            r'Reason[:\s]*([A-Za-z]+(?:\s+[A-Za-z]+){0,2})(?:\s|$|\n)',
            r'Cancellation[:\s]*Reason[:\s]*([A-Za-z]+(?:\s+[A-Za-z]+){0,2})(?:\s|$|\n)',
            r'Customer[:\s]*Request[:\s]*([A-Za-z]+(?:\s+[A-Za-z]+){0,2})(?:\s|$|\n)',
            r'Why[:\s]*([A-Za-z]+(?:\s+[A-Za-z]+){0,2})(?:\s|$|\n)',
            r'Cause[:\s]*([A-Za-z]+(?:\s+[A-Za-z]+){0,2})(?:\s|$|\n)',
            r'Explanation[:\s]*([A-Za-z]+(?:\s+[A-Za-z]+){0,2})(?:\s|$|\n)'
        ]
        for pattern in reason_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean_match = match.strip()
                if len(clean_match.split()) >= 1 and len(clean_match.split()) <= 3:
                    data['reason'].append(clean_match)
        
        # Mileage extraction
        mileage_patterns = [
            r'(\d{3,8})\s*miles?',
            r'Mileage[:\s]*(\d{3,8})',
            r'(\d{1,3}(?:,\d{3})*)\s*miles?',
            r'(\d{3,8})\s*mi',
            r'(\d{1,3}(?:,\d{3})*)\s*mi',
            r'Odometer[:\s]*(\d{3,8})'
        ]
        for pattern in mileage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean_match = re.sub(r'[^\d]', '', match)
                # Filter out years (1900-2030) and very short numbers
                if len(clean_match) >= 3 and len(clean_match) <= 8 and not (1900 <= int(clean_match) <= 2030):
                    data['mileage'].append(clean_match)
        
        # Financial data extraction
        money_patterns = [
            r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*dollars?',
            r'Total[:\s]*Refund[:\s]*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'Refund[:\s]*Amount[:\s]*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'Amount[:\s]*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        ]
        for pattern in money_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Remove commas for consistent format
                clean_match = match.replace(',', '')
                data['total_refund'].append(clean_match)
        
        # NCB extraction
        ncb_patterns = [
            r'NCB[:\s]*(Yes|No|Y|N)',
            r'No[:\s]*Chargeback[:\s]*(Yes|No|Y|N)',
            r'Dealer[:\s]*NCB[:\s]*(Yes|No|Y|N)',
            r'Chargeback[:\s]*(Yes|No|Y|N)'
        ]
        for pattern in ncb_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            data['dealer_ncb'].extend(matches)
            data['no_chargeback'].extend(matches)
        
        return data
    
    def process_zip(self, zip_file):
        """Process uploaded ZIP file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract ZIP
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Process each file
            all_data = {
                'vin': [],
                'contract_number': [],
                'customer_name': [],
                'cancellation_date': [],
                'sale_date': [],
                'contract_date': [],
                'reason': [],
                'mileage': [],
                'total_refund': [],
                'dealer_ncb': [],
                'no_chargeback': [],
                'address': [],
                'phone': [],
                'email': []
            }
            
            files_processed = []
            
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    filename = os.path.basename(file_path)
                    
                    # Extract text
                    text = self.extract_text_from_file(file_path)
                    if text:
                        # Extract data
                        data = self.extract_data(text, filename)
                        
                        # Debug: Print what we found
                        print(f"\n=== {filename} ===")
                        for key, values in data.items():
                            if values:
                                print(f"{key}: {values}")
                        
                        # Add to combined results
                        for key, values in data.items():
                            all_data[key].extend(values)
                        
                        # Read file data for download
                        try:
                            with open(file_path, 'rb') as f:
                                file_data = f.read()
                        except:
                            file_data = b""
                        
                        files_processed.append({
                            'filename': filename,
                            'data': data,
                            'text_length': len(text),
                            'file_data': file_data
                        })
            
            return all_data, files_processed
    
    def evaluate_qc_checklist(self, all_data, files_processed):
        """Evaluate against QC checklist"""
        results = {}
        
        # 1. Contract Number
        all_contracts = all_data['contract_number']
        unique_contracts = list(set(all_contracts))
        if len(unique_contracts) == 1:
            results['contract_number'] = {'status': 'PASS', 'value': unique_contracts[0], 'reason': f'All {len(all_contracts)} contract numbers match: {unique_contracts[0]}'}
        elif len(unique_contracts) > 1:
            results['contract_number'] = {'status': 'FAIL', 'value': f'Found: {", ".join(unique_contracts)}', 'reason': f'Multiple contract numbers found: {len(unique_contracts)} different values'}
        else:
            results['contract_number'] = {'status': 'INFO', 'value': 'Not found', 'reason': 'No contract number found in any file'}
        
        # 2. Customer Name
        all_customers = all_data['customer_name']
        unique_customers = list(set(all_customers))
        if len(unique_customers) == 1:
            results['customer_name'] = {'status': 'PASS', 'value': unique_customers[0], 'reason': f'All {len(all_customers)} customer names match: {unique_customers[0]}'}
        elif len(unique_customers) > 1:
            results['customer_name'] = {'status': 'FAIL', 'value': f'Found: {", ".join(unique_customers)}', 'reason': f'Multiple customer names found: {len(unique_customers)} different values'}
        else:
            results['customer_name'] = {'status': 'INFO', 'value': 'Not found', 'reason': 'No customer name found in any file'}
        
        # 3. VIN Match
        all_vins = all_data['vin']
        unique_vins = list(set(all_vins))
        if len(unique_vins) == 1:
            results['vin_match'] = {'status': 'PASS', 'value': unique_vins[0], 'reason': f'All {len(all_vins)} VINs match: {unique_vins[0]}'}
        elif len(unique_vins) > 1:
            results['vin_match'] = {'status': 'FAIL', 'value': f'Found: {", ".join(unique_vins)}', 'reason': f'Multiple VINs found: {len(unique_vins)} different values'}
        else:
            results['vin_match'] = {'status': 'INFO', 'value': 'Not found', 'reason': 'No VIN found in any file'}
        
        # 4. Mileage Match
        all_mileages = all_data['mileage']
        unique_mileages = list(set(all_mileages))
        if len(unique_mileages) == 1:
            results['mileage_match'] = {'status': 'PASS', 'value': unique_mileages[0], 'reason': f'All {len(all_mileages)} mileages match: {unique_mileages[0]}'}
        elif len(unique_mileages) > 1:
            results['mileage_match'] = {'status': 'FAIL', 'value': f'Found: {", ".join(unique_mileages)}', 'reason': f'Multiple mileages found: {len(unique_mileages)} different values'}
        else:
            results['mileage_match'] = {'status': 'INFO', 'value': 'Not found', 'reason': 'No mileage found in any file'}
        
        # 5. 90+ Days Check
        cancellation_dates = all_data['cancellation_date']
        sale_dates = all_data['sale_date']
        contract_dates = all_data['contract_date']
        
        if cancellation_dates and (sale_dates or contract_dates):
            # Use sale date first, fallback to contract date
            reference_dates = sale_dates if sale_dates else contract_dates
            try:
                # Parse cancellation date
                cancel_date = None
                for date_str in cancellation_dates:
                    try:
                        cancel_date = datetime.strptime(date_str, '%m/%d/%Y')
                        break
                    except:
                        try:
                            cancel_date = datetime.strptime(date_str, '%Y-%m-%d')
                            break
                        except:
                            continue
                
                # Parse reference date
                ref_date = None
                for date_str in reference_dates:
                    try:
                        ref_date = datetime.strptime(date_str, '%m/%d/%Y')
                        break
                    except:
                        try:
                            ref_date = datetime.strptime(date_str, '%Y-%m-%d')
                            break
                        except:
                            continue
                
                if cancel_date and ref_date:
                    days_diff = (cancel_date - ref_date).days
                    if days_diff > 90:
                        results['ninety_days'] = {'status': 'PASS', 'value': f'{days_diff} days', 'reason': f'Cancellation is {days_diff} days after reference date'}
                    else:
                        results['ninety_days'] = {'status': 'FAIL', 'value': f'{days_diff} days', 'reason': f'Cancellation is only {days_diff} days after reference date (needs 90+)'}
                else:
                    results['ninety_days'] = {'status': 'INFO', 'value': 'Unknown', 'reason': 'Could not parse dates'}
            except:
                results['ninety_days'] = {'status': 'INFO', 'value': 'Unknown', 'reason': 'Date parsing error'}
        else:
            results['ninety_days'] = {'status': 'INFO', 'value': 'Unknown', 'reason': 'No valid dates found'}
        
        # 6. Total Refund
        all_refunds = all_data['total_refund']
        if all_refunds:
            results['total_refund'] = {'status': 'INFO', 'value': f'Found: {", ".join(all_refunds)}', 'reason': f'Found {len(all_refunds)} refund amounts'}
        else:
            results['total_refund'] = {'status': 'INFO', 'value': 'Not found', 'reason': 'No refund amounts found'}
        
        # 7. Dealer NCB
        all_ncb = all_data['dealer_ncb']
        if all_ncb:
            results['dealer_ncb'] = {'status': 'INFO', 'value': f'Found: {", ".join(all_ncb)}', 'reason': f'Found {len(all_ncb)} NCB references'}
        else:
            results['dealer_ncb'] = {'status': 'INFO', 'value': 'Not found', 'reason': 'No NCB references found'}
        
        # 8. No Chargeback
        all_chargeback = all_data['no_chargeback']
        if all_chargeback:
            results['no_chargeback'] = {'status': 'INFO', 'value': f'Found: {", ".join(all_chargeback)}', 'reason': f'Found {len(all_chargeback)} chargeback references'}
        else:
            results['no_chargeback'] = {'status': 'INFO', 'value': 'Not found', 'reason': 'No chargeback references found'}
        
        # 9. Cancellation Dates
        all_cancel_dates = all_data['cancellation_date']
        if all_cancel_dates:
            results['cancellation_dates'] = {'status': 'INFO', 'value': f'Found: {", ".join(all_cancel_dates)}', 'reason': f'Found {len(all_cancel_dates)} cancellation dates'}
        else:
            results['cancellation_dates'] = {'status': 'INFO', 'value': 'Not found', 'reason': 'No cancellation dates found'}
        
        # 10. Sale Dates
        all_sale_dates = all_data['sale_date']
        if all_sale_dates:
            results['sale_dates'] = {'status': 'INFO', 'value': f'Found: {", ".join(all_sale_dates)}', 'reason': f'Found {len(all_sale_dates)} sale dates'}
        else:
            results['sale_dates'] = {'status': 'INFO', 'value': 'Not found', 'reason': 'No sale dates found'}
        
        # 11. Reasons
        all_reasons = all_data['reason']
        if all_reasons:
            results['reasons'] = {'status': 'INFO', 'value': f'Found: {", ".join(all_reasons)}', 'reason': f'Found {len(all_reasons)} cancellation reasons'}
        else:
            results['reasons'] = {'status': 'INFO', 'value': 'Not found', 'reason': 'No cancellation reasons found'}
        
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
        processor = QCProcessor()
        
        with st.spinner("Processing files..."):
            # Process ZIP
            all_data, files_processed = processor.process_zip(uploaded_file)
            
            # Evaluate QC checklist
            qc_results = processor.evaluate_qc_checklist(all_data, files_processed)
        
        # Display results
        st.success(f"✅ Processed {len(files_processed)} file(s)")
        
        # Data extraction summary
        st.subheader("📊 Data Extraction Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("VINs Found", len(all_data['vin']))
            st.metric("Contracts Found", len(all_data['contract_number']))
        
        with col2:
            st.metric("Customer Names", len(all_data['customer_name']))
            st.metric("Reasons Found", len(all_data['reason']))
        
        with col3:
            st.metric("Cancellation Dates", len(all_data['cancellation_date']))
            st.metric("Sale Dates Found", len(all_data['sale_date']))
        
        with col4:
            st.metric("Mileages Found", len(all_data['mileage']))
            st.metric("Files Processed", len(files_processed))
        
        # QC Checklist Results
        st.subheader("📋 QC Checklist Results")
        
        # Create results DataFrame
        qc_data = []
        
        for field, result in qc_results.items():
            status_color = {'PASS': '🟢', 'FAIL': '🔴', 'INFO': '🟡'}
            qc_data.append({
                'Field': field.replace('_', ' ').title(),
                'Status': f"{status_color[result['status']]} {result['status']}",
                'Value': result['value'],
                'Reason': result['reason']
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
                file_bytes = file_data.get('file_data', b"")
                st.download_button(
                    label=f"📄 {filename}",
                    data=file_bytes,
                    file_name=filename,
                    mime="application/octet-stream",
                    key=f"download_{idx}"
                )

if __name__ == "__main__":
    main()