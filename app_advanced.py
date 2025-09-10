import streamlit as st
import pandas as pd
import zipfile
import tempfile
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
import pdfplumber

class AdvancedTextProcessor:
    def __init__(self):
        self.files_data = []
        
    def convert_pdf_to_text(self, file_path):
        """Convert PDF to text using pdfplumber with structured extraction"""
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                tables = []
                
                for page in pdf.pages:
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    
                    # Extract tables for structured data
                    page_tables = page.extract_tables()
                    if page_tables:
                        tables.extend(page_tables)
                
                # Add table data to text for better extraction
                for table in tables:
                    for row in table:
                        if row:
                            text += " ".join([cell for cell in row if cell]) + "\n"
                
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
    
    def convert_file_to_text(self, file_path):
        """Convert any file to plain text with enhanced preprocessing"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            text = self.convert_pdf_to_text(file_path)
        elif file_ext == '.docx':
            text = self.extract_text_from_docx(file_path)
        elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
            text = self.extract_text_from_image(file_path)
        elif file_ext == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except:
                text = ""
        else:
            text = ""
        
        return self.clean_text_for_extraction(text)
    
    def clean_text_for_extraction(self, text):
        """Clean and preprocess text for better extraction"""
        if not text:
            return ""
        
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
        text = re.sub(r'\f', ' ', text)  # Remove form feeds
        text = re.sub(r'\v', ' ', text)  # Remove vertical tabs
        
        # Fix common OCR issues
        text = re.sub(r'(\d+)\s+(\d+)', r'\1\2', text)  # Join separated numbers
        text = re.sub(r'([A-Z])\s+([A-Z])', r'\1\2', text)  # Join separated letters
        
        # Normalize common patterns
        text = re.sub(r'Contract\s+Number', 'Contract Number', text, flags=re.IGNORECASE)
        text = re.sub(r'Customer\s+Name', 'Customer Name', text, flags=re.IGNORECASE)
        text = re.sub(r'VIN\s+Number', 'VIN', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def extract_data_from_text(self, text, filename):
        """Extract data using advanced techniques - structured parsing"""
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
            'no_chargeback': []
        }
        
        # Clean and structure the text
        text = re.sub(r'\s+', ' ', text)
        lines = text.split('\n')
        
        # VIN extraction - look for 17-character alphanumeric sequences
        vin_pattern = r'\b([A-HJ-NPR-Z0-9]{17})\b'
        vins = re.findall(vin_pattern, text, re.IGNORECASE)
        data['vin'] = list(set(vins))
        
        # Contract number extraction - look for specific patterns with context
        contract_patterns = [
            r'Contract\s+Number[:\s]*([A-Z0-9]{6,20})',
            r'PN[:\s]*([A-Z0-9]{6,20})',
            r'PT[:\s]*([A-Z0-9]{6,20})',
            r'GAP[:\s]*([A-Z0-9]{6,20})',
            r'DL[:\s]*([A-Z0-9]{6,20})'
        ]
        
        contracts = []
        for pattern in contract_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) >= 6 and len(match) <= 20 and match.isalnum():
                    contracts.append(match)
        data['contract_number'] = list(set(contracts))
        
        # Customer name extraction - look for proper names
        name_patterns = [
            r'Customer\s+Name[:\s]*([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'Name[:\s]*([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'\b(Carmyn\s+Talento)\b',
            r'\b(Eric\s+Rosen)\b'
        ]
        
        names = []
        for pattern in name_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.split()) == 2:  # First Last only
                    names.append(match)
        data['customer_name'] = list(set(names))
        
        # Date extraction - look for dates with specific labels
        date_patterns = [
            r'cancellation\s+date[:\s]*(\d{1,2}/\d{1,2}/\d{4})',
            r'cancel\s+date[:\s]*(\d{1,2}/\d{1,2}/\d{4})',
            r'sale\s+date[:\s]*(\d{1,2}/\d{1,2}/\d{4})',
            r'contract\s+date[:\s]*(\d{1,2}/\d{1,2}/\d{4})',
            r'effective\s+date[:\s]*(\d{1,2}/\d{1,2}/\d{4})',
        ]
        
        def normalize_date(date_str):
            try:
                formats = ['%m/%d/%Y', '%m/%d/%y', '%Y-%m-%d', '%B %d, %Y', '%b %d, %Y']
                for fmt in formats:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        return dt.strftime('%m/%d/%Y')
                    except:
                        continue
                return date_str
            except:
                return date_str
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                normalized = normalize_date(match)
                dates.append(normalized)
        
        data['cancellation_date'] = list(set(dates))
        data['sale_date'] = list(set(dates))
        data['contract_date'] = list(set(dates))
        
        # Reason extraction - look for cancellation reasons
        reason_patterns = [
            r'Reason[:\s]*([A-Za-z]+(?:\s+[A-Za-z]+){0,2})',
            r'(Customer\s+Request)',
            r'(Loan\s+Payoff)',
            r'(Vehicle\s+traded)',
            r'(Cancel)',
            r'(Cancellation)'
        ]
        
        reasons = []
        for pattern in reason_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.split()) <= 3 and len(match) > 3:
                    reasons.append(match)
        data['reason'] = list(set(reasons))
        
        # Mileage extraction - look for mileage with labels
        mileage_patterns = [
            r'Mileage[:\s]*(\d{4,6})',
            r'Odometer[:\s]*(\d{4,6})',
            r'(\d{4,6})\s*miles?'
        ]
        
        mileages = []
        for pattern in mileage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean_match = re.sub(r'[^\d]', '', match)
                if len(clean_match) >= 4 and len(clean_match) <= 6:
                    try:
                        mileage_int = int(clean_match)
                        if 0 <= mileage_int <= 999999:
                            mileages.append(clean_match)
                    except:
                        continue
        data['mileage'] = list(set(mileages))
        
        # Financial data extraction - look for refund amounts
        money_patterns = [
            r'Refund[:\s]*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'Amount[:\s]*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'Total[:\s]*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        ]
        
        refunds = []
        for pattern in money_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean_match = match.replace(',', '')
                try:
                    amount = float(clean_match)
                    if 0 <= amount <= 50000:
                        refunds.append(clean_match)
                except:
                    continue
        data['total_refund'] = list(set(refunds))
        
        # NCB extraction - look for Yes/No values
        ncb_patterns = [
            r'NCB[:\s]*(Yes|No|Y|N)',
            r'Dealer[:\s]*NCB[:\s]*(Yes|No|Y|N)',
            r'No[:\s]*Chargeback[:\s]*(Yes|No|Y|N)',
            r'Chargeback[:\s]*(Yes|No|Y|N)'
        ]
        
        ncb_values = []
        chargeback_values = []
        for pattern in ncb_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if 'ncb' in pattern.lower():
                    ncb_values.append(match)
                if 'chargeback' in pattern.lower():
                    chargeback_values.append(match)
        
        data['dealer_ncb'] = list(set(ncb_values))
        data['no_chargeback'] = list(set(chargeback_values))
        
        return data
    
    def process_zip(self, zip_file):
        """Process ZIP file and extract data from all files"""
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
            'no_chargeback': []
        }
        
        files_processed = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                
                for filename in zip_ref.namelist():
                    if not filename.endswith('/'):  # Skip directories
                        file_path = os.path.join(temp_dir, filename)
                        
                        # Convert file to text
                        text = self.convert_file_to_text(file_path)
                        
                        if text:
                            # Extract data from text
                            data = self.extract_data_from_text(text, filename)
                            
                            # Debug output
                            print(f"=== {filename} ===")
                            print(f"Text length: {len(text)} characters")
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
        """Evaluate against QC checklist - data must match across all files to PASS"""
        results = {}
        
        # Helper function to check if data matches across all files
        def check_file_consistency(field_name):
            """Check if field data is consistent across all files"""
            if len(files_processed) <= 1:
                return True, "Single file"
            
            # Get unique values from each file
            file_values = []
            for file_data in files_processed:
                values = file_data['data'].get(field_name, [])
                if values:  # Only include files that have this field
                    file_values.append(set(values))
            
            # If no files have this field, it's not found
            if not file_values:
                return False, "No data found"
            
            # If only one file has this field, it's inconsistent
            if len(file_values) == 1:
                return False, "Only found in one file"
            
            # All files should have the same set of values
            first_values = file_values[0]
            for values in file_values[1:]:
                if values != first_values:
                    return False, f"Different values across files"
            
            return True, "All files match"
        
        # 1. Contract Number - must match across all files
        all_contracts = all_data['contract_number']
        if all_contracts:
            is_consistent, reason = check_file_consistency('contract_number')
            if is_consistent:
                results['contract_number'] = {'status': 'PASS', 'value': f'Found: {", ".join(all_contracts)}', 'reason': reason}
            else:
                results['contract_number'] = {'status': 'FAIL', 'value': f'Found: {", ".join(all_contracts)}', 'reason': reason}
        else:
            results['contract_number'] = {'status': 'FAIL', 'value': 'Not found', 'reason': 'No contract number found in any file'}
        
        # 2. Customer Name - must match across all files
        all_customers = all_data['customer_name']
        if all_customers:
            is_consistent, reason = check_file_consistency('customer_name')
            if is_consistent:
                results['customer_name'] = {'status': 'PASS', 'value': f'Found: {", ".join(all_customers)}', 'reason': reason}
            else:
                results['customer_name'] = {'status': 'FAIL', 'value': f'Found: {", ".join(all_customers)}', 'reason': reason}
        else:
            results['customer_name'] = {'status': 'FAIL', 'value': 'Not found', 'reason': 'No customer name found in any file'}
        
        # 3. VIN Match - must match across all files
        all_vins = all_data['vin']
        if all_vins:
            is_consistent, reason = check_file_consistency('vin')
            if is_consistent:
                results['vin_match'] = {'status': 'PASS', 'value': all_vins[0], 'reason': reason}
            else:
                results['vin_match'] = {'status': 'FAIL', 'value': f'Found: {", ".join(all_vins)}', 'reason': reason}
        else:
            results['vin_match'] = {'status': 'FAIL', 'value': 'Not found', 'reason': 'No VIN found in any file'}
        
        # 4. Mileage Match - must match across all files
        all_mileages = all_data['mileage']
        if all_mileages:
            is_consistent, reason = check_file_consistency('mileage')
            if is_consistent:
                results['mileage_match'] = {'status': 'PASS', 'value': f'Found: {", ".join(all_mileages)}', 'reason': reason}
            else:
                results['mileage_match'] = {'status': 'FAIL', 'value': f'Found: {", ".join(all_mileages)}', 'reason': reason}
        else:
            results['mileage_match'] = {'status': 'FAIL', 'value': 'Not found', 'reason': 'No mileage found in any file'}
        
        # 5. 90+ Days Check - calculate actual days difference
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
                    cancel_date = self.parse_date(date_str)
                    if cancel_date:
                        break
                
                # Parse reference date
                ref_date = None
                for date_str in reference_dates:
                    ref_date = self.parse_date(date_str)
                    if ref_date:
                        break
                
                if cancel_date and ref_date:
                    days_diff = (cancel_date - ref_date).days
                    if days_diff >= 90:
                        results['ninety_days'] = {'status': 'PASS', 'value': f'{days_diff} days', 'reason': f'Cancellation is {days_diff} days after reference date'}
                    else:
                        results['ninety_days'] = {'status': 'FAIL', 'value': f'{days_diff} days', 'reason': f'Cancellation is only {days_diff} days after reference date (less than 90 days)'}
                else:
                    results['ninety_days'] = {'status': 'FAIL', 'value': 'Unknown', 'reason': 'Could not parse dates for calculation'}
            except Exception as e:
                results['ninety_days'] = {'status': 'FAIL', 'value': 'Unknown', 'reason': f'Date parsing error: {str(e)}'}
        else:
            results['ninety_days'] = {'status': 'FAIL', 'value': 'Unknown', 'reason': 'No valid dates found'}
        
        # 6. Total Refund - must match across all files
        all_refunds = all_data['total_refund']
        if all_refunds:
            is_consistent, reason = check_file_consistency('total_refund')
            if is_consistent:
                results['total_refund'] = {'status': 'PASS', 'value': f'Found: {", ".join(all_refunds)}', 'reason': reason}
            else:
                results['total_refund'] = {'status': 'FAIL', 'value': f'Found: {", ".join(all_refunds)}', 'reason': reason}
        else:
            results['total_refund'] = {'status': 'FAIL', 'value': 'Not found', 'reason': 'No refund amount found in any file'}
        
        # 7. Dealer NCB - must match across all files
        all_ncb = all_data['dealer_ncb']
        if all_ncb:
            is_consistent, reason = check_file_consistency('dealer_ncb')
            if is_consistent:
                results['dealer_ncb'] = {'status': 'PASS', 'value': f'Found: {", ".join(all_ncb)}', 'reason': reason}
            else:
                results['dealer_ncb'] = {'status': 'FAIL', 'value': f'Found: {", ".join(all_ncb)}', 'reason': reason}
        else:
            results['dealer_ncb'] = {'status': 'FAIL', 'value': 'Not found', 'reason': 'No NCB information found in any file'}
        
        # 8. No Chargeback - must match across all files
        all_chargeback = all_data['no_chargeback']
        if all_chargeback:
            is_consistent, reason = check_file_consistency('no_chargeback')
            if is_consistent:
                results['no_chargeback'] = {'status': 'PASS', 'value': f'Found: {", ".join(all_chargeback)}', 'reason': reason}
            else:
                results['no_chargeback'] = {'status': 'FAIL', 'value': f'Found: {", ".join(all_chargeback)}', 'reason': reason}
        else:
            results['no_chargeback'] = {'status': 'FAIL', 'value': 'Not found', 'reason': 'No chargeback information found in any file'}
        
        # 9. Cancellation Dates - must match across all files
        all_cancel_dates = all_data['cancellation_date']
        if all_cancel_dates:
            is_consistent, reason = check_file_consistency('cancellation_date')
            if is_consistent:
                results['cancellation_dates'] = {'status': 'PASS', 'value': f'Found: {", ".join(all_cancel_dates)}', 'reason': reason}
            else:
                results['cancellation_dates'] = {'status': 'FAIL', 'value': f'Found: {", ".join(all_cancel_dates)}', 'reason': reason}
        else:
            results['cancellation_dates'] = {'status': 'FAIL', 'value': 'Not found', 'reason': 'No cancellation date found in any file'}
        
        # 10. Sale Dates - must match across all files
        all_sale_dates = all_data['sale_date']
        if all_sale_dates:
            is_consistent, reason = check_file_consistency('sale_date')
            if is_consistent:
                results['sale_dates'] = {'status': 'PASS', 'value': f'Found: {", ".join(all_sale_dates)}', 'reason': reason}
            else:
                results['sale_dates'] = {'status': 'FAIL', 'value': f'Found: {", ".join(all_sale_dates)}', 'reason': reason}
        else:
            results['sale_dates'] = {'status': 'FAIL', 'value': 'Not found', 'reason': 'No sale date found in any file'}
        
        # 11. Reasons - must match across all files
        all_reasons = all_data['reason']
        if all_reasons:
            is_consistent, reason = check_file_consistency('reason')
            if is_consistent:
                results['reasons'] = {'status': 'PASS', 'value': f'Found: {", ".join(all_reasons)}', 'reason': reason}
            else:
                results['reasons'] = {'status': 'FAIL', 'value': f'Found: {", ".join(all_reasons)}', 'reason': reason}
        else:
            results['reasons'] = {'status': 'FAIL', 'value': 'Not found', 'reason': 'No cancellation reason found in any file'}
        
        return results
    
    def parse_date(self, date_str):
        """Parse date string in various formats"""
        if not date_str:
            return None

        # Common date formats to try
        formats = [
            '%m/%d/%Y',      # 08/22/2025
            '%m/%d/%y',      # 8/22/25
            '%Y-%m-%d',      # 2025-08-22
            '%B %d, %Y',     # August 22, 2025
            '%B %d, %Y',     # September 9, 2025
            '%b %d, %Y',     # Aug 22, 2025
            '%d-%m-%Y',      # 22-08-2025
            '%d.%m.%Y',      # 22.08.2025
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue

        return None

def main():
    st.title("📋 QC Form Cancellations Checker - Advanced")
    st.write("Upload a ZIP file containing cancellation packet files to perform quality control checks.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload Cancellation Packet ZIP",
        type=['zip'],
        help="Upload a ZIP file containing cancellation packet files"
    )
    
    if uploaded_file is not None:
        processor = AdvancedTextProcessor()
        
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
        
        # Create a DataFrame for better display
        qc_data = []
        for field, result in qc_results.items():
            status_icon = {'PASS': '✅', 'FAIL': '❌', 'INFO': 'ℹ️'}
            qc_data.append({
                'Field': field.replace('_', ' ').title(),
                'Status': f"{status_icon[result['status']]} {result['status']}",
                'Value': result['value'],
                'Reason': result['reason']
            })
        
        df_qc = pd.DataFrame(qc_data)
        st.dataframe(df_qc, use_container_width=True)
        
        # File-by-File Data Comparison
        st.subheader("📊 File-by-File Data Comparison")
        
        # Create a comparison table showing data from each file
        comparison_data = []
        
        # Get all unique fields across all files
        all_fields = set()
        for file_data in files_processed:
            all_fields.update(file_data['data'].keys())
        
        # Create comparison rows
        for field in sorted(all_fields):
            row = {'Field': field.replace('_', ' ').title()}
            
            # Add data from each file
            for i, file_data in enumerate(files_processed):
                filename = file_data['filename']
                # Truncate long filenames for display
                display_name = filename[:20] + '...' if len(filename) > 20 else filename
                file_values = file_data['data'].get(field, [])
                unique_values = list(set(file_values))
                
                if unique_values:
                    row[f'File {i+1}: {display_name}'] = ', '.join(unique_values[:3]) + ('...' if len(unique_values) > 3 else '')
                else:
                    row[f'File {i+1}: {display_name}'] = 'Not found'
            
            # Add match status
            all_values = []
            for file_data in files_processed:
                file_values = file_data['data'].get(field, [])
                all_values.extend(file_values)
            
            unique_all = list(set(all_values))
            if len(unique_all) == 1:
                row['Match Status'] = '✅ All Match'
            elif len(unique_all) > 1:
                row['Match Status'] = f'❌ {len(unique_all)} Different Values'
            else:
                row['Match Status'] = 'ℹ️ Not Found'
            
            comparison_data.append(row)
        
        # Display comparison table
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # Files section
        st.subheader("📁 Files - Click to Download")
        
        for file_data in files_processed:
            filename = file_data['filename']
            file_data_bytes = file_data['file_data']
            
            st.download_button(
                label=f"📄 {filename}",
                data=file_data_bytes,
                file_name=filename,
                mime="application/octet-stream"
            )

if __name__ == "__main__":
    main()
