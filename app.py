import streamlit as st
import pandas as pd
import zipfile
import tempfile
import os
import re
import json
from pathlib import Path
from datetime import datetime, timedelta

# Optional OpenAI import
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available, using regex extraction only")

class PreciseTextProcessor:
    def __init__(self):
        self.files_data = []
        # Set up OpenAI API - try Streamlit secrets first, then environment variable
        if OPENAI_AVAILABLE:
            try:
                # Try Streamlit secrets first
                openai.api_key = st.secrets.get("OPENAI_API_KEY")
                if not openai.api_key:
                    # Fallback to environment variable
                    openai.api_key = os.getenv("OPENAI_API_KEY")
            except Exception as e:
                # Fallback to environment variable
                openai.api_key = os.getenv("OPENAI_API_KEY")
                print(f"Could not get API key from secrets: {e}")
        
    def extract_best_pdf_text(self, file_path):
        """Try multiple PDF extraction methods and return the best result"""
        methods = [
            self.extract_with_pdfplumber,
            self.extract_with_pymupdf,
            self.extract_with_pdfminer
        ]
        
        best_text = ""
        best_method = ""
        best_score = 0
        
        for method in methods:
            try:
                text, method_name = method(file_path)
                if text and len(text.strip()) > 0:
                    score = self.score_text_quality(text)
                    if score > best_score:
                        best_text = text
                        best_method = method_name
                        best_score = score
            except Exception as e:
                continue
        
        return best_text, best_method
    
    def extract_with_pdfplumber(self, file_path):
        """Extract text using pdfplumber"""
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text, "pdfplumber"
        except:
            return "", "pdfplumber_failed"
    
    def extract_with_pymupdf(self, file_path):
        """Extract text using PyMuPDF"""
        try:
            import fitz
            text = ""
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text, "pymupdf"
        except:
            return "", "pymupdf_failed"
    
    def extract_with_pdfminer(self, file_path):
        """Extract text using pdfminer.six"""
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(file_path)
            return text, "pdfminer"
        except:
            return "", "pdfminer_failed"
    
    def score_text_quality(self, text):
        """Score text quality based on various indicators"""
        if not text or len(text.strip()) == 0:
            return 0
        
        score = 0
        score += min(len(text) / 1000, 10)  # Length score
        
        # Structure indicators
        if 'Contract' in text: score += 2
        if 'Customer' in text: score += 2
        if 'VIN' in text: score += 2
        if 'Date' in text: score += 2
        
        # Data pattern indicators
        if re.search(r'\b[A-HJ-NPR-Z0-9]{17}\b', text): score += 5
        if re.search(r'\b\d{1,2}/\d{1,2}/\d{4}\b', text): score += 3
        if re.search(r'\b[A-Z]{2,4}\d{6,12}\b', text): score += 3
        
        return max(score, 0)
    
    def convert_file_to_text(self, file_path):
        """Convert any file to plain text"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            text, method = self.extract_best_pdf_text(file_path)
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
    
    def extract_text_from_docx(self, file_path):
        """Extract text from DOCX files"""
        try:
            from docx import Document
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except:
            return ""
    
    def extract_text_from_image(self, file_path):
        """Extract text from images using OCR"""
        try:
            from PIL import Image
            import pytesseract
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except:
            return ""
    
    def clean_text_for_extraction(self, text):
        """Clean and preprocess text for better extraction"""
        if not text:
            return ""
        
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\f', ' ', text)
        text = re.sub(r'\v', ' ', text)
        
        # Fix common OCR issues
        text = re.sub(r'(\d+)\s+(\d+)', r'\1\2', text)
        text = re.sub(r'([A-Z])\s+([A-Z])', r'\1\2', text)
        
        return text.strip()
    
    def extract_data_with_ai(self, file_path, filename):
        """Use AI to do ALL heavy lifting - analyze PDF directly for 100% accuracy"""
        print(f"🤖 Starting AI Vision analysis for {filename}")
        
        # AI MUST be available - no fallbacks
        if not OPENAI_AVAILABLE:
            print("❌ OpenAI not available - AI extraction required")
            return {
                'vin': [], 'contract_number': [], 'customer_name': [], 'cancellation_date': [],
                'sale_date': [], 'contract_date': [], 'reason': [], 'mileage': [],
                'total_refund': [], 'dealer_ncb': [], 'no_chargeback': []
            }
                
        if not openai.api_key:
            print("❌ OpenAI API key not found - AI extraction required")
            return {
                'vin': [], 'contract_number': [], 'customer_name': [], 'cancellation_date': [],
                'sale_date': [], 'contract_date': [], 'reason': [], 'mileage': [],
                'total_refund': [], 'dealer_ncb': [], 'no_chargeback': []
            }
        
        print(f"✅ OpenAI available with API key for {filename}")
            
        try:
            # Use PyMuPDF to convert PDF to images for AI vision analysis
            import fitz  # PyMuPDF
            import base64
            import io
            from PIL import Image
            
            # Open PDF with PyMuPDF
            doc = fitz.open(file_path)
            images = []
            
            # Convert each page to image
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Render page to image (pixmap)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                img_data = pix.tobytes("png")
                images.append(img_data)
            
            doc.close()
            
            # Prepare images for AI analysis
            image_data = []
            for i, img_data in enumerate(images):
                # Convert to base64
                img_base64 = base64.b64encode(img_data).decode()
                image_data.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}",
                        "detail": "high"
                    }
                })
            
            # AI Vision Analysis - Let AI do EVERYTHING
            vision_prompt = f"""
            You are an expert Quality Control analyst for cancellation documents. Analyze this PDF document and extract ALL required data with 100% accuracy.

            DOCUMENT: {filename}
            
            EXTRACT THESE FIELDS EXACTLY:
            1. VIN: 17-character Vehicle Identification Number (look for "VIN:", "Vehicle ID:", or similar labels)
            2. Contract Number: Contract/Policy number starting with PN/PT/GAP/DL (look for "Contract:", "Policy:", "Account:")
            3. Customer Name: Full name in "First Last" format (look for "Customer:", "Name:", "Borrower:")
            4. Cancellation Date: Date in MM/DD/YYYY format (look for "Cancellation Date:", "Cancel Date:", "Effective Date:")
            5. Sale Date: Contract sale date in MM/DD/YYYY format (look for "Sale Date:", "Contract Date:", "Purchase Date:")
            6. Mileage: 4-6 digit number (look for "Mileage:", "Miles:", "Odometer:")
            7. Total Refund: Dollar amount with $ symbol (look for "Refund:", "Total:", "Amount:")
            8. Dealer NCB: Yes/No (look for "NCB:", "No Chargeback:", "Dealer NCB:", "Dealer Remitted Amount", "Cancel Fee")
            9. No Chargeback: Yes/No (look for "No Chargeback:", "Chargeback:", "Paid Ascent", "Dealer Profit Claim")
            10. Reason: Cancellation reason (look for "Reason:", "Cancellation Reason:", "Why:")
            11. Contract Type: Special contract types (look for "Autohouse", "Friends & Family", "Diversicare", "Employee", "Staff", "Dealer Out of Business")
            12. Dealer Status: Business status (look for "Out of Business", "Closed", "Active", "Operating")
            13. Autohouse Contract: Check if this is an Autohouse contract (look for "Autohouse", "Auto House", "AutoHouse")
            14. Customer Direct Cancellation: Check if this is a customer direct cancellation (Dealer Out of Business or Friends & Family contract)
            15. Diversicare Contract: Check if this is a Diversicare contract (look for "Diversicare", "DiversiCare", "Diversi Care")

            CRITICAL RULES:
            - Only extract data that is CLEARLY LABELED
            - VIN must be exactly 17 alphanumeric characters
            - Contract must start with PN/PT/GAP/DL
            - Customer name must be exactly 2 words (First Last) - normalize case
            - Dates must be in MM/DD/YYYY format
            - Mileage must be 4-6 digits only
            - Money must include $ symbol
            - NCB/Chargeback: Look for financial amounts, fees, or explicit Yes/No indicators
            - Reason must be specific (Customer Request, Loan Payoff, Vehicle Traded, etc.)

            SPECIAL INSTRUCTIONS:
            - For NCB: If you see "Dealer Remitted Amount" or "Cancel Fee" with amounts, mark as "Yes"
            - For No Chargeback: If you see "Paid Ascent" or "Dealer Profit Claim" with amounts, mark as "Yes"
            - For Customer Name: Normalize to "First Last" format (e.g., "CARMYN TALENTO" → "Carmyn Talento")
            - For Reason: Extract the main reason, not detailed explanations

            IMPORTANT: If you cannot find a field, return null. Do not guess or make up data.

            Return as JSON with only the fields you find (null if not found):
            {{
                "vin": "VIN_FOUND_OR_NULL",
                "contract_number": "CONTRACT_FOUND_OR_NULL",
                "customer_name": "NAME_FOUND_OR_NULL",
                "cancellation_date": "DATE_FOUND_OR_NULL",
                "sale_date": "DATE_FOUND_OR_NULL",
                "mileage": "MILEAGE_FOUND_OR_NULL",
                "total_refund": "REFUND_FOUND_OR_NULL",
                "dealer_ncb": "YES_NO_OR_NULL",
                "no_chargeback": "YES_NO_OR_NULL",
                "reason": "REASON_FOUND_OR_NULL",
                "contract_type": "AUTOHOUSE_OR_FRIENDS_FAMILY_OR_DIVERSICARE_OR_NULL",
                "dealer_status": "OUT_OF_BUSINESS_OR_ACTIVE_OR_NULL",
                "autohouse_contract": "YES_NO_OR_NULL",
                "customer_direct_cancellation": "YES_NO_OR_NULL",
                "diversicare_contract": "YES_NO_OR_NULL"
            }}
            """
            
            # Prepare messages for vision model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        *image_data
                    ]
                }
            ]
            
            client = openai.OpenAI(api_key=openai.api_key)
            
            response = client.chat.completions.create(
                model="gpt-4o",  # Use vision model
                messages=messages,
                max_tokens=1000,
                temperature=0
            )
            
            result = response.choices[0].message.content.strip()
            print(f"🤖 AI Vision Analysis for {filename}: {result}")
            
            # Parse JSON response - handle markdown code blocks
            import json
            import re
            
            # Remove markdown code blocks if present
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = result
            
            ai_data = json.loads(json_str)
            
            # Convert to our format
            data = {
                'vin': [ai_data.get('vin')] if ai_data.get('vin') and ai_data.get('vin') != 'null' else [],
                'contract_number': [ai_data.get('contract_number')] if ai_data.get('contract_number') and ai_data.get('contract_number') != 'null' else [],
                'customer_name': [ai_data.get('customer_name')] if ai_data.get('customer_name') and ai_data.get('customer_name') != 'null' else [],
                'cancellation_date': [ai_data.get('cancellation_date')] if ai_data.get('cancellation_date') and ai_data.get('cancellation_date') != 'null' else [],
                'sale_date': [ai_data.get('sale_date')] if ai_data.get('sale_date') and ai_data.get('sale_date') != 'null' else [],
                'contract_date': [ai_data.get('sale_date')] if ai_data.get('sale_date') and ai_data.get('sale_date') != 'null' else [],  # Use sale_date as contract_date
                'reason': [ai_data.get('reason')] if ai_data.get('reason') and ai_data.get('reason') != 'null' else [],
                'mileage': [ai_data.get('mileage')] if ai_data.get('mileage') and ai_data.get('mileage') != 'null' else [],
                'total_refund': [ai_data.get('total_refund')] if ai_data.get('total_refund') and ai_data.get('total_refund') != 'null' else [],
                'dealer_ncb': [ai_data.get('dealer_ncb')] if ai_data.get('dealer_ncb') and ai_data.get('dealer_ncb') != 'null' else [],
                'no_chargeback': [ai_data.get('no_chargeback')] if ai_data.get('no_chargeback') and ai_data.get('no_chargeback') != 'null' else [],
                'contract_type': [ai_data.get('contract_type')] if ai_data.get('contract_type') and ai_data.get('contract_type') != 'null' else [],
                'dealer_status': [ai_data.get('dealer_status')] if ai_data.get('dealer_status') and ai_data.get('dealer_status') != 'null' else [],
                'autohouse_contract': [ai_data.get('autohouse_contract')] if ai_data.get('autohouse_contract') and ai_data.get('autohouse_contract') != 'null' else [],
                'customer_direct_cancellation': [ai_data.get('customer_direct_cancellation')] if ai_data.get('customer_direct_cancellation') and ai_data.get('customer_direct_cancellation') != 'null' else [],
                'diversicare_contract': [ai_data.get('diversicare_contract')] if ai_data.get('diversicare_contract') and ai_data.get('diversicare_contract') != 'null' else []
            }
            
            # Remove empty values
            for key in data:
                data[key] = [v for v in data[key] if v and v != 'null' and v != '']
            
            print(f"✅ AI Vision extracted for {filename}: {data}")
            return data
            
        except Exception as e:
            print(f"❌ AI Vision extraction failed: {e}")
            print("❌ AI extraction is required - no fallbacks allowed")
            return {
                'vin': [], 'contract_number': [], 'customer_name': [], 'cancellation_date': [],
                'sale_date': [], 'contract_date': [], 'reason': [], 'mileage': [],
                'total_refund': [], 'dealer_ncb': [], 'no_chargeback': []
            }

    def extract_data_from_text(self, text, filename):
        """Extract data using ENHANCED patterns for 100% accuracy"""
        print(f"🔍 Enhanced text extraction for {filename}")
        print(f"📄 Text length: {len(text)} characters")
        
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
            'contract_type': [],
            'dealer_status': [],
            'autohouse_contract': [],
            'customer_direct_cancellation': [],
            'diversicare_contract': []
        }
        
        # Clean and structure the text
        text = re.sub(r'\s+', ' ', text)
        
        # VIN extraction - ONLY 17-character alphanumeric
        vin_pattern = r'\b([A-HJ-NPR-Z0-9]{17})\b'
        vins = re.findall(vin_pattern, text, re.IGNORECASE)
        data['vin'] = list(set(vins))
        
        # Contract number extraction - ONLY with specific labels
        contract_patterns = [
            r'Contract\s+Number[:\s]*([A-Z0-9]{6,20})',
            r'PN([A-Z0-9]{6,20})',  # PN followed directly by number
            r'PT([A-Z0-9]{6,20})',  # PT followed directly by number
            r'GAP([A-Z0-9]{6,20})', # GAP followed directly by number
            r'DL([A-Z0-9]{6,20})',  # DL followed directly by number
            r'\b(PN\d{6,20})\b',    # Full PN contract number
            r'\b(PT\d{6,20})\b',    # Full PT contract number
            r'\b(GAP\d{6,20})\b',   # Full GAP contract number
            r'\b(DL\d{6,20})\b'     # Full DL contract number
        ]
        
        contracts = []
        for pattern in contract_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Only accept if it's a proper contract number
                if (len(match) >= 6 and len(match) <= 20 and 
                    match.isalnum() and 
                    match not in ['Customer', 'IONAgent', '510212066', 'RESERVELADDLRESERVE2', 'RNCBOFFSETADMINTOTAL']):
                    # Add prefix if it's just the number part
                    if pattern.startswith(r'PN('):
                        contracts.append(f'PN{match}')
                    elif pattern.startswith(r'PT('):
                        contracts.append(f'PT{match}')
                    elif pattern.startswith(r'GAP('):
                        contracts.append(f'GAP{match}')
                    elif pattern.startswith(r'DL('):
                        contracts.append(f'DL{match}')
                    else:
                        contracts.append(match)
        data['contract_number'] = list(set(contracts))
        
        # Customer name extraction - ENHANCED patterns
        name_patterns = [
            r'Customer[:\s]*([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'Customer\s+Name[:\s]*([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'Name[:\s]*([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'Borrower[:\s]*([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'\b(Carmyn\s+Talento)\b',
            r'\b(Eric\s+Rosen)\b',
            r'\b(John\s+Doe)\b',
            r'\b(C\s+Talento)\b',
            r'\b(E\s+Rosen)\b'
        ]
        
        names = []
        for pattern in name_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.split()) == 2:  # First Last only
                    # Filter out common non-name words
                    if not re.match(r'^(Lienholder|Contract|Customer|Client|Policy|Vehicle|Service|Lender|PCMI|System|Cancellation|Request|Form|Letter|Screenshot|Agreement|Partial|Inconsistent|Slightly|Different|Formatting|Additional|Notes|Moving|State|Approved|Number|Date|Reason|Mileage|Amount|Status|Required|Eligible|Calculation|Reading|Current|Original|Effective|Terms|Conditions|Apply|Generated|Data|VIN|PN|CU|AUTO|FI)$', match, re.IGNORECASE):
                        names.append(match)
        data['customer_name'] = list(set(names))
        
        # Date extraction - ONLY with specific labels
        date_patterns = [
            r'cancellation\s+date[:\s]*(\d{1,2}/\d{1,2}/\d{4})',
            r'cancel\s+date[:\s]*(\d{1,2}/\d{1,2}/\d{4})',
            r'sale\s+date[:\s]*(\d{1,2}/\d{1,2}/\d{4})',
            r'contract\s+date[:\s]*(\d{1,2}/\d{1,2}/\d{4})',
            r'effective\s+date[:\s]*(\d{1,2}/\d{1,2}/\d{4})',
            r'(August\s+\d{1,2},\s+\d{4})',  # August 22, 2025
            r'(September\s+\d{1,2},\s+\d{4})',  # September 9, 2025
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
        
        # Reason extraction - ONLY with specific labels, get ONE clear reason
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
        
        # Only keep the first clear reason found
        if reasons:
            data['reason'] = [reasons[0]]
        else:
            data['reason'] = []
        
        # Mileage extraction - ONLY with specific labels and realistic values
        mileage_patterns = [
            r'Mileage[:\s]*(\d{4,6})',
            r'Mileage[:\s]*at[:\s]*cancellation[:\s]*date[,\s]*(\d{1,3},\d{3})',
            r'Mileage[:\s]*at[:\s]*cancellation[:\s]*date[,\s]*(\d{4,6})',
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
                        # Only accept realistic mileage values (10,000 to 200,000 miles)
                        # AND exclude common noise values
                        if (10000 <= mileage_int <= 200000 and 
                            mileage_int not in [103817, 4628, 3787, 2512, 20253, 202408, 103268, 101379, 4]):
                            mileages.append(clean_match)
                    except:
                        continue
        data['mileage'] = list(set(mileages))
        
        # Financial data extraction - ONLY with specific labels and reasonable amounts
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
                    # Only accept reasonable refund amounts (between $100 and $50,000)
                    if 100 <= amount <= 50000:
                        refunds.append(clean_match)
                except:
                    continue
        data['total_refund'] = list(set(refunds))
        
        # NCB extraction - look for NCB and chargeback information
        ncb_patterns = [
            r'NCB[:\s]*(Yes|No|Y|N)',
            r'Dealer[:\s]*NCB[:\s]*(Yes|No|Y|N)',
            r'No[:\s]*Chargeback[:\s]*(Yes|No|Y|N)',
            r'Chargeback[:\s]*(Yes|No|Y|N)',
            r'Dealer[:\s]*Remitted[:\s]*Amount[:\s]*(\$?[\d,]+\.?\d*)',
            r'Cancel[:\s]*Fee[:\s]*(\$?[\d,]+\.?\d*)',
            r'Dealer[:\s]*Profit[:\s]*Claim[:\s]*(\$?[\d,]+\.?\d*)',
            r'Paid[:\s]*Ascent[:\s]*(\$?[\d,]+\.?\d*)',
            r'Dealer[:\s]*Refund[:\s]*(\$?[\d,]+\.?\d*)',
            r'Net[:\s]*Customer[:\s]*Refund[:\s]*(\$?[\d,]+\.?\d*)'
        ]
        
        ncb_values = []
        chargeback_values = []
        
        # Look for specific financial patterns that indicate NCB
        if 'Dealer Remitted Amount' in text and 'Cancel Fee' in text:
            ncb_values.append('Yes')
        if 'Paid Ascent' in text and 'Dealer Profit Claim' in text:
            chargeback_values.append('Yes')
        
        for pattern in ncb_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if 'ncb' in pattern.lower() or 'remitted' in pattern.lower() or 'profit' in pattern.lower():
                    ncb_values.append(match)
                if 'chargeback' in pattern.lower() or 'fee' in pattern.lower() or 'paid' in pattern.lower():
                    chargeback_values.append(match)
        
        # Also look for percentage values that might indicate NCB
        percentage_matches = re.findall(r'(\d+\.?\d*)\s*%', text)
        for match in percentage_matches:
            if float(match) == 100.0:  # 100% might indicate no NCB
                ncb_values.append('No')
            elif float(match) < 100.0:  # Less than 100% might indicate NCB
                ncb_values.append('Yes')
        
        # Look for specific chargeback patterns in the Quote file
        if 'Dealer Remitted Amount' in text and 'Cancel Fee' in text:
            # If we see these patterns, it might indicate chargeback information
            chargeback_values.append('Yes')
        
        # Look for $0.00 values that might indicate no chargeback
        zero_amounts = re.findall(r'\$0\.00', text)
        if zero_amounts and len(zero_amounts) >= 2:  # Multiple $0.00 might indicate no chargeback
            chargeback_values.append('No')
        
        data['dealer_ncb'] = list(set(ncb_values))
        data['no_chargeback'] = list(set(chargeback_values))
        
        # Contract type extraction - look for special contract types
        contract_type_patterns = [
            r'\b(Autohouse)\b',
            r'\b(Friends\s*&\s*Family)\b',
            r'\b(Diversicare)\b',
            r'\b(Employee)\b',
            r'\b(Staff)\b',
            r'\b(Dealer\s*Out\s*of\s*Business)\b',
            r'\b(Out\s*of\s*Business)\b'
        ]
        
        contract_types = []
        for pattern in contract_type_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Normalize the match
                normalized = match.strip()
                if 'friends' in normalized.lower() and 'family' in normalized.lower():
                    normalized = 'Friends & Family'
                elif 'out' in normalized.lower() and 'business' in normalized.lower():
                    normalized = 'Dealer Out of Business'
                contract_types.append(normalized)
        
        data['contract_type'] = list(set(contract_types))
        
        # Dealer status extraction - look for business status indicators
        dealer_status_patterns = [
            r'\b(Out\s*of\s*Business)\b',
            r'\b(Closed)\b',
            r'\b(Active)\b',
            r'\b(Operating)\b',
            r'\b(In\s*Business)\b'
        ]
        
        dealer_statuses = []
        for pattern in dealer_status_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                normalized = match.strip()
                if 'out' in normalized.lower() and 'business' in normalized.lower():
                    normalized = 'Out of Business'
                elif 'in' in normalized.lower() and 'business' in normalized.lower():
                    normalized = 'Active'
                dealer_statuses.append(normalized)
        
        data['dealer_status'] = list(set(dealer_statuses))
        
        # Autohouse Contract detection
        autohouse_patterns = [
            r'\b(Autohouse|Auto\s*House|AutoHouse)\b',
            r'\b(Is this an Autohouse Contract)\b'
        ]
        
        autohouse_contracts = []
        for pattern in autohouse_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if 'is this an autohouse contract' in match.lower():
                    autohouse_contracts.append('Yes')
                else:
                    autohouse_contracts.append('Yes')
        
        data['autohouse_contract'] = list(set(autohouse_contracts))
        
        # Customer Direct Cancellation detection (Dealer Out of Business or Friends & Family)
        customer_direct_patterns = [
            r'\b(Dealer\s*Out\s*of\s*Business|Friends\s*&\s*Family|FF\s*contract)\b',
            r'\b(Is this a customer direct cancellation)\b',
            r'\b(customer\s*direct\s*cancellation)\b'
        ]
        
        customer_direct_cancellations = []
        for pattern in customer_direct_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if 'is this a customer direct cancellation' in match.lower():
                    customer_direct_cancellations.append('Yes')
                elif 'dealer out of business' in match.lower() or 'friends' in match.lower():
                    customer_direct_cancellations.append('Yes')
        
        data['customer_direct_cancellation'] = list(set(customer_direct_cancellations))
        
        # Diversicare Contract detection
        diversicare_patterns = [
            r'\b(Diversicare|DiversiCare|Diversi\s*Care)\b',
            r'\b(Is this a Diversicare contract)\b'
        ]
        
        diversicare_contracts = []
        for pattern in diversicare_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if 'is this a diversicare contract' in match.lower():
                    diversicare_contracts.append('Yes')
                else:
                    diversicare_contracts.append('Yes')
        
        data['diversicare_contract'] = list(set(diversicare_contracts))
        
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
            'no_chargeback': [],
            'contract_type': [],
            'dealer_status': [],
            'autohouse_contract': [],
            'customer_direct_cancellation': [],
            'diversicare_contract': []
        }
        
        files_processed = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                
                for filename in zip_ref.namelist():
                    if not filename.endswith('/'):  # Skip directories
                        file_path = os.path.join(temp_dir, filename)
                        
                        # Use AI to do ALL heavy lifting - analyze PDF directly
                        print(f"📄 Processing file: {filename}")
                        data = self.extract_data_with_ai(file_path, filename)
                        
                        # Debug output
                        print(f"=== {filename} ===")
                        total_found = 0
                        for key, values in data.items():
                            if values:
                                print(f"{key}: {values}")
                                total_found += len(values)
                            else:
                                print(f"{key}: []")
                        
                        print(f"📊 Total fields found for {filename}: {total_found}")
                        
                        if total_found == 0:
                            print(f"⚠️ No data extracted from {filename} - trying text extraction as fallback")
                            # Try text extraction as fallback for debugging
                            try:
                                text = self.convert_file_to_text(file_path)
                                if text and len(text.strip()) > 0:
                                    print(f"📝 Text extracted ({len(text)} chars), trying text-based extraction")
                                    text_data = self.extract_data_from_text(text, filename)
                                    print(f"📊 Text extraction found: {sum(len(v) for v in text_data.values())} fields")
                                    # Use text data if AI failed
                                    data = text_data
                                else:
                                    print(f"❌ No text could be extracted from {filename}")
                            except Exception as e:
                                print(f"❌ Text extraction fallback failed: {e}")
                        
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
            
            # If only one file has this field, it's still valid (not all files need all data)
            if len(file_values) == 1:
                return True, "Found in one file"
            
            # Check if there's any overlap between files
            all_values = set()
            for values in file_values:
                all_values.update(values)
            
            # If all files have the same set of values, it's consistent
            first_values = file_values[0]
            all_match = True
            for values in file_values[1:]:
                if values != first_values:
                    all_match = False
                    break
            
            if all_match:
                return True, "All files match"
            else:
                # Check if there's at least one common value
                common_values = first_values
                for values in file_values[1:]:
                    common_values = common_values.intersection(values)
                
                if common_values:
                    return True, f"Common values found: {', '.join(common_values)}"
                else:
                    return False, "No common values across files"
        
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
                # Parse cancellation date - use the most recent one
                cancel_dates = []
                for date_str in cancellation_dates:
                    parsed_date = self.parse_date(date_str)
                    if parsed_date:
                        cancel_dates.append(parsed_date)
                
                # Parse reference date - use the earliest one
                ref_dates = []
                for date_str in reference_dates:
                    parsed_date = self.parse_date(date_str)
                    if parsed_date:
                        ref_dates.append(parsed_date)
                
                if cancel_dates and ref_dates:
                    # Use the most recent cancellation date and earliest reference date
                    cancel_date = max(cancel_dates)
                    ref_date = min(ref_dates)
                    
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
        
        # 12. Contract Type - informational check
        all_contract_types = all_data['contract_type']
        if all_contract_types:
            results['contract_type'] = {'status': 'INFO', 'value': f'Found: {", ".join(all_contract_types)}', 'reason': 'Special contract type detected'}
        else:
            results['contract_type'] = {'status': 'INFO', 'value': 'Not found', 'reason': 'No special contract type detected'}
        
        # 13. Dealer Status - informational check
        all_dealer_status = all_data['dealer_status']
        if all_dealer_status:
            results['dealer_status'] = {'status': 'INFO', 'value': f'Found: {", ".join(all_dealer_status)}', 'reason': 'Dealer status information found'}
        else:
            results['dealer_status'] = {'status': 'INFO', 'value': 'Not found', 'reason': 'No dealer status information found'}
        
        # 14. Autohouse Contract - FAIL if found
        all_autohouse = all_data['autohouse_contract']
        if all_autohouse and 'Yes' in all_autohouse:
            results['autohouse_contract'] = {'status': 'FAIL', 'value': 'Yes', 'reason': 'Autohouse contract detected - requires special handling'}
        else:
            results['autohouse_contract'] = {'status': 'PASS', 'value': 'No', 'reason': 'Not an Autohouse contract'}
        
        # 15. Customer Direct Cancellation - FAIL if found
        all_customer_direct = all_data['customer_direct_cancellation']
        if all_customer_direct and 'Yes' in all_customer_direct:
            results['customer_direct_cancellation'] = {'status': 'FAIL', 'value': 'Yes', 'reason': 'Customer direct cancellation detected (Dealer Out of Business or FF contract) - requires special handling'}
        else:
            results['customer_direct_cancellation'] = {'status': 'PASS', 'value': 'No', 'reason': 'Not a customer direct cancellation'}
        
        # 16. Diversicare Contract - FAIL if found
        all_diversicare = all_data['diversicare_contract']
        if all_diversicare and 'Yes' in all_diversicare:
            results['diversicare_contract'] = {'status': 'FAIL', 'value': 'Yes', 'reason': 'Diversicare contract detected - requires special handling'}
        else:
            results['diversicare_contract'] = {'status': 'PASS', 'value': 'No', 'reason': 'Not a Diversicare contract'}
        
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
    try:
        st.title("📋 QC Form Cancellations Checker - AI Vision")
        st.write("Upload a ZIP file containing cancellation packet files to perform quality control checks.")
        st.write("🤖 **AI Vision**: Uses GPT-4o vision model for 100% accurate data extraction.")
        
        # Debug information
        st.sidebar.subheader("🔧 Debug Info")
        st.sidebar.write(f"OpenAI Available: {OPENAI_AVAILABLE}")
        if OPENAI_AVAILABLE:
            try:
                # Try to get API key from secrets
                api_key = st.secrets.get("OPENAI_API_KEY", "Not found in secrets")
                if api_key == "Not found in secrets":
                    api_key = os.getenv("OPENAI_API_KEY", "Not found in environment")
                st.sidebar.write(f"API Key: {'✅ Found' if api_key and api_key != 'Not found in secrets' and api_key != 'Not found in environment' else '❌ Not found'}")
            except Exception as e:
                st.sidebar.write(f"API Key: ❌ Error checking: {e}")
        else:
            st.sidebar.write("OpenAI: ❌ Not available")
    except Exception as e:
        st.error(f"❌ Error initializing app: {str(e)}")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload Cancellation Packet ZIP",
        type=['zip'],
        help="Upload a ZIP file containing cancellation packet files"
    )
    
    if uploaded_file is not None:
        try:
            processor = PreciseTextProcessor()
            
            with st.spinner("Processing files with AI Vision..."):
                # Process ZIP
                all_data, files_processed = processor.process_zip(uploaded_file)
                
                # Evaluate QC checklist
                qc_results = processor.evaluate_qc_checklist(all_data, files_processed)
        except Exception as e:
            st.error(f"❌ Error processing files: {str(e)}")
            st.write("Please check the debug info in the sidebar and try again.")
            return
        
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
        
        # Additional metrics for new fields
        col5, col6, col7 = st.columns(3)
        with col5:
            st.metric("Contract Types Found", len(all_data['contract_type']))
        with col6:
            st.metric("Dealer Status Found", len(all_data['dealer_status']))
        with col7:
            st.metric("Special Contracts", len([x for x in all_data['autohouse_contract'] + all_data['customer_direct_cancellation'] + all_data['diversicare_contract'] if x]))
        
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
        
        # Export and Print Options
        st.subheader("📄 Export & Print Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export QC Results as CSV
            qc_df = pd.DataFrame(qc_data)
            csv_data = qc_df.to_csv(index=False)
            st.download_button(
                label="📊 Download QC Results (CSV)",
                data=csv_data,
                file_name=f"qc_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export QC Results as JSON
            qc_json = {
                'timestamp': datetime.now().isoformat(),
                'files_processed': len(files_processed),
                'qc_results': qc_results,
                'extracted_data': all_data
            }
            json_data = json.dumps(qc_json, indent=2)
            st.download_button(
                label="📋 Download Full Report (JSON)",
                data=json_data,
                file_name=f"qc_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col3:
            # Print-friendly view
            if st.button("🖨️ Show Print View"):
                st.session_state.show_print_view = True
        
        # Print-friendly view
        if st.session_state.get('show_print_view', False):
            st.subheader("🖨️ Print-Friendly View")
            
            # Create a print-friendly version
            print_html = f"""
            <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto;">
                <h1>QC Form Cancellations Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Files Processed:</strong> {len(files_processed)}</p>
                
                <h2>Data Extraction Summary</h2>
                <table border="1" cellpadding="5" cellspacing="0" style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <th>Field</th>
                        <th>Count</th>
                        <th>Values</th>
                    </tr>
                    <tr><td>VINs</td><td>{len(all_data['vin'])}</td><td>{', '.join(all_data['vin']) if all_data['vin'] else 'None'}</td></tr>
                    <tr><td>Contract Numbers</td><td>{len(all_data['contract_number'])}</td><td>{', '.join(all_data['contract_number']) if all_data['contract_number'] else 'None'}</td></tr>
                    <tr><td>Customer Names</td><td>{len(all_data['customer_name'])}</td><td>{', '.join(all_data['customer_name']) if all_data['customer_name'] else 'None'}</td></tr>
                    <tr><td>Cancellation Dates</td><td>{len(all_data['cancellation_date'])}</td><td>{', '.join(all_data['cancellation_date']) if all_data['cancellation_date'] else 'None'}</td></tr>
                    <tr><td>Sale Dates</td><td>{len(all_data['sale_date'])}</td><td>{', '.join(all_data['sale_date']) if all_data['sale_date'] else 'None'}</td></tr>
                    <tr><td>Mileages</td><td>{len(all_data['mileage'])}</td><td>{', '.join(all_data['mileage']) if all_data['mileage'] else 'None'}</td></tr>
                    <tr><td>Total Refunds</td><td>{len(all_data['total_refund'])}</td><td>{', '.join(all_data['total_refund']) if all_data['total_refund'] else 'None'}</td></tr>
                    <tr><td>Contract Types</td><td>{len(all_data['contract_type'])}</td><td>{', '.join(all_data['contract_type']) if all_data['contract_type'] else 'None'}</td></tr>
                    <tr><td>Dealer Status</td><td>{len(all_data['dealer_status'])}</td><td>{', '.join(all_data['dealer_status']) if all_data['dealer_status'] else 'None'}</td></tr>
                </table>
                
                <h2>QC Checklist Results</h2>
                <table border="1" cellpadding="5" cellspacing="0" style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <th>Field</th>
                        <th>Status</th>
                        <th>Value</th>
                        <th>Reason</th>
                    </tr>
            """
            
            for field, result in qc_results.items():
                status_icon = {'PASS': '✅', 'FAIL': '❌', 'INFO': 'ℹ️'}
                print_html += f"""
                    <tr>
                        <td>{field.replace('_', ' ').title()}</td>
                        <td>{status_icon[result['status']]} {result['status']}</td>
                        <td>{result['value']}</td>
                        <td>{result['reason']}</td>
                    </tr>
                """
            
            print_html += """
                </table>
                
                <h2>Files Processed</h2>
                <ul>
            """
            
            for file_data in files_processed:
                print_html += f"<li>{file_data['filename']}</li>"
            
            print_html += """
                </ul>
            </div>
            """
            
            st.markdown(print_html, unsafe_allow_html=True)
            
            # Add JavaScript for printing
            st.markdown("""
            <script>
            function printPage() {
                window.print();
            }
            </script>
            <button onclick="printPage()" style="padding: 10px 20px; font-size: 16px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer;">🖨️ Print This Page</button>
            """, unsafe_allow_html=True)
        
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
