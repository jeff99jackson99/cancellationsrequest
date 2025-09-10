import streamlit as st
import pandas as pd
import zipfile
import tempfile
import os
import re
import json
from datetime import datetime, timedelta
from pathlib import Path
import pdfplumber
from docx import Document
from PIL import Image
import pytesseract
import io
import requests
import ipaddress
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    # Fallback for environments where OpenCV is not available
    import numpy as np

# Page configuration
st.set_page_config(
    page_title="QC Form Cancellations Checker",
    page_icon="📋",
    layout="wide"
)

# Title and description
st.title("📋 QC Form Cancellations Checker")
st.markdown("Upload a ZIP file containing cancellation packet files to perform quality control checks. You can also upload individual bucket screenshots for NCB fee analysis.")

# Check for OpenCV availability
if not OPENCV_AVAILABLE:
    st.warning("⚠️ OpenCV is not available. Screenshot processing will use basic OCR without image preprocessing.")

class IPSecurityChecker:
    """IP address lookup and VPN detection for security validation"""
    
    def __init__(self):
        self.vpn_indicators = [
            'vpn', 'proxy', 'tor', 'anonymizer', 'privacy', 'secure',
            'private', 'anonymous', 'hide', 'mask', 'shield', 'guard',
            'express', 'nord', 'surfshark', 'cyberghost', 'windscribe',
            'proton', 'mullvad', 'ivpn', 'perfect', 'privacy', 'tunnelbear'
        ]
        
        self.known_vpn_providers = [
            'nordvpn', 'expressvpn', 'surfshark', 'cyberghost', 'windscribe',
            'protonvpn', 'mullvad', 'ivpn', 'perfect-privacy', 'tunnelbear',
            'private internet access', 'pia', 'hotspot shield', 'zenmate',
            'ipvanish', 'vyprvpn', 'hidemyass', 'hma', 'purevpn'
        ]
    
    def get_client_ip(self):
        """Get the client's IP address from Streamlit session"""
        try:
            # Try to get IP from Streamlit's session state
            if hasattr(st, 'session_state') and 'client_ip' in st.session_state:
                return st.session_state.client_ip
            
            # Fallback: try to get from request headers
            import streamlit.web.server.server as server
            if hasattr(server, 'get_current_request'):
                request = server.get_current_request()
                if request:
                    # Check for forwarded IP first (common with proxies)
                    forwarded_for = request.headers.get('X-Forwarded-For')
                    if forwarded_for:
                        return forwarded_for.split(',')[0].strip()
                    
                    # Check for real IP header
                    real_ip = request.headers.get('X-Real-IP')
                    if real_ip:
                        return real_ip
                    
                    # Fallback to remote address
                    return request.environ.get('REMOTE_ADDR')
            
            return None
        except Exception as e:
            print(f"Error getting client IP: {e}")
            return None
    
    def lookup_ip_info(self, ip_address):
        """Look up IP address information using multiple services"""
        if not ip_address:
            return None
            
        try:
            # Validate IP address
            ipaddress.ip_address(ip_address)
        except ValueError:
            return None
        
        # Try multiple IP lookup services for redundancy
        services = [
            self._lookup_ipapi(ip_address),
            self._lookup_ipinfo(ip_address),
            self._lookup_ipgeolocation(ip_address)
        ]
        
        # Return the first successful lookup
        for result in services:
            if result:
                return result
        
        return None
    
    def _lookup_ipapi(self, ip_address):
        """Lookup using ipapi.co"""
        try:
            response = requests.get(f"http://ipapi.co/{ip_address}/json/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    'ip': data.get('ip'),
                    'city': data.get('city'),
                    'region': data.get('region'),
                    'country': data.get('country_name'),
                    'country_code': data.get('country_code'),
                    'org': data.get('org'),
                    'isp': data.get('org'),
                    'timezone': data.get('timezone'),
                    'latitude': data.get('latitude'),
                    'longitude': data.get('longitude'),
                    'service': 'ipapi.co'
                }
        except Exception as e:
            print(f"ipapi.co lookup failed: {e}")
        return None
    
    def _lookup_ipinfo(self, ip_address):
        """Lookup using ipinfo.io (free tier)"""
        try:
            response = requests.get(f"https://ipinfo.io/{ip_address}/json", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    'ip': data.get('ip'),
                    'city': data.get('city'),
                    'region': data.get('region'),
                    'country': data.get('country'),
                    'country_code': data.get('country'),
                    'org': data.get('org'),
                    'isp': data.get('org'),
                    'timezone': data.get('timezone'),
                    'latitude': data.get('loc', '').split(',')[0] if data.get('loc') else None,
                    'longitude': data.get('loc', '').split(',')[1] if data.get('loc') else None,
                    'service': 'ipinfo.io'
                }
        except Exception as e:
            print(f"ipinfo.io lookup failed: {e}")
        return None
    
    def _lookup_ipgeolocation(self, ip_address):
        """Lookup using ipgeolocation.io (free tier)"""
        try:
            response = requests.get(f"https://api.ipgeolocation.io/ipgeo?apiKey=free&ip={ip_address}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    'ip': data.get('ip'),
                    'city': data.get('city'),
                    'region': data.get('state_prov'),
                    'country': data.get('country_name'),
                    'country_code': data.get('country_code2'),
                    'org': data.get('organization'),
                    'isp': data.get('isp'),
                    'timezone': data.get('time_zone', {}).get('name'),
                    'latitude': data.get('latitude'),
                    'longitude': data.get('longitude'),
                    'service': 'ipgeolocation.io'
                }
        except Exception as e:
            print(f"ipgeolocation.io lookup failed: {e}")
        return None
    
    def detect_vpn_proxy(self, ip_info):
        """Detect if IP is likely a VPN or proxy"""
        if not ip_info:
            return {'is_vpn': False, 'confidence': 0, 'reasons': []}
        
        reasons = []
        confidence = 0
        
        # Check organization/ISP name for VPN indicators
        org = ip_info.get('org', '').lower()
        isp = ip_info.get('isp', '').lower()
        
        for indicator in self.vpn_indicators:
            if indicator in org or indicator in isp:
                reasons.append(f"Organization/ISP contains '{indicator}'")
                confidence += 20
        
        # Check for known VPN providers
        for provider in self.known_vpn_providers:
            if provider in org or provider in isp:
                reasons.append(f"Known VPN provider: {provider}")
                confidence += 40
        
        # Check for datacenter/hosting indicators
        datacenter_indicators = [
            'datacenter', 'data center', 'hosting', 'cloud', 'server',
            'amazon', 'aws', 'google cloud', 'azure', 'digital ocean',
            'linode', 'vultr', 'ovh', 'hetzner', 'contabo'
        ]
        
        for indicator in datacenter_indicators:
            if indicator in org or indicator in isp:
                reasons.append(f"Datacenter/hosting provider: {indicator}")
                confidence += 15
        
        # Check for suspicious patterns
        if 'tor' in org or 'tor' in isp:
            reasons.append("Tor network detected")
            confidence += 50
        
        # Determine if likely VPN/proxy
        is_vpn = confidence >= 30
        
        return {
            'is_vpn': is_vpn,
            'confidence': min(confidence, 100),
            'reasons': reasons,
            'org': org,
            'isp': isp
        }
    
    def validate_dealer_source(self, ip_info, vpn_detection):
        """Validate if the request appears to come from a legitimate dealer"""
        if not ip_info:
            return {
                'is_valid': False,
                'status': 'UNKNOWN',
                'message': 'Unable to determine IP location'
            }
        
        if vpn_detection['is_vpn']:
            return {
                'is_valid': False,
                'status': 'SUSPICIOUS',
                'message': f"VPN/Proxy detected (confidence: {vpn_detection['confidence']}%)",
                'reasons': vpn_detection['reasons']
            }
        
        # Check if location makes sense for a dealer
        country = ip_info.get('country', '').lower()
        city = ip_info.get('city', '').lower()
        
        # US-based dealers are most common
        if country in ['united states', 'us', 'usa']:
            return {
                'is_valid': True,
                'status': 'VALID',
                'message': f"US-based request from {city}, {ip_info.get('region', 'Unknown Region')}"
            }
        
        # Other countries might be valid but require review
        return {
            'is_valid': True,
            'status': 'REVIEW_NEEDED',
            'message': f"International request from {city}, {country}",
            'country': country,
            'city': city
        }

class ScreenshotProcessor:
    """Enhanced screenshot processing for bucket files"""
    
    def __init__(self):
        self.ncb_patterns = {
            'agent_ncb': [
                r'agent\s*ncb[:\s]*\$?([0-9,]+\.?[0-9]*)',
                r'agent\s*no\s*chargeback[:\s]*\$?([0-9,]+\.?[0-9]*)',
                r'agent\s*fee[:\s]*\$?([0-9,]+\.?[0-9]*)',
                r'agent[:\s]*\$?([0-9,]+\.?[0-9]*)'
            ],
            'dealer_ncb': [
                r'dealer\s*ncb[:\s]*\$?([0-9,]+\.?[0-9]*)',
                r'dealer\s*no\s*chargeback[:\s]*\$?([0-9,]+\.?[0-9]*)',
                r'dealer\s*fee[:\s]*\$?([0-9,]+\.?[0-9]*)',
                r'dealer[:\s]*\$?([0-9,]+\.?[0-9]*)'
            ],
            'total_fees': [
                r'total[:\s]*\$?([0-9,]+\.?[0-9]*)',
                r'amount[:\s]*\$?([0-9,]+\.?[0-9]*)',
                r'fee[:\s]*\$?([0-9,]+\.?[0-9]*)'
            ]
        }
    
    def preprocess_image(self, image):
        """Preprocess image for better OCR, especially for handwritten text"""
        if not OPENCV_AVAILABLE:
            # Fallback: return original image if OpenCV is not available
            return image
            
        try:
            # Convert PIL to OpenCV format
            img_array = np.array(image)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply denoising (more aggressive for handwritten text)
            denoised = cv2.fastNlMeansDenoising(gray, h=10)
            
            # Apply bilateral filter to reduce noise while preserving edges
            filtered = cv2.bilateralFilter(denoised, 9, 75, 75)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphological operations to clean up handwritten text
            kernel = np.ones((2,2), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to PIL
            return Image.fromarray(cleaned)
        except Exception as e:
            print(f"Image preprocessing failed: {e}")
            return image
    
    def extract_ncb_data(self, image_path):
        """Extract NCB fee data from bucket screenshot"""
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            processed_image = self.preprocess_image(image)
            
            # Extract text using OCR with settings optimized for handwritten text
            try:
                text = pytesseract.image_to_string(processed_image, config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:()[]{}\"/- ')
            except Exception as e:
                print(f"OCR failed with processed image: {e}")
                text = ""
            
            # Also try with different OCR settings for handwritten text
            try:
                text_alt = pytesseract.image_to_string(image, config='--psm 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:()[]{}\"/- ')
            except Exception as e:
                print(f"OCR failed with original image: {e}")
                text_alt = ""
            
            # Combine both results
            combined_text = text + "\n" + text_alt
            
            # Extract NCB data
            ncb_data = {
                'agent_ncb_amount': None,
                'dealer_ncb_amount': None,
                'total_amount': None,
                'has_agent_ncb': False,
                'has_dealer_ncb': False,
                'sale_date': None,
                'contract_number': None,
                'raw_text': combined_text
            }
            
            # Debug output for NCB extraction
            print(f"NCB extraction from {image_path}:")
            print(f"  Text sample: {combined_text[:300]}...")
            
            # Extract Agent NCB
            for pattern in self.ncb_patterns['agent_ncb']:
                matches = re.findall(pattern, combined_text, re.IGNORECASE)
                if matches:
                    try:
                        amount = float(matches[0].replace(',', ''))
                        ncb_data['agent_ncb_amount'] = amount
                        ncb_data['has_agent_ncb'] = amount > 0
                        break
                    except ValueError:
                        continue
            
            # Extract Dealer NCB
            for pattern in self.ncb_patterns['dealer_ncb']:
                matches = re.findall(pattern, combined_text, re.IGNORECASE)
                if matches:
                    try:
                        amount = float(matches[0].replace(',', ''))
                        ncb_data['dealer_ncb_amount'] = amount
                        ncb_data['has_dealer_ncb'] = amount > 0
                        break
                    except ValueError:
                        continue
            
            # Extract total amount
            for pattern in self.ncb_patterns['total_fees']:
                matches = re.findall(pattern, combined_text, re.IGNORECASE)
                if matches:
                    try:
                        amount = float(matches[0].replace(',', ''))
                        ncb_data['total_amount'] = amount
                        break
                    except ValueError:
                        continue
            
            # Extract sale date from PCMI screenshots
            sale_date_patterns = [
                r'sale\s*date[:\s]*(\d{1,2}/\d{1,2}/\d{4})',
                r'effect\s*date[:\s]*(\d{1,2}/\d{1,2}/\d{4})',
                r'(?:sale|effect)\s*date[:\s]*(\d{1,2}/\d{1,2}/\d{2,4})',
                r'sale[:\s]*(\d{1,2}/\d{1,2}/\d{4})',
                r'effect[:\s]*(\d{1,2}/\d{1,2}/\d{4})'
            ]
            
            for pattern in sale_date_patterns:
                match = re.search(pattern, combined_text, re.IGNORECASE)
                if match:
                    ncb_data['sale_date'] = match.group(1)
                    break
            
            # Extract contract number from PCMI screenshots
            contract_patterns = [
                r'contract\s*number[:\s]*([A-Z0-9]+)',
                r'contract[:\s]*([A-Z0-9]+)',
                r'pn(\d+)',
                r'edit\s*contract\s*([A-Z0-9]+)'
            ]
            
            for pattern in contract_patterns:
                match = re.search(pattern, combined_text, re.IGNORECASE)
                if match:
                    contract_num = match.group(1)
                    if not contract_num.startswith('PN'):
                        contract_num = 'PN' + contract_num
                    ncb_data['contract_number'] = contract_num
                    break
            
            # Debug output for results
            print(f"  Agent NCB: {ncb_data['agent_ncb_amount']}")
            print(f"  Dealer NCB: {ncb_data['dealer_ncb_amount']}")
            print(f"  Total: {ncb_data['total_amount']}")
            print(f"  Sale Date: {ncb_data['sale_date']}")
            print(f"  Contract: {ncb_data['contract_number']}")
            print(f"  Has Agent NCB: {ncb_data['has_agent_ncb']}")
            print(f"  Has Dealer NCB: {ncb_data['has_dealer_ncb']}")
            
            return ncb_data
            
        except Exception as e:
            st.warning(f"Error processing screenshot {image_path}: {e}")
            return {
                'agent_ncb_amount': None,
                'dealer_ncb_amount': None,
                'total_amount': None,
                'has_agent_ncb': False,
                'has_dealer_ncb': False,
                'sale_date': None,
                'contract_number': None,
                'raw_text': ''
            }

class CancellationProcessor:
    def __init__(self):
        self.files_data = []
        self.packets = {}
        self.screenshot_processor = ScreenshotProcessor()
        
    def extract_text_from_file(self, file_path):
        """Extract text from various file types"""
        text = ""
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.pdf':
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
            elif file_ext in ['.docx']:
                doc = Document(file_path)
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
            elif file_ext in ['.doc']:
                # Fallback for .doc files
                with open(file_path, 'rb') as f:
                    text = f.read().decode('utf-8', errors='ignore')
            elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff']:
                # Check if it's a bucket screenshot
                if self.is_bucket_screenshot(file_path):
                    ncb_data = self.screenshot_processor.extract_ncb_data(file_path)
                    text = ncb_data['raw_text']
                else:
                    try:
                        image = Image.open(file_path)
                        try:
                            text = pytesseract.image_to_string(image, config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:()[]{}\"/- ')
                        except Exception as ocr_error:
                            print(f"OCR failed for {file_path}: {ocr_error}")
                            text = ""
                    except Exception as e:
                        st.warning(f"Error processing image {file_path}: {e}")
                        text = ""
            elif file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
        except Exception as e:
            st.warning(f"Error extracting text from {file_path}: {e}")
            
        return text
    
    def is_bucket_screenshot(self, file_path):
        """Determine if image is likely a bucket screenshot"""
        try:
            image = Image.open(file_path)
            # Quick OCR to check for bucket-related keywords
            try:
                text = pytesseract.image_to_string(image, config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:()[]{}\"/- ')
            except Exception as ocr_error:
                print(f"OCR failed in bucket detection for {file_path}: {ocr_error}")
                return False
            
            # Enhanced bucket detection keywords
            bucket_keywords = [
                'ncb', 'bucket', 'fee', 'agent', 'dealer', 'chargeback', 'pcmi',
                'admin', 'reserve', 'commission', 'rate details', 'written', 'cancelled',
                'balance', 'baseadmin', 'ncbagent', 'ncbdealer', 'agentcomm',
                'subagent', 'basereserve', 'surcharge', 'total'
            ]
            
            text_lower = text.lower()
            keyword_matches = sum(1 for keyword in bucket_keywords if keyword in text_lower)
            
            # Also check for specific patterns that indicate PCMI screenshots
            pcmi_patterns = [
                'edit contract', 'contract number', 'sale date', 'effect date',
                'rate type', 'net rate', 'admin total', 'reserves total',
                'hampstead', 'preowned', 'odom', 'deal', 'billed date'
            ]
            
            pattern_matches = sum(1 for pattern in pcmi_patterns if pattern in text_lower)
            
            # Debug output
            print(f"Bucket detection for {file_path}:")
            print(f"  Keywords found: {keyword_matches}")
            print(f"  Patterns found: {pattern_matches}")
            print(f"  Text sample: {text_lower[:200]}...")
            
            # Consider it a bucket screenshot if we have multiple keyword matches or PCMI patterns
            is_bucket = keyword_matches >= 3 or pattern_matches >= 2
            print(f"  Is bucket: {is_bucket}")
            return is_bucket
        except Exception as e:
            print(f"Error in bucket detection for {file_path}: {e}")
            return False
    
    def is_handwritten_document(self, file_path, text):
        """Determine if document is likely handwritten based on OCR characteristics"""
        if not text or len(text.strip()) < 10:
            return False
            
        # Indicators of handwritten text
        handwritten_indicators = [
            # OCR confidence patterns (handwritten text often has lower confidence)
            len(text.split()) < 20,  # Very few words extracted
            len([c for c in text if c.isalpha()]) / max(len(text), 1) < 0.3,  # Low letter ratio
            text.count(' ') / max(len(text.split()), 1) > 0.5,  # Many spaces (OCR struggling)
            # Common OCR errors with handwritten text
            text.count('|') > 2,  # Vertical lines often misread
            text.count('l') > text.count('I') * 2,  # Lowercase l vs uppercase I confusion
            text.count('0') > text.count('O') * 2,  # Zero vs O confusion
            # Inconsistent spacing and formatting
            len([word for word in text.split() if len(word) == 1]) > len(text.split()) * 0.3,
            # Very short or very long words (OCR struggling with handwriting)
            any(len(word) > 20 for word in text.split()),
            # Repeated characters (common in handwritten OCR)
            any(text.count(char) > len(text) * 0.1 for char in set(text) if char.isalpha())
        ]
        
        # If multiple indicators are present, likely handwritten
        return sum(handwritten_indicators) >= 3
    
    def extract_fields(self, text, filename, file_path=None):
        """Extract all relevant fields from text using regex patterns"""
        # Check if document is handwritten
        is_handwritten = self.is_handwritten_document(file_path, text) if file_path else False
        
        fields = {
            'filename': filename,
            'vins': [],
            'contracts': [],
            'reasons': [],
            'cancellation_dates': [],
            'sale_dates': [],
            'refund_addresses': [],
            'mileages': [],
            'customer_names': [],
            'has_agent_ncb': False,
            'has_dealer_ncb': False,
            'is_autohouse': False,
            'is_customer_direct': False,
            'is_diversicare': False,
            'has_signature': False,
            'has_pcmi_hint': False,
            'is_lender_letter': False,
            'is_handwritten': is_handwritten,
            'raw_text': text,
            'agent_ncb_amount': None,
            'dealer_ncb_amount': None,
            'total_ncb_amount': None
        }
        
        # VIN extraction
        vin_pattern = r'\b([A-HJ-NPR-Z0-9]{17})\b'
        fields['vins'] = re.findall(vin_pattern, text.upper())
        
        # Contract ID extraction
        gap_pattern = r'\bGAP\d{8}\b'
        common_prefix_pattern = r'\b(?:PN|EL|AD|OP|PT|PPM|EW|TW)\d{8}\b'
        generic_contract_pattern = r'(?:Contract\s*(?:No\.?|Number)?|CN|Contract#)\s*[:#]?\s*([A-Z]{1,3}\d{6,})'
        
        contracts = []
        contracts.extend(re.findall(gap_pattern, text.upper()))
        contracts.extend(re.findall(common_prefix_pattern, text.upper()))
        contracts.extend(re.findall(generic_contract_pattern, text.upper()))
        fields['contracts'] = list(set(contracts))  # Remove duplicates
        
        # Reason extraction - more comprehensive patterns
        reason_patterns = [
            r'(?:vehicle\s+traded|trade\s*in|sold|repossession|total\s*loss|dealer\s*buyback|customer\s*request)',
            r'(?:reason|cause)[:\s]+([^.\n]+)',
            r'(?:cancellation|cancel)[:\s]+(?:reason|cause)[:\s]+([^.\n]+)',
            r'(?:why|because)[:\s]+([^.\n]+)'
        ]
        
        reasons = []
        for pattern in reason_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                reason = match.strip()
                if len(reason) > 3 and not any(char.isdigit() for char in reason[:10]):
                    reasons.append(reason)
        
        fields['reasons'] = list(set(reasons))  # Remove duplicates
        
        # Date extraction - more comprehensive patterns
        date_pattern = r'([0-1]?\d[\/\-][0-3]?\d[\/\-](?:20)?\d{2})'
        
        # Cancellation date patterns
        cancellation_patterns = [
            r'(?:cancellation|cancel|effective)\s*date[ :\-]*' + date_pattern,
            r'(?:date\s+of\s+cancellation|cancellation\s+date)[ :\-]*' + date_pattern,
            r'(?:effective\s+date|date\s+effective)[ :\-]*' + date_pattern,
            r'(?:canceled|cancelled)\s+on[ :\-]*' + date_pattern
        ]
        
        cancellation_dates = []
        for pattern in cancellation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            cancellation_dates.extend(matches)
        fields['cancellation_dates'] = list(set(cancellation_dates))
        
        # Sale date patterns - enhanced for PCMI screenshots
        sale_patterns = [
            r'(?:contract\s*sale|sale)\s*date[ :\-]*' + date_pattern,
            r'(?:date\s+of\s+sale|sale\s+date)[ :\-]*' + date_pattern,
            r'(?:purchase\s+date|date\s+purchased)[ :\-]*' + date_pattern,
            r'(?:sold\s+on|sale\s+on)[ :\-]*' + date_pattern,
            r'sale\s*date[ :\-]*' + date_pattern,  # Simple "Sale Date: MM/DD/YYYY"
            r'effect\s*date[ :\-]*' + date_pattern,  # "Effect Date: MM/DD/YYYY"
            r'(?:contract|sale)\s*date[ :\-]*' + date_pattern,  # "Contract Date: MM/DD/YYYY"
            r'purchase[ :\-]*' + date_pattern,  # "Purchase: MM/DD/YYYY"
            r'sold[ :\-]*' + date_pattern,  # "Sold: MM/DD/YYYY"
            r'(?:in\s*service|service)\s*date[ :\-]*' + date_pattern,  # "In-Service Date: MM/DD/YYYY"
            r'billed\s*date[ :\-]*' + date_pattern,  # "Billed Date: MM/DD/YYYY"
            # More flexible patterns for table data
            r'(?:sale|effect|contract|purchase|billed)\s*date[:\s]*(\d{1,2}/\d{1,2}/\d{2,4})',
            r'(?:sale|effect|contract|purchase|billed)[:\s]*(\d{1,2}/\d{1,2}/\d{2,4})',
            # Pattern for PCMI table format
            r'(?:sale\s*date|effect\s*date)[:\s]*(\d{1,2}/\d{1,2}/\d{4})',
        ]
        
        sale_dates = []
        for pattern in sale_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            sale_dates.extend(matches)
        
        # Debug output for sale dates
        if sale_dates:
            print(f"Found sale dates in {file_path}: {sale_dates}")
        
        fields['sale_dates'] = list(set(sale_dates))
        
        # Refund address extraction - more comprehensive patterns
        address_patterns = [
            r'(?:remit|send|mail)\s+(?:refund|check)\s+to[: ]\s*(.+?)(?:\n|$)',
            r'(?:refund|check)\s+(?:to|address)[: ]\s*(.+?)(?:\n|$)',
            r'(?:send|mail)\s+(?:to|address)[: ]\s*(.+?)(?:\n|$)',
            r'(?:return\s+to|refund\s+address)[: ]\s*(.+?)(?:\n|$)',
            r'(?:different\s+address|alternate\s+address)[: ]\s*(.+?)(?:\n|$)'
        ]
        
        addresses = []
        for pattern in address_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                address = match.strip()
                if len(address) > 5:  # Basic validation
                    addresses.append(address)
        
        fields['refund_addresses'] = list(set(addresses))
        
        # Mileage extraction
        mileage_pattern = r'(?:mileage|odom(?:eter)?)\s*[:#]?\s*([0-9]{1,6}(?:,[0-9]{3})?)'
        fields['mileages'] = re.findall(mileage_pattern, text, re.IGNORECASE)
        
        # Customer name extraction
        customer_patterns = [
            r'(?:customer|client|name)[:\s]+([A-Za-z\s]+?)(?:\n|$)',
            r'(?:customer|client|name)[:\s]+([A-Za-z\s]+?)(?:\n|$)',
            r'Name[:\s]+([A-Za-z\s]+?)(?:\n|$)',
            r'Customer[:\s]+([A-Za-z\s]+?)(?:\n|$)'
        ]
        
        for pattern in customer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.strip()
                if len(name) > 2 and not any(char.isdigit() for char in name):
                    fields['customer_names'].append(name)
        
        # Flag detection with comprehensive patterns
        # Agent NCB patterns - NCB means No Chargeback (agent is NOT charged back)
        agent_ncb_patterns = [
            r'Agent\s+NCB',
            r'No\s*Chargeback.*Agent',
            r'Agent.*No\s*Chargeback',
            r'Agent.*NCB'
        ]
        # Note: Agent NCB = Agent No Chargeback (agent is NOT charged back)
        fields['has_agent_ncb'] = any(re.search(pattern, text, re.IGNORECASE) for pattern in agent_ncb_patterns)
        
        # Dealer NCB patterns - NCB means No Chargeback (dealer is NOT charged back)
        dealer_ncb_patterns = [
            r'Dealer\s+NCB',
            r'No\s*Chargeback.*Dealer',
            r'Dealer.*No\s*Chargeback',
            r'Dealer.*NCB'
        ]
        # Note: Dealer NCB = Dealer No Chargeback (dealer is NOT charged back)
        fields['has_dealer_ncb'] = any(re.search(pattern, text, re.IGNORECASE) for pattern in dealer_ncb_patterns)
        
        # Autohouse patterns
        autohouse_patterns = [
            r'auto\s*house',
            r'autohouse',
            r'auto\s*house\s*llc',
            r'auto\s*house\s*inc'
        ]
        fields['is_autohouse'] = any(re.search(pattern, text, re.IGNORECASE) for pattern in autohouse_patterns)
        
        # Customer direct patterns
        customer_direct_patterns = [
            r'customer\s+direct',
            r'dealer\s+out\s+of\s+business',
            r'\bFF\b',
            r'dealer\s+closed',
            r'business\s+closed'
        ]
        fields['is_customer_direct'] = any(re.search(pattern, text, re.IGNORECASE) for pattern in customer_direct_patterns)
        
        # Diversicare patterns
        diversicare_patterns = [
            r'diversicare',
            r'diversi\s*care',
            r'diversi-care'
        ]
        fields['is_diversicare'] = any(re.search(pattern, text, re.IGNORECASE) for pattern in diversicare_patterns)
        
        # Enhanced signature detection
        signature_patterns = [
            r'signature',
            r'signed',
            r'sign\s*here',
            r'authorized\s*signature',
            r'customer\s*signature',
            r'_________________',
            r'________________',
            r'_______________',
            r'______________',
            r'_____________',
            r'____________',
            r'___________',
            r'__________',
            r'_________',
            r'________',
            r'_______',
            r'______',
            r'_____',
            r'____',
            r'___',
            r'__',
            r'_'
        ]
        
        # Basic signature text detection
        has_signature_text = any(re.search(pattern, text, re.IGNORECASE) for pattern in signature_patterns)
        
        # Enhanced signature detection for images
        signature_quality = "none"
        signature_image_path = None
        
        if file_path and os.path.splitext(file_path)[1].lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
            signature_quality, signature_image_path = self.detect_signature_in_image(file_path)
        
        fields['has_signature'] = has_signature_text or signature_quality != "none"
        fields['signature_quality'] = signature_quality
        fields['signature_image_path'] = signature_image_path
        
        # PCMI hint patterns
        pcmi_patterns = [
            r'pcmi',
            r'ncb\s*(fee|bucket)',
            r'fee\s*bucket',
            r'chargeback\s*bucket',
            r'ncbfee',
            r'ncb\s*fee'
        ]
        fields['has_pcmi_hint'] = any(re.search(pattern, text, re.IGNORECASE) for pattern in pcmi_patterns)
        
        # Lender letter patterns
        lender_letter_patterns = [
            r'lender\s+letter',
            r'payoff\s+letter',
            r'addressed\s+to\s+ascent',
            r'ascent\s+addressed',
            r'payoff\s+notice',
            r'lender\s+notice'
        ]
        fields['is_lender_letter'] = any(re.search(pattern, text, re.IGNORECASE) for pattern in lender_letter_patterns)
        
        # Enhanced NCB amount extraction for bucket screenshots
        if file_path and self.is_bucket_screenshot(file_path):
            ncb_data = self.screenshot_processor.extract_ncb_data(file_path)
            fields['agent_ncb_amount'] = ncb_data['agent_ncb_amount']
            fields['dealer_ncb_amount'] = ncb_data['dealer_ncb_amount']
            fields['total_ncb_amount'] = ncb_data['total_amount']
            fields['has_agent_ncb'] = ncb_data['has_agent_ncb']
            fields['has_dealer_ncb'] = ncb_data['has_dealer_ncb']
            
            # Also extract sale date and contract number from NCB data
            if ncb_data['sale_date']:
                fields['sale_dates'].append(ncb_data['sale_date'])
            if ncb_data['contract_number']:
                fields['contracts'].append(ncb_data['contract_number'])
        
        return fields
    
    def group_files_into_packets(self):
        """Group all files into a single packet for cross-reference checking"""
        # Put all files into one packet for comprehensive checking
        file_count = len(self.files_data)
        self.packets = {f"ALL_FILES_{file_count}": self.files_data}
    
    def normalize_reason(self, reason):
        """Normalize reason to canonical form"""
        reason_map = {
            'vehicle traded': 'Vehicle Traded',
            'trade in': 'Trade In',
            'sold': 'Sold',
            'repossession': 'Repossession',
            'total loss': 'Total Loss',
            'dealer buyback': 'Dealer Buyback',
            'customer request': 'Customer Request'
        }
        return reason_map.get(reason.lower(), reason.title())
    
    def filter_handwritten_reasons(self, reasons):
        """Filter and prioritize reasons from handwritten documents"""
        if not reasons:
            return reasons
            
        # Common cancellation reasons in order of priority
        priority_reasons = [
            'customer request',
            'cancellation request',
            'customer cancellation',
            'voluntary cancellation',
            'total loss',
            'repossession',
            'loan payoff',
            'vehicle traded',
            'early payoff',
            'refinance'
        ]
        
        # Clean and normalize all reasons
        cleaned_reasons = []
        for reason in reasons:
            if reason and reason.strip():
                cleaned = reason.strip().lower()
                # Remove common OCR noise
                if len(cleaned) > 2 and not cleaned in ['→', 'upload', 'lienholder', 'payoff', 'letter']:
                    cleaned_reasons.append(cleaned)
        
        if not cleaned_reasons:
            return reasons
            
        # Find the best match based on priority
        best_reason = None
        for priority_reason in priority_reasons:
            for cleaned_reason in cleaned_reasons:
                if priority_reason in cleaned_reason or cleaned_reason in priority_reason:
                    best_reason = cleaned_reason
                    break
            if best_reason:
                break
        
        # If no priority match found, use the longest/shortest meaningful reason
        if not best_reason:
            # Filter out very short or very long reasons (likely OCR noise)
            meaningful_reasons = [r for r in cleaned_reasons if 3 <= len(r) <= 50]
            if meaningful_reasons:
                # Prefer shorter, more specific reasons
                best_reason = min(meaningful_reasons, key=len)
        
        return [best_reason] if best_reason else reasons
    
    def detect_signature_in_image(self, image_path):
        """Detect and analyze signatures in images"""
        try:
            if not OPENCV_AVAILABLE:
                return "none", None
                
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return "none", None
                
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply preprocessing for signature detection
            # Use adaptive thresholding to better detect handwritten signatures
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            signature_areas = []
            signature_quality = "none"
            
            for contour in contours:
                # Calculate contour properties
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Signature characteristics:
                # - Reasonable size (not too small, not too large)
                # - Reasonable aspect ratio (not too square, not too thin)
                # - Complex shape (not just a dot or line)
                if (area > 500 and area < 50000 and  # Size range
                    0.2 < aspect_ratio < 5.0 and  # Aspect ratio range
                    len(contour) > 20):  # Complex shape (enough points)
                    
                    # Calculate complexity (perimeter^2 / area ratio)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        complexity = (perimeter * perimeter) / area
                        
                        # Signatures typically have moderate to high complexity
                        if complexity > 10:  # Threshold for signature-like complexity
                            signature_areas.append((x, y, w, h, complexity))
            
            if signature_areas:
                # Sort by complexity (most complex first)
                signature_areas.sort(key=lambda x: x[4], reverse=True)
                
                # Get the best signature area
                best_sig = signature_areas[0]
                x, y, w, h, complexity = best_sig
                
                # Determine quality based on complexity and size
                if complexity > 50 and area > 2000:
                    signature_quality = "excellent"
                elif complexity > 30 and area > 1000:
                    signature_quality = "good"
                elif complexity > 15 and area > 500:
                    signature_quality = "fair"
                else:
                    signature_quality = "poor"
                
                # Extract signature region
                signature_region = image[y:y+h, x:x+w]
                
                # Save signature image for display
                signature_filename = f"signature_{os.path.basename(image_path)}"
                signature_path = os.path.join(os.path.dirname(image_path), signature_filename)
                cv2.imwrite(signature_path, signature_region)
                
                return signature_quality, signature_path
            
            return "none", None
            
        except Exception as e:
            print(f"Error detecting signature in {image_path}: {e}")
            return "none", None
    
    def create_thumbnail(self, file_data, temp_dir, size=(200, 150)):
        """Create a thumbnail for file display"""
        filename = file_data['filename']
        file_path = os.path.join(temp_dir, filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        print(f"Creating thumbnail for {filename} at {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")
        
        try:
            if file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
                # For images, create a thumbnail
                image = Image.open(file_path)
                image.thumbnail(size, Image.Resampling.LANCZOS)
                print(f"Created image thumbnail for {filename}")
                return image
            elif file_ext == '.pdf':
                # For PDFs, create a simple icon
                from PIL import ImageDraw, ImageFont
                img = Image.new('RGB', size, color='white')
                draw = ImageDraw.Draw(img)
                try:
                    # Try to use a default font
                    font = ImageFont.load_default()
                except:
                    font = None
                draw.text((10, 10), "PDF", fill='red', font=font)
                draw.text((10, 30), filename[:20], fill='black', font=font)
                print(f"Created PDF thumbnail for {filename}")
                return img
            elif file_ext in ['.docx', '.doc']:
                # For Word docs, create a simple icon
                from PIL import ImageDraw, ImageFont
                img = Image.new('RGB', size, color='white')
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
                draw.text((10, 10), "DOC", fill='blue', font=font)
                draw.text((10, 30), filename[:20], fill='black', font=font)
                print(f"Created DOC thumbnail for {filename}")
                return img
            elif file_ext == '.txt':
                # For text files, create a simple icon
                from PIL import ImageDraw, ImageFont
                img = Image.new('RGB', size, color='white')
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
                draw.text((10, 10), "TXT", fill='green', font=font)
                draw.text((10, 30), filename[:20], fill='black', font=font)
                print(f"Created TXT thumbnail for {filename}")
                return img
            else:
                # Default icon for unknown file types
                from PIL import ImageDraw, ImageFont
                img = Image.new('RGB', size, color='lightgray')
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
                draw.text((10, 10), "FILE", fill='black', font=font)
                draw.text((10, 30), filename[:20], fill='black', font=font)
                print(f"Created default thumbnail for {filename}")
                return img
        except Exception as e:
            # Error thumbnail
            from PIL import ImageDraw, ImageFont
            img = Image.new('RGB', size, color='red')
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.load_default()
            except:
                font = None
            draw.text((10, 10), "ERROR", fill='white', font=font)
            draw.text((10, 30), str(e)[:20], fill='white', font=font)
            print(f"Error creating thumbnail for {filename}: {e}")
            return img
    
    def display_file_thumbnails(self, files_data, temp_dir):
        """Display all files as thumbnails in a clean grid"""
        if not files_data:
            st.info("No files to display")
            return
        
        st.subheader("📁 File Thumbnails - Quick Review")
        st.markdown("All uploaded files displayed as thumbnails for easy review")
        
        # Calculate grid layout (3 columns)
        num_files = len(files_data)
        cols_per_row = 3
        
        for i in range(0, num_files, cols_per_row):
            # Create columns for this row
            cols = st.columns(cols_per_row)
            
            for j, col in enumerate(cols):
                file_idx = i + j
                if file_idx < num_files:
                    file_data = files_data[file_idx]
                    filename = file_data['filename']
                    file_ext = os.path.splitext(filename)[1].lower()
                    
                    with col:
                        # Create thumbnail
                        try:
                            thumbnail = self.create_thumbnail(file_data, temp_dir)
                            if thumbnail:
                                st.image(thumbnail, caption=filename, use_container_width=True)
                            else:
                                st.write("❌ Thumbnail creation failed")
                        except Exception as e:
                            st.write(f"❌ Error creating thumbnail: {e}")
                            # Fallback: show file info without thumbnail
                            st.write(f"📄 {filename}")
                        
                        # File info
                        st.write(f"**{filename}**")
                        st.write(f"Type: {file_ext.upper()}")
                        
                        # Status indicators
                        status_indicators = []
                        if file_data.get('is_handwritten', False):
                            status_indicators.append("⚠️ Handwritten")
                        if file_data.get('has_signature', False):
                            quality = file_data.get('signature_quality', 'unknown')
                            status_indicators.append(f"✍️ Signature ({quality})")
                        if file_data.get('is_lender_letter', False):
                            status_indicators.append("📄 Lender Letter")
                        if file_data.get('has_pcmi_hint', False):
                            status_indicators.append("📊 PCMI Screenshot")
                        
                        if status_indicators:
                            for indicator in status_indicators:
                                st.write(indicator)
                        
                        # Quick data preview
                        with st.expander("📊 Quick Data"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if file_data.get('vins'):
                                    st.write(f"**VINs:** {len(file_data['vins'])}")
                                if file_data.get('contracts'):
                                    st.write(f"**Contracts:** {len(file_data['contracts'])}")
                                if file_data.get('reasons'):
                                    st.write(f"**Reasons:** {len(file_data['reasons'])}")
                                    
                            with col2:
                                if file_data.get('cancellation_dates'):
                                    st.write(f"**Dates:** {len(file_data['cancellation_dates'])}")
                                if file_data.get('customer_names'):
                                    st.write(f"**Names:** {len(file_data['customer_names'])}")
                                if file_data.get('mileages'):
                                    st.write(f"**Mileages:** {len(file_data['mileages'])}")
                        
                        # Show signature image if available
                        if file_data.get('signature_image_path') and os.path.exists(file_data['signature_image_path']):
                            st.write("**Detected Signature:**")
                            try:
                                sig_image = Image.open(file_data['signature_image_path'])
                                st.image(sig_image, width=150)
                            except Exception as e:
                                st.write(f"Error loading signature: {e}")
                        
                        st.divider()
    
    def parse_date(self, date_str):
        """Parse date string in various formats"""
        if not date_str or not date_str.strip():
            return None
            
        date_str = date_str.strip()
        
        # Try multiple date formats
        formats = [
            '%m/%d/%Y', '%m-%d-%Y', '%m/%d/%y', '%m-%d-%y',
            '%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y', '%d-%m-%Y',
            '%B %d, %Y', '%b %d, %Y', '%d %B %Y', '%d %b %Y',
            '%m/%d', '%m-%d', '%d/%m', '%d-%m'
        ]
        
        for fmt in formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                # If year is missing, assume current year
                if parsed_date.year == 1900:
                    parsed_date = parsed_date.replace(year=datetime.now().year)
                return parsed_date
            except ValueError:
                continue
        
        # Try to extract date from text using regex
        date_patterns = [
            r'(\d{1,2})/(\d{1,2})/(\d{4})',  # MM/DD/YYYY
            r'(\d{1,2})-(\d{1,2})-(\d{4})',  # MM-DD-YYYY
            r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
            r'(\d{1,2})/(\d{1,2})/(\d{2})',  # MM/DD/YY
            r'(\d{1,2})-(\d{1,2})-(\d{2})',  # MM-DD-YY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    groups = match.groups()
                    if len(groups) == 3:
                        month, day, year = groups
                        year = int(year)
                        if year < 100:  # Two-digit year
                            year += 2000 if year < 50 else 1900
                        return datetime(int(year), int(month), int(day))
                except ValueError:
                    continue
        
        return None
    
    def validate_refund_calculations(self, text):
        """Validate refund calculations from bucket screenshots"""
        calculations = {
            'retail_purchase_price': None,
            'dealer_remitted_amount': None,
            'cancel_fee': None,
            'refund_percentage': None,
            'ascent_refund': None,
            'dealer_refund': None,
            'net_customer_refund': None,
            'total_refund': None
        }
        
        # Extract monetary values
        money_patterns = [
            r'\$([0-9,]+\.?[0-9]*)',
            r'([0-9,]+\.?[0-9]*)\s*dollars?',
            r'([0-9,]+\.?[0-9]*)\s*USD'
        ]
        
        # Extract percentage
        percent_pattern = r'([0-9,]+\.?[0-9]*)\s*%'
        
        # Find all monetary values
        money_values = []
        for pattern in money_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    value = float(match.replace(',', ''))
                    money_values.append(value)
                except ValueError:
                    continue
        
        # Find percentage
        percent_match = re.search(percent_pattern, text)
        if percent_match:
            try:
                calculations['refund_percentage'] = float(percent_match.group(1))
            except ValueError:
                pass
        
        # Try to identify values by context
        lines = text.split('\n')
        for line in lines:
            line_lower = line.lower()
            
            # Look for specific labels
            if 'retail' in line_lower and 'purchase' in line_lower and 'price' in line_lower:
                for value in money_values:
                    if value > 1000:  # Reasonable retail price
                        calculations['retail_purchase_price'] = value
                        break
            
            elif 'dealer' in line_lower and 'remitted' in line_lower:
                for value in money_values:
                    if 500 < value < 5000:  # Reasonable dealer amount
                        calculations['dealer_remitted_amount'] = value
                        break
            
            elif 'cancel' in line_lower and 'fee' in line_lower:
                for value in money_values:
                    if 0 < value < 1000:  # Reasonable cancel fee
                        calculations['cancel_fee'] = value
                        break
            
            elif 'ascent' in line_lower and 'refund' in line_lower:
                for value in money_values:
                    if value > 0:
                        calculations['ascent_refund'] = value
                        break
            
            elif 'dealer' in line_lower and 'refund' in line_lower and 'ascent' not in line_lower:
                for value in money_values:
                    if value >= 0:
                        calculations['dealer_refund'] = value
                        break
            
            elif 'net' in line_lower and 'customer' in line_lower and 'refund' in line_lower:
                for value in money_values:
                    if value > 0:
                        calculations['net_customer_refund'] = value
                        break
        
        # If we couldn't identify by context, try to assign by value ranges
        if not calculations['retail_purchase_price'] and money_values:
            # Largest value is likely retail price
            calculations['retail_purchase_price'] = max(money_values)
        
        if not calculations['dealer_remitted_amount'] and len(money_values) >= 2:
            # Second largest is likely dealer amount
            sorted_values = sorted(money_values, reverse=True)
            calculations['dealer_remitted_amount'] = sorted_values[1]
        
        if not calculations['cancel_fee'] and len(money_values) >= 3:
            # Third largest or smallest non-zero is likely cancel fee
            sorted_values = sorted(money_values, reverse=True)
            for value in sorted_values:
                if 0 < value < 1000:
                    calculations['cancel_fee'] = value
                    break
        
        # Calculate total refund
        if calculations['ascent_refund'] and calculations['dealer_refund']:
            calculations['total_refund'] = calculations['ascent_refund'] + calculations['dealer_refund']
        elif calculations['net_customer_refund']:
            calculations['total_refund'] = calculations['net_customer_refund']
        
        return calculations
    
    def check_calculation_accuracy(self, calculations):
        """Check if the refund calculations are mathematically correct"""
        issues = []
        status = "PASS"
        
        retail = calculations.get('retail_purchase_price')
        dealer_remitted = calculations.get('dealer_remitted_amount')
        cancel_fee = calculations.get('cancel_fee')
        refund_pct = calculations.get('refund_percentage')
        ascent_refund = calculations.get('ascent_refund')
        dealer_refund = calculations.get('dealer_refund')
        net_customer_refund = calculations.get('net_customer_refund')
        total_refund = calculations.get('total_refund')
        
        # Check 1: Refund percentage should be 100% for full refunds
        if refund_pct is not None and abs(refund_pct - 100.0) > 0.1:
            issues.append(f"Refund percentage is {refund_pct}% instead of expected 100%")
            status = "MANUAL_REVIEW_NEEDED"
        
        # Check 2: Total refund should equal retail purchase price
        if retail and total_refund and abs(total_refund - retail) > 0.01:
            issues.append(f"Total refund (${total_refund:.2f}) doesn't match retail price (${retail:.2f})")
            status = "MANUAL_REVIEW_NEEDED"
        
        # Check 3: Dealer remitted + cancel fee should equal retail price
        if dealer_remitted and cancel_fee and retail:
            expected_retail = dealer_remitted + cancel_fee
            if abs(expected_retail - retail) > 0.01:
                issues.append(f"Dealer remitted (${dealer_remitted:.2f}) + Cancel fee (${cancel_fee:.2f}) = ${expected_retail:.2f}, but retail price is ${retail:.2f}")
                status = "MANUAL_REVIEW_NEEDED"
        
        # Check 4: Net customer refund should equal total refund
        if net_customer_refund and total_refund and abs(net_customer_refund - total_refund) > 0.01:
            issues.append(f"Net customer refund (${net_customer_refund:.2f}) doesn't match total refund (${total_refund:.2f})")
            status = "MANUAL_REVIEW_NEEDED"
        
        # Check 5: Ascent refund + dealer refund should equal total refund
        if ascent_refund is not None and dealer_refund is not None and total_refund:
            expected_total = ascent_refund + dealer_refund
            if abs(expected_total - total_refund) > 0.01:
                issues.append(f"Ascent refund (${ascent_refund:.2f}) + Dealer refund (${dealer_refund:.2f}) = ${expected_total:.2f}, but total refund is ${total_refund:.2f}")
                status = "MANUAL_REVIEW_NEEDED"
        
        # Check 6: All values should be positive (except dealer refund which can be 0)
        for key, value in calculations.items():
            if value is not None and value < 0 and key != 'dealer_refund':
                issues.append(f"{key.replace('_', ' ').title()} is negative: ${value:.2f}")
                status = "MANUAL_REVIEW_NEEDED"
        
        return {
            'status': status,
            'issues': issues,
            'calculations': calculations
        }
    
    def reconcile_values(self, values):
        """Reconcile multiple values and return status and canonical value"""
        # Remove empty values and duplicates while preserving order
        clean_values = list(dict.fromkeys([v for v in values if v and v.strip()]))
        
        if len(clean_values) == 0:
            return 'INFO', ''
        elif len(clean_values) == 1:
            return 'PASS', clean_values[0]
        else:
            return 'FAIL', '; '.join(clean_values)
    
    def validate_checklist_completeness(self, result):
        """Validate that all checklist items are properly evaluated"""
        required_fields = [
            'Vin Match on all forms',
            'Contract Match on all forms and Google sheet',
            'Reason Match across all forms',
            'Cancellation date match across all forms. (Favor lender letter if applicable)',
            'Is the cancellation effective date past 90 days from contract sale date?',
            'Is there an Agent NCB Fee?',
            'Is there a Dealer NCB Fee?',
            'Is there a different address to send the refund? (only applicable if the request included a lender letter addressed to Ascent)',
            'All necessary signatures collected?',
            'Is this an Autohouse Contract?',
            'Is this a customer direct cancellation? (Dealer Out of Business or FF contract)',
            'Is this a Diversicare contract?',
            'PCMI Screenshot (Of NCB fee buckets)'
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in result:
                missing_fields.append(field)
        
        if missing_fields:
            st.warning(f"Missing checklist fields: {', '.join(missing_fields)}")
        
        return len(missing_fields) == 0
    
    def choose_cxl_date_with_lender_preference(self, files, all_cxl_dates):
        """
        If any record looks like a lender letter file, prefer its cancellation date.
        Otherwise fall back to reconcile(all_cxl_dates).
        Returns (status, chosen_value)
        """
        # Heuristic: lender-letter record has is_lender_letter True
        lender_dates = []
        for file_data in files:
            if file_data.get("is_lender_letter") and file_data.get("cancellation_dates"):
                lender_dates.extend(file_data["cancellation_dates"])

        lender_dates = [d for d in lender_dates if d and d.strip()]
        lender_dates = list(dict.fromkeys(lender_dates))  # de-dup, order-preserving

        if len(lender_dates) == 1:
            return ("PASS", lender_dates[0])
        elif len(lender_dates) > 1:
            # Conflicting dates even within lender letters
            return ("FAIL", "; ".join(lender_dates))

        # No lender-letter dates; use normal reconcile across all dates
        return self.reconcile_values(all_cxl_dates)
    
    def evaluate_packet(self, packet_key, files):
        """Evaluate a packet against QC checklist"""
        # Initialize result
        result = {
            'Packet Key': packet_key,
            'Files': ', '.join([f['filename'] for f in files]),
            'Contract Number': '',
            'Customer Name': ''
        }
        
        # Create detailed source tracking
        source_data = {
            'vins': [],
            'contracts': [],
            'reasons': [],
            'cancellation_dates': [],
            'sale_dates': [],
            'refund_addresses': [],
            'mileages': [],
            'customer_names': []
        }
        
        # Collect all values from all files in packet with source tracking
        all_vins = []
        all_contracts = []
        all_reasons = []
        all_cancellation_dates = []
        all_sale_dates = []
        all_refund_addresses = []
        all_mileages = []
        all_customer_names = []
        
        has_agent_ncb = False
        has_dealer_ncb = False
        is_autohouse = False
        is_customer_direct = False
        is_diversicare = False
        has_signature = False
        has_pcmi_hint = False
        is_lender_letter = False
        
        # NCB amounts from screenshots
        agent_ncb_amounts = []
        dealer_ncb_amounts = []
        total_ncb_amounts = []
        
        for file_data in files:
            filename = file_data['filename']
            
            # Track sources for each data type
            for vin in file_data['vins']:
                all_vins.append(vin)
                source_data['vins'].append(f"{vin} (from {filename})")
            
            for contract in file_data['contracts']:
                all_contracts.append(contract)
                source_data['contracts'].append(f"{contract} (from {filename})")
            
            for reason in file_data['reasons']:
                all_reasons.append(reason)
                source_data['reasons'].append(f"{reason} (from {filename})")
            
            for date in file_data['cancellation_dates']:
                all_cancellation_dates.append(date)
                source_data['cancellation_dates'].append(f"{date} (from {filename})")
            
            for date in file_data['sale_dates']:
                all_sale_dates.append(date)
                source_data['sale_dates'].append(f"{date} (from {filename})")
            
            for address in file_data['refund_addresses']:
                all_refund_addresses.append(address)
                source_data['refund_addresses'].append(f"{address} (from {filename})")
            
            for mileage in file_data['mileages']:
                all_mileages.append(mileage)
                source_data['mileages'].append(f"{mileage} (from {filename})")
            
            for name in file_data['customer_names']:
                all_customer_names.append(name)
                source_data['customer_names'].append(f"{name} (from {filename})")
            
            has_agent_ncb = has_agent_ncb or file_data['has_agent_ncb']
            has_dealer_ncb = has_dealer_ncb or file_data['has_dealer_ncb']
            is_autohouse = is_autohouse or file_data['is_autohouse']
            is_customer_direct = is_customer_direct or file_data['is_customer_direct']
            is_diversicare = is_diversicare or file_data['is_diversicare']
            has_signature = has_signature or file_data['has_signature']
            has_pcmi_hint = has_pcmi_hint or file_data['has_pcmi_hint']
            is_lender_letter = is_lender_letter or file_data['is_lender_letter']
            
            # Collect NCB amounts
            if file_data['agent_ncb_amount'] is not None:
                agent_ncb_amounts.append(file_data['agent_ncb_amount'])
            if file_data['dealer_ncb_amount'] is not None:
                dealer_ncb_amounts.append(file_data['dealer_ncb_amount'])
            if file_data['total_ncb_amount'] is not None:
                total_ncb_amounts.append(file_data['total_ncb_amount'])
        
        # Remove duplicates and empty values
        all_vins = list(set([v for v in all_vins if v.strip()]))
        all_contracts = list(set([c for c in all_contracts if c.strip()]))
        all_reasons = list(set([r for r in all_reasons if r.strip()]))
        all_cancellation_dates = list(set([d for d in all_cancellation_dates if d.strip()]))
        all_sale_dates = list(set([d for d in all_sale_dates if d.strip()]))
        all_refund_addresses = list(set([a for a in all_refund_addresses if a.strip()]))
        all_mileages = list(set([m for m in all_mileages if m.strip()]))
        all_customer_names = list(set([n for n in all_customer_names if n.strip()]))
        
        # Set customer name
        if all_customer_names:
            result['Customer Name'] = all_customer_names[0]  # Take first one
        
        # Additional fields from JotForm structure
        result['Contact Person'] = ''
        result['Phone Number'] = ''
        result['Email'] = ''
        result['Selling Dealership'] = ''
        result['Today\'s Date'] = ''
        
        # VIN evaluation
        vin_status, vin_value = self.reconcile_values(all_vins)
        result['Vin Match on all forms'] = vin_status
        result['VIN (canonical)'] = vin_value
        
        # Contract evaluation
        contract_status, contract_value = self.reconcile_values(all_contracts)
        result['Contract Match on all forms and Google sheet'] = contract_status
        result['Contract (canonical)'] = contract_value
        result['Contract Number'] = contract_value  # Set the contract number for display
        
        # Check if any files are handwritten
        has_handwritten = any(file_data.get('is_handwritten', False) for file_data in files)
        
        # Reason evaluation with handwritten document filtering
        if has_handwritten:
            # For handwritten documents, flag for manual review
            result['Reason Match across all forms'] = 'MANUAL_REVIEW_NEEDED'
            result['Reason (canonical)'] = 'HANDWRITTEN - Manual Review Required'
            result['Handwritten Files'] = [f['filename'] for f in files if f.get('is_handwritten', False)]
        else:
            filtered_reasons = self.filter_handwritten_reasons(all_reasons)
            reason_status, reason_value = self.reconcile_values(filtered_reasons)
            result['Reason Match across all forms'] = reason_status
            if reason_value:
                result['Reason (canonical)'] = '; '.join([self.normalize_reason(r) for r in reason_value.split('; ')])
            else:
                result['Reason (canonical)'] = reason_value
        
        # Cancellation date evaluation with lender letter preference
        cxl_status, cxl_value = self.choose_cxl_date_with_lender_preference(files, all_cancellation_dates)
        result['Cancellation date match across all forms. (Favor lender letter if applicable)'] = cxl_status
        result['Cancellation Effective Date'] = cxl_value
        
        # Sale date
        if len(all_sale_dates) == 0:
            result['Sale Date'] = ''
        else:
            result['Sale Date'] = all_sale_dates[0]  # Take first one
        
        # 90-day check with improved date parsing
        cxl_date = self.parse_date(result['Cancellation Effective Date'])
        sale_date = self.parse_date(result['Sale Date'])
        
        if cxl_date and sale_date:
            days_diff = (cxl_date - sale_date).days
            result['Is the cancellation effective date past 90 days from contract sale date?'] = 'Yes' if days_diff > 90 else 'No'
            result['Days Difference'] = f"{days_diff} days"
        else:
            # Try to parse dates from all available date fields
            all_cxl_dates = [d for f in files for d in f.get('cancellation_dates', []) if d.strip()]
            all_sale_dates = [d for f in files for d in f.get('sale_dates', []) if d.strip()]
            
            # Try parsing each date combination
            for cxl_date_str in all_cxl_dates:
                for sale_date_str in all_sale_dates:
                    cxl_date = self.parse_date(cxl_date_str)
                    sale_date = self.parse_date(sale_date_str)
                    if cxl_date and sale_date:
                        days_diff = (cxl_date - sale_date).days
                        result['Is the cancellation effective date past 90 days from contract sale date?'] = 'Yes' if days_diff > 90 else 'No'
                        result['Days Difference'] = f"{days_diff} days"
                        break
                if 'Days Difference' in result:
                    break
            
            # If still no valid dates, check if we have a future cancellation date
            if 'Days Difference' not in result and cxl_date:
                current_date = datetime.now()
                if cxl_date > current_date:
                    result['Is the cancellation effective date past 90 days from contract sale date?'] = 'Future Date - Cannot Calculate'
                    result['Days Difference'] = f"Future date: {cxl_date.strftime('%m/%d/%Y')}"
                else:
                    result['Is the cancellation effective date past 90 days from contract sale date?'] = 'Unknown - No sale date found'
                    result['Days Difference'] = 'N/A'
            elif 'Days Difference' not in result:
                result['Is the cancellation effective date past 90 days from contract sale date?'] = 'Unknown - No valid dates found'
                result['Days Difference'] = 'N/A'
        
        # Enhanced NCB flags with amounts (NCB = No Chargeback)
        if agent_ncb_amounts:
            result['Is there an Agent NCB Fee?'] = f'Yes - No Chargeback (${sum(agent_ncb_amounts):.2f})' if sum(agent_ncb_amounts) > 0 else 'No'
        else:
            result['Is there an Agent NCB Fee?'] = 'Yes - No Chargeback' if has_agent_ncb else 'No'
            
        if dealer_ncb_amounts:
            result['Is there a Dealer NCB Fee?'] = f'Yes - No Chargeback (${sum(dealer_ncb_amounts):.2f})' if sum(dealer_ncb_amounts) > 0 else 'No'
        else:
            result['Is there a Dealer NCB Fee?'] = 'Yes - No Chargeback' if has_dealer_ncb else 'No'
        
        # Refund address
        if is_lender_letter and all_refund_addresses:
            result['Is there a different address to send the refund? (only applicable if the request included a lender letter addressed to Ascent)'] = 'Yes'
            result['Alt Refund Address (if any)'] = '; '.join(all_refund_addresses[:2])  # First 2 addresses
        else:
            result['Is there a different address to send the refund? (only applicable if the request included a lender letter addressed to Ascent)'] = 'No'
            result['Alt Refund Address (if any)'] = ''
        
        # Enhanced signature evaluation
        signature_qualities = [f.get('signature_quality', 'none') for f in files if f.get('has_signature', False)]
        signature_images = [f.get('signature_image_path') for f in files if f.get('signature_image_path')]
        
        if has_signature:
            if any(q in ['excellent', 'good'] for q in signature_qualities):
                result['All necessary signatures collected?'] = 'Yes - High Quality'
            elif any(q in ['fair', 'poor'] for q in signature_qualities):
                result['All necessary signatures collected?'] = 'Yes - Low Quality (Review Recommended)'
            else:
                result['All necessary signatures collected?'] = 'Yes - Text Only'
            
            # Store signature quality and images for display
            result['Signature Quality'] = ', '.join([q for q in signature_qualities if q != 'none'])
            result['Signature Images'] = [img for img in signature_images if img]
        else:
            result['All necessary signatures collected?'] = 'No'
            result['Signature Quality'] = 'None'
            result['Signature Images'] = []
        
        # Contract types
        result['Is this an Autohouse Contract?'] = 'Yes' if is_autohouse else 'No'
        result['Is this a customer direct cancellation? (Dealer Out of Business or FF contract)'] = 'Yes' if is_customer_direct else 'No'
        result['Is this a Diversicare contract?'] = 'Yes' if is_diversicare else 'No'
        
        # PCMI Screenshot with enhanced detection
        if has_pcmi_hint or agent_ncb_amounts or dealer_ncb_amounts:
            result['PCMI Screenshot (Of NCB fee buckets)'] = 'Present (hint detected)'
            if total_ncb_amounts:
                result['Total NCB Amount'] = f'${sum(total_ncb_amounts):.2f}'
        else:
            result['PCMI Screenshot (Of NCB fee buckets)'] = 'Not found'
            result['Total NCB Amount'] = ''
        
        # Refund calculation validation
        calculation_status = "INFO"
        calculation_issues = []
        calculation_details = {}
        
        # Check if any files contain bucket/refund information
        bucket_files = [f for f in files if f.get('has_pcmi_hint', False) or 'bucket' in f.get('filename', '').lower()]
        
        if bucket_files:
            # Extract calculations from bucket files
            for file_data in bucket_files:
                if file_data.get('raw_text'):
                    calculations = self.validate_refund_calculations(file_data['raw_text'])
                    if any(calculations.values()):
                        calculation_details = calculations
                        validation_result = self.check_calculation_accuracy(calculations)
                        calculation_status = validation_result['status']
                        calculation_issues = validation_result['issues']
                        break
        
        result['Refund Calculation Status'] = calculation_status
        result['Refund Calculation Issues'] = '; '.join(calculation_issues) if calculation_issues else ''
        result['Refund Calculation Details'] = calculation_details
        
        # Mileage - show all detected mileages
        if all_mileages:
            # Remove duplicates while preserving order
            unique_mileages = list(dict.fromkeys(all_mileages))
            result['Mileage values found'] = ', '.join(unique_mileages)
            result['Mileage Match Status'] = 'PASS' if len(unique_mileages) == 1 else 'FAIL' if len(unique_mileages) > 1 else 'INFO'
        else:
            result['Mileage values found'] = ''
            result['Mileage Match Status'] = 'INFO'
        
        # Add source tracking data
        result['_source_data'] = source_data
        
        # Validate checklist completeness
        self.validate_checklist_completeness(result)
        
        return result
    
    def process_zip(self, zip_file):
        """Process uploaded ZIP file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract ZIP
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Process all files
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.isfile(file_path):
                        text = self.extract_text_from_file(file_path)
                        if text.strip():  # Only process files with content
                            fields = self.extract_fields(text, file, file_path)
                            self.files_data.append(fields)
            
            # Group files into packets
            self.group_files_into_packets()
            
            # Evaluate each packet
            results = []
            for packet_key, files in self.packets.items():
                result = self.evaluate_packet(packet_key, files)
                results.append(result)
            
            # Display thumbnails before temp_dir is cleaned up
            self.display_file_thumbnails(self.files_data, temp_dir)
            
            return results

# Main app
def main():
    processor = CancellationProcessor()
    
    # Initialize IP security checker
    ip_checker = IPSecurityChecker()
    
    # Get client IP and perform security check
    client_ip = ip_checker.get_client_ip()
    ip_info = None
    vpn_detection = None
    security_validation = None
    
    if client_ip:
        with st.spinner("🔍 Checking IP address and security..."):
            ip_info = ip_checker.lookup_ip_info(client_ip)
            if ip_info:
                vpn_detection = ip_checker.detect_vpn_proxy(ip_info)
                security_validation = ip_checker.validate_dealer_source(ip_info, vpn_detection)
    
    # Display security status
    if security_validation:
        if security_validation['status'] == 'SUSPICIOUS':
            st.error(f"🚨 **SECURITY ALERT**: {security_validation['message']}")
            if security_validation.get('reasons'):
                for reason in security_validation['reasons']:
                    st.write(f"• {reason}")
        elif security_validation['status'] == 'REVIEW_NEEDED':
            st.warning(f"⚠️ **REVIEW NEEDED**: {security_validation['message']}")
        elif security_validation['status'] == 'VALID':
            st.success(f"✅ **VERIFIED**: {security_validation['message']}")
        else:
            st.info(f"ℹ️ **STATUS**: {security_validation['message']}")
    
    # Create two columns for uploads
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Upload Cancellation Packet ZIP")
        uploaded_file = st.file_uploader(
            "Upload a ZIP file containing cancellation packet files",
            type=['zip'],
            help="Upload a ZIP file containing PDF, DOCX, DOC, image, or text files. Screenshots of bucket files will be automatically processed for NCB fee data.",
            key="zip_uploader"
        )
    
    with col2:
        st.subheader("📸 Upload Bucket Screenshot")
        uploaded_screenshot = st.file_uploader(
            "Upload a bucket screenshot for NCB fee analysis",
            type=['png', 'jpg', 'jpeg', 'tiff'],
            help="Upload a screenshot of the PCMI NCB fee buckets to extract fee amounts.",
            key="screenshot_uploader"
        )
    
    # Process individual screenshot if uploaded
    screenshot_data = None
    if uploaded_screenshot is not None:
        with st.spinner("Processing bucket screenshot..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_screenshot.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_screenshot.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Process the screenshot
                screenshot_data = processor.screenshot_processor.extract_ncb_data(tmp_file_path)
                
                # Clean up temp file
                os.unlink(tmp_file_path)
                
                st.success("Screenshot processed successfully!")
                
                # Display screenshot results with visual indicators
                st.subheader("📊 Screenshot Analysis Results")
                
                # Visual checklist for screenshot
                st.markdown("### Bucket Screenshot Analysis")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    agent_amount = screenshot_data['agent_ncb_amount']
                    agent_color = "🟢" if agent_amount and agent_amount > 0 else "🔴" if agent_amount == 0 else "🟡"
                    st.markdown(f"{agent_color} Agent NCB: ${agent_amount:.2f}" if agent_amount is not None else f"{agent_color} Agent NCB: N/A")
                
                with col2:
                    dealer_amount = screenshot_data['dealer_ncb_amount']
                    dealer_color = "🟢" if dealer_amount and dealer_amount > 0 else "🔴" if dealer_amount == 0 else "🟡"
                    st.markdown(f"{dealer_color} Dealer NCB: ${dealer_amount:.2f}" if dealer_amount is not None else f"{dealer_color} Dealer NCB: N/A")
                
                with col3:
                    total_amount = screenshot_data['total_amount']
                    total_color = "🟢" if total_amount and total_amount > 0 else "🔴" if total_amount == 0 else "🟡"
                    st.markdown(f"{total_color} Total Amount: ${total_amount:.2f}" if total_amount is not None else f"{total_color} Total Amount: N/A")
                
                # Show extracted text
                with st.expander("View Extracted Text"):
                    st.text(screenshot_data['raw_text'])
                    
            except Exception as e:
                st.error(f"Error processing screenshot: {str(e)}")
    
    # Process ZIP file if uploaded
    if uploaded_file is not None:
        with st.spinner("Processing files..."):
            try:
                results = processor.process_zip(uploaded_file)
                
                if results:
                    # Create DataFrame
                    df = pd.DataFrame(results)
                    
                    # Display results
                    st.success(f"✅ Processed {len(results)} packet(s) from {len(processor.files_data)} file(s)")
                    
                    # Show extraction summary
                    st.subheader("📊 Data Extraction Summary")
                    total_vins = sum(len(f.get('vins', [])) for f in processor.files_data)
                    total_contracts = sum(len(f.get('contracts', [])) for f in processor.files_data)
                    total_reasons = sum(len(f.get('reasons', [])) for f in processor.files_data)
                    total_dates = sum(len(f.get('cancellation_dates', [])) for f in processor.files_data)
                    total_addresses = sum(len(f.get('refund_addresses', [])) for f in processor.files_data)
                    total_mileages = sum(len(f.get('mileages', [])) for f in processor.files_data)
                    total_customers = sum(len(f.get('customer_names', [])) for f in processor.files_data)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("VINs Found", total_vins)
                        st.metric("Contracts Found", total_contracts)
                    with col2:
                        st.metric("Reasons Found", total_reasons)
                        st.metric("Dates Found", total_dates)
                    with col3:
                        st.metric("Addresses Found", total_addresses)
                        st.metric("Mileages Found", total_mileages)
                    with col4:
                        st.metric("Customer Names", total_customers)
                        st.metric("Files Processed", len(processor.files_data))
                    
                    # Summary statistics - more prominent
                    st.subheader("📊 QC Analysis Summary")
                    
                    # Display IP security information
                    if ip_info and security_validation:
                        st.subheader("🔒 Security Verification")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("IP Address", client_ip)
                        with col2:
                            st.metric("Location", f"{ip_info.get('city', 'Unknown')}, {ip_info.get('country', 'Unknown')}")
                        with col3:
                            status_color = "🟢" if security_validation['status'] == 'VALID' else "🟡" if security_validation['status'] == 'REVIEW_NEEDED' else "🔴"
                            st.metric("Security Status", f"{status_color} {security_validation['status']}")
                        
                        if vpn_detection and vpn_detection['is_vpn']:
                            st.warning(f"⚠️ VPN/Proxy detected with {vpn_detection['confidence']}% confidence")
                            for reason in vpn_detection['reasons']:
                                st.write(f"• {reason}")
                        
                        st.divider()
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Total Packets", len(results))
                    
                    with col2:
                        pass_count = len([r for r in results if r['Vin Match on all forms'] == 'PASS'])
                        st.metric("VIN Matches (PASS)", pass_count)
                    
                    with col3:
                        fail_count = len([r for r in results if r['Vin Match on all forms'] == 'FAIL'])
                        st.metric("VIN Conflicts (FAIL)", fail_count)
                    
                    with col4:
                        screenshot_count = len([r for r in results if 'Present (hint detected)' in str(r.get('PCMI Screenshot (Of NCB fee buckets)', ''))])
                        st.metric("Screenshots Processed", screenshot_count)
                    
                    with col5:
                        info_count = len([r for r in results if r['Vin Match on all forms'] == 'INFO'])
                        st.metric("Needs Review (INFO)", info_count)
                    
                    # Visual QC Checklist Results
                    st.subheader("📋 QC Checklist Results")
                    
                    # Create visual checklist for each packet
                    for i, result in enumerate(results):
                        st.markdown(f"### Packet {i+1}: {result['Packet Key']}")
                        
                        # Show Contract Number and Customer Name prominently
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Contract Number:** {result.get('Contract Number', 'N/A')}")
                        with col2:
                            st.markdown(f"**Customer Name:** {result.get('Customer Name', 'N/A')}")
                        
                        st.markdown(f"**Files:** {result['Files']}")
                        
                        # Show source data if available
                        if '_source_data' in result:
                            with st.expander("📋 View Data Sources"):
                                source_data = result['_source_data']
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if source_data['vins']:
                                        st.write("**VINs Found:**")
                                        for vin_source in source_data['vins']:
                                            st.write(f"• {vin_source}")
                                    
                                    if source_data['contracts']:
                                        st.write("**Contracts Found:**")
                                        for contract_source in source_data['contracts']:
                                            st.write(f"• {contract_source}")
                                    
                                    if source_data['reasons']:
                                        st.write("**Reasons Found:**")
                                        for reason_source in source_data['reasons']:
                                            st.write(f"• {reason_source}")
                                
                                with col2:
                                    if source_data['cancellation_dates']:
                                        st.write("**Cancellation Dates Found:**")
                                        for date_source in source_data['cancellation_dates']:
                                            st.write(f"• {date_source}")
                                    
                                    if source_data['sale_dates']:
                                        st.write("**Sale Dates Found:**")
                                        for date_source in source_data['sale_dates']:
                                            st.write(f"• {date_source}")
                                    
                                    if source_data['refund_addresses']:
                                        st.write("**Refund Addresses Found:**")
                                        for addr_source in source_data['refund_addresses']:
                                            st.write(f"• {addr_source}")
                                    
                                    if source_data['mileages']:
                                        st.write("**Mileage Found:**")
                                        for mileage_source in source_data['mileages']:
                                            st.write(f"• {mileage_source}")
                        
                        # Create visual checklist with color coding
                        checklist_cols = st.columns(3)
                        
                        with checklist_cols[0]:
                            st.markdown("**Basic Information:**")
                            
                            # VIN Match
                            vin_status = result.get('Vin Match on all forms', 'INFO')
                            vin_color = "🟢" if vin_status == "PASS" else "🔴" if vin_status == "FAIL" else "🟡"
                            st.markdown(f"{vin_color} Vin Match on all forms: {vin_status}")
                            if result.get('VIN (canonical)'):
                                st.markdown(f"   └─ VIN: {result.get('VIN (canonical)')}")
                            
                            # Contract Match
                            contract_status = result.get('Contract Match on all forms and Google sheet', 'INFO')
                            contract_color = "🟢" if contract_status == "PASS" else "🔴" if contract_status == "FAIL" else "🟡"
                            st.markdown(f"{contract_color} Contract Match on all forms and Google sheet: {contract_status}")
                            if result.get('Contract (canonical)'):
                                st.markdown(f"   └─ Contract: {result.get('Contract (canonical)')}")
                            
                            # Reason Match
                            reason_status = result.get('Reason Match across all forms', 'INFO')
                            if reason_status == 'MANUAL_REVIEW_NEEDED':
                                reason_color = "🟡"
                                st.markdown(f"{reason_color} Reason Match across all forms: MANUAL REVIEW NEEDED")
                                st.markdown(f"   └─ Reason: {result.get('Reason (canonical)')}")
                                if result.get('Handwritten Files'):
                                    st.markdown(f"   └─ Handwritten Files: {', '.join(result['Handwritten Files'])}")
                            else:
                                reason_color = "🟢" if reason_status == "PASS" else "🔴" if reason_status == "FAIL" else "🟡"
                                st.markdown(f"{reason_color} Reason Match across all forms: {reason_status}")
                                if result.get('Reason (canonical)'):
                                    st.markdown(f"   └─ Reason: {result.get('Reason (canonical)')}")
                            
                            # Date Match
                            date_status = result.get('Cancellation date match across all forms. (Favor lender letter if applicable)', 'INFO')
                            date_color = "🟢" if date_status == "PASS" else "🔴" if date_status == "FAIL" else "🟡"
                            st.markdown(f"{date_color} Cancellation date match across all forms. (Favor lender letter if applicable): {date_status}")
                            if result.get('Cancellation Effective Date'):
                                st.markdown(f"   └─ Date: {result.get('Cancellation Effective Date')}")
                        
                        with checklist_cols[1]:
                            st.markdown("**Time & Fees:**")
                            
                            # 90 Day Check
                            days_status = result.get('Is the cancellation effective date past 90 days from contract sale date?', 'Unknown')
                            days_color = "🟢" if days_status == "Yes" else "🔴" if days_status == "No" else "🟡"
                            st.markdown(f"{days_color} 90+ Days: {days_status}")
                            if result.get('Days Difference'):
                                st.markdown(f"   └─ Days: {result.get('Days Difference')}")
                            if result.get('Sale Date'):
                                st.markdown(f"   └─ Sale Date: {result.get('Sale Date')}")
                            if result.get('Cancellation Effective Date'):
                                st.markdown(f"   └─ Cancellation Date: {result.get('Cancellation Effective Date')}")
                            
                            # Agent NCB (No Chargeback)
                            agent_ncb = result.get('Is there an Agent NCB Fee?', 'No')
                            agent_color = "🟢" if "Yes" in agent_ncb else "🔴" if "No" in agent_ncb else "🟡"
                            st.markdown(f"{agent_color} Agent NCB (No Chargeback): {agent_ncb}")
                            
                            # Dealer NCB (No Chargeback)
                            dealer_ncb = result.get('Is there a Dealer NCB Fee?', 'No')
                            # "No" means no chargeback (good), "Yes" means there is NCB protection (also good)
                            dealer_color = "🟢" if "No" in dealer_ncb or "Yes" in dealer_ncb else "🟡"
                            st.markdown(f"{dealer_color} Dealer NCB (No Chargeback): {dealer_ncb}")
                            
                            # Refund Address
                            refund_status = result.get('Is there a different address to send the refund? (only applicable if the request included a lender letter addressed to Ascent)', 'No')
                            refund_color = "🟢" if refund_status == "Yes" else "🔴" if refund_status == "No" else "🟡"
                            st.markdown(f"{refund_color} Is there a different address to send the refund? (only applicable if the request included a lender letter addressed to Ascent): {refund_status}")
                            if result.get('Alt Refund Address (if any)'):
                                st.markdown(f"   └─ Address: {result.get('Alt Refund Address (if any)')}")
                        
                        with checklist_cols[2]:
                            st.markdown("**Flags & Screenshots:**")
                            
                            # Signatures
                            sig_status = result.get('All necessary signatures collected?', 'Needs manual check')
                            sig_color = "🟢" if "Likely" in sig_status else "🔴" if "Needs manual check" in sig_status else "🟡"
                            st.markdown(f"{sig_color} Signatures: {sig_status}")
                            
                            # Autohouse
                            autohouse = result.get('Is this an Autohouse Contract?', 'No')
                            autohouse_color = "🟢" if autohouse == "Yes" else "🔴" if autohouse == "No" else "🟡"
                            st.markdown(f"{autohouse_color} Autohouse: {autohouse}")
                            
                            # Customer Direct
                            customer_direct = result.get('Is this a customer direct cancellation? (Dealer Out of Business or FF contract)', 'No')
                            customer_color = "🟢" if customer_direct == "Yes" else "🔴" if customer_direct == "No" else "🟡"
                            st.markdown(f"{customer_color} Customer Direct: {customer_direct}")
                            
                            # Diversicare
                            diversicare = result.get('Is this a Diversicare contract?', 'No')
                            diversicare_color = "🟢" if diversicare == "Yes" else "🔴" if diversicare == "No" else "🟡"
                            st.markdown(f"{diversicare_color} Diversicare: {diversicare}")
                            
                            # PCMI Screenshot
                            pcmi_status = result.get('PCMI Screenshot (Of NCB fee buckets)', 'Not found')
                            pcmi_color = "🟢" if "Present" in pcmi_status else "🔴" if "Not found" in pcmi_status else "🟡"
                            st.markdown(f"{pcmi_color} PCMI Screenshot: {pcmi_status}")
                            
                            # Show NCB amounts if available from screenshots
                            if result.get('Total NCB Amount'):
                                st.markdown(f"   └─ Total NCB: {result.get('Total NCB Amount')}")
                            
                            # Mileage
                            if result.get('Mileage values found'):
                                mileage_status = result.get('Mileage Match Status', 'INFO')
                                mileage_color = "🟢" if mileage_status == "PASS" else "🔴" if mileage_status == "FAIL" else "🟡"
                                st.markdown(f"{mileage_color} Mileage: {result.get('Mileage values found')}")
                                if mileage_status == "FAIL":
                                    st.markdown(f"   └─ Multiple different mileages found - needs review")
                                elif mileage_status == "PASS":
                                    st.markdown(f"   └─ All mileages match")
                            
                            # Refund Calculations
                            calc_status = result.get('Refund Calculation Status', 'INFO')
                            if calc_status != "INFO":
                                calc_color = "🟢" if calc_status == "PASS" else "🔴" if calc_status == "MANUAL_REVIEW_NEEDED" else "🟡"
                                st.markdown(f"{calc_color} Refund Calculations: {calc_status}")
                                if result.get('Refund Calculation Issues'):
                                    st.markdown(f"   └─ Issues: {result.get('Refund Calculation Issues')}")
                                
                                # Show calculation details
                                calc_details = result.get('Refund Calculation Details', {})
                                if calc_details:
                                    with st.expander("💰 Calculation Details"):
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            if calc_details.get('retail_purchase_price'):
                                                st.write(f"**Retail Price:** ${calc_details['retail_purchase_price']:.2f}")
                                            if calc_details.get('dealer_remitted_amount'):
                                                st.write(f"**Dealer Remitted:** ${calc_details['dealer_remitted_amount']:.2f}")
                                            if calc_details.get('cancel_fee'):
                                                st.write(f"**Cancel Fee:** ${calc_details['cancel_fee']:.2f}")
                                        with col2:
                                            if calc_details.get('ascent_refund'):
                                                st.write(f"**Ascent Refund:** ${calc_details['ascent_refund']:.2f}")
                                            if calc_details.get('dealer_refund'):
                                                st.write(f"**Dealer Refund:** ${calc_details['dealer_refund']:.2f}")
                                            if calc_details.get('net_customer_refund'):
                                                st.write(f"**Net Customer Refund:** ${calc_details['net_customer_refund']:.2f}")
                                            if calc_details.get('refund_percentage'):
                                                st.write(f"**Refund %:** {calc_details['refund_percentage']:.1f}%")
                        
                        # Show screenshot details if this packet has bucket screenshots
                        packet_files = result.get('Files', '').split(', ')
                        screenshot_files = [f for f in processor.files_data if f['filename'] in packet_files and (f.get('agent_ncb_amount') is not None or f.get('dealer_ncb_amount') is not None)]
                        
                        if screenshot_files:
                            st.markdown("**📸 Bucket Screenshot Details:**")
                            for file_data in screenshot_files:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Agent NCB", f"${file_data['agent_ncb_amount']:.2f}" if file_data['agent_ncb_amount'] else "N/A")
                                with col2:
                                    st.metric("Dealer NCB", f"${file_data['dealer_ncb_amount']:.2f}" if file_data['dealer_ncb_amount'] else "N/A")
                                with col3:
                                    st.metric("Total Amount", f"${file_data['total_ncb_amount']:.2f}" if file_data['total_ncb_amount'] else "N/A")
                                
                                with st.expander(f"View extracted text from {file_data['filename']}"):
                                    st.text(file_data.get('raw_text', 'No text extracted'))
                        
                        st.markdown("---")
                    
                    # Also show the data table for reference
                    st.subheader("📊 Data Table View")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download buttons
                    st.subheader("💾 Download Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📄 Download CSV",
                            data=csv,
                            file_name=f"qc_cancellations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        json_data = df.to_json(orient='records', indent=2)
                        st.download_button(
                            label="📄 Download JSON",
                            data=json_data,
                            file_name=f"qc_cancellations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    # Show screenshot processing details
                    screenshot_files = [f for f in processor.files_data if f.get('agent_ncb_amount') is not None or f.get('dealer_ncb_amount') is not None]
                    if screenshot_files:
                        st.subheader("📸 Screenshot Processing Details")
                        screenshot_data = []
                        for file_data in screenshot_files:
                            screenshot_data.append({
                                'File': file_data['filename'],
                                'Agent NCB Amount': f"${file_data['agent_ncb_amount']:.2f}" if file_data['agent_ncb_amount'] else 'N/A',
                                'Dealer NCB Amount': f"${file_data['dealer_ncb_amount']:.2f}" if file_data['dealer_ncb_amount'] else 'N/A',
                                'Total Amount': f"${file_data['total_ncb_amount']:.2f}" if file_data['total_ncb_amount'] else 'N/A'
                            })
                        
                        if screenshot_data:
                            st.dataframe(pd.DataFrame(screenshot_data), use_container_width=True)
                    
                    # Detailed analysis for each packet
                    st.subheader("🔍 Detailed Packet Analysis")
                    for i, result in enumerate(results):
                        with st.expander(f"Packet {i+1}: {result['Packet Key']} ({result['Files']})"):
                            # Create a more detailed view
                            detail_cols = st.columns(2)
                            
                            with detail_cols[0]:
                                st.write("**Basic Information:**")
                                st.write(f"• VIN: {result.get('VIN (canonical)', 'N/A')}")
                                st.write(f"• Contract: {result.get('Contract (canonical)', 'N/A')}")
                                st.write(f"• Reason: {result.get('Reason (canonical)', 'N/A')}")
                                st.write(f"• Cancellation Date: {result.get('Cancellation Effective Date', 'N/A')}")
                                st.write(f"• Sale Date: {result.get('Sale Date', 'N/A')}")
                            
                            with detail_cols[1]:
                                st.write("**QC Status:**")
                                st.write(f"• VIN Match: {result.get('VIN match on all forms', 'N/A')}")
                                st.write(f"• Contract Match: {result.get('Contract match on all forms and Google sheet', 'N/A')}")
                                st.write(f"• Reason Match: {result.get('Reason match across all forms', 'N/A')}")
                                st.write(f"• Date Match: {result.get('Cancellation date match across all forms (favor lender letter if applicable)', 'N/A')}")
                                st.write(f"• 90+ Days: {result.get('Is the cancellation effective date past 90 days from contract sale date?', 'N/A')}")
                            
                            st.write("**Fees & Flags:**")
                            st.write(f"• Agent NCB: {result.get('Is there an Agent NCB Fee?', 'N/A')}")
                            st.write(f"• Dealer NCB: {result.get('Is there a Dealer NCB Fee?', 'N/A')}")
                            st.write(f"• Refund Address: {result.get('Is there a different address to send the refund? (only if lender letter addressed to Ascent)', 'N/A')}")
                            st.write(f"• Signatures: {result.get('All necessary signatures collected?', 'N/A')}")
                            st.write(f"• Autohouse: {result.get('Is this an Autohouse Contract?', 'N/A')}")
                            st.write(f"• Customer Direct: {result.get('Is this a customer direct cancellation? (Dealer Out of Business or FF contract)', 'N/A')}")
                            st.write(f"• Diversicare: {result.get('Is this a Diversicare contract?', 'N/A')}")
                            st.write(f"• PCMI Screenshot: {result.get('PCMI Screenshot (Of NCB fee buckets)', 'N/A')}")
                            
                            if result.get('Mileage values found'):
                                mileage_status = result.get('Mileage Match Status', 'INFO')
                                mileage_color = "🟢" if mileage_status == "PASS" else "🔴" if mileage_status == "FAIL" else "🟡"
                                st.write(f"• Mileage: {mileage_color} {result.get('Mileage values found', 'N/A')}")
                                if mileage_status == "FAIL":
                                    st.write(f"  └─ Multiple different mileages found - needs review")
                                elif mileage_status == "PASS":
                                    st.write(f"  └─ All mileages match")
                    
                    # File thumbnails are displayed in process_zip method
                
                else:
                    st.warning("No valid files found in the ZIP archive.")
                    
            except Exception as e:
                st.error(f"Error processing files: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()
