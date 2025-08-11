#!/usr/bin/env python3
"""
Email Summary Script with Ollama Integration
Reads emails from a specific folder and provides AI-powered summaries of important content.
"""

import imaplib
import email
from email.header import decode_header
import requests
import json
from datetime import datetime, timedelta
import re
import os
from typing import List, Dict, Any
import argparse

class EmailSummarizer:
    def __init__(self, email_server: str, email_user: str, email_password: str, ollama_url: str = "https://ollama.valiantlynx.com"):
        self.email_server = email_server
        self.email_user = email_user
        self.email_password = email_password
        self.ollama_url = ollama_url
        self.imap = None
    
    def connect_to_email(self):
        """Connect to email server via IMAP"""
        try:
            self.imap = imaplib.IMAP4_SSL(self.email_server)
            self.imap.login(self.email_user, self.email_password)
            print(f"✓ Connected to {self.email_server}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to email: {e}")
            return False
    
    def get_folders(self):
        """List available email folders"""
        try:
            status, folders = self.imap.list()
            if status == 'OK':
                folder_list = []
                for folder in folders:
                    folder_name = folder.decode().split('"/"')[-1].strip(' "')
                    folder_list.append(folder_name)
                return folder_list
        except Exception as e:
            print(f"Error getting folders: {e}")
            return []
    
    def decode_email_subject(self, subject):
        """Decode email subject from various encodings"""
        if subject is None:
            return "No Subject"
        
        decoded_fragments = decode_header(subject)
        decoded_subject = ""
        
        for fragment, encoding in decoded_fragments:
            if isinstance(fragment, bytes):
                if encoding:
                    decoded_subject += fragment.decode(encoding)
                else:
                    decoded_subject += fragment.decode('utf-8', errors='ignore')
            else:
                decoded_subject += fragment
        
        return decoded_subject
    
    def extract_email_content(self, msg):
        """Extract text content from email message"""
        content = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    try:
                        body = part.get_payload(decode=True)
                        if body:
                            content += body.decode('utf-8', errors='ignore')
                    except Exception as e:
                        print(f"Error decoding part: {e}")
                        continue
        else:
            try:
                body = msg.get_payload(decode=True)
                if body:
                    content = body.decode('utf-8', errors='ignore')
            except Exception as e:
                print(f"Error decoding message: {e}")
        
        return content.strip()
    
    def clean_email_content(self, content: str) -> str:
        """Clean email content by removing unnecessary parts"""
        # Remove quoted text (lines starting with >)
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip quoted text, signatures, and common email footers
            if (line.startswith('>') or 
                line.startswith('On ') and 'wrote:' in line or
                line == '--' or
                'unsubscribe' in line.lower() or
                'confidential' in line.lower()):
                continue
            cleaned_lines.append(line)
        
        # Join and remove excessive whitespace
        cleaned_content = '\n'.join(cleaned_lines)
        cleaned_content = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_content)
        
        return cleaned_content.strip()
    
    def read_emails_from_folder(self, folder_name: str, days_back: int = 7, max_emails: int = 50):
        """Read emails from specified folder"""
        try:
            # Select the folder
            status, messages = self.imap.select(folder_name)
            if status != 'OK':
                print(f"✗ Failed to select folder: {folder_name}")
                return []
            
            # Calculate date range
            since_date = (datetime.now() - timedelta(days=days_back)).strftime("%d-%b-%Y")
            
            # Search for emails
            search_criteria = f'(SINCE "{since_date}")'
            status, message_ids = self.imap.search(None, search_criteria)
            
            if status != 'OK':
                print(f"✗ Failed to search emails")
                return []
            
            email_ids = message_ids[0].split()
            email_ids = email_ids[-max_emails:]  # Get most recent emails
            
            emails = []
            
            print(f"📧 Found {len(email_ids)} emails in '{folder_name}' from last {days_back} days")
            
            for email_id in email_ids:
                try:
                    status, msg_data = self.imap.fetch(email_id, '(RFC822)')
                    if status != 'OK':
                        continue
                    
                    msg = email.message_from_bytes(msg_data[0][1])
                    
                    # Extract email details
                    subject = self.decode_email_subject(msg.get('Subject'))
                    sender = msg.get('From')
                    date = msg.get('Date')
                    content = self.extract_email_content(msg)
                    content = self.clean_email_content(content)
                    
                    # Skip if content is too short
                    if len(content) < 50:
                        continue
                    
                    emails.append({
                        'id': email_id.decode(),
                        'subject': subject,
                        'sender': sender,
                        'date': date,
                        'content': content[:2000]  # Limit content length for processing
                    })
                    
                except Exception as e:
                    print(f"Error processing email {email_id}: {e}")
                    continue
            
            return emails
            
        except Exception as e:
            print(f"✗ Error reading emails: {e}")
            return []
    
    def call_ollama(self, prompt: str, model: str = "gpt-oss") -> str:
        """Call Ollama API for text generation"""
        try:
            data = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(f"{self.ollama_url}/api/generate", 
                                   json=data, 
                                   timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                return f"Error: HTTP {response.status_code}"
                
        except Exception as e:
            return f"Error calling Ollama: {e}"
    
    def categorize_and_summarize_emails(self, emails: List[Dict[str, Any]]) -> str:
        """Use Ollama to categorize and summarize emails"""
        if not emails:
            return "No emails found to summarize."
        
        # Prepare email data for analysis
        email_summaries = []
        
        for email in emails:
            email_text = f"Subject: {email['subject']}\nFrom: {email['sender']}\nContent: {email['content'][:800]}"
            
            # Get importance and category from Ollama
            analysis_prompt = f"""
Analyze this email and provide:
1. Importance level (Critical/High/Medium/Low)
2. Category (Work/Personal/Promotional/News/Other)
3. Key points (2-3 bullet points)

Email:
{email_text}

Response format:
IMPORTANCE: [level]
CATEGORY: [category]
KEY POINTS:
- [point 1]
- [point 2]
- [point 3]
"""
            
            analysis = self.call_ollama(analysis_prompt)
            
            email_summaries.append({
                'subject': email['subject'],
                'sender': email['sender'],
                'date': email['date'],
                'analysis': analysis
            })
        
        # Generate overall summary
        overall_prompt = f"""
Based on the following email analyses, create a comprehensive summary report:

{chr(10).join([f"Email: {es['subject']}\nAnalysis: {es['analysis']}\n---" for es in email_summaries[:10]])}

Please provide:
1. EXECUTIVE SUMMARY (2-3 sentences)
2. CRITICAL ITEMS (anything marked as Critical/High importance)
3. CATEGORIES BREAKDOWN (count by category)
4. ACTION ITEMS (things that likely need your attention)
5. KEY HIGHLIGHTS (important information to know)

Keep it concise but comprehensive.
"""
        
        return self.call_ollama(overall_prompt)
    
    def disconnect(self):
        """Close IMAP connection"""
        if self.imap:
            self.imap.close()
            self.imap.logout()

def main():
    parser = argparse.ArgumentParser(description='Email Summary Script with Ollama')
    parser.add_argument('--server', required=True, help='IMAP server (e.g., imap.gmail.com)')
    parser.add_argument('--email', required=True, help='Your email address')
    parser.add_argument('--password', help='Your email password (or use EMAIL_PASSWORD env var)') # from for example https://myaccount.google.com/apppasswords for google
    parser.add_argument('--folder', default='INBOX', help='Email folder to read (default: INBOX)')
    parser.add_argument('--days', type=int, default=7, help='Number of days back to read (default: 7)')
    parser.add_argument('--max-emails', type=int, default=50, help='Maximum number of emails to process (default: 50)')
    parser.add_argument('--ollama-url', default='https://ollama.valiantlynx.com', help='Ollama server URL')
    parser.add_argument('--model', default='gpt-oss', help='Ollama model to use')
    parser.add_argument('--list-folders', action='store_true', help='List available folders and exit')
    
    args = parser.parse_args()
    
    # Get password from environment variable if not provided
    password = args.password or os.getenv('EMAIL_PASSWORD')
    if not password:
        print("Error: Password required. Use --password or set EMAIL_PASSWORD environment variable")
        return
    
    # Create summarizer
    summarizer = EmailSummarizer(args.server, args.email, password, args.ollama_url)
    
    if not summarizer.connect_to_email():
        return
    
    try:
        # List folders if requested
        if args.list_folders:
            folders = summarizer.get_folders()
            print("\n📁 Available folders:")
            for folder in folders:
                print(f"  - {folder}")
            return
        
        # Read and summarize emails
        print(f"\n📖 Reading emails from folder: {args.folder}")
        emails = summarizer.read_emails_from_folder(args.folder, args.days, args.max_emails)
        
        if not emails:
            print("No emails found to process.")
            return
        
        print(f"\n🤖 Analyzing {len(emails)} emails with Ollama...")
        summary = summarizer.categorize_and_summarize_emails(emails)
        
        print("\n" + "="*60)
        print("📊 EMAIL SUMMARY REPORT")
        print("="*60)
        print(summary)
        print("="*60)
        
    finally:
        summarizer.disconnect()

if __name__ == "__main__":
    main()