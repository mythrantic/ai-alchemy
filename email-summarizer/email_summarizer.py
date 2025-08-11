#!/usr/bin/env python3
"""
Email Summary Script with Multiple LLM Support
Reads emails from a specific folder and provides AI-powered summaries using Groq or Ollama.
"""

import imaplib
import email
from email.header import decode_header
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from datetime import datetime, timedelta
import re
import os
from typing import List, Dict, Any
import argparse
import mistune
import markdownify
from html import unescape
from langchain_core.messages import HumanMessage
from langchain_core.runnables import ConfigurableField
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from pydantic import BaseModel, SecretStr
from dotenv import load_dotenv

load_dotenv()

def html_to_text(html_content: str) -> str:
    """Convert HTML to clean plain text using markdownify then to text"""
    if not html_content:
        return ""
    
    try:
        # First convert HTML to Markdown using markdownify
        markdown_content = markdownify.markdownify(
            html_content, 
            heading_style="ATX",  # Use # style headers
            strip=['script', 'style', 'meta', 'link', 'head'],  # Remove unwanted tags
            convert=['p', 'div', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li']
        )
        
        # Clean up the markdown and convert to plain text
        # Remove markdown formatting for plain text
        text = markdown_content
        # Remove markdown headers
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        # Remove markdown links but keep text
        text = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', text)
        # Remove markdown emphasis
        text = re.sub(r'\*\*([^\*]*)\*\*', r'\1', text)
        text = re.sub(r'\*([^\*]*)\*', r'\1', text)
        # Clean up whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
        
    except Exception:
        # Fallback: simple regex-based HTML stripping
        text = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)
        text = unescape(text)
        return ' '.join(text.split())

GROQ_API_KEY = SecretStr(os.getenv("GROQ_API_KEY", ""))
OLLAMA_API_URL = SecretStr(os.getenv("OLLAMA_API_URL", "http://localhost:11434"))
MODEL_NAME = SecretStr(os.getenv("MODEL_NAME", "llama3.1"))
DISCORD_WEBHOOK_URL = SecretStr(os.getenv("DISCORD_WEBHOOK_URL", ""))
SMTP_SERVER = SecretStr(os.getenv("SMTP_SERVER", ""))
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = SecretStr(os.getenv("SMTP_USER", ""))
SMTP_PASSWORD = SecretStr(os.getenv("SMTP_PASSWORD", ""))

# Configuration class to help switch between Ollama and Groq
class AgentConfig(BaseModel):
    model: str = "groq"  # or "ollama"
    ollama_api_url: str = OLLAMA_API_URL.get_secret_value()
    model_name: str = MODEL_NAME.get_secret_value()
    groq_api_key: str = GROQ_API_KEY.get_secret_value()
    supported_models: list[str] = ["groq", "ollama"]
    
    # Notification settings
    discord_webhook_url: str = DISCORD_WEBHOOK_URL.get_secret_value()
    smtp_server: str = SMTP_SERVER.get_secret_value()
    smtp_port: int = SMTP_PORT
    smtp_user: str = SMTP_USER.get_secret_value()
    smtp_password: str = SMTP_PASSWORD.get_secret_value()

    class Config:
        arbitrary_types_allowed = True

def get_llm(config: AgentConfig):
    """Get the appropriate LLM based on configuration"""
    if config.model == "groq":
        llm = ChatGroq(
            model=config.model_name,
            api_key=config.groq_api_key,
            temperature=0.0,
            streaming=False  # For email summarization we don't need streaming
        )
    elif config.model == "ollama":
        llm = ChatOllama(
            model=config.model_name,
            base_url=config.ollama_api_url,
            temperature=0.0,
        )
    else:
        raise ValueError(f"Unsupported model: {config.model}. Supported models are: {config.supported_models}")
    
    return llm.configurable_fields(
        callbacks=ConfigurableField(
            id="callbacks",
            name="Callbacks",
            description="List of callbacks to use for streaming",
        )
    )

class EmailSummarizer:
    def __init__(self, email_server: str, email_user: str, email_password: str, agent_config: AgentConfig):
        self.email_server = email_server
        self.email_user = email_user
        self.email_password = email_password
        self.config = agent_config
        self.llm = get_llm(agent_config)
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
        """Extract text content from email message with HTML fallback"""
        content = ""
        html_content = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                
                if "attachment" not in content_disposition:
                    try:
                        body = part.get_payload(decode=True)
                        if body:
                            decoded_body = body.decode('utf-8', errors='ignore')
                            
                            if content_type == "text/plain":
                                content += decoded_body
                            elif content_type == "text/html":
                                html_content += decoded_body
                    except Exception as e:
                        print(f"Error decoding part: {e}")
                        continue
        else:
            try:
                body = msg.get_payload(decode=True)
                if body:
                    decoded_body = body.decode('utf-8', errors='ignore')
                    content_type = msg.get_content_type()
                    
                    if content_type == "text/plain":
                        content = decoded_body
                    elif content_type == "text/html":
                        html_content = decoded_body
            except Exception as e:
                print(f"Error decoding message: {e}")
        
        # If no plain text content found, convert HTML to text
        if not content.strip() and html_content.strip():
            print(f"  📄 No plain text found, converting HTML to text...")
            content = html_to_text(html_content)
        
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
                    
                    # Debug info for content extraction
                    content_length = len(content)
                    if content_length < 50:
                        print(f"  ⚠️  Skipping '{subject[:50]}...' - content too short ({content_length} chars)")
                        continue
                    else:
                        print(f"  ✅ Processing '{subject[:50]}...' - {content_length} chars")
                    
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
    
    def call_llm(self, prompt: str) -> str:
        """Call the configured LLM for text generation"""
        try:
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except Exception as e:
            return f"Error calling {self.config.model}: {e}"
    
    def call_ollama_direct(self, prompt: str, model: str = None) -> str:
        """Direct call to Ollama API (legacy method, kept for compatibility)"""
        try:
            model = model or self.config.model_name
            data = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(f"{self.config.ollama_api_url}/api/generate", 
                                   json=data, 
                                   timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                return f"Error: HTTP {response.status_code}"
                
        except Exception as e:
            return f"Error calling Ollama directly: {e}"
    
    def categorize_and_summarize_emails(self, emails: List[Dict[str, Any]], use_direct_api: bool = False) -> str:
        """Use LLM to categorize and summarize emails"""
        if not emails:
            return "No emails found to summarize."
        
        # Prepare email data for analysis
        email_summaries = []
        
        print(f"🤖 Using {self.config.model.upper()} model: {self.config.model_name}")
        
        for i, email in enumerate(emails):
            email_text = f"Subject: {email['subject']}\nFrom: {email['sender']}\nContent: {email['content'][:800]}"
            
            # Get importance and category from LLM
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
            
            # Choose between LangChain LLM or direct API call
            if use_direct_api and self.config.model == "ollama":
                analysis = self.call_ollama_direct(analysis_prompt)
            else:
                analysis = self.call_llm(analysis_prompt)
            
            print(f"  Analyzed email {i+1}/{len(emails)}: {email['subject'][:50]}...")
            
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
        
        # Generate final summary
        if use_direct_api and self.config.model == "ollama":
            return self.call_ollama_direct(overall_prompt)
        else:
            return self.call_llm(overall_prompt)
    
    def send_discord_notification(self, summary: str, email_count: int) -> bool:
        """Send summary to Discord via webhook"""
        if not self.config.discord_webhook_url:
            print("❌ Discord webhook URL not configured")
            return False
        
        try:
            # Create Discord embed
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Split summary if it's too long for Discord (2000 char limit)
            if len(summary) > 1900:
                summary_parts = [summary[i:i+1900] for i in range(0, len(summary), 1900)]
                summary = summary_parts[0] + "\n\n... (truncated, see email for full report)"
            
            embed = {
                "title": f"📊 Email Summary Report - {email_count} emails analyzed",
                "description": summary,
                "color": 0x00ff00,  # Green color
                "footer": {
                    "text": f"Generated at {timestamp}"
                },
                "fields": [
                    {
                        "name": "📧 Email Count",
                        "value": str(email_count),
                        "inline": True
                    },
                    {
                        "name": "🤖 AI Model",
                        "value": f"{self.config.model.upper()} - {self.config.model_name}",
                        "inline": True
                    }
                ]
            }
            
            payload = {
                "embeds": [embed],
                "username": "Email Summary Bot"
            }
            
            response = requests.post(
                self.config.discord_webhook_url,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 204:
                print("✅ Discord notification sent successfully!")
                return True
            else:
                print(f"❌ Discord notification failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error sending Discord notification: {e}")
            return False
    
    def send_email_notification(self, summary: str, email_count: int, recipient_email: str) -> bool:
        """Send summary via email using mistune for Markdown to HTML conversion"""
        if not all([self.config.smtp_server, self.config.smtp_user, self.config.smtp_password]):
            print("❌ SMTP configuration incomplete")
            return False
        
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.config.smtp_user
            msg['To'] = recipient_email
            msg['Subject'] = f"📊 Email Summary Report - {email_count} emails analyzed"
            
            # Convert markdown summary to HTML using mistune
            # Using mistune.html() which has all features enabled by default
            html_summary = mistune.html(summary)
            
            # Create HTML email body with Tailwind CSS classes
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            html_body = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <script src="https://cdn.tailwindcss.com"></script>
                <script>
                tailwind.config = {{
                    theme: {{
                        extend: {{
                            fontFamily: {{
                                'sans': ['ui-sans-serif', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'Noto Sans', 'sans-serif'],
                            }}
                        }}
                    }}
                }}
                </script>
            </head>
            <body class="bg-gray-50 font-sans">
                <div class="max-w-4xl mx-auto p-4">
                    <!-- Header -->
                    <div class="bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-t-lg p-6 text-center">
                        <h1 class="text-2xl font-bold m-0">📊 Email Summary Report</h1>
                    </div>
                    
                    <!-- Meta Info -->
                    <div class="bg-blue-50 border-l-4 border-blue-500 p-4 mb-0">
                        <div class="space-y-2 text-sm">
                            <p><span class="font-semibold">📅 Generated:</span> {timestamp}</p>
                            <p><span class="font-semibold">📧 Emails Analyzed:</span> {email_count}</p>
                            <p><span class="font-semibold">🤖 AI Model:</span> {self.config.model.upper()} - {self.config.model_name}</p>
                        </div>
                    </div>
                    
                    <!-- Content -->
                    <div class="bg-white border border-gray-200 rounded-b-lg p-6">
                        <div class="bg-gray-50 border-l-4 border-green-500 rounded-lg p-5 my-4">
                            <div class="prose prose-gray max-w-none">
                                {html_summary}
                            </div>
                        </div>
                    </div>
                    
                    <!-- Footer -->
                    <div class="text-center p-4 bg-gray-50 text-gray-600 text-sm rounded-b-lg">
                        <p class="italic">This report was automatically generated by your Email Summary Bot.</p>
                        <p class="text-xs mt-1">Powered by {self.config.model.upper()} AI</p>
                    </div>
                </div>
                
                <style>
                    /* Custom styles for email content */
                    .prose h1 {{ @apply text-2xl font-bold text-gray-800 mt-6 mb-4; }}
                    .prose h2 {{ @apply text-xl font-semibold text-gray-700 mt-5 mb-3; }}
                    .prose h3 {{ @apply text-lg font-medium text-gray-600 mt-4 mb-2; }}
                    .prose p {{ @apply mb-3 text-gray-700 leading-relaxed; }}
                    .prose ul {{ @apply list-disc list-inside mb-3 space-y-1; }}
                    .prose ol {{ @apply list-decimal list-inside mb-3 space-y-1; }}
                    .prose li {{ @apply text-gray-700; }}
                    .prose blockquote {{ @apply border-l-4 border-blue-500 pl-4 italic text-gray-600 my-4; }}
                    .prose code {{ @apply bg-gray-100 text-red-600 px-1 py-0.5 rounded text-sm; }}
                    .prose pre {{ @apply bg-gray-100 p-4 rounded-lg overflow-x-auto; }}
                    .prose table {{ @apply w-full border-collapse border border-gray-300 my-4; }}
                    .prose th {{ @apply bg-gray-100 border border-gray-300 p-3 text-left font-semibold; }}
                    .prose td {{ @apply border border-gray-300 p-3; }}
                    .prose hr {{ @apply border-0 h-px bg-gradient-to-r from-blue-500 to-purple-600 my-6; }}
                    .prose del {{ @apply line-through text-gray-500; }}
                    .prose a {{ @apply text-blue-600 hover:text-blue-800 underline; }}
                    .prose strong {{ @apply font-semibold text-gray-800; }}
                    .prose em {{ @apply italic; }}
                    
                    /* Footnote styles */
                    .footnote {{ @apply text-sm text-gray-500; }}
                    .footnote-ref {{ @apply text-blue-600 no-underline hover:underline; }}
                </style>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_user, self.config.smtp_password)
                server.send_message(msg)
            
            print("✅ Email notification sent successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error sending email notification: {e}")
            return False
    
    def send_slack_notification(self, summary: str, email_count: int, slack_webhook_url: str) -> bool:
        """Send summary to Slack via webhook"""
        if not slack_webhook_url:
            print("❌ Slack webhook URL not provided")
            return False
        
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create Slack message
            payload = {
                "text": f"📊 Email Summary Report - {email_count} emails analyzed",
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": "📊 Email Summary Report"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Emails Analyzed:* {email_count}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*AI Model:* {self.config.model.upper()} - {self.config.model_name}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Generated:* {timestamp}"
                            }
                        ]
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"```{summary[:2900]}```"  # Slack has message limits
                        }
                    }
                ]
            }
            
            response = requests.post(
                slack_webhook_url,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                print("✅ Slack notification sent successfully!")
                return True
            else:
                print(f"❌ Slack notification failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error sending Slack notification: {e}")
            return False
    
    def disconnect(self):
        """Close IMAP connection"""
        if self.imap:
            try:
                self.imap.close()
                self.imap.logout()
            except:
                pass  # Ignore errors when disconnecting

def main():
    parser = argparse.ArgumentParser(description='Email Summary Script with Multiple LLM Support')
    parser.add_argument('--server', required=True, help='IMAP server (e.g., imap.gmail.com)')
    parser.add_argument('--email', required=True, help='Your email address')
    parser.add_argument('--password', help='Your email password (or use EMAIL_PASSWORD env var)')
    parser.add_argument('--folder', default='INBOX', help='Email folder to read (default: INBOX)')
    parser.add_argument('--days', type=int, default=7, help='Number of days back to read (default: 7)')
    parser.add_argument('--max-emails', type=int, default=50, help='Maximum number of emails to process (default: 50)')
    
    # LLM Configuration
    parser.add_argument('--model-type', choices=['groq', 'ollama'], default='ollama', 
                       help='LLM provider to use (default: ollama)')
    parser.add_argument('--model-name', default='llama3.1', 
                       help='Model name to use (default: llama3.1)')
    parser.add_argument('--ollama-url', default='http://localhost:11434', 
                       help='Ollama server URL (default: http://localhost:11434)')
    parser.add_argument('--groq-api-key', 
                       help='Groq API key (or use GROQ_API_KEY env var)')
    parser.add_argument('--use-direct-api', action='store_true',
                       help='Use direct API calls instead of LangChain (Ollama only)')
    
    # Notification options
    parser.add_argument('--send-discord', action='store_true',
                       help='Send summary to Discord (requires DISCORD_WEBHOOK_URL)')
    parser.add_argument('--send-email', 
                       help='Send summary to this email address')
    parser.add_argument('--send-slack',
                       help='Send summary to Slack (provide webhook URL)')
    parser.add_argument('--discord-webhook',
                       help='Discord webhook URL (or use DISCORD_WEBHOOK_URL env var)')
    parser.add_argument('--smtp-server',
                       help='SMTP server for email notifications (or use SMTP_SERVER env var)')
    parser.add_argument('--smtp-user',
                       help='SMTP username (or use SMTP_USER env var)')
    parser.add_argument('--smtp-password',
                       help='SMTP password (or use SMTP_PASSWORD env var)')
    
    parser.add_argument('--list-folders', action='store_true', help='List available folders and exit')
    
    args = parser.parse_args()
    
    # Get password from environment variable if not provided
    password = args.password or os.getenv('EMAIL_PASSWORD')
    if not password:
        print("Error: Password required. Use --password or set EMAIL_PASSWORD environment variable")
        print("For Gmail, create an App Password: https://myaccount.google.com/apppasswords")
        return
    
    # Set up agent configuration
    agent_config = AgentConfig(
        model=args.model_type,
        ollama_api_url=args.ollama_url,
        model_name=args.model_name,
        groq_api_key=args.groq_api_key or os.getenv('GROQ_API_KEY', ''),
        discord_webhook_url=args.discord_webhook or os.getenv('DISCORD_WEBHOOK_URL', ''),
        smtp_server=args.smtp_server or os.getenv('SMTP_SERVER', ''),
        smtp_port=int(os.getenv('SMTP_PORT', '587')),
        smtp_user=args.smtp_user or os.getenv('SMTP_USER', ''),
        smtp_password=args.smtp_password or os.getenv('SMTP_PASSWORD', '')
    )
    
    # Validate configuration
    if agent_config.model == 'groq' and not agent_config.groq_api_key:
        print("Error: Groq API key required for Groq model. Use --groq-api-key or set GROQ_API_KEY env var")
        return
    
    # Create summarizer
    try:
        summarizer = EmailSummarizer(args.server, args.email, password, agent_config)
    except Exception as e:
        print(f"Error initializing summarizer: {e}")
        return
    
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
        
        print(f"\n🤖 Analyzing {len(emails)} emails with {agent_config.model.upper()}...")
        summary = summarizer.categorize_and_summarize_emails(emails, args.use_direct_api)
        
        print("\n" + "="*60)
        print("📊 EMAIL SUMMARY REPORT")
        print("="*60)
        print(summary)
        print("="*60)
        
        # Send notifications if requested
        notification_sent = False
        
        if args.send_discord:
            print("\n📱 Sending Discord notification...")
            if summarizer.send_discord_notification(summary, len(emails)):
                notification_sent = True
        
        if args.send_email:
            print(f"\n📧 Sending email notification to {args.send_email}...")
            if summarizer.send_email_notification(summary, len(emails), args.send_email):
                notification_sent = True
        
        if args.send_slack:
            print("\n💬 Sending Slack notification...")
            if summarizer.send_slack_notification(summary, len(emails), args.send_slack):
                notification_sent = True
        
        if not notification_sent and any([args.send_discord, args.send_email, args.send_slack]):
            print("\n⚠️  No notifications were sent successfully. Check your configuration.")
        elif notification_sent:
            print(f"\n🎉 Summary analysis complete! Notifications sent successfully.")
        
    finally:
        summarizer.disconnect()

if __name__ == "__main__":
    main()