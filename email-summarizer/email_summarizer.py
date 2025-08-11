#!/usr/bin/env python3
"""
Production-Ready Email Summary Script with Multiple LLM Support
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
import logging
import sys
from typing import List, Dict, Any, Optional, Union
import argparse
import mistune
import markdownify
from html import unescape
from dataclasses import dataclass
from pathlib import Path
import json
import time
from contextlib import contextmanager

from langchain_core.messages import HumanMessage
from langchain_core.runnables import ConfigurableField
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from pydantic import BaseModel, SecretStr, Field, field_validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration"""
    logger = logging.getLogger("email_summarizer")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

@dataclass
class EmailData:
    """Data class for email information"""
    id: str
    subject: str
    sender: str
    date: str
    content: str
    importance: Optional[str] = None
    category: Optional[str] = None
    analysis: Optional[str] = None

class EmailProcessingError(Exception):
    """Custom exception for email processing errors"""
    pass

class NotificationError(Exception):
    """Custom exception for notification errors"""
    pass

class LLMError(Exception):
    """Custom exception for LLM errors"""
    pass

class AgentConfig(BaseModel):
    """Configuration class with validation"""
    model: str = Field(default="groq", pattern="^(groq|ollama)$")
    ollama_api_url: str = Field(default="http://localhost:11434")
    model_name: str = Field(default="llama3.1")
    groq_api_key: str = Field(default="")
    supported_models: List[str] = Field(default=["groq", "ollama"])
    
    # Notification settings
    discord_webhook_url: str = Field(default="")
    smtp_server: str = Field(default="")
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_user: str = Field(default="")
    smtp_password: str = Field(default="")
    
    # Processing settings
    max_retries: int = Field(default=3, ge=1, le=10)
    request_timeout: int = Field(default=60, ge=10, le=300)
    rate_limit_delay: float = Field(default=1.0, ge=0.1, le=10.0)
    
    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        if v not in ["groq", "ollama"]:
            raise ValueError("Model must be 'groq' or 'ollama'")
        return v
    
    @classmethod
    def from_env(cls) -> 'AgentConfig':
        """Create configuration from environment variables"""
        return cls(
            model=os.getenv("MODEL_TYPE", "groq"),
            ollama_api_url=os.getenv("OLLAMA_API_URL", "http://localhost:11434"),
            model_name=os.getenv("MODEL_NAME", "llama3.1"),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL", ""),
            smtp_server=os.getenv("SMTP_SERVER", ""),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_user=os.getenv("SMTP_USER", ""),
            smtp_password=os.getenv("SMTP_PASSWORD", ""),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "60")),
            rate_limit_delay=float(os.getenv("RATE_LIMIT_DELAY", "1.0"))
        )
    
    class Config:
        arbitrary_types_allowed = True

class HTMLToTextConverter:
    """Utility class for HTML to text conversion"""
    
    @staticmethod
    def convert(html_content: str) -> str:
        """Convert HTML to clean plain text using markdownify"""
        if not html_content:
            return ""
        
        try:
            # First convert HTML to Markdown using markdownify
            # Note: Cannot use both 'strip' and 'convert' parameters together
            markdown_content = markdownify.markdownify(
                html_content, 
                heading_style="ATX",
                strip=['script', 'style', 'meta', 'link', 'head']
                # Removed 'convert' parameter to avoid conflict
            )
            
            # Clean up the markdown and convert to plain text
            text = markdown_content
            text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
            text = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', text)
            text = re.sub(r'\*\*([^\*]*)\*\*', r'\1', text)
            text = re.sub(r'\*([^\*]*)\*', r'\1', text)
            text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            
            return text.strip()
            
        except Exception as e:
            logger.warning(f"Markdownify conversion failed: {e}, falling back to regex")
            # Fallback: simple regex-based HTML stripping
            text = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', '', text)
            text = unescape(text)
            return ' '.join(text.split())

class LLMProvider:
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM based on configuration"""
        try:
            if self.config.model == "groq":
                if not self.config.groq_api_key:
                    raise LLMError("Groq API key is required but not provided")
                
                self.llm = ChatGroq(
                    model=self.config.model_name,
                    api_key=self.config.groq_api_key,
                    temperature=0.0,
                    streaming=False,
                    timeout=self.config.request_timeout
                )
            elif self.config.model == "ollama":
                self.llm = ChatOllama(
                    model=self.config.model_name,
                    base_url=self.config.ollama_api_url,
                    temperature=0.0,
                    timeout=self.config.request_timeout
                )
            else:
                raise LLMError(f"Unsupported model: {self.config.model}")
            
            logger.info(f"Initialized {self.config.model.upper()} LLM with model {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise LLMError(f"LLM initialization failed: {e}")
    
    def generate(self, prompt: str) -> str:
        """Generate text using the configured LLM with retry logic"""
        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"LLM generation attempt {attempt + 1}/{self.config.max_retries}")
                
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                
                if hasattr(response, 'content'):
                    result = response.content
                else:
                    result = str(response)
                
                if not result or not result.strip():
                    raise LLMError("Empty response from LLM")
                
                logger.debug(f"LLM generation successful, response length: {len(result)}")
                return result.strip()
                
            except Exception as e:
                error_msg = f"LLM generation attempt {attempt + 1} failed: {str(e)}"
                logger.warning(error_msg)
                
                if attempt == self.config.max_retries - 1:
                    raise LLMError(f"LLM generation failed after {self.config.max_retries} attempts: {str(e)}")
                
                # Add delay between retries with exponential backoff
                delay = self.config.rate_limit_delay * (2 ** attempt)
                logger.debug(f"Waiting {delay}s before retry...")
                time.sleep(delay)
        
        raise LLMError("LLM generation failed after all retry attempts")

class EmailClient:
    """IMAP email client with connection management"""
    
    def __init__(self, server: str, user: str, password: str):
        self.server = server
        self.user = user
        self.password = password
        self.imap = None
        
    @contextmanager
    def connection(self):
        """Context manager for IMAP connection"""
        try:
            self.connect()
            yield self.imap
        finally:
            self.disconnect()
    
    def connect(self) -> bool:
        """Connect to email server via IMAP"""
        try:
            logger.info(f"Connecting to IMAP server: {self.server}")
            self.imap = imaplib.IMAP4_SSL(self.server)
            self.imap.login(self.user, self.password)
            logger.info("IMAP connection established successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to email server: {e}")
            raise EmailProcessingError(f"IMAP connection failed: {e}")
    
    def disconnect(self):
        """Close IMAP connection"""
        if self.imap:
            try:
                self.imap.close()
                self.imap.logout()
                logger.info("IMAP connection closed")
            except Exception as e:
                logger.warning(f"Error closing IMAP connection: {e}")
    
    def get_folders(self) -> List[str]:
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
            logger.error(f"Error getting folders: {e}")
            raise EmailProcessingError(f"Failed to get folders: {e}")
        return []

class EmailProcessor:
    """Email processing and content extraction"""
    
    def __init__(self):
        self.html_converter = HTMLToTextConverter()
    
    def decode_subject(self, subject: Optional[str]) -> str:
        """Decode email subject from various encodings"""
        if subject is None:
            return "No Subject"
        
        try:
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
        except Exception as e:
            logger.warning(f"Error decoding subject: {e}")
            return str(subject)
    
    def extract_content(self, msg) -> str:
        """Extract text content from email message"""
        content = ""
        html_content = ""
        
        try:
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
                            logger.warning(f"Error decoding part: {e}")
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
                    logger.warning(f"Error decoding message: {e}")
            
            # If no plain text content found, convert HTML to text
            if not content.strip() and html_content.strip():
                logger.debug("No plain text found, converting HTML to text")
                content = self.html_converter.convert(html_content)
            
            return content.strip()
            
        except Exception as e:
            logger.error(f"Error extracting email content: {e}")
            return ""
    
    def clean_content(self, content: str) -> str:
        """Clean email content by removing unnecessary parts"""
        try:
            lines = content.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                # Skip quoted text, signatures, and common email footers
                if (line.startswith('>') or 
                    (line.startswith('On ') and 'wrote:' in line) or
                    line == '--' or
                    'unsubscribe' in line.lower() or
                    'confidential' in line.lower()):
                    continue
                cleaned_lines.append(line)
            
            # Join and remove excessive whitespace
            cleaned_content = '\n'.join(cleaned_lines)
            cleaned_content = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_content)
            
            return cleaned_content.strip()
        except Exception as e:
            logger.warning(f"Error cleaning content: {e}")
            return content

class NotificationService:
    """Service for sending notifications"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
    
    def send_discord(self, summary: str, email_count: int) -> bool:
        """Send summary to Discord via webhook"""
        if not self.config.discord_webhook_url:
            logger.warning("Discord webhook URL not configured")
            return False
        
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Split summary if too long for Discord
            if len(summary) > 1900:
                summary = summary[:1900] + "\n\n... (truncated, see email for full report)"
            
            embed = {
                "title": f"📊 Email Summary Report - {email_count} emails analyzed",
                "description": summary,
                "color": 0x00ff00,
                "footer": {"text": f"Generated at {timestamp}"},
                "fields": [
                    {"name": "📧 Email Count", "value": str(email_count), "inline": True},
                    {"name": "🤖 AI Model", "value": f"{self.config.model.upper()} - {self.config.model_name}", "inline": True}
                ]
            }
            
            payload = {"embeds": [embed], "username": "Email Summary Bot"}
            
            response = requests.post(
                self.config.discord_webhook_url,
                json=payload,
                timeout=self.config.request_timeout
            )
            
            if response.status_code == 204:
                logger.info("Discord notification sent successfully")
                return True
            else:
                logger.error(f"Discord notification failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Discord notification: {e}")
            raise NotificationError(f"Discord notification failed: {e}")
    
    def send_email(self, summary: str, email_count: int, recipient_email: str) -> bool:
        """Send summary via email"""
        if not all([self.config.smtp_server, self.config.smtp_user, self.config.smtp_password]):
            logger.error("SMTP configuration incomplete")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.smtp_user
            msg['To'] = recipient_email
            msg['Subject'] = f"📊 Email Summary Report - {email_count} emails analyzed"
            
            # Convert markdown to HTML
            html_summary = mistune.html(summary)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            html_body = self._create_html_template(html_summary, timestamp, email_count)
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_user, self.config.smtp_password)
                server.send_message(msg)
            
            logger.info("Email notification sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            raise NotificationError(f"Email notification failed: {e}")
    
    def _create_html_template(self, html_summary: str, timestamp: str, email_count: int) -> str:
        """Create HTML email template"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-gray-50 font-sans">
            <div class="max-w-4xl mx-auto p-4">
                <div class="bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-t-lg p-6 text-center">
                    <h1 class="text-2xl font-bold m-0">📊 Email Summary Report</h1>
                </div>
                
                <div class="bg-blue-50 border-l-4 border-blue-500 p-4">
                    <div class="space-y-2 text-sm">
                        <p><span class="font-semibold">📅 Generated:</span> {timestamp}</p>
                        <p><span class="font-semibold">📧 Emails Analyzed:</span> {email_count}</p>
                        <p><span class="font-semibold">🤖 AI Model:</span> {self.config.model.upper()} - {self.config.model_name}</p>
                    </div>
                </div>
                
                <div class="bg-white border border-gray-200 rounded-b-lg p-6">
                    <div class="bg-gray-50 border-l-4 border-green-500 rounded-lg p-5">
                        <div class="prose prose-gray max-w-none">
                            {html_summary}
                        </div>
                    </div>
                </div>
                
                <div class="text-center p-4 bg-gray-50 text-gray-600 text-sm">
                    <p class="italic">Generated by Email Summary Bot</p>
                    <p class="text-xs mt-1">Powered by {self.config.model.upper()} AI</p>
                </div>
            </div>
        </body>
        </html>
        """

class EmailSummarizer:
    """Main email summarization service"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm_provider = LLMProvider(config)
        self.processor = EmailProcessor()
        self.notification_service = NotificationService(config)
        
    def read_emails(self, email_client: EmailClient, folder_name: str, 
                   days_back: int = 7, max_emails: int = 50) -> List[EmailData]:
        """Read emails from specified folder"""
        try:
            # Select the folder
            status, messages = email_client.imap.select(folder_name)
            if status != 'OK':
                raise EmailProcessingError(f"Failed to select folder: {folder_name}")
            
            # Calculate date range
            since_date = (datetime.now() - timedelta(days=days_back)).strftime("%d-%b-%Y")
            
            # Search for emails
            search_criteria = f'(SINCE "{since_date}")'
            status, message_ids = email_client.imap.search(None, search_criteria)
            
            if status != 'OK':
                raise EmailProcessingError("Failed to search emails")
            
            email_ids = message_ids[0].split()
            email_ids = email_ids[-max_emails:]  # Get most recent emails
            
            emails = []
            logger.info(f"Found {len(email_ids)} emails in '{folder_name}' from last {days_back} days")
            
            for i, email_id in enumerate(email_ids):
                try:
                    status, msg_data = email_client.imap.fetch(email_id, '(RFC822)')
                    if status != 'OK':
                        continue
                    
                    msg = email.message_from_bytes(msg_data[0][1])
                    
                    # Extract email details
                    subject = self.processor.decode_subject(msg.get('Subject'))
                    sender = msg.get('From', 'Unknown')
                    date = msg.get('Date', 'Unknown')
                    content = self.processor.extract_content(msg)
                    content = self.processor.clean_content(content)
                    
                    # Skip emails with insufficient content
                    if len(content) < 50:
                        logger.debug(f"Skipping email '{subject[:50]}...' - content too short ({len(content)} chars)")
                        continue
                    
                    logger.debug(f"Processing email {i+1}/{len(email_ids)}: {subject[:50]}... - {len(content)} chars")
                    
                    email_data = EmailData(
                        id=email_id.decode(),
                        subject=subject,
                        sender=sender,
                        date=date,
                        content=content[:2000]  # Limit content length
                    )
                    emails.append(email_data)
                    
                    # Rate limiting
                    time.sleep(self.config.rate_limit_delay)
                    
                except Exception as e:
                    logger.warning(f"Error processing email {email_id}: {e}")
                    continue
            
            logger.info(f"Successfully processed {len(emails)} emails")
            return emails
            
        except Exception as e:
            logger.error(f"Error reading emails: {e}")
            raise EmailProcessingError(f"Failed to read emails: {e}")
    
    def analyze_emails(self, emails: List[EmailData]) -> str:
        """Analyze and summarize emails using LLM"""
        if not emails:
            return "No emails found to summarize."
        
        logger.info(f"Analyzing {len(emails)} emails with {self.config.model.upper()}")
        
        # Analyze individual emails
        for i, email_data in enumerate(emails):
            try:
                email_text = f"Subject: {email_data.subject}\nFrom: {email_data.sender}\nContent: {email_data.content[:800]}"
                
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
                
                analysis = self.llm_provider.generate(analysis_prompt)
                email_data.analysis = analysis
                
                logger.debug(f"Analyzed email {i+1}/{len(emails)}: {email_data.subject[:50]}...")
                
                # Rate limiting between LLM calls
                time.sleep(self.config.rate_limit_delay)
                
            except Exception as e:
                logger.warning(f"Error analyzing email {i+1}: {e}")
                email_data.analysis = f"Analysis failed: {e}"
        
        # Generate overall summary
        analyses_text = "\n".join([
            f"Email: {email.subject}\nAnalysis: {email.analysis}\n---"
            for email in emails[:10]
        ])
        
        overall_prompt = f"""
Based on the following email analyses, create a comprehensive summary report:

{analyses_text}

Please provide:
1. EXECUTIVE SUMMARY (2-3 sentences)
2. CRITICAL ITEMS (anything marked as Critical/High importance)
3. CATEGORIES BREAKDOWN (count by category)
4. ACTION ITEMS (things that likely need your attention)
5. KEY HIGHLIGHTS (important information to know)

Keep it concise but comprehensive.
"""
        
        try:
            summary = self.llm_provider.generate(overall_prompt)
            logger.info("Email analysis completed successfully")
            return summary
        except Exception as e:
            logger.error(f"Error generating overall summary: {e}")
            raise LLMError(f"Failed to generate summary: {e}")
    
    def process_folder(self, server: str, user: str, password: str, 
                      folder_name: str, days_back: int = 7, max_emails: int = 50) -> str:
        """Process emails from a specific folder"""
        email_client = EmailClient(server, user, password)
        
        with email_client.connection():
            emails = self.read_emails(email_client, folder_name, days_back, max_emails)
            summary = self.analyze_emails(emails)
            return summary
    
    def send_notifications(self, summary: str, email_count: int, 
                          discord: bool = False, email_recipient: Optional[str] = None) -> bool:
        """Send notifications based on configuration"""
        success = True
        
        if discord:
            try:
                self.notification_service.send_discord(summary, email_count)
            except NotificationError as e:
                logger.error(f"Discord notification failed: {e}")
                success = False
        
        if email_recipient:
            try:
                self.notification_service.send_email(summary, email_count, email_recipient)
            except NotificationError as e:
                logger.error(f"Email notification failed: {e}")
                success = False
        
        return success

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description='Production-Ready Email Summary Script with Multiple LLM Support',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--server', required=True, help='IMAP server (e.g., imap.gmail.com)')
    parser.add_argument('--email', required=True, help='Your email address')
    parser.add_argument('--password', help='Email password (or use EMAIL_PASSWORD env var)')
    
    # Email processing options
    parser.add_argument('--folder', default='INBOX', help='Email folder to read (default: INBOX)')
    parser.add_argument('--days', type=int, default=7, help='Number of days back to read (default: 7)')
    parser.add_argument('--max-emails', type=int, default=50, help='Maximum emails to process (default: 50)')
    
    # LLM Configuration
    parser.add_argument('--model-type', choices=['groq', 'ollama'], default='groq',
                       help='LLM provider to use (default: groq)')
    parser.add_argument('--model-name', default='llama3.1', help='Model name (default: llama3.1)')
    parser.add_argument('--ollama-url', default='http://localhost:11434',
                       help='Ollama server URL (default: http://localhost:11434)')
    parser.add_argument('--groq-api-key', help='Groq API key (or use GROQ_API_KEY env var)')
    
    # Notification options
    parser.add_argument('--send-discord', action='store_true', help='Send summary to Discord')
    parser.add_argument('--send-email', help='Send summary to this email address')
    parser.add_argument('--discord-webhook', help='Discord webhook URL')
    
    # SMTP configuration
    parser.add_argument('--smtp-server', help='SMTP server for email notifications')
    parser.add_argument('--smtp-user', help='SMTP username')
    parser.add_argument('--smtp-password', help='SMTP password')
    
    # Utility options
    parser.add_argument('--list-folders', action='store_true', help='List available folders and exit')
    parser.add_argument('--config-file', help='Load configuration from JSON file')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--log-file', help='Log file path (optional)')
    
    args = parser.parse_args()
    
    # Setup logging
    global logger
    logger = setup_logging(args.log_level, args.log_file)
    
    try:
        # Load configuration
        if args.config_file and Path(args.config_file).exists():
            logger.info(f"Loading configuration from {args.config_file}")
            with open(args.config_file, 'r') as f:
                config_data = json.load(f)
            config = AgentConfig(**config_data)
        else:
            # Create configuration from environment and arguments
            config = AgentConfig.from_env()
            
            # Override with command line arguments
            if args.model_type:
                config.model = args.model_type
            if args.model_name:
                config.model_name = args.model_name
            if args.ollama_url:
                config.ollama_api_url = args.ollama_url
            if args.groq_api_key:
                config.groq_api_key = args.groq_api_key
            if args.discord_webhook:
                config.discord_webhook_url = args.discord_webhook
            if args.smtp_server:
                config.smtp_server = args.smtp_server
            if args.smtp_user:
                config.smtp_user = args.smtp_user
            if args.smtp_password:
                config.smtp_password = args.smtp_password
        
        # Get email password
        password = args.password or os.getenv('EMAIL_PASSWORD')
        if not password:
            # Prompt for password if not provided
            import getpass
            try:
                password = getpass.getpass("Enter your email password (or app password): ")
            except KeyboardInterrupt:
                logger.info("Password input cancelled by user")
                return 130
            
            if not password:
                logger.error("Email password is required")
                logger.info("For Gmail, create an App Password: https://myaccount.google.com/apppasswords")
                return 1
        
        # Validate configuration
        if config.model == 'groq' and not config.groq_api_key:
            logger.error("Groq API key required for Groq model. Use --groq-api-key or set GROQ_API_KEY env var")
            return 1
        
        # Initialize email client and summarizer
        email_client = EmailClient(args.server, args.email, password)
        summarizer = EmailSummarizer(config)
        
        # List folders if requested
        if args.list_folders:
            logger.info("Listing available email folders...")
            with email_client.connection():
                folders = email_client.get_folders()
                print("\n📁 Available folders:")
                for folder in folders:
                    print(f"  - {folder}")
            return 0
        
        # Process emails
        logger.info(f"Starting email processing from folder: {args.folder}")
        logger.info(f"Configuration: {config.model.upper()} model '{config.model_name}', "
                   f"{args.days} days back, max {args.max_emails} emails")
        
        with email_client.connection():
            emails = summarizer.read_emails(email_client, args.folder, args.days, args.max_emails)
            
            if not emails:
                logger.warning("No emails found to process")
                print("No emails found to process.")
                return 0
            
            # Analyze emails
            logger.info(f"Analyzing {len(emails)} emails...")
            summary = summarizer.analyze_emails(emails)
            
            # Display summary
            print("\n" + "="*60)
            print("📊 EMAIL SUMMARY REPORT")
            print("="*60)
            print(summary)
            print("="*60)
            
            # Send notifications
            notification_success = True
            if args.send_discord or args.send_email:
                logger.info("Sending notifications...")
                notification_success = summarizer.send_notifications(
                    summary, 
                    len(emails),
                    discord=args.send_discord,
                    email_recipient=args.send_email
                )
            
            if notification_success:
                logger.info("✅ Email analysis completed successfully!")
                print(f"\n🎉 Analysis complete! Processed {len(emails)} emails.")
            else:
                logger.warning("⚠️ Analysis completed but some notifications failed")
                print(f"\n⚠️ Analysis complete but some notifications failed. Check logs for details.")
            
            return 0 if notification_success else 1
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        print("\n⚠️ Process interrupted by user")
        return 130
        
    except EmailProcessingError as e:
        logger.error(f"Email processing error: {e}")
        print(f"❌ Email processing failed: {e}")
        return 1
        
    except LLMError as e:
        logger.error(f"LLM error: {e}")
        print(f"❌ AI analysis failed: {e}")
        return 1
        
    except NotificationError as e:
        logger.error(f"Notification error: {e}")
        print(f"❌ Notification failed: {e}")
        return 1
        
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"❌ Unexpected error occurred: {e}")
        return 1

def create_sample_config():
    """Create a sample configuration file"""
    config = AgentConfig.from_env()
    sample_config = {
        "model": "groq",
        "model_name": "llama3.1",
        "ollama_api_url": "http://localhost:11434",
        "groq_api_key": "your_groq_api_key_here",
        "discord_webhook_url": "https://discord.com/api/webhooks/your_webhook_here",
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "smtp_user": "your_email@gmail.com",
        "smtp_password": "your_app_password_here",
        "max_retries": 3,
        "request_timeout": 60,
        "rate_limit_delay": 1.0
    }
    
    with open("email_summarizer_config.json", "w") as f:
        json.dump(sample_config, f, indent=2)
    
    print("📄 Sample configuration file created: email_summarizer_config.json")
    print("Edit this file with your actual configuration values.")

if __name__ == "__main__":
    import sys
    
    # Handle special commands
    if len(sys.argv) > 1 and sys.argv[1] == "create-config":
        create_sample_config()
        sys.exit(0)
    
    # Run main application
    exit_code = main()
    sys.exit(exit_code)