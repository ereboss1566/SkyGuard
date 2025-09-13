"""
Alert Manager Module for SkyGuard

This module handles the generation and delivery of weather alerts.
"""

import logging
import json
import os
import smtplib
import requests
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Alert:
    """Class representing a weather alert."""
    
    def __init__(self, alert_type: str, severity: str, message: str, 
                 location: Dict[str, Union[str, float]], timestamp: Optional[datetime] = None):
        """Initialize a weather alert.
        
        Args:
            alert_type: Type of alert (e.g., 'storm', 'flood', 'tornado')
            severity: Severity level ('low', 'medium', 'high', 'critical')
            message: Alert message
            location: Dictionary with location information
            timestamp: Alert timestamp (defaults to current time)
        """
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.location = location
        self.timestamp = timestamp or datetime.now()
        self.alert_id = f"{self.alert_type}_{self.timestamp.strftime('%Y%m%d%H%M%S')}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary.
        
        Returns:
            Dictionary representation of the alert
        """
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'location': self.location,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """Create alert from dictionary.
        
        Args:
            data: Dictionary with alert data
            
        Returns:
            Alert instance
        """
        timestamp = datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else None
        return cls(
            alert_type=data['alert_type'],
            severity=data['severity'],
            message=data['message'],
            location=data['location'],
            timestamp=timestamp
        )
    
    def __str__(self) -> str:
        """String representation of the alert.
        
        Returns:
            Formatted alert string
        """
        return f"[{self.severity.upper()}] {self.alert_type.capitalize()} Alert: {self.message}"


class AlertChannel:
    """Base class for alert delivery channels."""
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert through the channel.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if alert was sent successfully, False otherwise
        """
        raise NotImplementedError("Subclasses must implement send_alert method")


class EmailAlertChannel(AlertChannel):
    """Channel for sending alerts via email."""
    
    def __init__(self, smtp_server: str, smtp_port: int, 
                 username: str, password: str, sender_email: str):
        """Initialize email alert channel.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            sender_email: Sender email address
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.sender_email = sender_email
    
    def send_alert(self, alert: Alert, recipient_emails: List[str]) -> bool:
        """Send alert via email.
        
        Args:
            alert: Alert to send
            recipient_emails: List of recipient email addresses
            
        Returns:
            True if alert was sent successfully, False otherwise
        """
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(recipient_emails)
            msg['Subject'] = f"SkyGuard {alert.severity.upper()} Weather Alert: {alert.alert_type.capitalize()}"
            
            # Create HTML content
            html = f"""
            <html>
              <head></head>
              <body>
                <h2 style="color: {'red' if alert.severity in ['high', 'critical'] else 'orange'}">
                  {alert.alert_type.capitalize()} Alert - {alert.severity.upper()}
                </h2>
                <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Location:</strong> {alert.location.get('name', 'Unknown')}</p>
                <p><strong>Message:</strong> {alert.message}</p>
                <p><strong>Coordinates:</strong> Lat {alert.location.get('lat', 'N/A')}, Lon {alert.location.get('lon', 'N/A')}</p>
                <hr>
                <p><em>This is an automated alert from the SkyGuard Weather Alert System.</em></p>
              </body>
            </html>
            """
            
            msg.attach(MIMEText(html, 'html'))
            
            # Connect to server and send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
                
            logger.info(f"Email alert sent to {len(recipient_emails)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
            return False


class SMSAlertChannel(AlertChannel):
    """Channel for sending alerts via SMS."""
    
    def __init__(self, api_key: str, api_url: str):
        """Initialize SMS alert channel.
        
        Args:
            api_key: API key for SMS service
            api_url: API URL for SMS service
        """
        self.api_key = api_key
        self.api_url = api_url
    
    def send_alert(self, alert: Alert, phone_numbers: List[str]) -> bool:
        """Send alert via SMS.
        
        Args:
            alert: Alert to send
            phone_numbers: List of recipient phone numbers
            
        Returns:
            True if alert was sent successfully, False otherwise
        """
        try:
            # Format message
            message = f"SkyGuard {alert.severity.upper()} ALERT: {alert.alert_type.capitalize()} - {alert.message}"
            
            # Send to each recipient
            success_count = 0
            for phone in phone_numbers:
                payload = {
                    'api_key': self.api_key,
                    'to': phone,
                    'message': message
                }
                
                response = requests.post(self.api_url, json=payload)
                
                if response.status_code == 200:
                    success_count += 1
                else:
                    logger.warning(f"Failed to send SMS to {phone}: {response.text}")
            
            logger.info(f"SMS alerts sent successfully to {success_count}/{len(phone_numbers)} recipients")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed to send SMS alert: {str(e)}")
            return False


class WebhookAlertChannel(AlertChannel):
    """Channel for sending alerts via webhook."""
    
    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        """Initialize webhook alert channel.
        
        Args:
            webhook_url: URL to send webhook to
            headers: Optional HTTP headers
        """
        self.webhook_url = webhook_url
        self.headers = headers or {}
        if 'Content-Type' not in self.headers:
            self.headers['Content-Type'] = 'application/json'
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if alert was sent successfully, False otherwise
        """
        try:
            # Convert alert to JSON
            payload = json.dumps(alert.to_dict())
            
            # Send webhook
            response = requests.post(self.webhook_url, headers=self.headers, data=payload)
            
            if response.status_code in [200, 201, 202, 204]:
                logger.info(f"Webhook alert sent successfully: {response.status_code}")
                return True
            else:
                logger.warning(f"Failed to send webhook alert: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {str(e)}")
            return False


class AlertManager:
    """Manager for generating and sending alerts."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the alert manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.channels = {}
        self.alert_history = []
        self.config = {}
        
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> None:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                
            # Initialize channels from config
            if 'channels' in self.config:
                for channel_type, channel_config in self.config['channels'].items():
                    if channel_type == 'email' and channel_config.get('enabled', False):
                        self.add_email_channel(
                            smtp_server=channel_config['smtp_server'],
                            smtp_port=channel_config['smtp_port'],
                            username=channel_config['username'],
                            password=channel_config['password'],
                            sender_email=channel_config['sender_email']
                        )
                    elif channel_type == 'sms' and channel_config.get('enabled', False):
                        self.add_sms_channel(
                            api_key=channel_config['api_key'],
                            api_url=channel_config['api_url']
                        )
                    elif channel_type == 'webhook' and channel_config.get('enabled', False):
                        self.add_webhook_channel(
                            webhook_url=channel_config['webhook_url'],
                            headers=channel_config.get('headers')
                        )
                        
            logger.info(f"Loaded configuration from {config_path}")
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
    
    def add_channel(self, channel_name: str, channel: AlertChannel) -> None:
        """Add an alert channel.
        
        Args:
            channel_name: Name of the channel
            channel: AlertChannel instance
        """
        self.channels[channel_name] = channel
        logger.info(f"Added alert channel: {channel_name}")
    
    def add_email_channel(self, smtp_server: str, smtp_port: int, 
                         username: str, password: str, sender_email: str) -> None:
        """Add an email alert channel.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            sender_email: Sender email address
        """
        channel = EmailAlertChannel(
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            username=username,
            password=password,
            sender_email=sender_email
        )
        self.add_channel('email', channel)
    
    def add_sms_channel(self, api_key: str, api_url: str) -> None:
        """Add an SMS alert channel.
        
        Args:
            api_key: API key for SMS service
            api_url: API URL for SMS service
        """
        channel = SMSAlertChannel(api_key=api_key, api_url=api_url)
        self.add_channel('sms', channel)
    
    def add_webhook_channel(self, webhook_url: str, 
                           headers: Optional[Dict[str, str]] = None) -> None:
        """Add a webhook alert channel.
        
        Args:
            webhook_url: URL to send webhook to
            headers: Optional HTTP headers
        """
        channel = WebhookAlertChannel(webhook_url=webhook_url, headers=headers)
        self.add_channel('webhook', channel)
    
    def create_alert(self, alert_type: str, severity: str, message: str, 
                    location: Dict[str, Union[str, float]]) -> Alert:
        """Create a new alert.
        
        Args:
            alert_type: Type of alert
            severity: Severity level
            message: Alert message
            location: Dictionary with location information
            
        Returns:
            Created Alert instance
        """
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            location=location
        )
        
        self.alert_history.append(alert)
        logger.info(f"Created alert: {alert}")
        
        return alert
    
    def send_alert(self, alert: Alert, channels: Optional[List[str]] = None) -> Dict[str, bool]:
        """Send alert through specified channels.
        
        Args:
            alert: Alert to send
            channels: List of channel names (if None, send through all channels)
            
        Returns:
            Dictionary mapping channel names to success status
        """
        if channels is None:
            channels = list(self.channels.keys())
            
        results = {}
        
        for channel_name in channels:
            if channel_name not in self.channels:
                logger.warning(f"Unknown channel: {channel_name}")
                results[channel_name] = False
                continue
                
            channel = self.channels[channel_name]
            
            # Get recipients from config if available
            recipients = None
            if 'recipients' in self.config and channel_name in self.config['recipients']:
                recipients = self.config['recipients'][channel_name]
            
            # Send alert
            try:
                if channel_name == 'email' and recipients:
                    success = channel.send_alert(alert, recipients)
                elif channel_name == 'sms' and recipients:
                    success = channel.send_alert(alert, recipients)
                else:
                    success = channel.send_alert(alert)
                    
                results[channel_name] = success
                
            except Exception as e:
                logger.error(f"Error sending alert through {channel_name}: {str(e)}")
                results[channel_name] = False
        
        return results
    
    def get_alert_history(self, limit: Optional[int] = None) -> List[Alert]:
        """Get alert history.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of Alert instances
        """
        if limit is not None:
            return self.alert_history[-limit:]
        return self.alert_history