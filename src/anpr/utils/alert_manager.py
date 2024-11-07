import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
import os

# Desktop notifications - optional import
try:
    from plyer import notification as desktop_notification
    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False
    desktop_notification = None


logger = logging.getLogger(__name__)


class AlertType(Enum):
    PLATE_DETECTED = "plate_detected"
    NEW_PLATE = "new_plate"
    REPEATED_PLATE = "repeated_plate"
    WATCHLIST_MATCH = "watchlist_match"
    LOW_CONFIDENCE = "low_confidence"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    SYSTEM_ERROR = "system_error"
    PERFORMANCE_ISSUE = "performance_issue"


@dataclass
class Alert:
    """Data structure for an alert."""
    alert_type: AlertType
    plate_text: str
    camera_id: str
    timestamp: datetime
    confidence: float
    message: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AlertRules:
    """Configuration for alert triggering rules."""
    watchlist_enabled: bool = True


class AlertManager:
    """
    Manages alert generation and distribution for the ANPR system.
    Handles various types of alerts including plate detection, watchlist matches,
    repeated detections, and low confidence warnings.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.alert_handlers: List['AlertHandler'] = []
        self.enabled = self.config.get('enabled', True)
        
        # Broadcast callback for WebSocket real-time alerts
        self._broadcast_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
        
        # Alert rules configuration
        rules_config = self.config.get('rules', {})
        self.rules = AlertRules(
            watchlist_enabled=rules_config.get('watchlist_enabled', True)
        )
        
        # Watchlist/blacklist management
        self.watchlist: Set[str] = set()
        self._load_watchlist()
        
        # Initialize alert handlers based on config
        self._setup_alert_handlers()
        
        logger.info("AlertManager initialized with desktop notifications: %s", PLYER_AVAILABLE)
    
    def set_broadcast_callback(self, callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        """
        Set the broadcast callback for WebSocket real-time alerts.
        
        Args:
            callback: Async function that broadcasts alert data to WebSocket clients
        """
        self._broadcast_callback = callback
        logger.info("WebSocket broadcast callback registered for real-time alerts")
    
    def _load_watchlist(self):
        """Load watchlist/blacklist from file."""
        # Load from config blacklist array
        blacklist_items = self.config.get('blacklist', [])
        if blacklist_items:
            self.watchlist.update(plate.upper().replace(' ', '') for plate in blacklist_items)
        
        # Load from blacklist file if specified
        blacklist_path = self.config.get('blacklist_path')
        if blacklist_path:
            expanded_path = os.path.expanduser(blacklist_path)
            if Path(expanded_path).exists():
                try:
                    with open(expanded_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            plate = line.strip().upper().replace(' ', '')
                            if plate and not plate.startswith('#'):
                                self.watchlist.add(plate)
                    logger.info(f"Loaded {len(self.watchlist)} plates from watchlist")
                except Exception as e:
                    logger.error(f"Failed to load watchlist from {blacklist_path}: {e}")
    
    def add_to_watchlist(self, plate_text: str):
        """Add a plate to the watchlist."""
        normalized = plate_text.upper().replace(' ', '')
        self.watchlist.add(normalized)
        logger.info(f"Added {normalized} to watchlist")
    
    def remove_from_watchlist(self, plate_text: str):
        """Remove a plate from the watchlist."""
        normalized = plate_text.upper().replace(' ', '')
        self.watchlist.discard(normalized)
        logger.info(f"Removed {normalized} from watchlist")
    
    def is_in_watchlist(self, plate_text: str) -> bool:
        """Check if a plate is in the watchlist."""
        normalized = plate_text.upper().replace(' ', '')
        return normalized in self.watchlist
    
    def _setup_alert_handlers(self):
        """Setup various alert handlers based on configuration."""
        channels = self.config.get('channels', ['console', 'log'])
        
        # Desktop notification handler
        if 'desktop' in channels and PLYER_AVAILABLE:
            desktop_handler = DesktopAlertHandler(self.config.get('desktop_config', {}))
            self.alert_handlers.append(desktop_handler)
            logger.info("Desktop notification handler enabled")
        elif 'desktop' in channels and not PLYER_AVAILABLE:
            logger.warning("Desktop notifications requested but plyer is not available. Install with: pip install plyer")
        
        # Console handler (always enabled for logging)
        if 'console' in channels:
            console_handler = ConsoleAlertHandler()
            self.alert_handlers.append(console_handler)
        
        # File handler
        if 'log' in channels:
            file_config = self.config.get('file_config', {})
            if not file_config.get('file_path'):
                file_config['file_path'] = self.config.get('log_path', 'alerts.log')
            file_handler = FileAlertHandler(file_config)
            self.alert_handlers.append(file_handler)
        
        # Email handler
        if self.config.get('email_enabled', False) or 'email' in channels:
            email_handler = EmailAlertHandler(self.config.get('email_config', {}))
            self.alert_handlers.append(email_handler)
        
        logger.info(f"Setup {len(self.alert_handlers)} alert handlers")
    
    async def trigger_alert(self, alert: Alert):
        """Trigger an alert and send it to all registered handlers."""
        if not self.enabled:
            return
        
        logger.info(f"Triggering alert: {alert.alert_type.value} for plate {alert.plate_text}")
        
        # Broadcast to WebSocket clients for real-time notifications
        if self._broadcast_callback:
            try:
                alert_data = {
                    "alert_type": alert.alert_type.value,
                    "plate_text": alert.plate_text,
                    "camera_id": alert.camera_id,
                    "timestamp": alert.timestamp.isoformat(),
                    "confidence": alert.confidence,
                    "message": alert.message,
                    "metadata": alert.metadata or {}
                }
                await self._broadcast_callback(alert_data)
            except Exception as e:
                logger.error(f"Error broadcasting alert to WebSocket: {e}")
        
        # Process alert through all handlers concurrently
        tasks = []
        for handler in self.alert_handlers:
            task = asyncio.create_task(handler.handle_alert(alert))
            tasks.append(task)
        
        # Wait for all handlers to process the alert
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in alert handling: {e}")
    
    def check_and_trigger(self, plate_text: str, camera_id: str, confidence: float = 0.0):
        """
        Check if an alert should be triggered for a detected plate.
        Only triggers alerts for watchlist matches to avoid flooding in busy deployments.
        
        This is a sync wrapper that schedules async alert triggering.
        """
        current_time = datetime.now()
        normalized_plate = plate_text.upper().replace(' ', '')
        
        # Only trigger alerts for watchlist matches
        if self.rules.watchlist_enabled and self.is_in_watchlist(normalized_plate):
            alert = Alert(
                alert_type=AlertType.WATCHLIST_MATCH,
                plate_text=plate_text,
                camera_id=camera_id,
                timestamp=current_time,
                confidence=confidence,
                message=f"WATCHLIST MATCH: Plate {plate_text} detected on camera {camera_id}",
                metadata={
                    'detection_confidence': confidence,
                    'camera_id': camera_id,
                    'is_watchlist': True,
                    'priority': 'high'
                }
            )
            self._schedule_alerts([alert])
    
    def _schedule_alerts(self, alerts: List[Alert]):
        """Schedule alert triggering in the appropriate event loop."""
        for alert in alerts:
            try:
                # Try to get the running event loop (works in async context)
                loop = asyncio.get_running_loop()
                # Schedule the async task
                loop.create_task(self.trigger_alert(alert))
            except RuntimeError:
                # No running loop - we're in a sync context
                # Use run_coroutine_threadsafe if there's a loop in another thread
                # Or fall back to sync execution
                try:
                    # Try to get event loop that might exist but not running in this thread
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Loop exists and is running in another thread
                        asyncio.run_coroutine_threadsafe(self.trigger_alert(alert), loop)
                    else:
                        # Loop exists but not running - run the coroutine
                        loop.run_until_complete(self.trigger_alert(alert))
                except RuntimeError:
                    # No event loop at all - create one and run
                    # Note: WebSocket broadcast won't work in this isolated context
                    logger.warning("No event loop available - running alert in isolated context")
                    asyncio.run(self.trigger_alert(alert))
    
    def enable(self):
        """Enable alert generation."""
        self.enabled = True
        logger.info("AlertManager enabled")
    
    def disable(self):
        """Disable alert generation."""
        self.enabled = False
        logger.info("AlertManager disabled")
    
    def get_watchlist(self) -> List[str]:
        """Get the current watchlist."""
        return list(self.watchlist)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get alert system statistics."""
        return {
            'enabled': self.enabled,
            'handlers_count': len(self.alert_handlers),
            'watchlist_count': len(self.watchlist),
            'desktop_notifications_available': PLYER_AVAILABLE,
            'rules': {
                'watchlist_enabled': self.rules.watchlist_enabled
            }
        }
    
    async def trigger_test_alert(self, camera_id: str = "test_camera") -> Optional[str]:
        """
        Trigger a test watchlist alert for debugging notifications.
        
        Args:
            camera_id: Camera ID to use in test alert
            
        Returns:
            Alert ID if successful, None otherwise
        """
        import uuid
        
        test_plate = "TEST1234"
        alert_id = str(uuid.uuid4())
        
        alert = Alert(
            alert_type=AlertType.WATCHLIST_MATCH,
            plate_text=test_plate,
            camera_id=camera_id,
            timestamp=datetime.now(),
            confidence=0.95,
            message=f"TEST ALERT: This is a test watchlist alert for plate {test_plate}",
            metadata={
                "is_test": True,
                "alert_id": alert_id
            }
        )
        
        try:
            # Trigger alert through normal flow (handlers + broadcast)
            await self.trigger_alert(alert)
            logger.info(f"Test alert triggered successfully: {alert_id}")
            return alert_id
        except Exception as e:
            logger.error(f"Failed to trigger test alert: {e}")
            return None


class AlertHandler:
    """Base class for alert handlers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    async def handle_alert(self, alert: Alert):
        """Handle an alert. This should be implemented by subclasses."""
        raise NotImplementedError


class DesktopAlertHandler(AlertHandler):
    """Handles alerts by sending desktop notifications using plyer."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.app_name = self.config.get('app_name', 'ANPR System')
        self.timeout = self.config.get('timeout', 10)
    
    async def handle_alert(self, alert: Alert):
        """Send desktop notification."""
        if not PLYER_AVAILABLE or desktop_notification is None:
            logger.warning("Desktop notifications not available")
            return
        
        try:
            # Determine notification title based on alert type
            title_map = {
                AlertType.WATCHLIST_MATCH: f"WATCHLIST ALERT - {self.app_name}",
                AlertType.NEW_PLATE: f"New Plate - {self.app_name}",
                AlertType.REPEATED_PLATE: f"Repeated Plate - {self.app_name}",
                AlertType.LOW_CONFIDENCE: f"Low Confidence - {self.app_name}",
                AlertType.PLATE_DETECTED: f"Plate Detected - {self.app_name}",
                AlertType.SYSTEM_ERROR: f"System Error - {self.app_name}",
                AlertType.PERFORMANCE_ISSUE: f"Performance Issue - {self.app_name}",
            }
            
            title = title_map.get(alert.alert_type, f"Alert - {self.app_name}")
            
            # Create notification message
            message = f"Plate: {alert.plate_text}\n"
            message += f"Camera: {alert.camera_id}\n"
            message += f"Confidence: {alert.confidence:.1%}\n"
            message += f"Time: {alert.timestamp.strftime('%H:%M:%S')}"
            
            # Send notification
            desktop_notification.notify(
                title=title,
                message=message,
                app_name=self.app_name,
                timeout=self.timeout
            )
            
            logger.debug(f"Desktop notification sent for plate {alert.plate_text}")
            
        except Exception as e:
            logger.error(f"Failed to send desktop notification: {e}")


class ConsoleAlertHandler(AlertHandler):
    """Handles alerts by logging to console."""
    
    async def handle_alert(self, alert: Alert):
        # Use different formatting for high-priority alerts
        if alert.alert_type == AlertType.WATCHLIST_MATCH:
            print(f"\n{'='*60}")
            print(f"[WATCHLIST ALERT] {alert.message}")
            print(f"Plate: {alert.plate_text} | Camera: {alert.camera_id}")
            print(f"Time: {alert.timestamp} | Confidence: {alert.confidence:.2%}")
            print(f"{'='*60}\n")
        elif alert.alert_type == AlertType.LOW_CONFIDENCE:
            print(f"[LOW CONF] {alert.alert_type.value}: {alert.message} | "
                  f"Plate: {alert.plate_text} | Camera: {alert.camera_id} | "
                  f"Confidence: {alert.confidence:.2%}")
        else:
            print(f"[ALERT] {alert.alert_type.value}: {alert.message} | "
                  f"Plate: {alert.plate_text} | Camera: {alert.camera_id} | "
                  f"Time: {alert.timestamp} | Confidence: {alert.confidence:.2%}")


class FileAlertHandler(AlertHandler):
    """Handles alerts by writing to a JSON log file."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.file_path = os.path.expanduser(self.config.get('file_path', 'alerts.log'))
        
        # Ensure the directory exists
        Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)
    
    async def handle_alert(self, alert: Alert):
        try:
            with open(self.file_path, 'a', encoding='utf-8') as f:
                alert_data = {
                    'timestamp': alert.timestamp.isoformat(),
                    'type': alert.alert_type.value,
                    'plate': alert.plate_text,
                    'camera': alert.camera_id,
                    'confidence': alert.confidence,
                    'message': alert.message,
                    'metadata': alert.metadata
                }
                f.write(json.dumps(alert_data) + '\n')
        except Exception as e:
            logger.error(f"Failed to write alert to file: {e}")


class EmailAlertHandler(AlertHandler):
    """Handles alerts by sending emails."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.smtp_server = self.config.get('smtp_server', 'localhost')
        self.smtp_port = self.config.get('smtp_port', 587)
        self.username = self.config.get('username')
        self.password = self.config.get('password')
        self.sender_email = self.config.get('sender_email')
        self.recipients = self.config.get('recipients', [])
        
        # Only send emails for high-priority alerts by default
        self.alert_types_to_email = self.config.get('alert_types', [
            AlertType.WATCHLIST_MATCH.value,
            AlertType.SYSTEM_ERROR.value
        ])
    
    async def handle_alert(self, alert: Alert):
        if not self.recipients:
            logger.warning("No email recipients configured, skipping email alert")
            return
        
        # Only send emails for configured alert types
        if alert.alert_type.value not in self.alert_types_to_email:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = f"ANPR Alert: {alert.alert_type.value} - {alert.plate_text}"
            
            body = (f"ANPR System Alert\n\n"
                   f"Type: {alert.alert_type.value}\n"
                   f"Plate: {alert.plate_text}\n"
                   f"Camera: {alert.camera_id}\n"
                   f"Time: {alert.timestamp}\n"
                   f"Confidence: {alert.confidence:.2%}\n"
                   f"Message: {alert.message}\n\n"
                   f"Metadata: {json.dumps(alert.metadata, indent=2)}\n")
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.send_message(msg)
                
            logger.info(f"Alert email sent to {', '.join(self.recipients)}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
