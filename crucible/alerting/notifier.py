"""
Real-time Training Dashboard & Alerting System for crucible

This module provides:
1. Real-time training dashboard with interactive Plotly charts
2. Anomaly detection and alerting (Slack/email)
3. GPU monitoring and gradient norm tracking
4. Integration with existing LogCallback system

Usage:
    from crucible.alerting.notifier import DashboardCallback, TrainingDashboard
    
    # In training script
    if args.dashboard:
        dashboard = TrainingDashboard()
        callback = DashboardCallback(dashboard)
        trainer.add_callback(callback)
"""

import os
import sys
import json
import time
import threading
import queue
import logging
import smtplib
import socket
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

import numpy as np

# Optional imports with graceful fallback
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import psutil
    import GPUtil
    GPU_MONITORING = True
except ImportError:
    GPU_MONITORING = False

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False

# Import from existing crucible modules
try:
    from crucible.train.callbacks import LogCallback
    from crucible.utils.constants import LOGGING_PATH
    from crucible.utils.helpers import get_logger
    LLAMAFACTORY_AVAILABLE = True
except ImportError:
    LLAMAFACTORY_AVAILABLE = False
    # Fallback logger
    def get_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

logger = get_logger(__name__)


@dataclass
class AlertConfig:
    """Configuration for alerting system"""
    # Slack configuration
    slack_enabled: bool = False
    slack_token: Optional[str] = None
    slack_channel: str = "#training-alerts"
    
    # Email configuration
    email_enabled: bool = False
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_sender: Optional[str] = None
    email_password: Optional[str] = None
    email_recipients: List[str] = field(default_factory=list)
    
    # Alert thresholds
    loss_spike_threshold: float = 2.0  # Relative increase from moving average
    gpu_memory_threshold: float = 0.9  # 90% GPU memory usage
    gpu_temperature_threshold: float = 85.0  # Celsius
    gradient_norm_threshold: float = 100.0  # Gradient explosion threshold
    training_stall_seconds: int = 300  # 5 minutes without progress
    
    # Alert cooldowns (seconds)
    alert_cooldown: int = 600  # 10 minutes between same alerts
    
    @classmethod
    def from_env(cls) -> "AlertConfig":
        """Load configuration from environment variables"""
        config = cls()
        config.slack_enabled = os.getenv("LLAMAFACTORY_SLACK_ENABLED", "false").lower() == "true"
        config.slack_token = os.getenv("LLAMAFACTORY_SLACK_TOKEN")
        config.slack_channel = os.getenv("LLAMAFACTORY_SLACK_CHANNEL", "#training-alerts")
        
        config.email_enabled = os.getenv("LLAMAFACTORY_EMAIL_ENABLED", "false").lower() == "true"
        config.smtp_server = os.getenv("LLAMAFACTORY_SMTP_SERVER", "smtp.gmail.com")
        config.smtp_port = int(os.getenv("LLAMAFACTORY_SMTP_PORT", "587"))
        config.email_sender = os.getenv("LLAMAFACTORY_EMAIL_SENDER")
        config.email_password = os.getenv("LLAMAFACTORY_EMAIL_PASSWORD")
        config.email_recipients = os.getenv("LLAMAFACTORY_EMAIL_RECIPIENTS", "").split(",")
        
        return config


class Notifier:
    """Handles sending alerts via Slack and email"""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.last_alerts: Dict[str, float] = {}
        
        # Initialize Slack client
        self.slack_client = None
        if config.slack_enabled and SLACK_AVAILABLE and config.slack_token:
            try:
                self.slack_client = WebClient(token=config.slack_token)
                logger.info("Slack client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Slack client: {e}")
        
        # Email server connection pool
        self.email_server = None
        if config.email_enabled:
            self._connect_smtp()
    
    def _connect_smtp(self):
        """Establish SMTP connection"""
        try:
            self.email_server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            self.email_server.starttls()
            if self.config.email_sender and self.config.email_password:
                self.email_server.login(self.config.email_sender, self.config.email_password)
            logger.info("SMTP server connected")
        except Exception as e:
            logger.error(f"Failed to connect to SMTP server: {e}")
            self.email_server = None
    
    def _should_send_alert(self, alert_type: str) -> bool:
        """Check if we should send an alert based on cooldown"""
        now = time.time()
        last_sent = self.last_alerts.get(alert_type, 0)
        
        if now - last_sent < self.config.alert_cooldown:
            return False
        
        self.last_alerts[alert_type] = now
        return True
    
    def send_slack_alert(self, message: str, alert_type: str = "training_alert") -> bool:
        """Send alert to Slack"""
        if not self.slack_client or not self._should_send_alert(alert_type):
            return False
        
        try:
            hostname = socket.gethostname()
            full_message = f"🚨 *crucible Alert* [{alert_type}]\nHost: {hostname}\n{message}"
            
            response = self.slack_client.chat_postMessage(
                channel=self.config.slack_channel,
                text=full_message,
                unfurl_links=False
            )
            logger.info(f"Slack alert sent: {alert_type}")
            return True
        except SlackApiError as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def send_email_alert(self, subject: str, message: str, alert_type: str = "training_alert") -> bool:
        """Send alert via email"""
            if not self.email_server or not self.config.email_sender or not self._should_send_alert(alert_type):
                return False
            
            try:
                hostname = socket.gethostname()
                full_subject = f"[crucible Alert] {subject} - {hostname}"
                
                msg = MIMEMultipart()
                msg['From'] = self.config.email_sender
                msg['To'] = ', '.join(self.config.email_recipients)
                msg['Subject'] = full_subject
                
                body = f"""
                Training Alert from crucible
                
                Alert Type: {alert_type}
                Host: {hostname}
                Time: {datetime.now().isoformat()}
                
                Message:
                {message}
                """
                
                msg.attach(MIMEText(body, 'plain'))
                
                self.email_server.send_message(msg)
                logger.info(f"Email alert sent: {alert_type}")
                return True
            except Exception as e:
                logger.error(f"Failed to send email alert: {e}")
                # Try to reconnect
                self._connect_smtp()
                return False
    
    def send_alert(self, message: str, alert_type: str = "training_alert", 
                   subject: Optional[str] = None) -> None:
        """Send alert through all enabled channels"""
        if subject is None:
            subject = alert_type.replace("_", " ").title()
        
        # Send to Slack
        if self.config.slack_enabled:
            self.send_slack_alert(message, alert_type)
        
        # Send email
        if self.config.email_enabled:
            self.send_email_alert(subject, message, alert_type)
        
        # Log locally
        logger.warning(f"ALERT [{alert_type}]: {message}")
    
    def close(self):
        """Clean up resources"""
        if self.email_server:
            try:
                self.email_server.quit()
            except:
                pass


class AnomalyDetector:
    """Detects anomalies in training metrics"""
    
    def __init__(self, config: AlertConfig, notifier: Notifier):
        self.config = config
        self.notifier = notifier
        self.loss_history: List[float] = []
        self.last_progress_time = time.time()
        self.gradient_norms: List[float] = []
        self.gpu_stats_history: List[Dict] = []
        
    def update_loss(self, loss: float) -> None:
        """Update loss history and check for spikes"""
        self.loss_history.append(loss)
        
        # Keep last 100 values
        if len(self.loss_history) > 100:
            self.loss_history = self.loss_history[-100:]
        
        # Check for loss spike
        if len(self.loss_history) >= 10:
            recent_avg = np.mean(self.loss_history[-10:])
            if recent_avg > 0 and loss / recent_avg > self.config.loss_spike_threshold:
                self.notifier.send_alert(
                    f"Loss spike detected: {loss:.4f} (avg: {recent_avg:.4f}, ratio: {loss/recent_avg:.2f}x)",
                    "loss_spike"
                )
    
    def update_gradient_norm(self, grad_norm: float) -> None:
        """Update gradient norm history and check for explosions"""
        self.gradient_norms.append(grad_norm)
        
        # Keep last 50 values
        if len(self.gradient_norms) > 50:
            self.gradient_norms = self.gradient_norms[-50:]
        
        # Check for gradient explosion
        if grad_norm > self.config.gradient_norm_threshold:
            self.notifier.send_alert(
                f"Gradient explosion detected: {grad_norm:.4f} (threshold: {self.config.gradient_norm_threshold})",
                "gradient_explosion"
            )
    
    def update_gpu_stats(self, gpu_stats: Dict) -> None:
        """Update GPU statistics and check for issues"""
        self.gpu_stats_history.append(gpu_stats)
        
        # Keep last 20 values
        if len(self.gpu_stats_history) > 20:
            self.gpu_stats_history = self.gpu_stats_history[-20:]
        
        # Check GPU memory
        if gpu_stats.get("memory_util", 0) > self.config.gpu_memory_threshold:
            self.notifier.send_alert(
                f"High GPU memory usage: {gpu_stats['memory_util']*100:.1f}%",
                "gpu_memory_high"
            )
        
        # Check GPU temperature
        if gpu_stats.get("temperature", 0) > self.config.gpu_temperature_threshold:
            self.notifier.send_alert(
                f"High GPU temperature: {gpu_stats['temperature']}°C",
                "gpu_temperature_high"
            )
    
    def check_training_stall(self) -> None:
        """Check if training has stalled"""
        if time.time() - self.last_progress_time > self.config.training_stall_seconds:
            self.notifier.send_alert(
                f"Training appears stalled for {self.config.training_stall_seconds} seconds",
                "training_stall"
            )
            self.last_progress_time = time.time()  # Reset timer
    
    def update_progress(self) -> None:
        """Update last progress time"""
        self.last_progress_time = time.time()


class TrainingDashboard:
    """Real-time training dashboard with Plotly charts"""
    
    def __init__(self, config: Optional[AlertConfig] = None):
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio is required for the dashboard. Install with: pip install gradio")
        
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for the dashboard. Install with: pip install plotly")
        
        self.config = config or AlertConfig.from_env()
        self.notifier = Notifier(self.config)
        self.anomaly_detector = AnomalyDetector(self.config, self.notifier)
        
        # Data storage
        self.metrics_history: Dict[str, List[Dict]] = {
            "loss": [],
            "learning_rate": [],
            "gradient_norm": [],
            "gpu_utilization": [],
            "gpu_memory": [],
            "gpu_temperature": [],
            "throughput": []
        }
        
        self.current_step = 0
        self.start_time = time.time()
        self.last_update_time = time.time()
        
        # Thread-safe queue for updates
        self.update_queue = queue.Queue()
        
        # Dashboard state
        self.dashboard_running = False
        self.dashboard_thread = None
        
        logger.info("TrainingDashboard initialized")
    
    def update_metrics(self, logs: Dict[str, Any]) -> None:
        """Update metrics from training logs (thread-safe)"""
        self.update_queue.put(logs)
    
    def _process_updates(self) -> None:
        """Process updates from the queue"""
        while not self.update_queue.empty():
            try:
                logs = self.update_queue.get_nowait()
                self._process_logs(logs)
            except queue.Empty:
                break
    
    def _process_logs(self, logs: Dict[str, Any]) -> None:
        """Process training logs and update metrics"""
        timestamp = datetime.now().isoformat()
        self.current_step = logs.get("step", self.current_step)
        
        # Extract and store metrics
        if "loss" in logs:
            loss = logs["loss"]
            self.metrics_history["loss"].append({
                "step": self.current_step,
                "value": loss,
                "timestamp": timestamp
            })
            self.anomaly_detector.update_loss(loss)
        
        if "learning_rate" in logs:
            lr = logs["learning_rate"]
            self.metrics_history["learning_rate"].append({
                "step": self.current_step,
                "value": lr,
                "timestamp": timestamp
            })
        
        if "grad_norm" in logs:
            grad_norm = logs["grad_norm"]
            self.metrics_history["gradient_norm"].append({
                "step": self.current_step,
                "value": grad_norm,
                "timestamp": timestamp
            })
            self.anomaly_detector.update_gradient_norm(grad_norm)
        
        # Update GPU stats if available
        if GPU_MONITORING:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Monitor first GPU
                    gpu_stats = {
                        "utilization": gpu.load * 100,
                        "memory_util": gpu.memoryUtil,
                        "temperature": gpu.temperature
                    }
                    
                    self.metrics_history["gpu_utilization"].append({
                        "step": self.current_step,
                        "value": gpu_stats["utilization"],
                        "timestamp": timestamp
                    })
                    
                    self.metrics_history["gpu_memory"].append({
                        "step": self.current_step,
                        "value": gpu_stats["memory_util"] * 100,
                        "timestamp": timestamp
                    })
                    
                    self.metrics_history["gpu_temperature"].append({
                        "step": self.current_step,
                        "value": gpu_stats["temperature"],
                        "timestamp": timestamp
                    })
                    
                    self.anomaly_detector.update_gpu_stats(gpu_stats)
            except Exception as e:
                logger.debug(f"Failed to get GPU stats: {e}")
        
        # Calculate throughput
        current_time = time.time()
        time_elapsed = current_time - self.last_update_time
        if time_elapsed > 0:
            steps_per_second = 1 / time_elapsed if time_elapsed > 0 else 0
            self.metrics_history["throughput"].append({
                "step": self.current_step,
                "value": steps_per_second,
                "timestamp": timestamp
            })
        
        self.last_update_time = current_time
        self.anomaly_detector.update_progress()
        
        # Keep history manageable (last 1000 points per metric)
        for metric in self.metrics_history:
            if len(self.metrics_history[metric]) > 1000:
                self.metrics_history[metric] = self.metrics_history[metric][-1000:]
    
    def _create_plot(self, metric_name: str, title: str, y_label: str) -> go.Figure:
        """Create a Plotly figure for a metric"""
        data = self.metrics_history.get(metric_name, [])
        
        if not data:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title=title, height=300)
            return fig
        
        steps = [d["step"] for d in data]
        values = [d["value"] for d in data]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=steps,
            y=values,
            mode='lines+markers',
            name=metric_name,
            line=dict(width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Step",
            yaxis_title=y_label,
            height=300,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    def _create_dashboard(self) -> gr.Blocks:
        """Create the Gradio dashboard interface"""
        with gr.Blocks(title="crucible Training Dashboard", theme=gr.themes.Soft()) as dashboard:
            gr.Markdown("# 🚀 crucible Training Dashboard")
            gr.Markdown(f"**Current Step:** {self.current_step} | **Training Time:** {self._format_time(time.time() - self.start_time)}")
            
            with gr.Tabs():
                with gr.TabItem("📈 Training Metrics"):
                    with gr.Row():
                        with gr.Column():
                            loss_plot = gr.Plot(label="Loss")
                        with gr.Column():
                            lr_plot = gr.Plot(label="Learning Rate")
                    
                    with gr.Row():
                        with gr.Column():
                            grad_norm_plot = gr.Plot(label="Gradient Norm")
                        with gr.Column():
                            throughput_plot = gr.Plot(label="Throughput (steps/sec)")
                
                with gr.TabItem("🖥️ GPU Monitoring"):
                    with gr.Row():
                        with gr.Column():
                            gpu_util_plot = gr.Plot(label="GPU Utilization (%)")
                        with gr.Column():
                            gpu_memory_plot = gr.Plot(label="GPU Memory (%)")
                    
                    with gr.Row():
                        with gr.Column():
                            gpu_temp_plot = gr.Plot(label="GPU Temperature (°C)")
                        with gr.Column():
                            # Placeholder for additional GPU info
                            gpu_info = gr.JSON(label="GPU Information")
                
                with gr.TabItem("⚠️ Alerts & Status"):
                    with gr.Row():
                        with gr.Column():
                            alert_config = gr.JSON(
                                value=asdict(self.config),
                                label="Alert Configuration"
                            )
                        with gr.Column():
                            system_status = gr.JSON(
                                value=self._get_system_status(),
                                label="System Status"
                            )
                    
                    gr.Markdown("### Recent Alerts")
                    alerts_log = gr.Textbox(
                        label="Alert Log",
                        value=self._get_recent_alerts(),
                        interactive=False,
                        lines=5
                    )
            
            # Auto-refresh components
            dashboard.load(
                fn=self._update_dashboard,
                outputs=[loss_plot, lr_plot, grad_norm_plot, throughput_plot,
                        gpu_util_plot, gpu_memory_plot, gpu_temp_plot, gpu_info,
                        system_status, alerts_log]
            )
            
            # Refresh every 2 seconds
            gr.Timer(2).tick(
                fn=self._update_dashboard,
                outputs=[loss_plot, lr_plot, grad_norm_plot, throughput_plot,
                        gpu_util_plot, gpu_memory_plot, gpu_temp_plot, gpu_info,
                        system_status, alerts_log]
            )
        
        return dashboard
    
    def _update_dashboard(self) -> tuple:
        """Update all dashboard components"""
        self._process_updates()
        self.anomaly_detector.check_training_stall()
        
        # Update plots
        loss_plot = self._create_plot("loss", "Training Loss", "Loss")
        lr_plot = self._create_plot("learning_rate", "Learning Rate", "LR")
        grad_norm_plot = self._create_plot("gradient_norm", "Gradient Norm", "Norm")
        throughput_plot = self._create_plot("throughput", "Training Throughput", "Steps/sec")
        
        # GPU plots
        gpu_util_plot = self._create_plot("gpu_utilization", "GPU Utilization", "%")
        gpu_memory_plot = self._create_plot("gpu_memory", "GPU Memory Usage", "%")
        gpu_temp_plot = self._create_plot("gpu_temperature", "GPU Temperature", "°C")
        
        # System info
        gpu_info = self._get_gpu_info()
        system_status = self._get_system_status()
        alerts_log = self._get_recent_alerts()
        
        return (loss_plot, lr_plot, grad_norm_plot, throughput_plot,
                gpu_util_plot, gpu_memory_plot, gpu_temp_plot, gpu_info,
                system_status, alerts_log)
    
    def _get_gpu_info(self) -> Dict:
        """Get current GPU information"""
        if not GPU_MONITORING:
            return {"error": "GPU monitoring not available"}
        
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return {"error": "No GPUs detected"}
            
            gpu = gpus[0]
            return {
                "name": gpu.name,
                "load": f"{gpu.load * 100:.1f}%",
                "memory_used": f"{gpu.memoryUsed}MB",
                "memory_total": f"{gpu.memoryTotal}MB",
                "memory_util": f"{gpu.memoryUtil * 100:.1f}%",
                "temperature": f"{gpu.temperature}°C"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_system_status(self) -> Dict:
        """Get current system status"""
        status = {
            "current_step": self.current_step,
            "training_time": self._format_time(time.time() - self.start_time),
            "metrics_collected": {k: len(v) for k, v in self.metrics_history.items()},
            "dashboard_running": self.dashboard_running
        }
        
        # Add CPU and memory info
        try:
            status["cpu_percent"] = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            status["memory_used"] = f"{memory.used / (1024**3):.1f}GB"
            status["memory_total"] = f"{memory.total / (1024**3):.1f}GB"
            status["memory_percent"] = memory.percent
        except:
            pass
        
        return status
    
    def _get_recent_alerts(self) -> str:
        """Get recent alerts as formatted string"""
        # This would be populated from a log file or in-memory buffer
        # For now, return a placeholder
        return "No recent alerts"
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def start(self, port: int = 7860, share: bool = False) -> None:
        """Start the dashboard server"""
        if not GRADIO_AVAILABLE:
            logger.error("Cannot start dashboard: Gradio not installed")
            return
        
        if self.dashboard_running:
            logger.warning("Dashboard already running")
            return
        
        def run_dashboard():
            try:
                dashboard = self._create_dashboard()
                logger.info(f"Starting training dashboard on port {port}")
                dashboard.launch(
                    server_port=port,
                    share=share,
                    prevent_thread_lock=True,
                    show_error=True
                )
                self.dashboard_running = True
                
                # Keep the dashboard alive
                while self.dashboard_running:
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Dashboard failed: {e}")
            finally:
                self.dashboard_running = False
        
        self.dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        self.dashboard_thread.start()
        
        # Wait a moment for the server to start
        time.sleep(2)
        logger.info(f"Dashboard available at http://localhost:{port}")
    
    def stop(self) -> None:
        """Stop the dashboard server"""
        self.dashboard_running = False
        if self.dashboard_thread:
            self.dashboard_thread.join(timeout=5)
        logger.info("Dashboard stopped")
    
    def close(self) -> None:
        """Clean up resources"""
        self.stop()
        self.notifier.close()


class DashboardCallback:
    """
    Callback to integrate dashboard with crucible training
    
    This callback:
    1. Streams training metrics to the dashboard
    2. Triggers alerts for anomalies
    3. Can be launched with --dashboard flag
    """
    
    def __init__(self, dashboard: Optional[TrainingDashboard] = None, 
                 config: Optional[AlertConfig] = None):
        self.dashboard = dashboard
        self.config = config or AlertConfig.from_env()
        
        # Initialize dashboard if not provided
        if self.dashboard is None and GRADIO_AVAILABLE:
            self.dashboard = TrainingDashboard(self.config)
        
        self.enabled = self.dashboard is not None
        self.last_log_time = time.time()
        
        if self.enabled:
            logger.info("DashboardCallback initialized")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training"""
        if self.enabled:
            logger.info("Training started - dashboard active")
            # Start dashboard if not already running
            if not self.dashboard.dashboard_running:
                port = getattr(args, 'dashboard_port', 7860)
                share = getattr(args, 'dashboard_share', False)
                self.dashboard.start(port=port, share=share)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs"""
        if not self.enabled or logs is None:
            return
        
        # Add step information
        logs["step"] = state.global_step
        
        # Update dashboard
        self.dashboard.update_metrics(logs)
        
        # Throttle dashboard updates to avoid overwhelming
        current_time = time.time()
        if current_time - self.last_log_time < 1.0:  # Max 1 update per second
            return
        
        self.last_log_time = current_time
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training"""
        if self.enabled:
            logger.info("Training ended - stopping dashboard")
            self.dashboard.stop()
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each step"""
        # Update throughput calculation
        if self.enabled:
            current_time = time.time()
            self.dashboard.update_metrics({
                "step": state.global_step,
                "timestamp": datetime.now().isoformat()
            })


# Factory function for easy integration
def create_dashboard_callback(
    enable_slack: bool = False,
    enable_email: bool = False,
    dashboard_port: int = 7860,
    **alert_kwargs
) -> DashboardCallback:
    """
    Factory function to create a dashboard callback with common configurations
    
    Args:
        enable_slack: Enable Slack alerts
        enable_email: Enable email alerts
        dashboard_port: Port for the dashboard server
        **alert_kwargs: Additional alert configuration parameters
    
    Returns:
        Configured DashboardCallback instance
    """
    config = AlertConfig.from_env()
    config.slack_enabled = enable_slack or config.slack_enabled
    config.email_enabled = enable_email or config.email_enabled
    
    # Update config with any provided kwargs
    for key, value in alert_kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    dashboard = TrainingDashboard(config)
    return DashboardCallback(dashboard=dashboard, config=config)


# Command-line interface for standalone dashboard
def main():
    """Main entry point for standalone dashboard"""
    import argparse
    
    parser = argparse.ArgumentParser(description="crucible Training Dashboard")
    parser.add_argument("--port", type=int, default=7860, help="Dashboard port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--enable-slack", action="store_true", help="Enable Slack alerts")
    parser.add_argument("--enable-email", action="store_true", help="Enable email alerts")
    
    args = parser.parse_args()
    
    # Create and start dashboard
    callback = create_dashboard_callback(
        enable_slack=args.enable_slack,
        enable_email=args.enable_email,
        dashboard_port=args.port
    )
    
    try:
        print(f"Starting dashboard on port {args.port}...")
        print("Press Ctrl+C to stop")
        
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
    finally:
        callback.dashboard.close()


if __name__ == "__main__":
    main()