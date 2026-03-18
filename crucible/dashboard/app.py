"""
Real-Time Training Dashboard & Alerting for crucible
Extends LogCallback to provide live visualization and anomaly detection
"""

import os
import sys
import time
import json
import threading
import queue
import logging
import smtplib
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Try to import Gradio with fallback
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Warning: Gradio not installed. Dashboard will run in limited mode.")

# Try to import pynvml for GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False

# Import from existing crucible modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crucible.extras.callbacks import LogCallback
from crucible.extras.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AlertConfig:
    """Configuration for alert system"""
    slack_webhook_url: Optional[str] = None
    email_smtp_server: Optional[str] = None
    email_smtp_port: int = 587
    email_sender: Optional[str] = None
    email_password: Optional[str] = None
    email_recipients: List[str] = None
    loss_spike_threshold: float = 2.0  # Loss increase factor
    gpu_memory_threshold: float = 0.95  # GPU memory usage threshold
    gpu_temp_threshold: float = 85.0  # GPU temperature threshold
    gradient_norm_threshold: float = 100.0  # Gradient norm threshold
    check_interval: int = 60  # Seconds between checks
    cooldown_period: int = 300  # Seconds between repeated alerts


class MetricsBuffer:
    """Thread-safe buffer for storing training metrics"""
    
    def __init__(self, maxlen: int = 10000):
        self.metrics = {
            'loss': deque(maxlen=maxlen),
            'learning_rate': deque(maxlen=maxlen),
            'epoch': deque(maxlen=maxlen),
            'step': deque(maxlen=maxlen),
            'grad_norm': deque(maxlen=maxlen),
            'gpu_memory': deque(maxlen=maxlen),
            'gpu_utilization': deque(maxlen=maxlen),
            'gpu_temperature': deque(maxlen=maxlen),
            'timestamps': deque(maxlen=maxlen)
        }
        self.lock = threading.Lock()
        self.last_alert_time = {}
    
    def add_metrics(self, logs: Dict[str, Any]):
        """Add new metrics from training logs"""
        with self.lock:
            timestamp = datetime.now().isoformat()
            
            # Extract standard metrics
            if 'loss' in logs:
                self.metrics['loss'].append(logs['loss'])
            if 'learning_rate' in logs:
                self.metrics['learning_rate'].append(logs['learning_rate'])
            if 'epoch' in logs:
                self.metrics['epoch'].append(logs['epoch'])
            if 'step' in logs:
                self.metrics['step'].append(logs['step'])
            if 'grad_norm' in logs:
                self.metrics['grad_norm'].append(logs['grad_norm'])
            
            # Get GPU metrics if available
            if GPU_AVAILABLE:
                try:
                    gpu_metrics = self._get_gpu_metrics()
                    self.metrics['gpu_memory'].append(gpu_metrics['memory_used_percent'])
                    self.metrics['gpu_utilization'].append(gpu_metrics['gpu_utilization'])
                    self.metrics['gpu_temperature'].append(gpu_metrics['temperature'])
                except:
                    pass
            
            self.metrics['timestamps'].append(timestamp)
    
    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get current GPU metrics"""
        metrics = {}
        device_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics['memory_used_percent'] = mem_info.used / mem_info.total
            
            # Utilization rates
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            metrics['gpu_utilization'] = util.gpu / 100.0
            
            # Temperature
            metrics['temperature'] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Power usage (optional)
            try:
                metrics['power_usage'] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            except:
                metrics['power_usage'] = 0.0
        
        return metrics
    
    def get_recent_metrics(self, n: int = 100) -> Dict[str, List]:
        """Get n most recent metrics"""
        with self.lock:
            result = {}
            for key, values in self.metrics.items():
                result[key] = list(values)[-n:]
            return result
    
    def get_dataframe(self, n: int = 1000) -> pd.DataFrame:
        """Convert metrics to pandas DataFrame"""
        metrics = self.get_recent_metrics(n)
        
        # Ensure all lists have the same length
        min_len = min(len(v) for v in metrics.values() if v)
        if min_len == 0:
            return pd.DataFrame()
        
        data = {}
        for key, values in metrics.items():
            if values:
                data[key] = values[-min_len:]
        
        df = pd.DataFrame(data)
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
        return df


class AlertManager:
    """Manages alerting for training anomalies"""
    
    def __init__(self, config: AlertConfig, metrics_buffer: MetricsBuffer):
        self.config = config
        self.metrics_buffer = metrics_buffer
        self.alert_history = {}
        self.alert_thread = None
        self.stop_event = threading.Event()
    
    def start(self):
        """Start alert monitoring thread"""
        if self.config.slack_webhook_url or self.config.email_smtp_server:
            self.alert_thread = threading.Thread(target=self._alert_loop, daemon=True)
            self.alert_thread.start()
            logger.info("Alert monitoring started")
    
    def stop(self):
        """Stop alert monitoring"""
        self.stop_event.set()
        if self.alert_thread:
            self.alert_thread.join(timeout=5)
    
    def _alert_loop(self):
        """Main alert checking loop"""
        while not self.stop_event.is_set():
            try:
                self._check_anomalies()
            except Exception as e:
                logger.error(f"Error in alert loop: {e}")
            
            time.sleep(self.config.check_interval)
    
    def _check_anomalies(self):
        """Check for training anomalies"""
        metrics = self.metrics_buffer.get_recent_metrics(10)
        
        # Check for loss spikes
        if len(metrics['loss']) >= 2:
            recent_losses = [l for l in metrics['loss'] if l is not None]
            if len(recent_losses) >= 2:
                current_loss = recent_losses[-1]
                prev_loss = recent_losses[-2]
                
                if prev_loss > 0 and current_loss / prev_loss > self.config.loss_spike_threshold:
                    self._send_alert(
                        "Loss Spike Detected",
                        f"Loss increased from {prev_loss:.4f} to {current_loss:.4f} "
                        f"({current_loss/prev_loss:.2f}x increase)",
                        severity="warning"
                    )
        
        # Check for gradient explosion
        if metrics['grad_norm']:
            recent_grads = [g for g in metrics['grad_norm'] if g is not None]
            if recent_grads and max(recent_grads) > self.config.gradient_norm_threshold:
                self._send_alert(
                    "Gradient Explosion",
                    f"Gradient norm reached {max(recent_grads):.2f} "
                    f"(threshold: {self.config.gradient_norm_threshold})",
                    severity="critical"
                )
        
        # Check GPU metrics
        if GPU_AVAILABLE and metrics['gpu_memory']:
            recent_mem = [m for m in metrics['gpu_memory'] if m is not None]
            if recent_mem and max(recent_mem) > self.config.gpu_memory_threshold:
                self._send_alert(
                    "GPU Memory High",
                    f"GPU memory usage reached {max(recent_mem)*100:.1f}% "
                    f"(threshold: {self.config.gpu_memory_threshold*100:.1f}%)",
                    severity="warning"
                )
        
        if GPU_AVAILABLE and metrics['gpu_temperature']:
            recent_temps = [t for t in metrics['gpu_temperature'] if t is not None]
            if recent_temps and max(recent_temps) > self.config.gpu_temp_threshold:
                self._send_alert(
                    "GPU Temperature High",
                    f"GPU temperature reached {max(recent_temps):.1f}°C "
                    f"(threshold: {self.config.gpu_temp_threshold}°C)",
                    severity="critical"
                )
    
    def _send_alert(self, title: str, message: str, severity: str = "info"):
        """Send alert via configured channels"""
        alert_key = f"{title}:{message}"
        
        # Check cooldown
        if alert_key in self.alert_history:
            last_sent = self.alert_history[alert_key]
            if time.time() - last_sent < self.config.cooldown_period:
                return
        
        # Format alert
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] [{severity.upper()}] {title}\n{message}"
        
        # Send via Slack
        if self.config.slack_webhook_url:
            self._send_slack_alert(formatted_message, severity)
        
        # Send via email
        if self.config.email_smtp_server and self.config.email_recipients:
            self._send_email_alert(title, formatted_message, severity)
        
        # Log alert
        logger.warning(f"Alert sent: {formatted_message}")
        
        # Update history
        self.alert_history[alert_key] = time.time()
    
    def _send_slack_alert(self, message: str, severity: str):
        """Send alert to Slack"""
        try:
            import requests
            
            color_map = {
                "info": "#36a64f",
                "warning": "#ff9900",
                "critical": "#ff0000"
            }
            
            payload = {
                "attachments": [{
                    "color": color_map.get(severity, "#36a64f"),
                    "text": message,
                    "mrkdwn_in": ["text"]
                }]
            }
            
            response = requests.post(
                self.config.slack_webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to send Slack alert: {response.text}")
        
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
    
    def _send_email_alert(self, subject: str, message: str, severity: str):
        """Send alert via email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.email_sender
            msg['To'] = ', '.join(self.config.email_recipients)
            msg['Subject'] = f"[crucible {severity.upper()}] {subject}"
            
            msg.attach(MIMEText(message, 'plain'))
            
            with smtplib.SMTP(self.config.email_smtp_server, self.config.email_smtp_port) as server:
                server.starttls()
                if self.config.email_password:
                    server.login(self.config.email_sender, self.config.email_password)
                server.send_message(msg)
        
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")


class DashboardCallback(LogCallback):
    """Extended callback for real-time training dashboard"""
    
    def __init__(
        self,
        alert_config: Optional[AlertConfig] = None,
        port: int = 7860,
        share: bool = False,
        update_interval: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.metrics_buffer = MetricsBuffer()
        self.alert_config = alert_config or AlertConfig()
        self.alert_manager = AlertManager(self.alert_config, self.metrics_buffer)
        
        self.port = port
        self.share = share
        self.update_interval = update_interval
        
        self.dashboard_thread = None
        self.stop_event = threading.Event()
        self.is_training = False
        
        # Initialize dashboard if Gradio is available
        if GRADIO_AVAILABLE:
            self._init_dashboard()
        else:
            logger.warning("Gradio not available. Dashboard disabled.")
    
    def _init_dashboard(self):
        """Initialize Gradio dashboard"""
        # Create Gradio interface
        with gr.Blocks(
            title="crucible Training Dashboard",
            theme=gr.themes.Soft(),
            css="""
            .plot-container { height: 400px !important; }
            .alert-box { 
                background-color: #ffebee; 
                border-left: 4px solid #f44336; 
                padding: 10px; 
                margin: 10px 0;
            }
            """
        ) as self.demo:
            gr.Markdown("# 🦙 crucible Training Dashboard")
            gr.Markdown("Real-time monitoring of training metrics and GPU utilization")
            
            with gr.Tabs():
                with gr.TabItem("📈 Training Metrics"):
                    with gr.Row():
                        with gr.Column():
                            self.loss_plot = gr.Plot(label="Loss", show_label=True)
                        with gr.Column():
                            self.lr_plot = gr.Plot(label="Learning Rate", show_label=True)
                    
                    with gr.Row():
                        with gr.Column():
                            self.grad_norm_plot = gr.Plot(label="Gradient Norm", show_label=True)
                        with gr.Column():
                            self.epoch_step_plot = gr.Plot(label="Epoch/Step Progress", show_label=True)
                
                with gr.TabItem("🖥️ GPU Monitoring"):
                    if GPU_AVAILABLE:
                        with gr.Row():
                            with gr.Column():
                                self.gpu_memory_plot = gr.Plot(label="GPU Memory Usage", show_label=True)
                            with gr.Column():
                                self.gpu_util_plot = gr.Plot(label="GPU Utilization", show_label=True)
                        
                        with gr.Row():
                            with gr.Column():
                                self.gpu_temp_plot = gr.Plot(label="GPU Temperature", show_label=True)
                            with gr.Column():
                                self.gpu_power_plot = gr.Plot(label="GPU Power Usage", show_label=True)
                    else:
                        gr.Markdown("⚠️ GPU monitoring unavailable. Install pynvml for GPU metrics.")
                
                with gr.TabItem("🚨 Alerts & Status"):
                    with gr.Row():
                        with gr.Column():
                            self.status_display = gr.JSON(label="Training Status")
                        with gr.Column():
                            self.alerts_display = gr.Markdown(label="Recent Alerts")
                    
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Refresh Now", variant="primary")
                        export_btn = gr.Button("💾 Export Metrics")
            
            # Set up event handlers
            refresh_btn.click(
                fn=self._update_dashboard,
                outputs=[
                    self.loss_plot,
                    self.lr_plot,
                    self.grad_norm_plot,
                    self.epoch_step_plot,
                    self.gpu_memory_plot if GPU_AVAILABLE else gr.Plot(visible=False),
                    self.gpu_util_plot if GPU_AVAILABLE else gr.Plot(visible=False),
                    self.gpu_temp_plot if GPU_AVAILABLE else gr.Plot(visible=False),
                    self.gpu_power_plot if GPU_AVAILABLE else gr.Plot(visible=False),
                    self.status_display,
                    self.alerts_display
                ]
            )
            
            export_btn.click(
                fn=self._export_metrics,
                outputs=gr.File(label="Download Metrics")
            )
            
            # Auto-update every update_interval seconds
            self.demo.load(
                fn=self._update_dashboard,
                outputs=[
                    self.loss_plot,
                    self.lr_plot,
                    self.grad_norm_plot,
                    self.epoch_step_plot,
                    self.gpu_memory_plot if GPU_AVAILABLE else gr.Plot(visible=False),
                    self.gpu_util_plot if GPU_AVAILABLE else gr.Plot(visible=False),
                    self.gpu_temp_plot if GPU_AVAILABLE else gr.Plot(visible=False),
                    self.gpu_power_plot if GPU_AVAILABLE else gr.Plot(visible=False),
                    self.status_display,
                    self.alerts_display
                ],
                every=self.update_interval
            )
    
    def _create_loss_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create loss visualization"""
        if df.empty or 'loss' not in df.columns:
            return go.Figure().update_layout(title="No loss data available")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Training Loss', 'Loss Distribution'),
            horizontal_spacing=0.1
        )
        
        # Loss over time
        fig.add_trace(
            go.Scatter(
                x=df['step'] if 'step' in df.columns else df.index,
                y=df['loss'],
                mode='lines+markers',
                name='Loss',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # Add smoothed loss if enough data
        if len(df) > 10:
            smoothed = df['loss'].rolling(window=min(10, len(df)//2), center=True).mean()
            fig.add_trace(
                go.Scatter(
                    x=df['step'] if 'step' in df.columns else df.index,
                    y=smoothed,
                    mode='lines',
                    name='Smoothed Loss',
                    line=dict(color='#ff7f0e', width=3, dash='dash')
                ),
                row=1, col=1
            )
        
        # Loss distribution
        fig.add_trace(
            go.Histogram(
                x=df['loss'],
                nbinsx=30,
                name='Loss Distribution',
                marker_color='#2ca02c',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def _create_lr_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create learning rate visualization"""
        if df.empty or 'learning_rate' not in df.columns:
            return go.Figure().update_layout(title="No learning rate data available")
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df['step'] if 'step' in df.columns else df.index,
                y=df['learning_rate'],
                mode='lines',
                name='Learning Rate',
                line=dict(color='#9467bd', width=2),
                fill='tozeroy',
                fillcolor='rgba(148, 103, 189, 0.2)'
            )
        )
        
        fig.update_layout(
            title='Learning Rate Schedule',
            xaxis_title='Step',
            yaxis_title='Learning Rate',
            height=400,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def _create_grad_norm_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create gradient norm visualization"""
        if df.empty or 'grad_norm' not in df.columns:
            return go.Figure().update_layout(title="No gradient norm data available")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Gradient Norm Over Time', 'Gradient Norm Distribution'),
            horizontal_spacing=0.1
        )
        
        # Gradient norm over time
        fig.add_trace(
            go.Scatter(
                x=df['step'] if 'step' in df.columns else df.index,
                y=df['grad_norm'],
                mode='lines+markers',
                name='Gradient Norm',
                line=dict(color='#d62728', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # Add threshold line
        if self.alert_config.gradient_norm_threshold:
            fig.add_hline(
                y=self.alert_config.gradient_norm_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text="Alert Threshold",
                annotation_position="top left",
                row=1, col=1
            )
        
        # Gradient norm distribution
        fig.add_trace(
            go.Histogram(
                x=df['grad_norm'],
                nbinsx=30,
                name='Gradient Norm Distribution',
                marker_color='#e377c2',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def _create_epoch_step_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create epoch/step progress visualization"""
        if df.empty:
            return go.Figure().update_layout(title="No progress data available")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Training Progress', 'Steps per Epoch'),
            horizontal_spacing=0.1
        )
        
        # Training progress
        if 'epoch' in df.columns and 'step' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['step'],
                    y=df['epoch'],
                    mode='lines',
                    name='Progress',
                    line=dict(color='#17becf', width=2)
                ),
                row=1, col=1
            )
            
            # Calculate steps per epoch
            if len(df) > 1:
                epoch_changes = df['epoch'].diff().fillna(0)
                step_changes = df['step'].diff().fillna(1)
                steps_per_epoch = step_changes[epoch_changes > 0].mean()
                
                if not np.isnan(steps_per_epoch):
                    fig.add_trace(
                        go.Indicator(
                            mode="number+gauge",
                            value=steps_per_epoch,
                            title={"text": "Avg Steps/Epoch"},
                            gauge={'axis': {'range': [0, steps_per_epoch * 2]}},
                            domain={'row': 0, 'column': 1}
                        ),
                        row=1, col=2
                    )
        
        fig.update_layout(
            height=400,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def _create_gpu_plots(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create GPU monitoring plots"""
        plots = {}
        
        if not GPU_AVAILABLE or df.empty:
            return {
                'gpu_memory': go.Figure().update_layout(title="GPU monitoring unavailable"),
                'gpu_util': go.Figure().update_layout(title="GPU monitoring unavailable"),
                'gpu_temp': go.Figure().update_layout(title="GPU monitoring unavailable"),
                'gpu_power': go.Figure().update_layout(title="GPU monitoring unavailable")
            }
        
        # GPU Memory
        if 'gpu_memory' in df.columns:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df['timestamps'] if 'timestamps' in df.columns else df.index,
                    y=df['gpu_memory'] * 100,  # Convert to percentage
                    mode='lines',
                    name='GPU Memory',
                    line=dict(color='#8c564b', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(140, 86, 75, 0.2)'
                )
            )
            
            # Add threshold line
            fig.add_hline(
                y=self.alert_config.gpu_memory_threshold * 100,
                line_dash="dash",
                line_color="red",
                annotation_text="Alert Threshold"
            )
            
            fig.update_layout(
                title='GPU Memory Usage',
                yaxis_title='Usage (%)',
                yaxis_range=[0, 100],
                height=400,
                hovermode='x unified',
                template='plotly_white'
            )
            plots['gpu_memory'] = fig
        
        # GPU Utilization
        if 'gpu_utilization' in df.columns:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df['timestamps'] if 'timestamps' in df.columns else df.index,
                    y=df['gpu_utilization'] * 100,
                    mode='lines',
                    name='GPU Utilization',
                    line=dict(color='#7f7f7f', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(127, 127, 127, 0.2)'
                )
            )
            
            fig.update_layout(
                title='GPU Utilization',
                yaxis_title='Utilization (%)',
                yaxis_range=[0, 100],
                height=400,
                hovermode='x unified',
                template='plotly_white'
            )
            plots['gpu_util'] = fig
        
        # GPU Temperature
        if 'gpu_temperature' in df.columns:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df['timestamps'] if 'timestamps' in df.columns else df.index,
                    y=df['gpu_temperature'],
                    mode='lines',
                    name='GPU Temperature',
                    line=dict(color='#bcbd22', width=2)
                )
            )
            
            # Add threshold line
            fig.add_hline(
                y=self.alert_config.gpu_temp_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text="Alert Threshold"
            )
            
            fig.update_layout(
                title='GPU Temperature',
                yaxis_title='Temperature (°C)',
                height=400,
                hovermode='x unified',
                template='plotly_white'
            )
            plots['gpu_temp'] = fig
        
        return plots
    
    def _update_dashboard(self):
        """Update all dashboard plots and displays"""
        try:
            df = self.metrics_buffer.get_dataframe(500)
            
            # Create plots
            loss_plot = self._create_loss_plot(df)
            lr_plot = self._create_lr_plot(df)
            grad_norm_plot = self._create_grad_norm_plot(df)
            epoch_step_plot = self._create_epoch_step_plot(df)
            
            # Create GPU plots
            gpu_plots = self._create_gpu_plots(df)
            
            # Create status display
            status = {
                "training_active": self.is_training,
                "total_steps": len(df) if not df.empty else 0,
                "current_loss": float(df['loss'].iloc[-1]) if not df.empty and 'loss' in df.columns else None,
                "current_lr": float(df['learning_rate'].iloc[-1]) if not df.empty and 'learning_rate' in df.columns else None,
                "gpu_available": GPU_AVAILABLE,
                "last_update": datetime.now().isoformat()
            }
            
            # Create alerts display
            alerts_text = "### Recent Alerts\n"
            if hasattr(self, 'alert_manager') and self.alert_manager.alert_history:
                for alert_key, timestamp in list(self.alert_manager.alert_history.items())[-5:]:
                    alerts_text += f"- `{alert_key}` at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}\n"
            else:
                alerts_text += "✅ No recent alerts"
            
            return (
                loss_plot,
                lr_plot,
                grad_norm_plot,
                epoch_step_plot,
                gpu_plots.get('gpu_memory', go.Figure()),
                gpu_plots.get('gpu_util', go.Figure()),
                gpu_plots.get('gpu_temp', go.Figure()),
                gpu_plots.get('gpu_power', go.Figure()),
                status,
                alerts_text
            )
        
        except Exception as e:
            logger.error(f"Error updating dashboard: {e}")
            # Return empty figures on error
            empty_fig = go.Figure()
            return (
                empty_fig, empty_fig, empty_fig, empty_fig,
                empty_fig, empty_fig, empty_fig, empty_fig,
                {"error": str(e)},
                f"❌ Dashboard update failed: {str(e)}"
            )
    
    def _export_metrics(self):
        """Export metrics to CSV file"""
        try:
            df = self.metrics_buffer.get_dataframe()
            if df.empty:
                return None
            
            # Create temporary file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crucible_metrics_{timestamp}.csv"
            filepath = os.path.join(os.getcwd(), filename)
            
            df.to_csv(filepath, index=False)
            logger.info(f"Metrics exported to {filepath}")
            
            return filepath
        
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return None
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when training logs are produced"""
        super().on_log(args, state, control, logs, **kwargs)
        
        if logs:
            # Add step/epoch info if available
            if state:
                logs['step'] = state.global_step
                logs['epoch'] = state.epoch
            
            # Add to metrics buffer
            self.metrics_buffer.add_metrics(logs)
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training"""
        super().on_train_begin(args, state, control, **kwargs)
        
        self.is_training = True
        
        # Start alert manager
        self.alert_manager.start()
        
        # Start dashboard in separate thread
        if GRADIO_AVAILABLE and not self.dashboard_thread:
            self.dashboard_thread = threading.Thread(
                target=self._run_dashboard,
                daemon=True
            )
            self.dashboard_thread.start()
            logger.info(f"Dashboard started on http://localhost:{self.port}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training"""
        super().on_train_end(args, state, control, **kwargs)
        
        self.is_training = False
        
        # Stop alert manager
        self.alert_manager.stop()
        
        # Export final metrics
        if GRADIO_AVAILABLE:
            self._export_metrics()
        
        logger.info("Training ended. Dashboard remains accessible.")
    
    def _run_dashboard(self):
        """Run the Gradio dashboard"""
        try:
            self.demo.launch(
                server_name="0.0.0.0",
                server_port=self.port,
                share=self.share,
                prevent_thread_lock=True,
                show_error=True
            )
            
            # Keep dashboard alive
            while not self.stop_event.is_set():
                time.sleep(1)
        
        except Exception as e:
            logger.error(f"Error launching dashboard: {e}")
    
    def close(self):
        """Clean up resources"""
        self.stop_event.set()
        self.alert_manager.stop()
        
        if hasattr(self, 'demo'):
            try:
                self.demo.close()
            except:
                pass


def create_dashboard_callback(
    slack_webhook_url: Optional[str] = None,
    email_smtp_server: Optional[str] = None,
    email_sender: Optional[str] = None,
    email_password: Optional[str] = None,
    email_recipients: Optional[List[str]] = None,
    port: int = 7860,
    share: bool = False,
    **kwargs
) -> DashboardCallback:
    """Factory function to create dashboard callback with configuration"""
    
    # Load config from environment if not provided
    if not slack_webhook_url:
        slack_webhook_url = os.getenv("LLAMAFACTORY_SLACK_WEBHOOK")
    
    if not email_smtp_server:
        email_smtp_server = os.getenv("LLAMAFACTORY_EMAIL_SMTP_SERVER")
    
    if not email_sender:
        email_sender = os.getenv("LLAMAFACTORY_EMAIL_SENDER")
    
    if not email_password:
        email_password = os.getenv("LLAMAFACTORY_EMAIL_PASSWORD")
    
    if not email_recipients:
        recipients_str = os.getenv("LLAMAFACTORY_EMAIL_RECIPIENTS", "")
        email_recipients = [r.strip() for r in recipients_str.split(",") if r.strip()]
    
    alert_config = AlertConfig(
        slack_webhook_url=slack_webhook_url,
        email_smtp_server=email_smtp_server,
        email_sender=email_sender,
        email_password=email_password,
        email_recipients=email_recipients,
        **kwargs
    )
    
    return DashboardCallback(
        alert_config=alert_config,
        port=port,
        share=share
    )


# Command-line interface for standalone dashboard
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="crucible Training Dashboard")
    parser.add_argument("--port", type=int, default=7860, help="Port for dashboard")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--slack-webhook", type=str, help="Slack webhook URL")
    parser.add_argument("--email-smtp", type=str, help="Email SMTP server")
    parser.add_argument("--email-sender", type=str, help="Email sender address")
    parser.add_argument("--email-password", type=str, help="Email password")
    parser.add_argument("--email-recipients", type=str, nargs="+", help="Email recipients")
    
    args = parser.parse_args()
    
    # Create and launch dashboard
    callback = create_dashboard_callback(
        slack_webhook_url=args.slack_webhook,
        email_smtp_server=args.email_smtp,
        email_sender=args.email_sender,
        email_password=args.email_password,
        email_recipients=args.email_recipients,
        port=args.port,
        share=args.share
    )
    
    print(f"Dashboard running on http://localhost:{args.port}")
    print("Press Ctrl+C to stop")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
        callback.close()
        print("Dashboard stopped")