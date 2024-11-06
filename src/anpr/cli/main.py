"""
ANPR Command Line Interface

Provides CLI commands for processing videos, querying detections,
and managing the ANPR system.
"""
import click
import asyncio
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from src.anpr.database.manager import DatabaseManager
from src.anpr.database.utils import query_detections, get_detection_stats
from src.anpr.config.config import get_config


@click.group()
@click.version_option(version='1.0.0')
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to configuration file')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), default='INFO', help='Logging level')
def cli(config: Optional[str], log_level: str):
    """ANPR System Command Line Interface."""
    import logging
    logging.basicConfig(level=getattr(logging, log_level), 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if config:
        from src.anpr.config.config import load_config
        load_config(config)


@cli.command()
@click.argument('source', type=click.STRING)
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory for snapshots')
@click.option('--no-alerts', is_flag=True, help='Disable alerts')
def live(source: str, output_dir: Optional[str], no_alerts: bool):
    """Process live RTSP stream."""
    if not source:
        click.echo("Error: Source cannot be empty", err=True)
        sys.exit(1)
    
    if output_dir:
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            click.echo(f"Error creating output directory: {e}", err=True)
            sys.exit(1)
    
    job_id = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    click.echo(f"Starting live processing for source: {source}")
    click.echo(f"Job ID: {job_id}")
    click.echo("Press Ctrl+C to stop...")
    
    # TODO: Implement actual live processing
    click.echo("Live processing not yet implemented.")


@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory for snapshots')
@click.option('--skip-frames', '-s', type=int, default=5, help='Process every Nth frame')
@click.option('--export-report', type=click.Path(), help='Export detection report to file')
def process(video_path: str, output_dir: Optional[str], skip_frames: int, export_report: Optional[str]):
    """Process video file."""
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        click.echo(f"Error: Video file does not exist: {video_path}", err=True)
        sys.exit(1)
    
    if skip_frames <= 0:
        click.echo("Error: Skip frames must be a positive integer", err=True)
        sys.exit(1)
    
    job_id = f"video_{Path(video_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    click.echo(f"Processing video: {video_path}")
    click.echo(f"Job ID: {job_id}")
    click.echo(f"Frame skip interval: {skip_frames}")
    
    # TODO: Implement actual video processing
    click.echo("Video processing not yet implemented.")
    
    if export_report:
        report_data = {
            'job_id': job_id,
            'video_path': video_path,
            'timestamp': datetime.now().isoformat(),
            'detections': []
        }
        try:
            with open(export_report, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)
            click.echo(f"Report exported to: {export_report}")
        except Exception as e:
            click.echo(f"Error exporting report: {e}", err=True)


@cli.command()
@click.option('--plate', '-p', help='Filter by plate number')
@click.option('--start-date', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', help='End date (YYYY-MM-DD)')
@click.option('--min-confidence', type=float, help='Minimum confidence threshold')
@click.option('--export', type=click.Path(), help='Export results to file')
@click.option('--limit', type=int, help='Limit number of results')
def query(plate: Optional[str], start_date: Optional[str], end_date: Optional[str],
          min_confidence: Optional[float], export: Optional[str], limit: Optional[int]):
    """Query detection database."""
    start_time = None
    end_time = None
    
    if start_date:
        try:
            start_time = datetime.strptime(start_date, '%Y-%m-%d').isoformat()
        except ValueError:
            click.echo("Invalid start date format. Use YYYY-MM-DD.", err=True)
            sys.exit(1)
    
    if end_date:
        try:
            end_time = datetime.strptime(end_date, '%Y-%m-%d').isoformat()
        except ValueError:
            click.echo("Invalid end date format. Use YYYY-MM-DD.", err=True)
            sys.exit(1)
    
    try:
        db_manager = DatabaseManager()
        
        with db_manager.get_session() as session:
            results = query_detections(
                db_session=session,
                plate_text=plate,
                start_time=start_time,
                end_time=end_time,
                min_confidence=min_confidence,
                limit=limit
            )
        
        if not results:
            click.echo("No detections found matching the criteria.")
            return
        
        click.echo(f"Found {len(results)} detection(s):")
        click.echo("-" * 80)
        
        for detection in results:
            click.echo(f"ID: {detection.id}")
            click.echo(f"Plate: {detection.plate_text}")
            click.echo(f"Confidence: {detection.confidence:.2f}")
            click.echo(f"Timestamp: {detection.timestamp}")
            click.echo("-" * 80)
        
        if export:
            export_data = [{'id': d.id, 'plate_text': d.plate_text, 'confidence': d.confidence} for d in results]
            with open(export, 'w') as f:
                json.dump(export_data, f, indent=2)
            click.echo(f"Results exported to: {export}")
    
    except Exception as e:
        click.echo(f"Error querying database: {e}", err=True)
        sys.exit(1)


@cli.command()
def status():
    """System health check."""
    click.echo("ANPR System Status:")
    
    try:
        db_manager = DatabaseManager()
        db_connected = db_manager.test_connection()
        click.echo(f"Database Connection: {'Connected' if db_connected else 'Disconnected'}")
        
        if db_connected:
            with db_manager.get_session() as session:
                stats = get_detection_stats(session)
            click.echo(f"Total Detections: {stats.get('total_detections', 0)}")
    except Exception as e:
        click.echo(f"Database: Error - {e}")
    
    try:
        config = get_config()
        click.echo("Configuration: Loaded")
    except Exception as e:
        click.echo(f"Configuration: Error - {e}")


@cli.command()
def stats():
    """Detection statistics."""
    try:
        db_manager = DatabaseManager()
        
        with db_manager.get_session() as session:
            stats = get_detection_stats(session)
        
        click.echo("Detection Statistics:")
        click.echo(f"Total Detections: {stats.get('total_detections', 0)}")
        click.echo(f"Average Confidence: {stats.get('average_confidence', 0):.2f}")
        click.echo(f"Earliest Detection: {stats.get('earliest_detection', 'N/A')}")
        click.echo(f"Latest Detection: {stats.get('latest_detection', 'N/A')}")
        
    except Exception as e:
        click.echo(f"Error retrieving statistics: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--host', default='localhost', help='Host address')
@click.option('--port', default=8000, help='Port number')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host: str, port: int, reload: bool):
    """Start the API server."""
    click.echo(f"Starting ANPR API server on {host}:{port}")
    
    from src.anpr.api.main import run_api
    run_api(host=host, port=port, reload=reload)


if __name__ == '__main__':
    cli()
