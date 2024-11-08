"""
Main entry point for the ANPR system.

This module serves as the primary entry point for the application, providing
both CLI and API modes of operation. It initializes configuration, sets up
logging, and handles command-line arguments for different modes of operation.

Includes signal handlers for graceful shutdown on SIGINT/SIGTERM.
"""

import argparse
import logging
import signal
import atexit
import sys
import os
import platform
from pathlib import Path

import click

from src.anpr.config.config import load_config, get_config
from src.anpr.api.main import run_api
from src.anpr.cli.main import cli


# Module-level logger
logger = logging.getLogger(__name__)

# Shutdown state tracking
_shutdown_initiated = False


def _signal_handler(signum, frame):
    """
    Handle shutdown signals (SIGINT, SIGTERM) gracefully.
    
    This handler sets a flag and logs the signal. The actual cleanup
    is handled by FastAPI's shutdown event and uvicorn's signal handling.
    """
    global _shutdown_initiated
    
    if _shutdown_initiated:
        logger.warning("Shutdown already in progress, forcing exit...")
        sys.exit(1)
    
    _shutdown_initiated = True
    signal_name = signal.Signals(signum).name
    logger.info(f"Received shutdown signal ({signal_name}), initiating graceful shutdown...")
    
    # Let uvicorn handle the actual shutdown - it will call FastAPI's shutdown event
    # Raising SystemExit here allows uvicorn to catch it and shutdown gracefully
    raise SystemExit(0)


def _atexit_handler():
    """
    Handler for unexpected exits.
    
    This is a last-resort cleanup that runs when the Python interpreter
    is shutting down, catching cases where normal shutdown didn't occur.
    """
    global _shutdown_initiated
    
    if not _shutdown_initiated:
        logger.warning("Unexpected exit detected - atexit handler invoked")
        logger.info("Note: FastAPI shutdown event may not have run. Check for resource leaks.")


def _setup_signal_handlers():
    """
    Setup signal handlers for graceful shutdown.
    
    Registers handlers for:
    - SIGINT (Ctrl+C) - all platforms
    - SIGTERM - Unix/Linux/macOS
    """
    # SIGINT works on all platforms (Ctrl+C)
    signal.signal(signal.SIGINT, _signal_handler)
    
    # SIGTERM only works on Unix-like systems
    if platform.system() != "Windows":
        signal.signal(signal.SIGTERM, _signal_handler)
    
    # Register atexit handler for unexpected exits
    atexit.register(_atexit_handler)
    
    logger.debug(f"Signal handlers registered (platform: {platform.system()})")


def setup_logging(level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Add console handler
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def main():
    """Main entry point for the ANPR system."""
    parser = argparse.ArgumentParser(
        description="Automatic Number Plate Recognition (ANPR) System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode api --host 0.0.0.0 --port 800    # Start API server
  %(prog)s --mode cli -- --process video.mp4        # Process video via CLI
  %(prog)s --config my_config.yaml --mode api       # Start with custom config
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["api", "cli"], 
        default="cli",
        help="Operation mode: 'api' for REST API server, 'cli' for command-line interface (default: cli)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="DEBUG",
        help="Logging level (default: DEBUG)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file"
    )
    
    # API-specific arguments
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host address for API server (default: localhost)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for API server (default: 8000)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for API development"
    )
    
    # Parse known args to separate API/CLI specific arguments
    args, remaining = parser.parse_known_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Load configuration
    try:
        config = load_config(args.config)
        logging.info(f"Configuration loaded from: {args.config or 'default location'}")
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Set environment variables from config
    os.environ.setdefault('ANPR_API_HOST', config.api.host)
    os.environ.setdefault('ANPR_API_PORT', str(config.api.port))
    
    # Execute based on mode
    if args.mode == "api":
        logging.info(f"Starting ANPR API server on {args.host}:{args.port}")
        
        # Setup signal handlers for graceful shutdown
        _setup_signal_handlers()
        
        try:
            run_api(host=args.host, port=args.port, reload=args.reload)
        except KeyboardInterrupt:
            logging.info("API server stopped by user (KeyboardInterrupt)")
        except SystemExit as e:
            # Normal shutdown via signal handler
            if e.code == 0:
                logging.info("API server shutdown complete")
            else:
                logging.error(f"API server exited with code {e.code}")
                sys.exit(e.code)
        except Exception as e:
            logging.error(f"Error running API server: {e}")
            sys.exit(1)
        finally:
            global _shutdown_initiated
            _shutdown_initiated = True  # Prevent atexit warning on normal exit
            logging.info("ANPR system terminated")
    
    elif args.mode == "cli":
        # For CLI mode, we'll pass the remaining arguments to the CLI
        # Since the CLI uses click, we need to modify sys.argv to include
        # the remaining arguments after the '--' separator
        original_argv = sys.argv.copy()
        try:
            # Replace sys.argv with the remaining arguments for CLI processing
            sys.argv = [sys.argv[0]] + remaining
            # Call the CLI group function directly
            cli(standalone_mode=False)
        except click.exceptions.ClickException as e:
            e.show(file=sys.stderr)
            sys.exit(e.exit_code)
        except click.exceptions.Exit as e:
            sys.exit(e.exit_code)
        finally:
            # Restore original argv
            sys.argv = original_argv


if __name__ == "__main__":
    main()
