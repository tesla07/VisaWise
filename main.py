#!/usr/bin/env python3
"""Main entry point for VisaWise application."""

import asyncio
import uvicorn
import logging
from src.visawise.config import settings

logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run the VisaWise API server."""
    logger.info("Starting VisaWise API server...")
    logger.info(f"API will be available at http://{settings.api_host}:{settings.api_port}")
    logger.info(f"Metrics available at http://{settings.api_host}:{settings.api_port}/metrics")
    logger.info(f"Grafana dashboard at http://{settings.grafana_host}:{settings.grafana_port}")
    
    uvicorn.run(
        "src.visawise.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()
