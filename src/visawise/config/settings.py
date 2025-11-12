"""Configuration management for VisaWise."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""
    
    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4-turbo-preview"
    
    # USCIS Configuration
    uscis_api_base_url: str = "https://egov.uscis.gov/casestatus/mycasestatus.do"
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Grafana Configuration
    grafana_host: str = "localhost"
    grafana_port: int = 3000
    
    # Prometheus Configuration
    prometheus_port: int = 9090
    
    # Log Level
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
