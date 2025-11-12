"""Monitoring and metrics for Prometheus/Grafana."""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from prometheus_client import CollectorRegistry, push_to_gateway
import time
from typing import Dict
import logging


logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """Prometheus metrics for VisaWise."""
    
    def __init__(self):
        """Initialize Prometheus metrics."""
        # Request counters
        self.query_counter = Counter(
            'visawise_queries_total',
            'Total number of queries processed',
            ['agent_type', 'status']
        )
        
        self.case_check_counter = Counter(
            'visawise_case_checks_total',
            'Total number of case status checks',
            ['status']
        )
        
        # Response time histograms
        self.query_duration = Histogram(
            'visawise_query_duration_seconds',
            'Time spent processing queries',
            ['agent_type']
        )
        
        self.case_check_duration = Histogram(
            'visawise_case_check_duration_seconds',
            'Time spent checking case status'
        )
        
        # Current state gauges
        self.active_sessions = Gauge(
            'visawise_active_sessions',
            'Number of active user sessions'
        )
        
        self.cache_size = Gauge(
            'visawise_cache_size',
            'Number of items in cache'
        )
        
        # Error counter
        self.error_counter = Counter(
            'visawise_errors_total',
            'Total number of errors',
            ['error_type']
        )
    
    def record_query(self, agent_type: str, status: str, duration: float):
        """Record a query execution.
        
        Args:
            agent_type: Type of agent used (langchain, langgraph, crewai)
            status: Success or failure
            duration: Execution time in seconds
        """
        self.query_counter.labels(agent_type=agent_type, status=status).inc()
        self.query_duration.labels(agent_type=agent_type).observe(duration)
    
    def record_case_check(self, status: str, duration: float):
        """Record a case status check.
        
        Args:
            status: Check result (success, error, not_found)
            duration: Check time in seconds
        """
        self.case_check_counter.labels(status=status).inc()
        self.case_check_duration.observe(duration)
    
    def record_error(self, error_type: str):
        """Record an error.
        
        Args:
            error_type: Type of error
        """
        self.error_counter.labels(error_type=error_type).inc()
    
    def set_active_sessions(self, count: int):
        """Set the number of active sessions.
        
        Args:
            count: Number of active sessions
        """
        self.active_sessions.set(count)
    
    def set_cache_size(self, size: int):
        """Set the cache size.
        
        Args:
            size: Number of items in cache
        """
        self.cache_size.set(size)
    
    def get_metrics(self) -> bytes:
        """Get current metrics in Prometheus format.
        
        Returns:
            Metrics data
        """
        return generate_latest(REGISTRY)


# Global metrics instance
_metrics: PrometheusMetrics = None


def setup_metrics() -> PrometheusMetrics:
    """Set up and return the metrics instance.
    
    Returns:
        PrometheusMetrics instance
    """
    global _metrics
    if _metrics is None:
        _metrics = PrometheusMetrics()
        logger.info("Prometheus metrics initialized")
    return _metrics


def get_metrics() -> PrometheusMetrics:
    """Get the metrics instance.
    
    Returns:
        PrometheusMetrics instance
    """
    if _metrics is None:
        return setup_metrics()
    return _metrics
