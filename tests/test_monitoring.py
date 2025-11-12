"""Tests for monitoring and metrics."""

import pytest


def test_prometheus_metrics_imports():
    """Test that PrometheusMetrics can be imported and initialized."""
    from src.visawise.utils.monitoring import setup_metrics, get_metrics
    
    # Setup should work
    metrics = setup_metrics()
    assert metrics is not None
    
    # Get should return the same instance
    metrics2 = get_metrics()
    assert metrics2 is metrics


def test_record_operations():
    """Test recording various operations."""
    from src.visawise.utils.monitoring import get_metrics
    
    metrics = get_metrics()
    
    # These should not raise exceptions
    metrics.record_query("langgraph", "success", 1.5)
    metrics.record_case_check("success", 0.5)
    metrics.record_error("query_processing")
    metrics.set_active_sessions(10)
    metrics.set_cache_size(100)


def test_get_metrics_output():
    """Test that metrics can be exported."""
    from src.visawise.utils.monitoring import get_metrics
    
    metrics = get_metrics()
    metrics.record_query("langgraph", "success", 1.0)
    
    result = metrics.get_metrics()
    assert isinstance(result, bytes)
    assert b"visawise" in result
