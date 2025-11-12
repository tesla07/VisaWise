"""Tests for monitoring and metrics."""

import pytest
from src.visawise.utils.monitoring import PrometheusMetrics


@pytest.fixture
def metrics():
    """Create a metrics instance."""
    return PrometheusMetrics()


def test_record_query(metrics):
    """Test recording a query."""
    metrics.record_query("langgraph", "success", 1.5)
    # No assertion, just ensure no exception


def test_record_case_check(metrics):
    """Test recording a case check."""
    metrics.record_case_check("success", 0.5)
    # No assertion, just ensure no exception


def test_record_error(metrics):
    """Test recording an error."""
    metrics.record_error("query_processing")
    # No assertion, just ensure no exception


def test_set_active_sessions(metrics):
    """Test setting active sessions."""
    metrics.set_active_sessions(10)
    # No assertion, just ensure no exception


def test_set_cache_size(metrics):
    """Test setting cache size."""
    metrics.set_cache_size(100)
    # No assertion, just ensure no exception


def test_get_metrics(metrics):
    """Test getting metrics."""
    metrics.record_query("langgraph", "success", 1.0)
    result = metrics.get_metrics()
    assert isinstance(result, bytes)
    assert b"visawise" in result
