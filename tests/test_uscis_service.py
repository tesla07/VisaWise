"""Tests for USCIS service."""

import pytest
from src.visawise.services.uscis_service import USCISService


@pytest.fixture
def uscis_service():
    """Create a USCIS service instance."""
    return USCISService("https://egov.uscis.gov/casestatus/mycasestatus.do")


def test_validate_receipt_number_valid(uscis_service):
    """Test validation of valid receipt numbers."""
    assert uscis_service._validate_receipt_number("WAC2190012345")
    assert uscis_service._validate_receipt_number("EAC1234567890")
    assert uscis_service._validate_receipt_number("LIN9876543210")


def test_validate_receipt_number_invalid(uscis_service):
    """Test validation of invalid receipt numbers."""
    assert not uscis_service._validate_receipt_number("WAC219001234")  # Too short
    assert not uscis_service._validate_receipt_number("WAC21900123456")  # Too long
    assert not uscis_service._validate_receipt_number("123456789012")  # No letters
    assert not uscis_service._validate_receipt_number("WABC190012345")  # 4 letters
    assert not uscis_service._validate_receipt_number("WA2190012345")  # Only 2 letters


@pytest.mark.asyncio
async def test_check_case_status_invalid_format(uscis_service):
    """Test case status check with invalid receipt number."""
    result = await uscis_service.check_case_status("INVALID123")
    assert result["success"] is False
    assert "Invalid receipt number format" in result["error"]


def test_parse_response_received(uscis_service):
    """Test parsing of 'Received' status."""
    html = "<html>Case Was Received</html>"
    result = uscis_service._parse_response(html, "WAC2190012345")
    assert result["success"] is True
    assert result["status"] == "Received"
    assert result["receipt_number"] == "WAC2190012345"


def test_parse_response_approved(uscis_service):
    """Test parsing of 'Approved' status."""
    html = "<html>Case Was Approved</html>"
    result = uscis_service._parse_response(html, "WAC2190012345")
    assert result["success"] is True
    assert result["status"] == "Approved"


def test_parse_response_not_found(uscis_service):
    """Test parsing when case is not found."""
    html = "<html>Case not be found</html>"
    result = uscis_service._parse_response(html, "WAC2190012345")
    assert result["success"] is False
    assert "Case not found" in result["error"]
