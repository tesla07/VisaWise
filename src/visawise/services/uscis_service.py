"""USCIS case status checking service."""

import httpx
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import re


logger = logging.getLogger(__name__)


class USCISService:
    """Service for interacting with USCIS case status system."""
    
    def __init__(self, base_url: str):
        """Initialize USCIS service.
        
        Args:
            base_url: Base URL for USCIS case status API
        """
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def check_case_status(self, receipt_number: str) -> Dict[str, Any]:
        """Check the status of a USCIS case.
        
        Args:
            receipt_number: USCIS receipt number (e.g., WAC2190012345)
            
        Returns:
            Dictionary containing case status information
        """
        try:
            # Validate receipt number format
            if not self._validate_receipt_number(receipt_number):
                return {
                    "success": False,
                    "error": "Invalid receipt number format",
                    "receipt_number": receipt_number
                }
            
            # Make request to USCIS
            response = await self.client.post(
                self.base_url,
                data={"appReceiptNum": receipt_number}
            )
            
            if response.status_code != 200:
                logger.error(f"USCIS API returned status code: {response.status_code}")
                return {
                    "success": False,
                    "error": f"API request failed with status {response.status_code}",
                    "receipt_number": receipt_number
                }
            
            # Parse response
            status_info = self._parse_response(response.text, receipt_number)
            status_info["timestamp"] = datetime.utcnow().isoformat()
            
            return status_info
            
        except httpx.TimeoutException:
            logger.error(f"Timeout checking case {receipt_number}")
            return {
                "success": False,
                "error": "Request timeout",
                "receipt_number": receipt_number
            }
        except Exception as e:
            logger.error(f"Error checking case {receipt_number}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "receipt_number": receipt_number
            }
    
    def _validate_receipt_number(self, receipt_number: str) -> bool:
        """Validate USCIS receipt number format.
        
        Args:
            receipt_number: Receipt number to validate
            
        Returns:
            True if valid, False otherwise
        """
        # USCIS receipt numbers are typically 13 characters:
        # 3 letters + 10 digits (e.g., WAC2190012345)
        pattern = r'^[A-Z]{3}\d{10}$'
        return bool(re.match(pattern, receipt_number.upper()))
    
    def _parse_response(self, html: str, receipt_number: str) -> Dict[str, Any]:
        """Parse USCIS response HTML.
        
        Args:
            html: HTML response from USCIS
            receipt_number: Receipt number that was queried
            
        Returns:
            Parsed case status information
        """
        # Simple parsing - in production, use BeautifulSoup or similar
        result = {
            "success": True,
            "receipt_number": receipt_number,
            "status": "Unknown",
            "description": "",
            "case_type": ""
        }
        
        # Check for common status indicators in the response
        if "Case Was Received" in html:
            result["status"] = "Received"
        elif "Case Was Approved" in html:
            result["status"] = "Approved"
        elif "Request for Additional Evidence" in html:
            result["status"] = "RFE Issued"
        elif "Case Was Denied" in html:
            result["status"] = "Denied"
        elif "Case Was Transferred" in html:
            result["status"] = "Transferred"
        elif "not be found" in html or "does not exist" in html:
            result["success"] = False
            result["error"] = "Case not found"
        
        return result
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
