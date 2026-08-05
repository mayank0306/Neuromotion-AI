"""Data export utilities."""

import structlog
from datetime import datetime

logger = structlog.get_logger(__name__)


class DataExporter:
    """
    Export user data to various formats.
    
    Supports:
    - PDF reports
    - Excel spreadsheets
    - JSON exports
    """
    
    def __init__(self):
        """Initialize exporter."""
        self.logger = structlog.get_logger(__name__)

    def export_session_to_pdf(self, session_data: dict) -> bytes:
        """
        Export session data to PDF.
        
        Args:
            session_data: Session data to export.
            
        Returns:
            PDF file as bytes.
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            from io import BytesIO
            
            buffer = BytesIO()
            pdf = canvas.Canvas(buffer, pagesize=letter)
            
            # Add content
            pdf.setTitle(f"Movement Analysis Report - {datetime.now().strftime('%Y-%m-%d')}")
            pdf.drawString(100, 750, f"Session: {session_data.get('id', 'Unknown')}")
            pdf.drawString(100, 730, f"Score: {session_data.get('score', 0)}%")
            pdf.drawString(100, 710, f"Duration: {session_data.get('duration', 0)}s")
            
            pdf.save()
            buffer.seek(0)
            
            return buffer.read()
            
        except ImportError:
            logger.error("PDF export library not available")
            raise

    def export_progress_to_excel(self, progress_data: list) -> bytes:
        """
        Export progress history to Excel.
        
        Args:
            progress_data: List of progress records.
            
        Returns:
            Excel file as bytes.
        """
        try:
            from openpyxl import Workbook
            from io import BytesIO
            
            wb = Workbook()
            ws = wb.active
            ws.title = "Progress"
            
            # Headers
            ws.append(["Date", "Score", "Session Type", "Duration"])
            
            # Data
            for record in progress_data:
                ws.append([
                    record.get("date", ""),
                    record.get("score", 0),
                    record.get("type", ""),
                    record.get("duration", 0),
                ])
            
            buffer = BytesIO()
            wb.save(buffer)
            buffer.seek(0)
            
            return buffer.read()
            
        except ImportError:
            logger.error("Excel export library not available")
            raise
