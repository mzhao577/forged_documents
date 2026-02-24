"""
Metadata Analysis Module
Analyzes document metadata for signs of manipulation or forgery.
"""

import os
import re
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class MetadataResult:
    """Results from metadata analysis."""
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    author: Optional[str] = None
    producer: Optional[str] = None
    creator_tool: Optional[str] = None
    page_count: int = 0
    file_size_bytes: int = 0
    anomalies: list = field(default_factory=list)
    risk_score: float = 0.0


class MetadataAnalyzer:
    """Analyze document metadata for forgery indicators."""

    # Suspicious software patterns often used to forge documents
    SUSPICIOUS_TOOLS = [
        "photoshop", "gimp", "paint", "canva",
        "online pdf", "smallpdf", "ilovepdf",
        "pdf editor", "foxit phantompdf"
    ]

    # Legitimate medical document software
    LEGITIMATE_MEDICAL_SOFTWARE = [
        "epic", "cerner", "meditech", "allscripts",
        "athenahealth", "nextgen", "eclinicalworks",
        "microsoft word", "adobe acrobat"
    ]

    def analyze_pdf(self, pdf_path: str) -> MetadataResult:
        """
        Analyze PDF metadata.

        Args:
            pdf_path: Path to PDF file

        Returns:
            MetadataResult with findings
        """
        result = MetadataResult()
        result.file_size_bytes = os.path.getsize(pdf_path)

        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                result.page_count = len(pdf.pages)
                metadata = pdf.metadata or {}

                result.creation_date = self._parse_pdf_date(
                    metadata.get("CreationDate", "")
                )
                result.modification_date = self._parse_pdf_date(
                    metadata.get("ModDate", "")
                )
                result.author = metadata.get("Author")
                result.producer = metadata.get("Producer")
                result.creator_tool = metadata.get("Creator")

        except ImportError:
            result.anomalies.append("pdfplumber not installed")
            return result
        except Exception as e:
            result.anomalies.append(f"Failed to read PDF metadata: {str(e)}")
            return result

        # Check for anomalies
        self._check_metadata_anomalies(result)

        return result

    def analyze_image(self, image_path: str) -> MetadataResult:
        """
        Analyze image metadata (EXIF data).

        Args:
            image_path: Path to image file

        Returns:
            MetadataResult with findings
        """
        result = MetadataResult()
        result.file_size_bytes = os.path.getsize(image_path)

        try:
            from PIL import Image
            from PIL.ExifTags import TAGS

            with Image.open(image_path) as img:
                exif_data = img._getexif()

                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if tag == "DateTime":
                            result.creation_date = str(value)
                        elif tag == "DateTimeOriginal":
                            result.modification_date = str(value)
                        elif tag == "Software":
                            result.creator_tool = str(value)
                else:
                    result.anomalies.append(
                        "No EXIF data found - metadata may have been stripped"
                    )

        except ImportError:
            result.anomalies.append("Pillow not installed")
        except Exception as e:
            result.anomalies.append(f"Failed to read image metadata: {str(e)}")

        self._check_metadata_anomalies(result)
        return result

    def analyze_file(self, file_path: str) -> MetadataResult:
        """Auto-detect file type and analyze metadata."""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return self.analyze_pdf(file_path)
        elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            return self.analyze_image(file_path)
        else:
            result = MetadataResult()
            result.file_size_bytes = os.path.getsize(file_path)
            result.anomalies.append(f"Metadata analysis not supported for {ext}")
            return result

    def _parse_pdf_date(self, date_str: str) -> Optional[str]:
        """Parse PDF date format (D:YYYYMMDDHHmmss)."""
        if not date_str:
            return None
        # Remove 'D:' prefix if present
        date_str = date_str.replace("D:", "").split("+")[0].split("-")[0]
        try:
            if len(date_str) >= 8:
                dt = datetime.strptime(date_str[:14].ljust(14, "0"), "%Y%m%d%H%M%S")
                return dt.isoformat()
        except ValueError:
            pass
        return date_str

    def _check_metadata_anomalies(self, result: MetadataResult):
        """Check for suspicious patterns in metadata."""
        risk_score = 0.0

        # Check for suspicious creation tools
        if result.creator_tool:
            tool_lower = result.creator_tool.lower()
            for suspicious in self.SUSPICIOUS_TOOLS:
                if suspicious in tool_lower:
                    result.anomalies.append(
                        f"Document created with suspicious tool: {result.creator_tool}"
                    )
                    risk_score += 0.3
                    break

        # Check for missing metadata (often stripped to hide origin)
        missing_fields = []
        if not result.creation_date:
            missing_fields.append("creation_date")
        if not result.author:
            missing_fields.append("author")

        if missing_fields:
            result.anomalies.append(
                f"Missing metadata fields: {', '.join(missing_fields)}"
            )
            risk_score += 0.1 * len(missing_fields)

        # Check for date inconsistencies
        if result.creation_date and result.modification_date:
            try:
                created = datetime.fromisoformat(result.creation_date)
                modified = datetime.fromisoformat(result.modification_date)

                if modified < created:
                    result.anomalies.append(
                        "Modification date is before creation date"
                    )
                    risk_score += 0.4

                # Check if dates are in the future
                now = datetime.now()
                if created > now:
                    result.anomalies.append("Creation date is in the future")
                    risk_score += 0.5

            except (ValueError, TypeError):
                pass

        # Check file size anomalies
        if result.file_size_bytes < 1000 and result.page_count > 0:
            result.anomalies.append(
                "Unusually small file size for document with content"
            )
            risk_score += 0.2

        result.risk_score = min(risk_score, 1.0)
