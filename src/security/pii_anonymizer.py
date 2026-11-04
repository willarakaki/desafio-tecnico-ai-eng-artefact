"""LGPD PII anonymizer and masking engine compatible with Microsoft Presidio."""

from __future__ import annotations

import re
import threading
from typing import ClassVar

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig, RecognizerResult
from pydantic import BaseModel, Field


class PIIEntityResult(BaseModel):
    """Structured representation of a detected sensitive PII entity."""

    entity_type: str = Field(
        ...,
        description="Recognized entity type (CPF, EMAIL_ADDRESS, PHONE_NUMBER)",
    )
    start: int = Field(..., description="Character start index in the analyzed text")
    end: int = Field(..., description="Character end index in the analyzed text")
    score: float = Field(default=1.0, description="Detection confidence score (0.0 to 1.0)")
    text: str = Field(..., description="The raw substring detected as PII")


class PIIAnonymizer:
    """LGPD-compliant PII detection and masking engine.

    Uses high-precision Brazilian regex patterns and mathematical CPF Modulo 11
    validation, delegating redaction to Microsoft Presidio's AnonymizerEngine.
    """

    DEFAULT_REPLACEMENTS: ClassVar[dict[str, str]] = {
        "CPF": "***.***.***-**",
        "EMAIL_ADDRESS": "[EMAIL_REDACTED]",
        "PHONE_NUMBER": "[PHONE_REDACTED]",
    }

    # Official store institutional contacts (excluded from customer PII redaction)
    INSTITUTIONAL_PHONES_DIGITS: ClassVar[set[str]] = {
        "6733214500",
        "6733414444",
        "33214500",
        "33414444",
    }
    INSTITUTIONAL_EMAILS: ClassVar[set[str]] = {
        "contato@emporiodamusica.com.br",
        "contato@empóriodamusica.com.br",
    }

    # Formatted Brazilian CPF: 123.456.789-01
    FORMATTED_CPF_REGEX: ClassVar[re.Pattern[str]] = re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")

    # Raw 11-digit CPF candidate: 12345678901
    RAW_CPF_REGEX: ClassVar[re.Pattern[str]] = re.compile(r"\b\d{11}\b")

    # Context indicators for unformatted CPFs
    CPF_CONTEXT_REGEX: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(cpf|documento|doc|cadastro)\b", re.IGNORECASE
    )

    # Standard Email regex (RFC 5322 compatible subset)
    EMAIL_REGEX: ClassVar[re.Pattern[str]] = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )

    # Brazilian phone numbers:
    # 1. Formatted with DDD: (67) 99999-9999, (67) 3341-4444, 67 98888-7777, (67)98888-7777
    # 2. Unformatted mobile: [1-9]{2}9\d{8} (11 digits, where 3rd digit is 9)
    # 3. Unformatted landline: [1-9]{2}[2-5]\d{7} (10 digits)
    # 4. Local phone with prefix keyword: "fone 99999-8888" or "tel: 3341-4444"
    PHONE_REGEXES: ClassVar[list[re.Pattern[str]]] = [
        re.compile(
            r"(?:\+55\s*)?(?:\([1-9]{2}\)\s*|\b[1-9]{2}\s+)"
            r"(?:9\s*\d{4}[-\s]?\d{4}|[2-5]\d{3}[-\s]?\d{4})\b"
        ),
        re.compile(r"\b(?:\+55\s*)?[1-9]{2}9\d{8}\b"),
        re.compile(r"\b(?:\+55\s*)?[1-9]{2}[2-5]\d{7}\b"),
        re.compile(
            r"\b(?:telefone|fone|celular|tel|whats|whatsapp|contato)[:\s]{1,5}"
            r"((?:9\s*)?\d{4}[-\s]?\d{4})\b",
            re.IGNORECASE,
        ),
    ]

    def __init__(self, replacements: dict[str, str] | None = None) -> None:
        self.replacements = replacements or dict(self.DEFAULT_REPLACEMENTS)
        self.engine = AnonymizerEngine()
        self._operators = {
            entity: OperatorConfig("replace", {"new_value": val})
            for entity, val in self.replacements.items()
        }

    @staticmethod
    def validate_cpf_checksum(cpf: str) -> bool:
        """Validate Brazilian CPF using the official Modulo 11 check digits algorithm.

        Args:
            cpf: String containing at least 11 numeric characters.

        Returns:
            True if check digits match official mathematical rules; False otherwise.
        """
        digits = [int(c) for c in cpf if c.isdigit()]
        if len(digits) != 11:
            return False

        # Known invalid identical sequence (e.g., 000.000.000-00, 111.111.111-11)
        if len(set(digits)) == 1:
            return False

        # First check digit calculation
        s1 = sum(digits[i] * (10 - i) for i in range(9))
        d1 = (s1 * 10) % 11
        if d1 == 10:
            d1 = 0
        if digits[9] != d1:
            return False

        # Second check digit calculation
        s2 = sum(digits[i] * (11 - i) for i in range(10))
        d2 = (s2 * 10) % 11
        if d2 == 10:
            d2 = 0
        return digits[10] == d2

    def extract_entities(self, text: str) -> list[PIIEntityResult]:
        """Detect and return all Brazilian PII entities in the provided text.

        Args:
            text: Input string to inspect for personal data.

        Returns:
            List of PIIEntityResult sorted by starting character position.
        """
        entities: list[PIIEntityResult] = []
        occupied_spans: list[tuple[int, int]] = []

        def _is_overlapping(start: int, end: int) -> bool:
            return any(s < end and start < e for s, e in occupied_spans)

        # 1. Formatted CPF matching
        for match in self.FORMATTED_CPF_REGEX.finditer(text):
            start, end = match.span()
            matched_text = match.group()
            entities.append(
                PIIEntityResult(
                    entity_type="CPF",
                    start=start,
                    end=end,
                    score=1.0,
                    text=matched_text,
                )
            )
            occupied_spans.append((start, end))

        # 2. Unformatted 11-digit CPF matching
        for match in self.RAW_CPF_REGEX.finditer(text):
            start, end = match.span()
            if _is_overlapping(start, end):
                continue

            matched_text = match.group()
            # Inspect surrounding window for CPF keywords or validate modulo 11 checksum
            surrounding = text[max(0, start - 20) : min(len(text), end + 20)]
            has_context = bool(self.CPF_CONTEXT_REGEX.search(surrounding))
            is_valid_checksum = self.validate_cpf_checksum(matched_text)

            if is_valid_checksum or has_context:
                entities.append(
                    PIIEntityResult(
                        entity_type="CPF",
                        start=start,
                        end=end,
                        score=1.0 if is_valid_checksum else 0.8,
                        text=matched_text,
                    )
                )
                occupied_spans.append((start, end))

        # 3. Email Address matching
        for match in self.EMAIL_REGEX.finditer(text):
            start, end = match.span()
            if _is_overlapping(start, end):
                continue

            matched_text = match.group()
            if matched_text.strip().lower() in self.INSTITUTIONAL_EMAILS:
                continue

            entities.append(
                PIIEntityResult(
                    entity_type="EMAIL_ADDRESS",
                    start=start,
                    end=end,
                    score=1.0,
                    text=matched_text,
                )
            )
            occupied_spans.append((start, end))

        # 4. Brazilian Phone Numbers
        for phone_regex in self.PHONE_REGEXES:
            for match in phone_regex.finditer(text):
                if match.lastindex and match.lastindex >= 1:
                    start, end = match.span(1)
                    matched_text = match.group(1)
                else:
                    start, end = match.span()
                    matched_text = match.group()

                if _is_overlapping(start, end):
                    continue

                # Exclude official institutional phones from redaction
                digits = re.sub(r"\D", "", matched_text)
                if digits in self.INSTITUTIONAL_PHONES_DIGITS or any(
                    digits.endswith(p) for p in self.INSTITUTIONAL_PHONES_DIGITS
                ):
                    continue

                # Basic sanity check on digit count (between 8 and 13 digits)
                digit_count = sum(c.isdigit() for c in matched_text)
                if 8 <= digit_count <= 13:
                    entities.append(
                        PIIEntityResult(
                            entity_type="PHONE_NUMBER",
                            start=start,
                            end=end,
                            score=0.95,
                            text=matched_text,
                        )
                    )
                    occupied_spans.append((start, end))

        # Sort entities by starting index
        entities.sort(key=lambda e: e.start)
        return entities

    def anonymize(self, text: str) -> str:
        """Redact sensitive Brazilian PII entities from the text using Presidio Anonymizer.

        Args:
            text: Raw string containing possible personal identifiable information.

        Returns:
            Sanitized string with sensitive substrings replaced by safe placeholders.
        """
        entities = self.extract_entities(text)
        if not entities:
            return text

        analyzer_results = [
            RecognizerResult(
                entity_type=ent.entity_type,
                start=ent.start,
                end=ent.end,
                score=ent.score,
            )
            for ent in entities
        ]

        engine_result = self.engine.anonymize(
            text=text,
            analyzer_results=analyzer_results,
            operators=self._operators,
        )
        return engine_result.text

    def contains_pii(self, text: str) -> bool:
        """Return True if any Brazilian PII entity (CPF, email, phone) is present."""
        return len(self.extract_entities(text)) > 0

    def has_cpf(self, text: str) -> bool:
        """Return True if a CPF entity is detected in the input text."""
        return any(e.entity_type == "CPF" for e in self.extract_entities(text))

    def has_email(self, text: str) -> bool:
        """Return True if an email entity is detected in the input text."""
        return any(e.entity_type == "EMAIL_ADDRESS" for e in self.extract_entities(text))

    def has_phone(self, text: str) -> bool:
        """Return True if a phone number entity is detected in the input text."""
        return any(e.entity_type == "PHONE_NUMBER" for e in self.extract_entities(text))


_global_pii_anonymizer: PIIAnonymizer | None = None
_pii_anonymizer_lock = threading.Lock()


def get_pii_anonymizer() -> PIIAnonymizer:
    """Return a thread-safe singleton instance of PIIAnonymizer."""
    global _global_pii_anonymizer
    with _pii_anonymizer_lock:
        if _global_pii_anonymizer is None:
            _global_pii_anonymizer = PIIAnonymizer()
        return _global_pii_anonymizer


def reset_pii_anonymizer() -> None:
    """Reset the global PII anonymizer singleton."""
    global _global_pii_anonymizer
    with _pii_anonymizer_lock:
        _global_pii_anonymizer = None
