"""Clean PDF extraction and topic-aware cross-page chunking for store policies."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_POLICY_PDF_FILENAME = "políticas_da_loja.pdf"

RUNNING_HEADER_PATTERN = re.compile(
    r"Emp[oó]rio\s+da\s+M[uú]sica\s+Manual\s+de\s+Pol[ií]ticas\s+e\s+Procedimentos",
    re.IGNORECASE,
)
PAGE_FOOTER_PATTERN = re.compile(r"^P[aá]gina\s+\d+\s*$", re.IGNORECASE)
H1_SECTION_PATTERN = re.compile(r"^(?P<num>\d+)\.\s+(?P<title>[A-ZÀ-Ú][^:\n]+)$")
H2_SECTION_PATTERN = re.compile(r"^(?P<num>\d+\.\d+)\s+(?P<title>[A-ZÀ-Ú].+)$")
PHONE_SPACING_PATTERN = re.compile(r"\((\d{2})\)\s*(\d{4,5})\s*-\s*(\d{4})")

SECTION_TOPIC_MAP: dict[str, str] = {
    "0": "sobre_a_loja",
    "1": "sobre_a_loja",
    "1.1": "sobre_a_loja",
    "1.2": "endereco_e_contato",
    "2": "horario_funcionamento",
    "3": "formas_pagamento_e_parcelamento",
    "3.1": "formas_pagamento_e_parcelamento",
    "4": "trocas_e_devolucoes",
    "4.1": "trocas_e_devolucoes",
    "4.2": "trocas_e_devolucoes",
    "4.3": "trocas_e_devolucoes",
    "4.4": "trocas_e_devolucoes",
    "5": "frete_e_entregas",
    "5.1": "frete_e_entregas",
    "5.2": "frete_e_entregas",
    "5.3": "frete_e_entregas",
    "6": "promocoes_e_descontos",
    "6.1": "promocoes_e_descontos",
    "6.2": "promocoes_e_descontos",
    "7": "atendimento_whatsapp",
    "7.1": "atendimento_whatsapp",
    "7.2": "atendimento_whatsapp",
    "7.3": "atendimento_whatsapp",
    "8": "garantia",
    "8.1": "garantia",
    "8.2": "garantia",
    "8.3": "garantia",
    "9": "privacidade_e_lgpd",
    "10": "disposicoes_finais",
}


@dataclass
class _SpanBlock:
    """Internal representation of a typographically grouped PDF text block."""

    page_number: int
    role: str  # "cover" | "h1" | "h2" | "body"
    words: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return _clean_block_text(" ".join(self.words))


@dataclass
class _SectionAccumulator:
    """Accumulates cross-page content and metadata for a single logical policy section."""

    section_number: str
    section_title: str
    parent_section: str | None = None
    parent_intro: str = ""
    pages: set[int] = field(default_factory=set)
    body_parts: list[str] = field(default_factory=list)


def _clean_block_text(raw: str) -> str:
    """Normalize whitespace, bullet breaks, and phone formatting in extracted prose."""
    cleaned = RUNNING_HEADER_PATTERN.sub("", raw)
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    cleaned = PHONE_SPACING_PATTERN.sub(r"(\1) \2-\3", cleaned)
    # Place inline bullets on dedicated lines for readability and lexical clarity
    cleaned = re.sub(r"\s*•\s*", "\n• ", cleaned).strip()
    return cleaned


def _infer_topic(section_number: str, section_title: str) -> str:
    """Map section number or title keywords to a canonical policy topic."""
    if section_number in SECTION_TOPIC_MAP:
        return SECTION_TOPIC_MAP[section_number]
    root = section_number.split(".")[0]
    if root in SECTION_TOPIC_MAP:
        return SECTION_TOPIC_MAP[root]

    lower_title = section_title.lower()
    if "horário" in lower_title or "horario" in lower_title:
        return "horario_funcionamento"
    if "pagamento" in lower_title or "parcelamento" in lower_title:
        return "formas_pagamento_e_parcelamento"
    if "troca" in lower_title or "devolu" in lower_title:
        return "trocas_e_devolucoes"
    if "frete" in lower_title or "entrega" in lower_title:
        return "frete_e_entregas"
    if "garantia" in lower_title:
        return "garantia"
    return "politicas_gerais"


class PDFPolicyLoader:
    """Typography- and section-aware PDF loader for Empório da Música store policies.

    Concatenates spans across page breaks prior to section splitting so headers
    (such as Sections 5.2, 7.2, and 9) remain attached to their tables and bullet rules,
    while preserving exact 1-based `page_start`, `page_end`, and `pages` metadata.
    """

    def __init__(
        self,
        pdf_path: Path | str | None = None,
        max_chunk_size: int = 1400,
        chunk_overlap: int = 150,
    ) -> None:
        """Initialize the loader and validate the target PDF path.

        Args:
            pdf_path: Optional explicit path to the store policies PDF.
            max_chunk_size: Maximum character size before secondary overlap splitting.
            chunk_overlap: Character overlap applied if a section exceeds `max_chunk_size`.
        """
        if max_chunk_size <= 0:
            raise ValueError(f"max_chunk_size must be positive, got {max_chunk_size}")
        if chunk_overlap < 0 or chunk_overlap >= max_chunk_size:
            raise ValueError(f"chunk_overlap must be in [0, max_chunk_size), got {chunk_overlap}")

        self.pdf_path: Path = self._resolve_pdf_path(pdf_path)
        self.max_chunk_size: int = max_chunk_size
        self.chunk_overlap: int = chunk_overlap
        self.source_name: str = unicodedata.normalize("NFC", self.pdf_path.name)

    @staticmethod
    def _resolve_pdf_path(pdf_path: Path | str | None) -> Path:
        """Resolve the PDF file path safely across NFC/NFD Unicode filesystems."""
        if pdf_path is not None:
            candidate = Path(pdf_path)
            if candidate.is_file():
                return candidate.resolve()
            # Check NFC/NFD sibling match if parent directory exists
            if candidate.parent.is_dir():
                target_nfc = unicodedata.normalize("NFC", candidate.name)
                for item in candidate.parent.iterdir():
                    if item.is_file() and unicodedata.normalize("NFC", item.name) == target_nfc:
                        return item.resolve()
            raise FileNotFoundError(f"Store policies PDF not found at: {candidate}")

        settings_dir = Path(get_settings().data_dir)
        project_root_dir = Path(__file__).resolve().parent.parent.parent / "data"
        target_nfc = unicodedata.normalize("NFC", DEFAULT_POLICY_PDF_FILENAME)

        for search_dir in (settings_dir, project_root_dir):
            if not search_dir.is_dir():
                continue
            direct = search_dir / DEFAULT_POLICY_PDF_FILENAME
            if direct.is_file():
                return direct.resolve()
            for item in search_dir.glob("*.pdf"):
                if unicodedata.normalize("NFC", item.name) == target_nfc:
                    return item.resolve()

        raise FileNotFoundError(
            f"Store policies PDF '{DEFAULT_POLICY_PDF_FILENAME}' not found in "
            f"'{settings_dir}' or '{project_root_dir}'."
        )

    def _extract_raw_blocks(self) -> tuple[int, list[_SpanBlock]]:
        """Extract typographically classified text blocks across all PDF pages."""
        reader = PdfReader(str(self.pdf_path))
        total_pages = len(reader.pages)
        all_blocks: list[_SpanBlock] = []

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            raw_tokens: list[tuple[float, bool, str]] = []

            def _visitor(
                text: str,
                _cm: Any,
                _tm: Any,
                font_dict: dict[str, Any] | None,
                font_size: float,
                _tokens: list[tuple[float, bool, str]] = raw_tokens,
            ) -> None:
                stripped = text.strip()
                if not stripped:
                    return
                base_font = font_dict.get("/BaseFont", "") if font_dict else ""
                is_bold = "Bold" in str(base_font)
                _tokens.append((round(float(font_size or 0.0), 1), is_bold, stripped))

            page.extract_text(visitor_text=_visitor)

            for fs, is_bold, token in raw_tokens:
                # Strip running header & page footer on content pages (pages 2..N)
                if page_num >= 2 and fs <= 11.0:
                    continue
                if PAGE_FOOTER_PATTERN.match(token):
                    continue

                if page_num == 1 and fs >= 11.5:
                    role = "cover"
                elif fs >= 18.0 and is_bold:
                    role = "h1"
                elif 15.5 <= fs < 18.0 and is_bold:
                    role = "h2"
                else:
                    role = "body"

                if (
                    all_blocks
                    and all_blocks[-1].page_number == page_num
                    and all_blocks[-1].role == role
                ):
                    all_blocks[-1].words.append(token)
                else:
                    all_blocks.append(_SpanBlock(page_number=page_num, role=role, words=[token]))

        return total_pages, all_blocks

    def extract_pages(self) -> list[dict[str, Any]]:
        """Extract cleaned, human-readable text per page with running headers/footers removed.

        Returns:
            List of dictionaries with `page_number` (1-based) and cleaned `text`.
        """
        total_pages, blocks = self._extract_raw_blocks()
        by_page: dict[int, list[str]] = {p: [] for p in range(1, total_pages + 1)}

        for blk in blocks:
            txt = blk.text
            if txt and not PAGE_FOOTER_PATTERN.match(txt):
                by_page[blk.page_number].append(txt)

        return [
            {
                "page_number": page_num,
                "text": "\n\n".join(by_page[page_num]).strip(),
            }
            for page_num in range(1, total_pages + 1)
        ]

    def load(self) -> list[Document]:
        """Load and split the store policy manual into topic-aware LangChain Documents.

        Preserves cross-page continuity (e.g. Sections 5.2, 7.2, and 9) and injects
        controlled hierarchical parent context on subsections.

        Returns:
            List of LangChain `Document` chunks with section, topic, and page-span metadata.
        """
        _, blocks = self._extract_raw_blocks()

        # Fallback to plain regex line parsing if a synthetic PDF had no font-size hierarchy
        has_typographic_headers = any(b.role in ("h1", "h2") for b in blocks)
        if not has_typographic_headers:
            return self._fallback_regex_chunking()

        cover_parts: list[str] = []
        cover_pages: set[int] = set()

        sections: list[_SectionAccumulator] = []
        current_h1_num: str | None = None
        current_h1_title: str | None = None
        current_h1_intro_parts: list[str] = []
        current_h1_pages: set[int] = set()
        active_sub: _SectionAccumulator | None = None

        def _flush_h1_if_needed() -> None:
            """Emit standalone or substantive parent H1 section before starting a new H1."""
            nonlocal current_h1_num, current_h1_title, current_h1_intro_parts, current_h1_pages
            nonlocal active_sub

            if active_sub is not None:
                sections.append(active_sub)
                active_sub = None

            if current_h1_num is None or current_h1_title is None:
                return

            has_children = any(s.section_number.startswith(f"{current_h1_num}.") for s in sections)
            intro_text = "\n".join(p for p in current_h1_intro_parts if p).strip()

            # Emit H1 chunk when it has no subsections (e.g. 2, 9, 10) OR when its own body
            # contains standalone operational tables/scope rules (e.g. 1 and 3).
            if not has_children or current_h1_num in ("1", "3"):
                pages_set = set(current_h1_pages)
                body_list: list[str] = []
                if current_h1_num == "1" and cover_parts:
                    pages_set.update(cover_pages)
                    body_list.append("\n".join(cover_parts))
                if intro_text:
                    body_list.append(intro_text)
                if body_list:
                    sections.append(
                        _SectionAccumulator(
                            section_number=current_h1_num,
                            section_title=current_h1_title,
                            parent_section=None,
                            pages=pages_set,
                            body_parts=body_list,
                        )
                    )

            current_h1_num = None
            current_h1_title = None
            current_h1_intro_parts = []
            current_h1_pages = set()

        for blk in blocks:
            txt = blk.text
            if not txt or PAGE_FOOTER_PATTERN.match(txt):
                continue

            if blk.role == "cover":
                cover_parts.append(txt)
                cover_pages.add(blk.page_number)
                continue

            if blk.role == "h1":
                _flush_h1_if_needed()
                m1 = re.match(r"^(?P<num>\d+)\.?\s+(?P<rest>.+)$", txt)
                current_h1_num = m1.group("num") if m1 else txt.split()[0].rstrip(".")
                current_h1_title = txt
                current_h1_pages = {blk.page_number}
                current_h1_intro_parts = []
                continue

            if blk.role == "h2":
                if active_sub is not None:
                    sections.append(active_sub)
                m2 = re.match(r"^(?P<num>\d+\.\d+)\.?\s+(?P<rest>.+)$", txt)
                sub_num = m2.group("num") if m2 else txt.split()[0]
                # Controlled overlap: carry short parent intros (e.g. Section 4 CDC, 7 WhatsApp)
                parent_intro = ""
                if current_h1_num not in ("1", "3") and current_h1_intro_parts:
                    parent_intro = "\n".join(current_h1_intro_parts).strip()

                active_sub = _SectionAccumulator(
                    section_number=sub_num,
                    section_title=txt,
                    parent_section=current_h1_title,
                    parent_intro=parent_intro,
                    pages={blk.page_number},
                )
                continue

            # Body block: attach to active subsection if open, otherwise to current H1 intro
            if active_sub is not None:
                active_sub.body_parts.append(txt)
                active_sub.pages.add(blk.page_number)
            elif current_h1_num is not None:
                current_h1_intro_parts.append(txt)
                current_h1_pages.add(blk.page_number)
            else:
                cover_parts.append(txt)
                cover_pages.add(blk.page_number)

        _flush_h1_if_needed()

        # Sort sections in canonical document order (1, 1.1, 1.2, 2, 3, 3.1, ...)
        def _sec_sort_key(acc: _SectionAccumulator) -> tuple[int, ...]:
            return tuple(int(part) for part in acc.section_number.split(".") if part.isdigit())

        sections.sort(key=_sec_sort_key)

        documents: list[Document] = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n• ", "\n", ". ", " "],
        )

        for sec in sections:
            sorted_pages = sorted(sec.pages) if sec.pages else [1]
            header_prefix = (
                f"[{sec.parent_section} — {sec.section_title}]"
                if sec.parent_section
                else f"[{sec.section_title}]"
            )
            parts = [header_prefix]
            if sec.parent_intro:
                parts.append(sec.parent_intro)
            parts.extend(sec.body_parts)
            full_content = "\n".join(p for p in parts if p).strip()

            base_meta: dict[str, Any] = {
                "source": self.source_name,
                "section_number": sec.section_number,
                "section_title": sec.section_title,
                "parent_section": sec.parent_section,
                "topic": _infer_topic(sec.section_number, sec.section_title),
                "page_start": sorted_pages[0],
                "page_end": sorted_pages[-1],
                "pages": sorted_pages,
            }

            if len(full_content) <= self.max_chunk_size:
                documents.append(Document(page_content=full_content, metadata=base_meta))
            else:
                sub_texts = splitter.split_text(full_content)
                for chunk_idx, sub_text in enumerate(sub_texts):
                    content_with_header = (
                        sub_text if sub_text.startswith("[") else f"{header_prefix}\n{sub_text}"
                    )
                    chunk_meta = dict(base_meta)
                    chunk_meta["chunk_part"] = chunk_idx + 1
                    documents.append(
                        Document(page_content=content_with_header, metadata=chunk_meta)
                    )

        return documents

    def _fallback_regex_chunking(self) -> list[Document]:
        """Fallback regex-based section chunker for uniform-font synthetic PDFs."""
        pages = self.extract_pages()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        docs: list[Document] = []
        for page_info in pages:
            page_num = page_info["page_number"]
            text = page_info["text"]
            if not text:
                continue
            for idx, chunk in enumerate(splitter.split_text(text)):
                m_sub = H2_SECTION_PATTERN.search(chunk)
                m_top = H1_SECTION_PATTERN.search(chunk)
                sec_num = (
                    m_sub.group("num")
                    if m_sub
                    else (m_top.group("num") if m_top else f"p{page_num}.{idx + 1}")
                )
                sec_title = (
                    m_sub.group(0) if m_sub else (m_top.group(0) if m_top else f"Página {page_num}")
                )
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": self.source_name,
                            "section_number": sec_num,
                            "section_title": sec_title,
                            "parent_section": None,
                            "topic": _infer_topic(sec_num, sec_title),
                            "page_start": page_num,
                            "page_end": page_num,
                            "pages": [page_num],
                        },
                    )
                )
        return docs


def load_store_policy_documents(pdf_path: Path | str | None = None) -> list[Document]:
    """Convenience helper to load topic-aware policy chunks from `políticas_da_loja.pdf`."""
    return PDFPolicyLoader(pdf_path=pdf_path).load()
