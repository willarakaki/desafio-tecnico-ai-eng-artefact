"""Two-tier input guardrails blocking SQLi, Jailbreak, Prompt Leaking, and Out-of-Scope queries."""

from __future__ import annotations

import re
import threading
import unicodedata
from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, Field


class GuardrailCategory(StrEnum):
    """Categorization of incoming user messages by security guardrails."""

    SAFE = "safe"
    SQL_INJECTION = "sql_injection"
    PROMPT_INJECTION = "prompt_injection"
    SYSTEM_PROMPT_EXTRACTION = "system_prompt_extraction"
    OUT_OF_SCOPE = "out_of_scope"


class GuardrailResult(BaseModel):
    """Evaluation result produced by InputGuardrail."""

    is_safe: bool = Field(
        ...,
        description="Whether the query is safe and in-scope to process",
    )
    category: GuardrailCategory = Field(
        ...,
        description="Detailed categorization of the evaluated query",
    )
    blocked_reason: str | None = Field(
        default=None,
        description="Internal technical explanation if blocked",
    )
    refusal_message: str | None = Field(
        default=None,
        description="In-persona empathetic refusal response in Brazilian Portuguese",
    )


class InputGuardrail:
    """Two-tier input guardrail.

    Tier 1: High-speed deterministic regex & keyword filters for SQL Injection,
            Prompt Injection / Jailbreak, and System Prompt extraction.
    Tier 2: Domain scope validation keeping the assistant focused on musical
            instruments and Empório da Música store operations.
    """

    # Persona-aligned standardized refusal messages in Brazilian Portuguese
    REFUSAL_SQLI = (
        "Olá! Por motivos de segurança e governança de dados, não processamos comandos "
        "técnicos ou consultas estruturadas de banco de dados. Como posso te ajudar com nossos "
        "instrumentos musicais da Empório da Música?"
    )

    REFUSAL_JAILBREAK = (
        "Olá! Sou o especialista virtual da Empório da Música e sigo diretrizes rigorosas "
        "de atendimento e segurança. Não posso alterar minha identidade ou ignorar minhas regras. "
        "Como posso te auxiliar com seus instrumentos, compras ou pedidos?"
    )

    REFUSAL_SYSTEM_PROMPT = (
        "Olá! Minhas diretrizes internas de sistema e configuração de prompt são confidenciais. "
        "Estou à disposição para tirar qualquer dúvida sobre nossos violões, guitarras, teclados, "
        "pedidos e políticas da loja!"
    )

    REFUSAL_OUT_OF_SCOPE = (
        "Olá! Como especialista da Empório da Música, meu foco exclusivo é te atender sobre "
        "nossos instrumentos musicais, pedidos, trocas e informações da loja aqui em "
        "Campo Grande/MS. Não consigo te ajudar com receitas, programação, política ou outros "
        "temas fora do nosso universo musical!"
    )

    # Tier 1: Deterministic Security Regex Patterns
    SQLI_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(r"'\s*or\s+['\"]?1['\"]?\s*=\s*['\"]?1", re.IGNORECASE),
        re.compile(r"or\s+1\s*=\s*1\s*(--|#|/\*)", re.IGNORECASE),
        re.compile(r"\bunion\s+(all\s+)?select\b", re.IGNORECASE),
        re.compile(r"\b(drop|alter|truncate)\s+table\b", re.IGNORECASE),
        re.compile(r"\b(insert\s+into|delete\s+from|update\s+\w+\s+set)\b", re.IGNORECASE),
        re.compile(r"\bselect\s+.+\s+from\s+\w+", re.IGNORECASE),
        re.compile(r"\b(information_schema|sysdatabases|sysobjects)\b", re.IGNORECASE),
        re.compile(r";\s*(drop|delete|insert|update|alter)\b", re.IGNORECASE),
        re.compile(r"\bexec(ute)?\s*\(", re.IGNORECASE),
        re.compile(r"\bxp_cmdshell\b", re.IGNORECASE),
        re.compile(r"--\s*$", re.MULTILINE),
    ]

    JAILBREAK_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(r"\bignore\s+(all\s+)?previous\s+instructions\b", re.IGNORECASE),
        re.compile(r"\bdisregard\s+(all\s+)?previous\s+instructions\b", re.IGNORECASE),
        re.compile(r"\bforget\s+(all\s+)?previous\s+instructions\b", re.IGNORECASE),
        re.compile(
            r"\besque(ça|cer)\s+(todas\s+as\s+)?instru(ções|coes)\s+anteriores\b",
            re.IGNORECASE,
        ),
        re.compile(r"\bignore\s+(todas\s+as\s+)?instru(ções|coes)\s+anteriores\b", re.IGNORECASE),
        re.compile(r"\bdo\s+anything\s+now\b", re.IGNORECASE),
        re.compile(r"\byou\s+are\s+now\s+dan\b", re.IGNORECASE),
        re.compile(r"\b(jailbreak|dan\s+mode|unaligned\s+mode)\b", re.IGNORECASE),
        re.compile(r"\bpretend\s+you\s+are\s+(an\s+)?unrestricted\s+ai\b", re.IGNORECASE),
        re.compile(r"\broleplay\s+as\s+(an\s+)?evil\s+ai\b", re.IGNORECASE),
        re.compile(r"\bbypass\s+(your\s+)?(safety\s+)?rules\b", re.IGNORECASE),
        re.compile(r"\bdeveloper\s+mode\s+(enabled|on|activate)\b", re.IGNORECASE),
        re.compile(r"\bsystem\s+override\b", re.IGNORECASE),
        re.compile(r"\byou\s+are\s+no\s+longer\s+bound\s+by\b", re.IGNORECASE),
    ]

    SYSTEM_PROMPT_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(
            r"\b(show|reveal|display|print|repeat|give\s+me)\s+(me\s+)?(your\s+)?(system\s+)?prompt\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(developer|system)\s+instructions\b", re.IGNORECASE),
        re.compile(
            r"\bwhat\s+(is|are)\s+your\s+(system\s+prompt|initial\s+instructions|system\s+instructions|prompt)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\brepeat\s+the\s+instructions\s+above\b", re.IGNORECASE),
        re.compile(r"\bqual\s+(é|e)\s+o\s+seu\s+(prompt|prompt\s+de\s+sistema)\b", re.IGNORECASE),
        re.compile(
            r"\b(mostre|revele|imprima|exiba|repita)\s+(?:o\s+|as\s+)?(?:seu|suas|seus)?\s*"
            r"(prompt(\s+(de\s+sistema|inicial|confidencial))?|instru(ções|coes)(\s+(de\s+sistema|iniciais|completas))?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bquais\s+s(ã|a)o\s+suas\s+diretrizes\s+(secretas|iniciais|de\s+sistema)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\brevelar\s+instru(ções|coes)\s+do\s+sistema\b", re.IGNORECASE),
    ]

    # Tier 2: Domain Scope Keyword and Topic Matchers
    OUT_OF_SCOPE_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        # Cooking / Recipes
        re.compile(
            r"\b(receita\s+de|como\s+(fazer|cozinhar|assar|preparar)\s+(um\s+|uma\s+)?"
            r"(bolo|pizza|lasanha|feijoada|torta|pudim|almoco|jantar))\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(ingredientes\s+para\s+o\s+bolo|modo\s+de\s+preparo\s+do\s+bolo)\b",
            re.IGNORECASE,
        ),
        # Coding / General Programming
        re.compile(
            r"\b(escreva|crie|gere)\s+um\s+(script|codigo|programa)\s+(em\s+)?"
            r"(python|javascript|typescript|c\+\+|java|rust|html|css|sql|bash)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bcomo\s+(programar|codificar|desenvolver)\s+(em\s+)?"
            r"(python|rust|golang|javascript)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(funcao|algoritmo)\s+(em\s+)?python\s+para\b", re.IGNORECASE),
        # Politics / Elections
        re.compile(
            r"\b(em\s+quem\s+(devo\s+)?votar|qual\s+partido\b|elei(ção|cao)|voto\s+nulo|"
            r"governo\s+lula|governo\s+bolsonaro|presidente\s+da\s+rep[uú]blica)\b",
            re.IGNORECASE,
        ),
        # Medical Advice / Diagnosis
        re.compile(
            r"\b(estou\s+com\s+dor\s+no\s+peito|o\s+que\s+tomar\s+para\s+(gripe|dor|febre)|"
            r"sintomas\s+de\s+(dengue|covid)|diagnostico\s+medico|qual\s+remedio\s+tomar)\b",
            re.IGNORECASE,
        ),
        # Cryptocurrency / Investments
        re.compile(
            r"\b(devo\s+comprar\s+bitcoin|qual\s+criptomoeda\s+comprar|"
            r"como\s+investir\s+na\s+bolsa|dica\s+de\s+investimento|comprar\s+acoes)\b",
            re.IGNORECASE,
        ),
    ]

    # Valid in-scope musical instrument & store operation tokens (used for disambiguation)
    IN_SCOPE_TOKENS: ClassVar[set[str]] = {
        "violao",
        "violoes",
        "guitarra",
        "guitarras",
        "baixo",
        "baixos",
        "bateria",
        "baterias",
        "teclado",
        "teclados",
        "piano",
        "pianos",
        "ukulele",
        "ukuleles",
        "instrumento",
        "instrumentos",
        "musica",
        "musical",
        "cordas",
        "captador",
        "afinacao",
        "afinador",
        "palheta",
        "cabo",
        "takamine",
        "giannini",
        "yamaha",
        "ibanez",
        "fender",
        "tagima",
        "strinberg",
        "epiphone",
        "loja",
        "emporio",
        "endereco",
        "horario",
        "funcionamento",
        "aberto",
        "fechado",
        "campo grande",
        "14 de maio",
        "cep",
        "whatsapp",
        "telefone",
        "pedido",
        "pedidos",
        "rastreio",
        "rastreamento",
        "entrega",
        "frete",
        "motoboy",
        "correios",
        "pac",
        "sedex",
        "jadlog",
        "devolucao",
        "devolver",
        "troca",
        "trocar",
        "garantia",
        "defeito",
        "preco",
        "valor",
        "promocao",
        "desconto",
        "pix",
        "cartao",
        "parcelamento",
        "boleto",
        "estoque",
        "disponivel",
        "comprar",
        "venda",
    }

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Strip accents and lower-case text for robust keyword matching."""
        normalized = unicodedata.normalize("NFKD", text)
        stripped = "".join(c for c in normalized if not unicodedata.combining(c))
        return stripped.lower().strip()

    def validate(self, message: str) -> GuardrailResult:
        """Evaluate an incoming user query against all guardrail tiers.

        Args:
            message: Raw incoming text payload from the user.

        Returns:
            GuardrailResult with pass/fail decision, category, and refusal message.
        """
        normalized_message = self._normalize_text(message)

        # 1. Tier 1: SQL Injection Check
        for pattern in self.SQLI_PATTERNS:
            if pattern.search(message) or pattern.search(normalized_message):
                return GuardrailResult(
                    is_safe=False,
                    category=GuardrailCategory.SQL_INJECTION,
                    blocked_reason="Tentativa de injeção SQL ou comando não autorizado detectado.",
                    refusal_message=self.REFUSAL_SQLI,
                )

        # 2. Tier 1: Jailbreak & DAN Patterns
        for pattern in self.JAILBREAK_PATTERNS:
            if pattern.search(message) or pattern.search(normalized_message):
                return GuardrailResult(
                    is_safe=False,
                    category=GuardrailCategory.PROMPT_INJECTION,
                    blocked_reason="Tentativa de Jailbreak / Prompt Injection detectada.",
                    refusal_message=self.REFUSAL_JAILBREAK,
                )

        # 3. Tier 1: System Prompt Extraction
        for pattern in self.SYSTEM_PROMPT_PATTERNS:
            if pattern.search(message) or pattern.search(normalized_message):
                return GuardrailResult(
                    is_safe=False,
                    category=GuardrailCategory.SYSTEM_PROMPT_EXTRACTION,
                    blocked_reason="Tentativa de extração do System Prompt detectada.",
                    refusal_message=self.REFUSAL_SYSTEM_PROMPT,
                )

        # 4. Tier 2: Out-of-Scope Validation
        for pattern in self.OUT_OF_SCOPE_PATTERNS:
            if pattern.search(message) or pattern.search(normalized_message):
                # Pattern match on recipe, script, medical or political query triggers rejection
                return GuardrailResult(
                    is_safe=False,
                    category=GuardrailCategory.OUT_OF_SCOPE,
                    blocked_reason="Consulta fora do escopo musical e comercial da loja.",
                    refusal_message=self.REFUSAL_OUT_OF_SCOPE,
                )

        # Passed all checks
        return GuardrailResult(
            is_safe=True,
            category=GuardrailCategory.SAFE,
            blocked_reason=None,
            refusal_message=None,
        )


_global_input_guardrail: InputGuardrail | None = None
_input_guardrail_lock = threading.Lock()


def get_input_guardrail() -> InputGuardrail:
    """Return a thread-safe singleton instance of InputGuardrail."""
    global _global_input_guardrail
    with _input_guardrail_lock:
        if _global_input_guardrail is None:
            _global_input_guardrail = InputGuardrail()
        return _global_input_guardrail


def reset_input_guardrail() -> None:
    """Reset the global input guardrail singleton."""
    global _global_input_guardrail
    with _input_guardrail_lock:
        _global_input_guardrail = None
