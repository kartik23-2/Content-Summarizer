import re
from llm.model import ask_llm
from llm.prompt import build_query_reformulation_prompt

# Pronouns and ordinals indicating follow-ups or coreferences
COREFERENCE_PATTERNS = [
    r"\b(it|this|that|they|them|these|those)\b",
    r"\b(the\s+former|the\s+latter)\b",
    r"\b(first|second|third|fourth|fifth|last)\s+(one|condition|point|algorithm|state|rule)\b",
    r"\b(elaborate|explain\s+more|tell\s+me\s+more)\b",
    r"\bwhat\s+about\b",
    r"\band\s+the\s+(other|next|previous)\b",
]


class ConversationManager:
    """
    Manages multi-turn conversation history and selectively reformulates
    anaphoric or short follow-up questions to minimize unnecessary LLM invocations.
    """

    def __init__(self):
        self.history: list[dict] = []

    def add_turn(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def clear(self):
        self.history = []

    def get_history(self, max_turns: int = 6) -> list[dict]:
        return self.history[-max_turns:]

    def requires_reformulation(self, query: str) -> bool:
        """
        Determines whether a query needs LLM reformulation based on heuristics:
        1. History must not be empty.
        2. Query is short (<= 6 words, e.g., 'What about Doc B?', 'Explain it more') OR
        3. Query contains coreference pronouns or ordinals.
        """
        if not self.history:
            return False

        clean_q = query.strip()
        words = clean_q.split()

        # Short queries in a multi-turn session are typically contextual follow-ups
        if len(words) <= 6:
            return True

        # Check for coreference regex patterns
        lower_q = clean_q.lower()
        for pattern in COREFERENCE_PATTERNS:
            if re.search(pattern, lower_q):
                return True

        return False

    def get_search_query(self, raw_query: str) -> tuple[str, bool]:
        """
        Returns (search_query: str, was_reformulated: bool).
        Only invokes LLM if the heuristic determines reformulation is necessary.
        """
        if not self.requires_reformulation(raw_query):
            return raw_query, False

        prompt = build_query_reformulation_prompt(self.get_history(), raw_query)
        try:
            reformulated = ask_llm(prompt)
            clean_ref = reformulated.strip().strip('"').strip("'")
            if clean_ref and len(clean_ref) > 3:
                return clean_ref, True
        except Exception:
            pass

        return raw_query, False
