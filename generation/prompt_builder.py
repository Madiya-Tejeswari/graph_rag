"""
Prompt builder for GraphRAG-based generation.
Creates structured, domain-aware prompts optimized for PG&E document Q&A.
"""

import logging
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain-specific constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a PG&E Greenbook & Tariff expert assistant powered by a Graph-RAG pipeline.

## Your Mission
Answer questions about PG&E's Electric & Gas Service Requirements ("Greenbook") \
and Tariff schedules **using ONLY the retrieved evidence** provided below.
Your answers should read like the manual itself — present the actual content, \
not a summary or paraphrase.

## Rules (strict)
1. **Evidence-only answers** – Every claim MUST be traceable to the evidence. \
If the evidence does not contain the answer, say so explicitly.
2. **Cite page numbers** – Reference "(Page X)" after each fact you state.
3. **Precise numbers** – Quote exact values (kVA, amps, voltages, dimensions) \
from the evidence. Never round or approximate.
4. **Use the manual's own words** – Present the actual content from the manual \
as much as possible. Include the full text of relevant sections, requirements, \
specifications, and notes. Do NOT summarize or paraphrase when precision matters.
5. **Structured answers** – Use bullet points or numbered lists for multi-part \
answers. Use markdown tables when comparing specifications.
6. **Reference figures and tables by name** – When the evidence mentions a \
Figure (e.g., Figure 3-2, Figure 7-23) or a Table (e.g., Table 4-1), always \
include the figure/table name and page number. Say: "as shown in Figure X-Y (Page Z)".
7. **Implementation steps** – When the question asks about a process, installation, \
procedure, or how to do something, start your answer with the heading '[Procedure]'. \
Then give a connective introduction, followed by exactly 6 to 8 numbered steps \
about the content, exactly as described in the manual.
8. **Technical accuracy** – Use correct PG&E terminology: kVA, CT (current \
transformer), single-phase, three-phase, service lateral, meter socket, etc.
9. **No hallucination** – If evidence is partial, state what IS known and what \
is NOT covered by the provided evidence.
10. **Comprehensive answers** – Give a thorough answer that covers all relevant \
aspects from the evidence. Include all conditions, exceptions, and notes.

## Answer Format
- Start with a **direct answer** to the question.
- Present supporting content from the manual organized by relevance.
- When the evidence includes table data, reproduce it as a **markdown table**.
- When figures are referenced, mention them explicitly by name and page.
- For procedures, provide **numbered implementation steps**.
- End with consolidated page references: "(Pages X, Y, Z)"

## Domain Knowledge (for context only — still cite evidence)
- PG&E Greenbook (TD-7001M) covers service installation requirements.
- Tariff schedules define rates, rules, and service conditions.
- Common topics: transformer sizing, metering, conduit specs, service voltages, \
clearances, underground/overhead installation, load limits.
"""

QUERY_TYPE_HINTS = {
    "specification": (
        "The user is asking about a specific technical specification. "
        "Provide the exact value(s) with units, all conditions and exceptions, "
        "and reference any related Tables or Figures by name and page. "
        "Include the full text of the relevant specification from the manual."
    ),
    "procedure": (
        "The user is asking about an installation process or procedure. "
        "Format your answer starting with the heading '[Procedure]'. "
        "Follow with a short connective introduction, and then provide exactly 6 to 8 "
        "numbered steps or points that need to be followed based on the manual content. "
        "Include requirements, materials, dimensions, and conditions at each step. "
        "Reference any Figures showing the installation or process."
    ),
    "comparison": (
        "The user is comparing two or more options. "
        "Use a **markdown comparison table** with rows for each specification. "
        "Include all relevant values, conditions, and page references in the table."
    ),
    "requirement": (
        "The user is asking what is required or allowed. "
        "List all requirements clearly with any conditions, exceptions, or notes, "
        "exactly as stated in the manual. Reference applicable Tables and Figures."
    ),
    "visual": (
        "The user is asking about something visual — a diagram, layout, figure, "
        "or what something looks like. Describe the content referenced in the "
        "evidence and explicitly reference Figures by name and page number. "
        "Include all labels, notes, and dimensions shown in the figure."
    ),
    "general": (
        "Provide a comprehensive answer covering all relevant aspects from the "
        "evidence. Include the actual manual content, reference Figures/Tables "
        "by name and page, and provide any related implementation steps."
    ),
}

FEW_SHOT_EXAMPLES = [
    {
        "question": "What is the maximum transformer size for single-phase service?",
        "evidence": '[Page 52 - greenbook.pdf]\nFor any single-phase service, the maximum demand, as determined by PG&E, is limited to the capability of a 100-kVA transformer. If the load requires a transformer installation larger than 100 kVA, the service will be three-phase.\n\n[Page 54 - greenbook.pdf]\nTable 4-1 shows available voltages including 208Y/120, 240/120, or 480Y/277.',
        "answer": "The maximum transformer size for single-phase service is **100 kVA** (Page 52).\n\n* For any single-phase service, the maximum demand, as determined by PG&E, is limited to the capability of a 100-kVA transformer.\n* If the load requires a transformer installation larger than 100 kVA, the service will be three-phase.\n* Available voltages include 208Y/120, 240/120, or 480Y/277, as shown in Table 4-1 (Page 54).\n\n(Pages 52, 54)",
    },
    {
        "question": "What does a typical underground service connection look like?",
        "evidence": '[Page 58 - greenbook.pdf]\nA typical underground service connection is shown in Figure 3-2, "Underground to Underground Service Connection" on Page 3-8, and Figure 7-23 on Page 7-44. The connection includes:\n1. A termination facility\n2. Substructures such as conduit, boxes, and transformer pads\n3. A service lateral conductor connected to the applicant\'s termination facilities',
        "answer": "A typical underground service connection is shown in Figure 3-2, \"Underground to Underground Service Connection\" (Page 58), and Figure 7-23 (Page 58). The connection includes:\n\n1. A termination facility, typically on or within the building or structure.\n2. Substructures, such as conduit, boxes, and transformer pads.\n3. A service lateral conductor connected to the applicant's termination facilities by PG&E.\n\nas illustrated in Figure 3-2 (Page 58).\n\n(Pages 58)",
    },
]


class PromptBuilder:
    """Builds domain-optimized prompts for PG&E GraphRAG generation."""

    def __init__(self):
        """Initialize prompt builder with system prompt and examples."""
        self.system_prompt = SYSTEM_PROMPT
        self.few_shot_examples = FEW_SHOT_EXAMPLES

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, query: str, context: str) -> str:
        """
        Build the primary RAG prompt.

        Args:
            query: User question
            context: Retrieved evidence text

        Returns:
            Fully formatted prompt string
        """
        query_type = self._classify_query(query)
        hint = QUERY_TYPE_HINTS.get(query_type, QUERY_TYPE_HINTS["general"])

        examples_block = self._format_examples()

        prompt = (
            f"{self.system_prompt}\n"
            f"\n## Query Classification\n"
            f"Type: {query_type}\n"
            f"Guidance: {hint}\n"
            f"\n{examples_block}"
            f"\n{'=' * 60}\n"
            f"RETRIEVED EVIDENCE\n"
            f"{'=' * 60}\n\n"
            f"{context}\n\n"
            f"{'=' * 60}\n"
            f"USER QUESTION\n"
            f"{'=' * 60}\n\n"
            f"{query}\n\n"
            f"{'=' * 60}\n"
            f"YOUR ANSWER (follow the rules above)\n"
            f"{'=' * 60}\n\n"
        )
        return prompt

    def build_with_metadata(
        self, query: str, context: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt with additional metadata context."""
        prompt = self.build(query, context)
        if metadata:
            meta_str = "\n".join(f"- {k}: {v}" for k, v in metadata.items())
            prompt += f"\n[Additional context]\n{meta_str}\n"
        return prompt

    def build_condensed(self, query: str, context: str) -> str:
        """
        Build a shorter prompt for fast inference or smaller models.

        Args:
            query: User question
            context: Retrieved evidence

        Returns:
            Condensed prompt
        """
        return (
            "You are a PG&E document expert. Answer using ONLY the evidence below. "
            "Cite page numbers. Be precise with numbers and units.\n\n"
            f"Evidence:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

    def build_with_examples(
        self, query: str, context: str, examples: Optional[List[Dict]] = None
    ) -> str:
        """Build prompt with custom few-shot examples."""
        custom_examples = examples or self.few_shot_examples
        examples_block = ""
        for i, ex in enumerate(custom_examples[:3], 1):
            examples_block += (
                f"--- Example {i} ---\n"
                f"Q: {ex.get('question', '')}\n"
                f"Evidence: {ex.get('evidence', '')}\n"
                f"A: {ex.get('answer', '')}\n\n"
            )

        return (
            f"{self.system_prompt}\n\n"
            f"## Examples\n{examples_block}"
            f"{'=' * 60}\n"
            f"RETRIEVED EVIDENCE\n{'=' * 60}\n\n"
            f"{context}\n\n"
            f"{'=' * 60}\n"
            f"USER QUESTION\n{'=' * 60}\n\n"
            f"{query}\n\n"
            f"{'=' * 60}\n"
            f"YOUR ANSWER\n{'=' * 60}\n\n"
        )

    def build_extraction_prompt(self, context: str, extraction_type: str) -> str:
        """Build prompt for structured information extraction."""
        return (
            f"Extract all {extraction_type} from the following PG&E document text.\n"
            f"Format each item as a bullet point with page reference.\n\n"
            f"Text:\n{context}\n\n"
            f"Extracted {extraction_type}:\n"
        )

    def build_summarization_prompt(self, context: str) -> str:
        """Build prompt for document summarization."""
        return (
            "Summarize the following PG&E document content. "
            "Preserve all key specifications, requirements, and page references.\n\n"
            f"Content:\n{context}\n\n"
            "Summary:\n"
        )

    def build_follow_up_prompt(
        self, query: str, context: str, previous_answer: str
    ) -> str:
        """
        Build prompt for follow-up questions that reference a prior answer.

        Args:
            query: Follow-up question
            context: Newly retrieved evidence
            previous_answer: The prior answer for continuity

        Returns:
            Prompt with conversation context
        """
        return (
            f"{self.system_prompt}\n\n"
            f"## Previous Answer\n{previous_answer}\n\n"
            f"{'=' * 60}\n"
            f"NEW EVIDENCE\n{'=' * 60}\n\n"
            f"{context}\n\n"
            f"{'=' * 60}\n"
            f"FOLLOW-UP QUESTION\n{'=' * 60}\n\n"
            f"{query}\n\n"
            f"{'=' * 60}\n"
            f"YOUR ANSWER (build on the previous answer if relevant)\n"
            f"{'=' * 60}\n\n"
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def estimate_tokens(self, prompt: str) -> int:
        """Estimate token count (~4 chars per token)."""
        return len(prompt) // 4

    def validate_prompt(self, prompt: str) -> tuple:
        """
        Validate prompt structure.

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        if len(prompt) < 100:
            issues.append("Prompt is too short — may lack evidence")
        if "EVIDENCE" not in prompt and "Evidence" not in prompt:
            issues.append("Prompt missing evidence section")
        if "QUESTION" not in prompt and "Question" not in prompt:
            issues.append("Prompt missing question section")
        token_est = self.estimate_tokens(prompt)
        if token_est > 8000:
            issues.append(f"Prompt may be too long (~{token_est} tokens)")
        return (len(issues) == 0, issues)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_query(self, query: str) -> str:
        """
        Classify the query type to tailor the prompt guidance.

        Returns one of: specification, procedure, comparison, requirement, general
        """
        q = query.lower()

        spec_patterns = [
            r"what is the (maximum|minimum|size|rating|capacity|voltage|amperage)",
            r"how (much|many|large|long|thick|wide|deep)",
            r"\b(kva|amps?|volts?|inches?|feet|diameter)\b",
        ]
        if any(re.search(p, q) for p in spec_patterns):
            return "specification"

        procedure_patterns = [
            r"how (do|does|to|should|can) .*(install|connect|apply|request|get)",
            r"what (steps|process|procedure)",
            r"(steps|procedure|process) (for|to)",
            r"install(ation|ment)? steps?",
        ]
        if any(re.search(p, q) for p in procedure_patterns):
            return "procedure"

        comparison_patterns = [
            r"\bvs\.?\b",
            r"\bversus\b",
            r"(difference|compare|comparison) between",
            r"(single.phase|underground).*(three.phase|overhead)",
        ]
        if any(re.search(p, q) for p in comparison_patterns):
            return "comparison"

        visual_patterns = [
            r"(show|look|looks?) like",
            r"(diagram|figure|drawing|layout|illustration|schematic)",
            r"(what does|how does) .*(look|appear|connect)",
            r"(show me|display|visualize)",
        ]
        if any(re.search(p, q) for p in visual_patterns):
            return "visual"

        requirement_patterns = [
            r"(what|which) .*(require|need|must|shall|necessary)",
            r"(do i need|is it required|must i|shall)",
            r"(requirements?|regulations?|rules?|code) for",
        ]
        if any(re.search(p, q) for p in requirement_patterns):
            return "requirement"

        return "general"

    def _format_examples(self) -> str:
        """Format the built-in few-shot examples."""
        if not self.few_shot_examples:
            return ""

        block = "## Reference Examples\n\n"
        for i, ex in enumerate(self.few_shot_examples[:2], 1):
            block += (
                f"**Example {i}**\n"
                f"Q: {ex['question']}\n"
                f"Evidence: {ex['evidence'][:200]}...\n"
                f"A: {ex['answer']}\n\n"
            )
        return block
