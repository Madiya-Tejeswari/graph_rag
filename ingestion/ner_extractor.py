"""
Named Entity Recognition and Relation Extraction module.
Uses spaCy for NER and dependency-based relation extraction
to build a richer knowledge graph with entity-to-entity edges.
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Set, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

# Try to import spaCy; fall back gracefully
try:
    import spacy
    from spacy.tokens import Span, Doc
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not installed. NER will use rule-based fallback only.")


class NERExtractor:
    """
    Named Entity Recognition and Relation Extraction engine.

    Combines spaCy NER with domain-specific rules to extract:
    - Named entities (equipment, specifications, standards, organizations)
    - Relations between co-occurring entities (dependency-based + proximity)
    """

    # ── Domain-specific entity patterns ──────────────────────────────────
    DOMAIN_PATTERNS = {
        "Equipment": [
            r"\btransformer(?:s)?\b", r"\bconduit(?:s)?\b", r"\bmeter(?:s)?\b",
            r"\bsocket(?:s)?\b", r"\blateral(?:s)?\b", r"\benclosure(?:s)?\b",
            r"\bdisconnect(?:s)?\b", r"\bpanel(?:s)?\b", r"\bbushing(?:s)?\b",
            r"\bcable(?:s)?\b", r"\bcircuit breaker(?:s)?\b", r"\bfuse(?:s)?\b",
            r"\bswitch(?:es)?\b", r"\bpole(?:s)?\b", r"\bcapacitor(?:s)?\b",
            r"\brecloser(?:s)?\b", r"\bregulator(?:s)?\b", r"\briser(?:s)?\b",
            r"\bmanhole(?:s)?\b", r"\bhandhole(?:s)?\b", r"\bpedestal(?:s)?\b",
            r"\bsplice(?:s)?\b", r"\bjunction box(?:es)?\b", r"\bwire(?:s)?\b",
            r"\bconductor(?:s)?\b", r"\bgrounding\b", r"\bground rod(?:s)?\b",
        ],
        "ServiceType": [
            r"\bsingle[- ]phase\b", r"\bthree[- ]phase\b",
            r"\bsingle phase\b", r"\bthree phase\b",
            r"\bresidential service\b", r"\bcommercial service\b",
            r"\bindustrial service\b", r"\btemporary service\b",
            r"\bpermanent service\b", r"\bself-contained\b",
        ],
        "InstallationType": [
            r"\bunderground\b", r"\boverhead\b", r"\baerial\b",
            r"\bdirect[- ]buried?\b", r"\bpad[- ]mounted\b",
            r"\bsubsurface\b", r"\bsubmersible\b",
        ],
        "Specification": [
            r"\b\d+\s*kVA\b", r"\b\d+\s*kV\b", r"\b\d+\s*(?:V|volt(?:s)?)\b",
            r"\b\d+\s*(?:A|amp(?:s|ere)?(?:s)?)\b",
            r"\b\d+\s*(?:AWG|kcmil|MCM)\b",
            r"\b\d+(?:/\d+)?\s*(?:kV|V)\b",
            r"\b\d+\s*(?:HP|hp|horsepower)\b",
            r"\b\d+\s*(?:kW|MW|watts?)\b",
            r"\b\d+\s*(?:feet|ft|inches?|in)\b",
        ],
        "Standard": [
            r"\bNEC\b", r"\bCPUC\b", r"\bGO\s*\d+\b", r"\bRule\s+\d+\b",
            r"\bSection\s+\d+[\.\d]*\b", r"\bTable\s+\d+[-\.\d]*\b",
            r"\bFigure\s+\d+[-\.\d]*\b", r"\bArticle\s+\d+\b",
            r"\bTariff\s+\w+\b",
        ],
        "Material": [
            r"\bPVC\b", r"\bHDPE\b", r"\bcopper\b", r"\baluminum\b",
            r"\bsteel\b", r"\bfiberglass\b", r"\bconcrete\b",
            r"\bgalvanized\b",
        ],
    }

    # ── Relationship trigger patterns ────────────────────────────────────
    # These define verbs/phrases that imply a relationship between
    # a subject-entity and an object-entity in the same sentence.
    RELATION_TRIGGERS = {
        "REQUIRES": [
            r"\brequire[sd]?\b", r"\bmust have\b", r"\bshall have\b",
            r"\bneeds?\b", r"\bmandatory\b", r"\bmust be\b",
        ],
        "CONNECTS_TO": [
            r"\bconnect(?:s|ed|ing)?\s+to\b", r"\battach(?:es|ed)?\s+to\b",
            r"\bjoined?\s+to\b", r"\bcoupled?\s+to\b",
        ],
        "HAS_SPECIFICATION": [
            r"\brated\s+(?:at|for)\b", r"\bwith\s+a?\s*capacity\b",
            r"\bmaximum\s+(?:of|size|rating)\b", r"\bminimum\s+(?:of|size|rating)\b",
            r"\bnot\s+(?:to\s+)?exceed\b",
        ],
        "INSTALLED_IN": [
            r"\binstall(?:s|ed|ing)?\s+(?:in|on|at)\b",
            r"\bplaced?\s+(?:in|on|at)\b", r"\bmounted\s+(?:in|on|at)\b",
            r"\blocated?\s+(?:in|on|at)\b",
        ],
        "USED_FOR": [
            r"\bused\s+(?:for|in|with)\b", r"\bserves?\s+as\b",
            r"\bprovides?\b", r"\bsupplies?\b",
        ],
        "PART_OF": [
            r"\bpart\s+of\b", r"\bcomponent\s+of\b",
            r"\binclude[sd]?\b", r"\bconsist(?:s|ing)?\s+of\b",
        ],
        "SUPERSEDES": [
            r"\breplaces?\b", r"\bsupersedes?\b", r"\bobsoletes?\b",
        ],
        "REGULATED_BY": [
            r"\bper\s+\b", r"\bin\s+accordance\s+with\b",
            r"\bas\s+specified\s+in\b", r"\bcompliant?\s+with\b",
            r"\bconforms?\s+to\b",
        ],
    }

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the NER extractor.

        Args:
            spacy_model: spaCy model name. Use 'en_core_web_sm' for speed
                         or 'en_core_web_trf' for accuracy (requires torch).
        """
        self.nlp = None
        self.spacy_model_name = spacy_model

        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(spacy_model)
                logger.info(f"Loaded spaCy model: {spacy_model}")
            except OSError:
                logger.warning(
                    f"spaCy model '{spacy_model}' not found. "
                    f"Run: python -m spacy download {spacy_model}"
                )
                # Try downloading it
                try:
                    spacy.cli.download(spacy_model)
                    self.nlp = spacy.load(spacy_model)
                    logger.info(f"Downloaded and loaded spaCy model: {spacy_model}")
                except Exception as e:
                    logger.warning(f"Could not download spaCy model: {e}")

        # Compile regex patterns once
        self._compiled_domain = {}
        for etype, patterns in self.DOMAIN_PATTERNS.items():
            self._compiled_domain[etype] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        self._compiled_relations = {}
        for rtype, patterns in self.RELATION_TRIGGERS.items():
            self._compiled_relations[rtype] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    # ── Public API ───────────────────────────────────────────────────────

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.

        Combines spaCy NER (if available) with domain-specific regex patterns.
        Deduplicates and normalizes entity names.

        Args:
            text: Input text

        Returns:
            List of entity dicts with keys: name, type, start, end
        """
        entities: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        # ── Pass 1: spaCy NER ────────────────────────────────────────
        if self.nlp is not None:
            doc = self.nlp(text[:100_000])  # Cap to avoid OOM on huge texts
            for ent in doc.ents:
                # Map spaCy labels to our domain types
                mapped_type = self._map_spacy_label(ent.label_)
                if mapped_type is None:
                    continue

                norm_name = self._normalize_entity(ent.text)
                if not norm_name or len(norm_name) < 2:
                    continue

                dedup_key = f"{norm_name.lower()}|{mapped_type}"
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    entities.append({
                        "name": norm_name,
                        "type": mapped_type,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "source": "spacy",
                    })

        # ── Pass 2: Domain-specific regex NER ────────────────────────
        for etype, compiled_patterns in self._compiled_domain.items():
            for pattern in compiled_patterns:
                for match in pattern.finditer(text):
                    norm_name = self._normalize_entity(match.group())
                    if not norm_name or len(norm_name) < 2:
                        continue

                    dedup_key = f"{norm_name.lower()}|{etype}"
                    if dedup_key not in seen:
                        seen.add(dedup_key)
                        entities.append({
                            "name": norm_name,
                            "type": etype,
                            "start": match.start(),
                            "end": match.end(),
                            "source": "regex",
                        })

        return entities

    def extract_relations(
        self, text: str, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract relations between entities found in the same text.

        Strategy (layered, from highest to lowest confidence):
        1. **Dependency parse**: If spaCy is available, use syntactic
           dependencies (nsubj → ROOT → dobj) to find subject-verb-object
           triples where subject and object are known entities.
        2. **Trigger patterns**: Scan sentences for relationship trigger
           phrases. If a trigger sits between two entities in the same
           sentence, emit a relation.
        3. **Proximity co-occurrence**: If two entities appear within the
           same sentence with no trigger, emit a generic CO_OCCURS_WITH
           relation (lower confidence).

        Args:
            text: The source text
            entities: Entities extracted from this text (output of extract_entities)

        Returns:
            List of relation dicts with keys:
                source, target, type, evidence, confidence
        """
        if len(entities) < 2:
            return []

        relations: List[Dict[str, Any]] = []
        seen_rels: Set[str] = set()

        # Build entity lookup by character span for fast matching
        entity_spans = sorted(entities, key=lambda e: e["start"])

        # ── Strategy 1: Dependency-based (spaCy) ────────────────────
        if self.nlp is not None:
            dep_rels = self._extract_dependency_relations(text, entities)
            for rel in dep_rels:
                rel_key = f"{rel['source']}|{rel['target']}|{rel['type']}"
                if rel_key not in seen_rels:
                    seen_rels.add(rel_key)
                    relations.append(rel)

        # ── Strategy 2 & 3: Trigger + proximity (sentence-level) ────
        sentences = self._split_sentences(text)
        for sentence in sentences:
            sent_start = text.find(sentence)
            if sent_start == -1:
                continue
            sent_end = sent_start + len(sentence)

            # Find entities in this sentence
            sent_entities = [
                e for e in entity_spans
                if e["start"] >= sent_start and e["end"] <= sent_end
            ]

            if len(sent_entities) < 2:
                continue

            # Check for trigger patterns — only between adjacent or nearby entity pairs
            for i, ent_a in enumerate(sent_entities):
                # Only check the next few entities (limit combinatorial explosion)
                for ent_b in sent_entities[i + 1 : i + 4]:
                    # Text between the two entities
                    between_start = ent_a["end"] - sent_start
                    between_end = ent_b["start"] - sent_start
                    between_text = sentence[between_start:between_end]

                    # Try trigger patterns — only in text between entities
                    rel_found = False
                    for rtype, compiled_patterns in self._compiled_relations.items():
                        for pattern in compiled_patterns:
                            if pattern.search(between_text):
                                rel_key = f"{ent_a['name']}|{ent_b['name']}|{rtype}"
                                rev_key = f"{ent_b['name']}|{ent_a['name']}|{rtype}"
                                if rel_key not in seen_rels and rev_key not in seen_rels:
                                    seen_rels.add(rel_key)
                                    relations.append({
                                        "source": ent_a["name"],
                                        "source_type": ent_a["type"],
                                        "target": ent_b["name"],
                                        "target_type": ent_b["type"],
                                        "type": rtype,
                                        "evidence": sentence.strip()[:200],
                                        "confidence": 0.80,
                                    })
                                    rel_found = True
                                    break
                        if rel_found:
                            break

                    # Proximity co-occurrence fallback — only for immediately adjacent pairs
                    if not rel_found and ent_b is sent_entities[i + 1]:
                        rel_key = f"{ent_a['name']}|{ent_b['name']}|CO_OCCURS_WITH"
                        rev_key = f"{ent_b['name']}|{ent_a['name']}|CO_OCCURS_WITH"
                        if rel_key not in seen_rels and rev_key not in seen_rels:
                            seen_rels.add(rel_key)
                            relations.append({
                                "source": ent_a["name"],
                                "source_type": ent_a["type"],
                                "target": ent_b["name"],
                                "target_type": ent_b["type"],
                                "type": "CO_OCCURS_WITH",
                                "evidence": sentence.strip()[:200],
                                "confidence": 0.50,
                            })

        return relations

    # ── Private helpers ──────────────────────────────────────────────────

    def _extract_dependency_relations(
        self, text: str, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Use spaCy dependency parse to find SVO triples involving known entities."""
        relations = []
        if self.nlp is None:
            return relations

        doc = self.nlp(text[:100_000])

        # Build a set of normalized entity names for quick lookup
        entity_name_set = {e["name"].lower() for e in entities}
        entity_type_map = {e["name"].lower(): e["type"] for e in entities}

        for sent in doc.sents:
            # Find the root verb of the sentence
            root = None
            for token in sent:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    root = token
                    break

            if root is None:
                continue

            # Find subject and object
            subjects = []
            objects = []
            for child in root.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    # Get the full noun phrase
                    subj_text = self._get_noun_phrase(child)
                    subjects.append(subj_text)
                elif child.dep_ in ("dobj", "pobj", "attr"):
                    obj_text = self._get_noun_phrase(child)
                    objects.append(obj_text)

            # Also check for prepositional objects
            for child in root.children:
                if child.dep_ == "prep":
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            obj_text = self._get_noun_phrase(grandchild)
                            objects.append(obj_text)

            # Match subjects and objects to known entities
            for subj in subjects:
                subj_lower = subj.lower()
                matched_subj = self._find_matching_entity(subj_lower, entity_name_set)
                if matched_subj is None:
                    continue

                for obj in objects:
                    obj_lower = obj.lower()
                    matched_obj = self._find_matching_entity(obj_lower, entity_name_set)
                    if matched_obj is None or matched_obj == matched_subj:
                        continue

                    # Map verb to relation type
                    rel_type = self._map_verb_to_relation(root.lemma_)

                    relations.append({
                        "source": matched_subj,
                        "source_type": entity_type_map.get(matched_subj, "Unknown"),
                        "target": matched_obj,
                        "target_type": entity_type_map.get(matched_obj, "Unknown"),
                        "type": rel_type,
                        "evidence": sent.text.strip()[:200],
                        "confidence": 0.85,
                    })

        return relations

    def _get_noun_phrase(self, token) -> str:
        """Extract the full noun phrase from a dependency token."""
        phrase_parts = []
        for child in token.subtree:
            if child.pos_ in ("NOUN", "PROPN", "ADJ", "NUM", "DET") or child.dep_ == "compound":
                phrase_parts.append(child.text)
        return " ".join(phrase_parts) if phrase_parts else token.text

    def _find_matching_entity(
        self, text: str, entity_names: Set[str]
    ) -> Optional[str]:
        """Find which known entity name best matches the given text."""
        text_lower = text.lower().strip()
        # Exact match
        if text_lower in entity_names:
            return text_lower
        # Substring match
        for name in entity_names:
            if name in text_lower or text_lower in name:
                return name
        return None

    def _map_verb_to_relation(self, verb_lemma: str) -> str:
        """Map a verb lemma to a relationship type."""
        verb_map = {
            "require": "REQUIRES",
            "need": "REQUIRES",
            "connect": "CONNECTS_TO",
            "attach": "CONNECTS_TO",
            "join": "CONNECTS_TO",
            "install": "INSTALLED_IN",
            "place": "INSTALLED_IN",
            "mount": "INSTALLED_IN",
            "locate": "INSTALLED_IN",
            "use": "USED_FOR",
            "serve": "USED_FOR",
            "provide": "USED_FOR",
            "supply": "USED_FOR",
            "include": "PART_OF",
            "contain": "PART_OF",
            "consist": "PART_OF",
            "replace": "SUPERSEDES",
            "supersede": "SUPERSEDES",
            "rate": "HAS_SPECIFICATION",
            "specify": "REGULATED_BY",
            "comply": "REGULATED_BY",
            "conform": "REGULATED_BY",
            "exceed": "HAS_SPECIFICATION",
        }
        return verb_map.get(verb_lemma.lower(), "RELATED_TO")

    @staticmethod
    def _map_spacy_label(label: str) -> Optional[str]:
        """Map spaCy NER labels to domain entity types."""
        mapping = {
            "ORG": "Organization",
            "PERSON": "Person",
            "GPE": "Location",
            "LOC": "Location",
            "FAC": "Facility",
            "PRODUCT": "Equipment",
            "QUANTITY": "Specification",
            "CARDINAL": "Specification",
            "MONEY": "Cost",
            "LAW": "Standard",
            "DATE": "Date",
            "ORDINAL": None,        # Usually not useful
            "PERCENT": None,        # Usually not useful in this domain
            "TIME": None,
            "WORK_OF_ART": None,
            "LANGUAGE": None,
            "NORP": "Organization",  # Nationalities, groups
            "EVENT": None,
        }
        return mapping.get(label, "Other")

    @staticmethod
    def _normalize_entity(text: str) -> str:
        """Normalize entity text: strip, collapse whitespace, title-case."""
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        # Don't title-case abbreviations (all-caps) or specs with units
        if text.isupper() and len(text) <= 6:
            return text
        if re.match(r"^\d+\s*\w{1,4}$", text):
            return text
        return text

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Simple sentence splitter (fallback when spaCy not available)."""
        # Split on period followed by whitespace and an uppercase letter
        # This avoids splitting on abbreviations like "e.g.", "Fig.", "No."
        raw_splits = re.split(r"\.\s+(?=[A-Z])", text)
        sentences = []
        for s in raw_splits:
            s = s.strip()
            if len(s) > 10:
                sentences.append(s)
        return sentences
