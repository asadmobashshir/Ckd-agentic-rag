"""Medical guardrails.

Two enforcement points:

``pre_check``
    Runs before planning. Classifies the request as general information, a
    personalised clinical decision, an emergency, a possible self-harm
    situation, or non-medical. Only emergencies and self-harm short-circuit the
    pipeline; everything else proceeds with constraints attached.

``post_check``
    Runs on the drafted answer. Redacts individualised dosing instructions and
    definitive diagnostic statements that survived synthesis.

The goal is a useful medical *information* assistant, so a general question
("what is metformin used for?") is never blocked. A blanket disclaimer is not a
substitute for this logic; it is added only when the classification warrants it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Pattern

from agent.state import RiskCategory, SafetyAssessment
from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Rule:
    """A named regular-expression rule."""

    id: str
    pattern: Pattern[str]
    description: str

    def matches(self, text: str) -> bool:
        return bool(self.pattern.search(text))


def _rule(rule_id: str, pattern: str, description: str) -> Rule:
    return Rule(rule_id, re.compile(pattern, re.IGNORECASE), description)


EMERGENCY_RULES: tuple[Rule, ...] = (
    _rule("emg.cardiac", r"\b(chest pain|crushing chest|pain (in|down) my (left )?arm|heart attack)\b",
          "Possible acute cardiac event"),
    _rule("emg.stroke", r"\b(face drooping|slurred speech|sudden (weakness|numbness)|"
                        r"can'?t move (my|one) (arm|leg|side)|stroke symptoms)\b",
          "Possible stroke"),
    _rule("emg.breathing", r"\b(can'?t breathe|cannot breathe|struggling to breathe|"
                           r"stopped breathing|turning blue|anaphyla\w+|throat closing)\b",
          "Airway or breathing compromise"),
    _rule("emg.bleeding", r"\b(bleeding (heavily|won'?t stop|uncontrollably)|"
                          r"vomiting blood|coughing up blood|haemorrhag\w+|hemorrhag\w+)\b",
          "Severe bleeding"),
    _rule("emg.consciousness", r"\b(unconscious|unresponsive|passed out|won'?t wake up|"
                               r"seizure that (won'?t|will not) stop|convulsing)\b",
          "Altered consciousness or status epilepticus"),
    _rule("emg.overdose", r"\b(overdos\w+|took too (many|much)|poison\w+ (myself|someone))\b",
          "Possible overdose or poisoning"),
    _rule("emg.obstetric", r"\b(baby (isn'?t|not) moving|severe abdominal pain (and|with) pregnan\w+)\b",
          "Obstetric emergency"),
)

SELF_HARM_RULES: tuple[Rule, ...] = (
    _rule("sh.intent", r"\b(kill myself|end my life|take my own life|suicidal|"
                       r"want to die|don'?t want to (live|be here)|hurt myself|self[- ]harm)\b",
          "Possible self-harm or suicidal ideation"),
    _rule("sh.lethality", r"\b(lethal dose|how much .{0,30}(to|would) (kill|overdose)|"
                          r"fatal (dose|amount)|enough .{0,20}to die)\b",
          "Request for lethality information"),
)

PERSONAL_CONTEXT_RULES: tuple[Rule, ...] = (
    _rule("per.first_person_med", r"\b(i|i'?m|my|me|we)\b.{0,60}\b(take|taking|prescribed|"
                                  r"on|using|started|stopped)\b.{0,40}\b(medication|medicine|"
                                  r"drug|pill|tablet|dose|warfarin|ibuprofen|metformin|insulin)\b",
          "First-person medication context"),
    _rule("per.my_condition", r"\bmy (diagnosis|condition|symptoms?|results?|scan|blood test|"
                              r"inr|a1c|blood pressure|prescription)\b",
          "First-person clinical context"),
)

DOSING_REQUEST_RULES: tuple[Rule, ...] = (
    _rule("dose.how_much", r"\bhow (much|many)\b.{0,40}\b(should|can|do) (i|we|my \w+)\b",
          "Request for an individualised dose"),
    _rule("dose.adjust", r"\b(should i|can i|do i) (take|increase|decrease|double|split|"
                         r"skip|stop|halve|adjust)\b",
          "Request to change a personal medication regimen"),
    _rule("dose.calculate", r"\b(calculate|work out|what'?s? my) (the )?(dose|dosage|amount)\b",
          "Request to calculate a dose"),
)

DIAGNOSIS_REQUEST_RULES: tuple[Rule, ...] = (
    _rule("dx.what_do_i_have", r"\b(what('?s| is) wrong with me|do i have|what do i have|"
                               r"am i (having|suffering from)|diagnose (me|my))\b",
          "Request for a personal diagnosis"),
    _rule("dx.is_it_serious", r"\b(is (it|this|my \w+) (serious|dangerous|cancer|fatal))\b",
          "Request for personal risk determination"),
)

STOPPING_MEDICATION_RULES: tuple[Rule, ...] = (
    _rule("stop.med", r"\b(stop|quit|come off|discontinue|wean off) (taking |my )?"
                      r"(medication|medicine|pills?|tablets?|antidepressant|statin|"
                      r"insulin|warfarin|metformin)\b",
          "Stopping medication without clinician involvement"),
)

# ------------------------------------------------------------- post-check
DOSE_INSTRUCTION_RE = re.compile(
    r"(?:^|(?<=[.\n]))[^.\n]*\b(?:you|your)\b[^.\n]*?"
    r"\b(?:should|can|could|may|ought to|need to)\b[^.\n]*?"
    r"\b(?:take|use|start|increase|decrease|double|halve|stop|skip)\b[^.\n]*"
    r"(?:\d+\s*(?:mg|mcg|µg|g|ml|units?|tablets?|pills?)|\bdose\b)[^.\n]*\.?",
    re.IGNORECASE,
)
DIAGNOSIS_STATEMENT_RE = re.compile(
    r"(?:^|(?<=[.\n]))[^.\n]*\byou (?:have|are (?:suffering from|experiencing)|likely have)\b"
    r"[^.\n]*\.?",
    re.IGNORECASE,
)

EMERGENCY_RESPONSE = (
    "This sounds like it could be a medical emergency, and that is outside what an "
    "information tool should handle.\n\n"
    "Please contact emergency services now (for example 112 in the EU and India, 911 in "
    "the US, 999 in the UK, or your local emergency number), or go to the nearest "
    "emergency department. If someone is with you, tell them what is happening.\n\n"
    "This system does not triage symptoms and cannot assess how urgent a specific "
    "situation is. It has not attempted to retrieve evidence for this request."
)

SELF_HARM_RESPONSE = (
    "It sounds like you may be going through something very painful, and I am not able "
    "to help with this as an information-retrieval tool.\n\n"
    "Please reach out to someone who can help right now: your local emergency number, a "
    "crisis line in your country, or a trusted person nearby. In India, Tele-MANAS is "
    "available on 14416; in the US and Canada you can call or text 988; in the UK and "
    "Ireland, Samaritans is on 116 123. Findahelpline.com lists services for other "
    "countries.\n\n"
    "If you are in immediate danger, please contact emergency services."
)

PERSONALIZED_NOTE = (
    "This answer gives general reference information only. It is not a diagnosis and "
    "not a dosing recommendation for your situation - a clinician who knows your "
    "history needs to make that call."
)


class GuardrailEngine:
    """Classifies requests and sanitises drafted answers."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # ------------------------------------------------------------ pre-check
    def pre_check(self, query: str) -> SafetyAssessment:
        """Classify a query before any planning or retrieval happens."""
        text = (query or "").strip()

        if not text:
            return SafetyAssessment(
                risk_category=RiskCategory.NON_MEDICAL,
                block_pipeline=True,
                matched_rules=["input.empty"],
                warnings=["Empty query."],
                direct_response="Please enter a question.",
            )

        if len(text) > self.settings.max_query_chars:
            return SafetyAssessment(
                risk_category=RiskCategory.NON_MEDICAL,
                block_pipeline=True,
                matched_rules=["input.too_long"],
                warnings=[f"Query exceeds {self.settings.max_query_chars} characters."],
                direct_response=(
                    f"That query is too long ({len(text)} characters). Please shorten it to "
                    f"under {self.settings.max_query_chars} characters."
                ),
            )

        matched_self_harm = _matching(SELF_HARM_RULES, text)
        if matched_self_harm:
            return SafetyAssessment(
                risk_category=RiskCategory.SELF_HARM,
                block_pipeline=True,
                matched_rules=[r.id for r in matched_self_harm],
                warnings=["Request routed to crisis guidance; retrieval was not performed."],
                direct_response=SELF_HARM_RESPONSE,
            )

        matched_emergency = _matching(EMERGENCY_RULES, text)
        if matched_emergency:
            return SafetyAssessment(
                risk_category=RiskCategory.EMERGENCY,
                block_pipeline=True,
                matched_rules=[r.id for r in matched_emergency],
                warnings=[
                    "Possible emergency detected: " + "; ".join(r.description for r in matched_emergency)
                ],
                direct_response=EMERGENCY_RESPONSE,
            )

        personal = _matching(
            PERSONAL_CONTEXT_RULES + DOSING_REQUEST_RULES + DIAGNOSIS_REQUEST_RULES
            + STOPPING_MEDICATION_RULES,
            text,
        )
        if personal:
            warnings = [
                "Detected a patient-specific request: "
                + "; ".join(sorted({r.description for r in personal}))
            ]
            return SafetyAssessment(
                risk_category=RiskCategory.PERSONALIZED_CLINICAL_DECISION,
                block_pipeline=False,
                matched_rules=[r.id for r in personal],
                warnings=warnings,
                safety_note=PERSONALIZED_NOTE,
            )

        return SafetyAssessment(risk_category=RiskCategory.GENERAL_INFORMATION)

    def synthesis_constraints(self, assessment: SafetyAssessment | None) -> str:
        """Constraint text injected into the synthesizer prompt."""
        base = [
            "Provide general medical information grounded in the evidence block.",
            "Do not diagnose the user and do not recommend an individualised dose.",
        ]
        if assessment and assessment.risk_category is RiskCategory.PERSONALIZED_CLINICAL_DECISION:
            base.extend(
                [
                    "The user is asking about their own care. Explain what the general "
                    "evidence says, then state explicitly that the specific decision "
                    "belongs to their prescriber or pharmacist.",
                    "Do NOT produce a number of milligrams, a frequency, or a titration "
                    "schedule as a recommendation for this person.",
                ]
            )
        return "\n".join(f"- {line}" for line in base)

    # ----------------------------------------------------------- post-check
    def post_check(
        self, answer: str, assessment: SafetyAssessment | None = None
    ) -> tuple[str, list[str]]:
        """Redact unsafe patterns from a drafted answer.

        Returns ``(sanitised_answer, warnings)``.
        """
        warnings: list[str] = []
        sanitised = answer or ""

        if DOSE_INSTRUCTION_RE.search(sanitised):
            sanitised = DOSE_INSTRUCTION_RE.sub(
                "[Removed: an individualised dosing instruction. Dosing decisions for a "
                "specific person must come from a qualified prescriber.]",
                sanitised,
            )
            warnings.append("Removed an individualised dosing instruction from the answer.")

        if DIAGNOSIS_STATEMENT_RE.search(sanitised):
            sanitised = DIAGNOSIS_STATEMENT_RE.sub(
                "[Removed: a statement asserting a personal diagnosis. This system does "
                "not diagnose.]",
                sanitised,
            )
            warnings.append("Removed a personal diagnostic assertion from the answer.")

        if assessment and assessment.safety_note and assessment.safety_note not in sanitised:
            sanitised = f"{sanitised.rstrip()}\n\n{assessment.safety_note}"

        return sanitised.strip(), warnings


def _matching(rules: Iterable[Rule], text: str) -> list[Rule]:
    return [rule for rule in rules if rule.matches(text)]
