"""Structure, answer-tier, and error-breakdown metrics for holdout generations."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Optional


FINAL_ANSWER_RES = [
    re.compile(r"\*\*\s*Final Answer\s*:?\s*\*\*\s*:?\s*(.+)", re.IGNORECASE | re.DOTALL),
    re.compile(r"\*\*\s*Final Answer\s*\*\*\s*:?\s*(.+)", re.IGNORECASE | re.DOTALL),
    re.compile(r"Final Answer\s*:?\s*(.+)", re.IGNORECASE | re.DOTALL),
    re.compile(r"Therefore,? the answer is:?\s*(.+)", re.IGNORECASE | re.DOTALL),
]

PREAMBLE_RES = re.compile(
    r"^\s*(to determine|let'?s |we need to|the (question|task|figure|image)|based on the)",
    re.IGNORECASE,
)
REPETITION_RE = re.compile(r"(.{24,}?)\1{3,}", re.DOTALL)
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def normalize_answer(value: Any) -> str:
    """Same whitespace+lowercase rule as the holdout notebook exact-match."""
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = text.replace("\\%", "%")
    return " ".join(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?|[%+\-]", text))


def token_match(ground_truth: Any, candidate: Any) -> bool:
    """Whole-token containment; does not use raw substrings (so 4 does not match 94)."""
    expected = normalize_text(ground_truth)
    actual = normalize_text(candidate)
    if not expected or not actual:
        return False
    if expected == actual:
        return True
    exp_tokens = expected.split()
    act_tokens = actual.split()
    width = len(exp_tokens)
    return any(act_tokens[i : i + width] == exp_tokens for i in range(len(act_tokens) - width + 1))


def extract_final_answer(text: str) -> str:
    """Robust extractor: markdown Final Answer, plain marker, or 'Therefore, the answer is'."""
    for pattern in FINAL_ANSWER_RES:
        matches = pattern.findall(text or "")
        if matches:
            answer = matches[-1].strip().splitlines()[0].strip()
            answer = answer.strip("\"'`*:").strip()
            return answer
    return ""


ALLOWED_EXTRA_TOKENS = {
    "s",
    "l",
    "n",
    "k",
    "ua",
    "um",
    "mm",
    "cm",
    "kg",
    "pct",
    "hz",
    "db",
}
BANNED_EXTRA_TOKENS = {
    "no",
    "not",
    "none",
    "and",
    "or",
    "then",
    "vs",
    "except",
    "without",
}


def extra_token_ok(token: str) -> bool:
    if token in BANNED_EXTRA_TOKENS:
        return False
    if token in ALLOWED_EXTRA_TOKENS or token in {"%", "+"}:
        return True
    return len(token) == 1 and token.isalpha()


def extracted_match(ground_truth: Any, candidate: Any) -> bool:
    """True if the extracted answer is the GT, allowing units/labels but not extra entities."""
    if not str(candidate or "").strip():
        return False
    if normalize_answer(candidate) == normalize_answer(ground_truth):
        return True
    gt_n = normalize_text(ground_truth)
    pred_n = normalize_text(candidate)
    if not gt_n or not pred_n:
        return False
    if gt_n == pred_n:
        return True
    gt_toks = gt_n.split()
    pred_toks = pred_n.split()
    width = len(gt_toks)
    for i in range(len(pred_toks) - width + 1):
        if pred_toks[i : i + width] != gt_toks:
            continue
        leftover = pred_toks[:i] + pred_toks[i + width :]
        if leftover and all(extra_token_ok(tok) for tok in leftover):
            return True
    return False


def structure_flags(text: str) -> Dict[str, Any]:
    body = text or ""
    stripped = body.lstrip()
    step_ids = [int(n) for n in re.findall(r"(?im)^\s*step\s+(\d+)\s*:", body)]
    has_plain_final = bool(re.search(r"(?i)(?<!\*)Final Answer\s*:", body))
    has_md_final = bool(re.search(r"(?i)\*\*\s*Final Answer\s*\*\*", body))
    return {
        "starts_step1": stripped.startswith("Step 1:"),
        "has_step2": bool(re.search(r"(?im)^\s*step\s+2\s*:", body)),
        "n_steps": len(step_ids),
        "has_final_answer_plain": has_plain_final,
        "has_final_answer_markdown": has_md_final,
        "has_final_answer_any": has_plain_final or has_md_final or bool(extract_final_answer(body)),
        "has_preamble": bool(PREAMBLE_RES.search(stripped)),
        "has_repetition": bool(REPETITION_RE.search(body)),
        "n_chars": len(body),
    }


def score_generation(
    text: str,
    ground_truth: Any,
    official_pred: Optional[str] = None,
) -> Dict[str, Any]:
    """Score one generation for structure, answer tiers, and error type."""
    flags = structure_flags(text)
    robust_pred = extract_final_answer(text)
    official = (official_pred or "").strip()
    token_official = extracted_match(ground_truth, official)
    token_robust = extracted_match(ground_truth, robust_pred)
    token_pred = token_official or token_robust
    pred = robust_pred if token_robust and not token_official else (official or robust_pred)

    exact_official = bool(official) and normalize_answer(official) == normalize_answer(ground_truth)
    token_body = token_match(ground_truth, text)
    conclusion = text[-400:] if text else ""
    token_conclusion = token_match(ground_truth, conclusion)
    numeric_pred = extracted_match(ground_truth, pred) and bool(NUMBER_RE.search(str(ground_truth)))

    structure_hits = [
        flags["starts_step1"],
        flags["has_step2"],
        flags["has_final_answer_plain"],
        not flags["has_preamble"],
        flags["n_steps"] >= 2,
    ]
    structure_score = sum(structure_hits) / len(structure_hits)

    if exact_official or token_pred:
        error_type = "correct_extracted"
    elif token_conclusion and not pred:
        error_type = "correct_unextracted"
    elif token_body and pred:
        error_type = "mentions_gt_wrong_commit"
    elif pred:
        error_type = "wrong_committed"
    else:
        error_type = "no_answer"

    return {
        **flags,
        "official_pred": official,
        "robust_pred": robust_pred,
        "pred": pred,
        "exact_official": exact_official,
        "token_official": token_official,
        "token_pred": token_pred,
        "token_body": token_body,
        "token_conclusion": token_conclusion,
        "numeric_pred": numeric_pred,
        "structure_score": structure_score,
        "structured_correct": bool(flags["starts_step1"] and flags["has_final_answer_plain"] and token_pred),
        "error_type": error_type,
    }
