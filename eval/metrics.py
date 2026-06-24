"""Caption quality metrics vs reference list (BLEU-n, ROUGE-L, semantic similarity)."""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Sequence

# NLTK ``sentence_bleu(references, hypothesis)``:
# ``references`` = list of reference translations, each ref is a list of tokens.
# For multiple human captions of different lengths, we report **max over references**
# of smoothed sentence BLEU (common practice vs. ambiguous single multi-ref BP).


def _tokenize(s: str) -> List[str]:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return [t for t in s.split() if t]


def _normalize_reference_strings(references: Sequence[str]) -> List[str]:
    out: List[str] = []
    for ref in references:
        if ref is None:
            continue
        s = str(ref).strip()
        if s:
            out.append(s)
    return out


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(f"[eval.metrics] {message}", file=sys.stderr)


_WARNED: set[str] = set()
_SENTENCE_BERT_MODEL: Any | None = None
_SIMCSE_COMPONENTS: tuple[Any, Any, str] | None = None
_SENTENCE_BERT_LOAD_FAILED = False
_SIMCSE_LOAD_FAILED = False
_TEXT_METRIC_DEVICE = os.environ.get("SHAPELLM_TEXT_METRIC_DEVICE", "cpu")
_SEMANTIC_METRICS_ENABLED = os.environ.get("SHAPELLM_ENABLE_SEMANTIC_METRICS", "1").lower() not in {
    "0",
    "false",
    "no",
}
_SENTENCE_BERT_MODEL_ID = os.environ.get(
    "SHAPELLM_SENTENCE_BERT_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
_SIMCSE_MODEL_ID = os.environ.get(
    "SHAPELLM_SIMCSE_MODEL",
    "princeton-nlp/sup-simcse-roberta-base",
)


def _resolve_text_metric_device() -> str:
    if _TEXT_METRIC_DEVICE != "auto":
        return _TEXT_METRIC_DEVICE
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _get_sentence_bert_model() -> Any | None:
    global _SENTENCE_BERT_LOAD_FAILED, _SENTENCE_BERT_MODEL
    if not _SEMANTIC_METRICS_ENABLED or _SENTENCE_BERT_LOAD_FAILED:
        return None
    if _SENTENCE_BERT_MODEL is not None:
        return _SENTENCE_BERT_MODEL
    try:
        from sentence_transformers import SentenceTransformer

        _SENTENCE_BERT_MODEL = SentenceTransformer(
            _SENTENCE_BERT_MODEL_ID,
            device=_resolve_text_metric_device(),
        )
    except Exception as exc:
        _SENTENCE_BERT_LOAD_FAILED = True
        _warn_once(
            "sentence_bert_load",
            f"Sentence-BERT metric disabled: failed to load {_SENTENCE_BERT_MODEL_ID!r}: {type(exc).__name__}: {exc}",
        )
        return None
    return _SENTENCE_BERT_MODEL


def _get_simcse_components() -> tuple[Any, Any, str] | None:
    global _SIMCSE_COMPONENTS, _SIMCSE_LOAD_FAILED
    if not _SEMANTIC_METRICS_ENABLED or _SIMCSE_LOAD_FAILED:
        return None
    if _SIMCSE_COMPONENTS is not None:
        return _SIMCSE_COMPONENTS
    try:
        from transformers import AutoModel, AutoTokenizer

        device = _resolve_text_metric_device()
        tokenizer = AutoTokenizer.from_pretrained(_SIMCSE_MODEL_ID)
        model = AutoModel.from_pretrained(_SIMCSE_MODEL_ID).to(device)
        model.eval()
        _SIMCSE_COMPONENTS = (tokenizer, model, device)
    except Exception as exc:
        _SIMCSE_LOAD_FAILED = True
        _warn_once(
            "simcse_load",
            f"SimCSE metric disabled: failed to load {_SIMCSE_MODEL_ID!r}: {type(exc).__name__}: {exc}",
        )
        return None
    return _SIMCSE_COMPONENTS


def _sentence_bert_similarity(prediction: str, references: Sequence[str]) -> float | None:
    global _SENTENCE_BERT_LOAD_FAILED
    model = _get_sentence_bert_model()
    if model is None:
        return None
    try:
        import torch

        texts = [prediction, *references]
        embeddings = model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sims = torch.matmul(embeddings[0:1], embeddings[1:].T).view(-1)
        return float(torch.max(sims).item()) if sims.numel() else 0.0
    except Exception as exc:
        _SENTENCE_BERT_LOAD_FAILED = True
        _warn_once(
            "sentence_bert_score",
            f"Sentence-BERT metric disabled for this process: {type(exc).__name__}: {exc}",
        )
        return None


def _simcse_similarity(prediction: str, references: Sequence[str]) -> float | None:
    global _SIMCSE_LOAD_FAILED
    components = _get_simcse_components()
    if components is None:
        return None
    tokenizer, model, device = components
    try:
        import torch
        import torch.nn.functional as F

        texts = [prediction, *references]
        encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded, return_dict=True)
            embeddings = getattr(outputs, "pooler_output", None)
            if embeddings is None:
                embeddings = outputs.last_hidden_state[:, 0]
            embeddings = F.normalize(embeddings, p=2, dim=1)
            sims = torch.matmul(embeddings[0:1], embeddings[1:].T).view(-1)
        return float(torch.max(sims).item()) if sims.numel() else 0.0
    except Exception as exc:
        _SIMCSE_LOAD_FAILED = True
        _warn_once("simcse_score", f"SimCSE metric disabled for this process: {type(exc).__name__}: {exc}")
        return None


def _lcs_length(a: List[str], b: List[str]) -> int:
    """Longest common subsequence length (words)."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 0
    dp = [[0] * (nb + 1) for _ in range(na + 1)]
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[na][nb]


def rouge_l_f1(prediction: str, reference: str) -> float:
    """Unigram ROUGE-L F1 (recall-oriented LCS / token F1 variant)."""
    p = _tokenize(prediction)
    r = _tokenize(reference)
    if not p or not r:
        return 0.0
    lcs = _lcs_length(p, r)
    prec = lcs / len(p)
    rec = lcs / len(r)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _bleu_n_single_ref(candidate: List[str], ref: List[str], n: int) -> float:
    """Unsmoothed BLEU-n precision (one reference) for fallback."""
    from collections import Counter

    def ngrams(tokens: List[str], n: int):
        if len(tokens) < n:
            return []
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    c = ngrams(candidate, n)
    if not c:
        return 0.0
    rc = Counter(ngrams(ref, n))
    counts = Counter(c)
    clip = sum(min(counts[g], rc.get(g, 0)) for g in counts)
    return clip / max(len(c), 1)


def _bleu_max_over_refs_nltk(
    hyp: List[str], refs_tok: List[List[str]], weights: tuple[float, float, float, float]
) -> float:
    """Smoothed sentence BLEU with one reference; take max over references."""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    sm = SmoothingFunction().method1
    scores = []
    for ref in refs_tok:
        # One reference: ``references`` = [ref] (outer list length 1)
        s = sentence_bleu([ref], hyp, weights=weights, smoothing_function=sm)
        scores.append(float(s))
    return max(scores) if scores else 0.0


def compute_text_metrics(prediction: str, references: Sequence[str]) -> Dict[str, float | None]:
    """
    BLEU-1..4: max over reference captions of smoothed NLTK sentence BLEU
    (each caption is one reference; avoids multi-ref brevity / clip ambiguity).

    ROUGE-L: max F1 vs any reference string.

    Sentence-BERT / SimCSE: max cosine similarity vs any reference string.

    Fallback (no NLTK): max over refs of simple n-gram precision per order.
    """
    ref_strs = _normalize_reference_strings(references)
    refs_tok = [t for t in (_tokenize(r) for r in ref_strs) if t]
    hyp = _tokenize(prediction)
    out: Dict[str, float | None] = {}

    if not refs_tok or not hyp:
        for k in ("bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_l", "sentence_bert", "simcse"):
            out[k] = 0.0
        return out

    out["rouge_l"] = max(rouge_l_f1(prediction, ref) for ref in ref_strs)
    out["sentence_bert"] = _sentence_bert_similarity(prediction, ref_strs)
    out["simcse"] = _simcse_similarity(prediction, ref_strs)

    weights_list = [
        (1.0, 0, 0, 0),
        (0.5, 0.5, 0, 0),
        (1.0 / 3, 1.0 / 3, 1.0 / 3, 0),
        (0.25, 0.25, 0.25, 0.25),
    ]
    try:
        for i, w in enumerate(weights_list, start=1):
            out[f"bleu_{i}"] = _bleu_max_over_refs_nltk(hyp, refs_tok, w)
    except Exception:
        for n in range(1, 5):
            out[f"bleu_{n}"] = max(
                float(_bleu_n_single_ref(hyp, ref, n)) for ref in refs_tok
            )

    return out
