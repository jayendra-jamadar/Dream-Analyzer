"""
helpers.py  —  AI-Powered Dream Analysis System
Agents: Backend Coder (2) + Tester (3) + Quality (5)

Fixes:
  - ADDED get_cluster_name() — was missing, causing ImportError crash on startup
  - ADDED CLUSTER_PROFILES dict — drives all cluster name/emoji/color/meaning logic
  - clean_dream_text, compute_sentiment_vader, count_words — unchanged, stable
  - BertEmbedder — unchanged, stable
  - All VALID_* constants — unchanged

Optimizations (v2):
  - VADER SentimentIntensityAnalyzer is now a lazy singleton (no re-creation per call)
  - Added module-level _safe_value() helper (was inner function in parse_form)
  - Added logging instead of silent failures
"""

import re
import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("dream-ai")

# ─── Validation constants ─────────────────────────────────────────────────────

VALID_EMOTIONS    = {"Joy", "Fear", "Sadness", "Calm", "Neutral", "Anger", "Surprise", "Disgust"}
VALID_ACTIVITIES  = {"Talking", "Running", "Flying", "Falling", "Fighting",
                     "Walking", "Swimming", "Searching", "Other"}
VALID_SEASONS     = {"Summer", "Winter", "Spring", "Autumn", "Monsoon"}
VALID_STAGES      = {"REM", "N1", "N2", "N3", "Deep"}
VALID_LUCID       = {"Yes", "No"}


# ─── Module-level safe-value helper ───────────────────────────────────────────

def safe_value(val, valid_set, default):
    """Normalise a form value to Title Case and validate against a known set."""
    v = str(val or "").strip().title()
    return v if v in valid_set else default


# ─── Cluster identity system ──────────────────────────────────────────────────
#
# Each cluster profile has:
#   emoji       — the single visual anchor in the UI
#   name        — short human-readable label (replaces "Cluster 3")
#   subtitle    — one-line descriptor shown below the name
#   color       — CSS accent color for this cluster's cards/badges
#   bg          — subtle background tint (rgba string)
#   description — two-sentence psychological summary
#
# These are DEFAULTS.  get_cluster_name() also accepts a live cluster_meta
# dict and will enrich the label based on real data (top_emotion, avg_stress).

EMOTION_PROFILES = {
    # emotion → (emoji, name_prefix, subtitle, color, bg)
    "Fear":    ("😰", "Anxiety & Fear",   "Processing unresolved tension",
                "#ef5350", "rgba(239,83,80,.12)"),
    "Sadness": ("💙", "Grief & Longing",  "Emotional weight seeking release",
                "#5c85d6", "rgba(92,133,214,.12)"),
    "Joy":     ("✨", "Joy & Wonder",     "Your mind celebrating life",
                "#f9c84e", "rgba(249,200,78,.10)"),
    "Calm":    ("🌿", "Peace & Stillness","Restful, integrative processing",
                "#4ecdc4", "rgba(78,205,196,.10)"),
    "Neutral": ("🌙", "Quiet Processing", "Sorting everyday memories",
                "#8b7cf8", "rgba(139,124,248,.10)"),
    "Anger":   ("🔥", "Conflict & Heat",  "Tension looking for an outlet",
                "#ff7043", "rgba(255,112,67,.12)"),
    "Surprise":("⚡", "Unexpected Events","The mind meeting the unknown",
                "#ab47bc", "rgba(171,71,188,.10)"),
    "Disgust": ("🌫️", "Rejection Dreams", "Boundaries and discomfort explored",
                "#78909c", "rgba(120,144,156,.10)"),
}

_DEFAULT_PROFILE = ("🌙", "Dream Cluster",   "A unique pattern of night visions",
                    "#8b7cf8", "rgba(139,124,248,.10)")


def get_cluster_name(cluster_id, cluster_meta: dict = None) -> dict:
    """
    Return a rich cluster-identity dict for a given cluster_id.

    Parameters
    ----------
    cluster_id  : int | None
    cluster_meta: dict  — loaded from cluster_meta.json  { "0": {...}, "1": {...} }
                          Pass None to use emotion-only defaults.

    Returns
    -------
    dict with keys:
        id, name, subtitle, emoji, color, bg, description,
        top_emotion, avg_stress, avg_sentiment, lucid_pct, count
    """
    # Graceful None handling (GET request before any POST)
    if cluster_id is None:
        return {
            "id": None, "name": "—", "subtitle": "", "emoji": "🌙",
            "color": "#8b7cf8", "bg": "rgba(139,124,248,.08)",
            "description": "", "top_emotion": "Neutral",
            "avg_stress": 0.0, "avg_sentiment": 0.0,
            "lucid_pct": 0.0, "count": 0,
        }

    # Pull live stats from cluster_meta if available
    meta_entry = {}
    if cluster_meta:
        meta_entry = cluster_meta.get(str(cluster_id), {})

    top_emotion   = meta_entry.get("top_emotion",  "Neutral")
    avg_stress    = float(meta_entry.get("avg_stress",    0.5))
    avg_sentiment = float(meta_entry.get("avg_sentiment", 0.0))
    lucid_pct     = float(meta_entry.get("lucid_pct",     0.0))
    count         = int(meta_entry.get("count", 0))

    # Override emotion classification based on stress + sentiment
    # (cluster_meta emotion may be 'Neutral' even for anxious clusters)
    effective_emotion = top_emotion
    if avg_stress > 0.65 and top_emotion in ("Neutral", "Calm"):
        effective_emotion = "Fear"
    elif avg_sentiment > 0.6 and top_emotion in ("Neutral",):
        effective_emotion = "Joy"
    elif avg_sentiment < -0.5 and top_emotion in ("Neutral",):
        effective_emotion = "Sadness"

    emoji, name_prefix, subtitle, color, bg = EMOTION_PROFILES.get(
        effective_emotion, _DEFAULT_PROFILE
    )

    # Enrich name with stress context
    stress_suffix = ""
    if avg_stress > 0.70:
        stress_suffix = " (High Tension)"
    elif avg_stress < 0.15 and effective_emotion in ("Joy", "Calm", "Neutral"):
        stress_suffix = " (Very Relaxed)"

    # Lucid note
    lucid_note = " · Lucid" if lucid_pct > 15 else ""

    name = f"{name_prefix}{stress_suffix}{lucid_note}"

    # Two-sentence description
    descriptions = {
        "Fear":     ("These dreams carry the weight of unresolved anxiety or tension. "
                     "Your mind is actively working through challenges — that's a healthy sign."),
        "Sadness":  ("This cluster reflects emotional depth and a longing to process loss. "
                     "Dreams like these often signal that healing is quietly underway."),
        "Joy":      ("A beautiful space of optimism and wonder. "
                     "Your subconscious is celebrating connections and possibility."),
        "Calm":     ("Peaceful and integrative — your mind rests well. "
                     "These dreams suggest emotional clarity and inner balance."),
        "Neutral":  ("Routine processing of the day's thoughts and memories. "
                     "Quiet, orderly, and grounding — your mind is doing its job."),
        "Anger":    ("Suppressed frustration seeking a safe outlet during sleep. "
                     "Awareness of these patterns is the first step toward release."),
        "Surprise": ("Your mind is grappling with the unexpected. "
                     "These dreams are the psyche's way of rehearsing adaptability."),
        "Disgust":  ("Dreams of rejection or discomfort signal boundary awareness. "
                     "Your subconscious is clarifying what no longer belongs in your life."),
    }
    description = descriptions.get(effective_emotion,
                                   "A distinct pattern of dream experiences unique to this cluster.")

    return {
        "id":           cluster_id,
        "name":         name,
        "subtitle":     subtitle,
        "emoji":        emoji,
        "color":        color,
        "bg":           bg,
        "description":  description,
        "top_emotion":  top_emotion,
        "avg_stress":   round(avg_stress, 3),
        "avg_sentiment": round(avg_sentiment, 3),
        "lucid_pct":    lucid_pct,
        "count":        count,
    }


# ─── Text utilities ───────────────────────────────────────────────────────────

def clean_dream_text(text: str) -> str:
    """Remove noise while preserving emotional language."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([!?.]){3,}", r"\1", text)
    text = re.sub(r"[^\w\s'\".,!?;:()\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── VADER singleton ──────────────────────────────────────────────────────────

_vader_analyzer = None


def _get_vader():
    """Lazy-init VADER analyzer — created once, reused forever."""
    global _vader_analyzer
    if _vader_analyzer is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader_analyzer = SentimentIntensityAnalyzer()
            logger.info("VADER SentimentIntensityAnalyzer initialized")
        except ImportError:
            logger.warning("vaderSentiment not installed — sentiment will default to 0.0")
            return None
    return _vader_analyzer


def compute_sentiment_vader(text: str) -> float:
    """Compound sentiment in [-1, 1]. Falls back to 0.0."""
    analyzer = _get_vader()
    if analyzer is None:
        return 0.0
    try:
        score = analyzer.polarity_scores(clean_dream_text(text))["compound"]
        return round(float(score), 4)
    except Exception:
        return 0.0


def count_words(text: str) -> int:
    if not text or not isinstance(text, str):
        return 0
    return len(text.split())


# ─── BERT Embedder ────────────────────────────────────────────────────────────

class BertEmbedder(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible BERT sentence encoder.
    Uses all-MiniLM-L6-v2 → 384-dim dense vectors.
    Replaces TF-IDF on Top_Keywords.
    """

    def __init__(self, column_name: str = "Dream_Text",
                 model_name: str = "all-MiniLM-L6-v2"):
        self.column_name = column_name
        self.model_name  = model_name

    def _get_model(self) -> SentenceTransformer:
        if not hasattr(self, "_bert") or self._bert is None:
            logger.info("Loading BERT model '%s' ...", self.model_name)
            self._bert = SentenceTransformer(self.model_name)
            logger.info("BERT model loaded successfully")
        return self._bert

    def _extract_texts(self, X) -> list:
        if isinstance(X, pd.DataFrame):
            raw = X[self.column_name].astype(str).tolist()
        else:
            raw = [str(t) for t in X]
        return [clean_dream_text(t) or "unknown" for t in raw]

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        texts = self._extract_texts(X)
        model = self._get_model()
        return model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array([f"bert_{i}" for i in range(384)])


# ─── Legacy compat ────────────────────────────────────────────────────────────

class TextColumnSelector(BaseEstimator, TransformerMixin):
    """Kept so old model.pkl files still unpickle cleanly."""

    def __init__(self, column_name: str):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.column_name].astype(str).values
        return X
