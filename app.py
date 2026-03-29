"""
app.py  —  AI-Powered Dream Analysis System (v3 — Performance Optimized)

Optimizations vs v2:
  OPT 1: Cluster stats pre-computed at startup → _cluster_cache dict
  OPT 2: Similarity search limited to top-500 vectors per cluster
  OPT 3: BERT model pre-warmed at startup (no first-request delay)
  OPT 4: API clients created once (singleton pattern)
  OPT 5: tsne_data.json cached at startup
  OPT 6: /api/stats served from cache (no recomputation)
  OPT 7: Proper logging replaces print()
  OPT 8: Confidence score added to prediction
  OPT 9: Cluster route validation (graceful 404)
  OPT 10: Error handlers for 404/500
  OPT 11: DataFrame → records conversion done in Python (not Jinja)
  OPT 12: Submit button disabled during processing (template JS)
"""

import json
import logging
import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dream-ai")

# ── Single clean import block from helpers ────────────────────────────────────
# BertEmbedder MUST be imported before joblib.load so pickle resolves the class
from helpers import (                                           # noqa: F401
    BertEmbedder, TextColumnSelector,
    clean_dream_text, compute_sentiment_vader, count_words,
    get_cluster_name, safe_value,
    VALID_EMOTIONS, VALID_ACTIVITIES, VALID_SEASONS,
    VALID_STAGES, VALID_LUCID,
)

# ─── Paths ────────────────────────────────────────────────────────────────────

MODEL_PATH   = "model.pkl"
SCORED_PATH  = "scored_dreams.csv"
VECTORS_PATH = "data_vectors.npy"
META_PATH    = "cluster_meta.json"
TSNE_PATH    = "tsne_data.json"

# ─── Feature columns (must match train_model.py exactly) ─────────────────────

FEATURE_COLS = [
    "Sentiment", "Word_Count", "Stress_Before_Sleep",
    "Emotion", "Lucid", "Dominant_Activity",
    "Season", "Sleep_Stage", "Dream_Text",
]

# ─── App + artefact loading ───────────────────────────────────────────────────

app = Flask(__name__)


def _load_artefacts():
    missing = [p for p in (MODEL_PATH, SCORED_PATH, VECTORS_PATH) if not os.path.exists(p)]
    if missing:
        logger.critical("Missing model artefacts: %s", missing)
        logger.critical("Run:  python train_model.py --data <your_dataset.csv>")
        sys.exit(1)

    pipeline     = joblib.load(MODEL_PATH)
    df           = pd.read_csv(SCORED_PATH)
    X_vec        = np.load(VECTORS_PATH)
    cluster_meta = {}
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            cluster_meta = json.load(f)
    return pipeline, df, X_vec, cluster_meta


pipeline, df, X_vec, cluster_meta = _load_artefacts()


# ─── Pre-warm BERT model ─────────────────────────────────────────────────────
# Force BERT to load NOW so the first user request isn't slow

def _prewarm_model():
    logger.info("Pre-warming BERT model ...")
    t0 = time.time()
    warmup_row = {c: ("warmup test dream" if c == "Dream_Text" else
                       0.5 if c in ("Sentiment", "Stress_Before_Sleep") else
                       1.0 if c == "Word_Count" else
                       "Neutral" if c == "Emotion" else
                       "No" if c == "Lucid" else
                       "Other" if c == "Dominant_Activity" else
                       "Summer" if c == "Season" else
                       "REM")
                  for c in FEATURE_COLS}
    warmup_df = pd.DataFrame([warmup_row], columns=FEATURE_COLS)
    try:
        pipeline.named_steps["preprocessor"].transform(warmup_df)
        logger.info("BERT pre-warmed in %.1fs", time.time() - t0)
    except Exception as e:
        logger.warning("BERT pre-warm failed (non-fatal): %s", e)


_prewarm_model()


# ─── Cluster cache (pre-computed stats) ──────────────────────────────────────
# Avoids recomputing .mode(), .mean() on every request

_cluster_cache = {}
_valid_cluster_ids = set()
_api_stats_cache = None


def _build_cluster_cache():
    """Pre-compute all cluster statistics at startup."""
    global _cluster_cache, _valid_cluster_ids, _api_stats_cache

    logger.info("Building cluster cache ...")
    t0 = time.time()

    stats_list = []

    for cid in sorted(df["KMeans_Cluster"].dropna().unique()):
        cid = int(cid)
        _valid_cluster_ids.add(cid)

        mask = df["KMeans_Cluster"].values == cid
        indices = np.where(mask)[0]
        cdf = df[mask]

        top_emotion = (cdf["Emotion"].mode(dropna=True).iat[0]
                       if not cdf["Emotion"].mode(dropna=True).empty else "Neutral")
        avg_stress  = round(float(cdf["Stress_Before_Sleep"].astype(float).mean()), 3)
        avg_sent    = round(float(cdf["Sentiment"].astype(float).mean()), 3)
        top_activity = (cdf["Dominant_Activity"].mode(dropna=True).iat[0]
                        if not cdf["Dominant_Activity"].mode(dropna=True).empty else "Other")
        lucid_pct   = round(float((cdf["Lucid"] == "Yes").mean() * 100), 1)

        # Keywords
        kw = "N/A"
        if "Top_Keywords" in cdf.columns:
            kw = ", ".join(
                cdf["Top_Keywords"].astype(str).str.split(",").explode()
                .str.strip().value_counts().head(8).index.tolist()
            )

        # Cluster summary text for AI prompts
        summary_text = (
            f"Cluster {cid} ({top_emotion} dreams): "
            f"avg stress={avg_stress}, avg sentiment={avg_sent}, "
            f"dominant activity={top_activity}, keywords=[{kw}]."
        )

        # Cluster vectors (limit to 500 for fast similarity search)
        cluster_vectors = X_vec[indices]
        if len(indices) > 500:
            sample_idx = np.random.RandomState(42).choice(len(indices), 500, replace=False)
            cluster_vectors_limited = cluster_vectors[sample_idx]
            cluster_indices_limited = indices[sample_idx]
        else:
            cluster_vectors_limited = cluster_vectors
            cluster_indices_limited = indices

        # Psychological profile
        profile = _build_profile(cdf, top_emotion, avg_stress, avg_sent)

        _cluster_cache[cid] = {
            "indices":            indices,
            "vectors_limited":    cluster_vectors_limited,
            "indices_limited":    cluster_indices_limited,
            "top_emotion":        top_emotion,
            "avg_stress":         avg_stress,
            "avg_sentiment":      avg_sent,
            "top_activity":       top_activity,
            "lucid_pct":          lucid_pct,
            "keywords":           kw,
            "count":              int(len(cdf)),
            "summary_text":       summary_text,
            "profile":            profile,
        }

        # API stats entry
        info = get_cluster_name(cid, cluster_meta)
        stats_list.append({
            "cluster_id":    cid,
            "cluster_name":  info["name"],
            "emoji":         info["emoji"],
            "count":         int(len(cdf)),
            "top_emotion":   top_emotion,
            "avg_stress":    avg_stress,
            "avg_sentiment": avg_sent,
            "lucid_pct":     lucid_pct,
        })

    _api_stats_cache = {"clusters": stats_list, "total_dreams": int(len(df))}
    logger.info("Cluster cache built in %.1fs (%d clusters)", time.time() - t0, len(_cluster_cache))


def _build_profile(cluster_df, top_emotion, avg_stress, avg_sentiment):
    """Full psychological profile dict for cluster_profile route."""
    top_activity = (
        cluster_df["Dominant_Activity"].mode(dropna=True).iat[0]
        if not cluster_df["Dominant_Activity"].mode(dropna=True).empty else "Other"
    )
    lucid_pct = round(float((cluster_df["Lucid"] == "Yes").mean() * 100), 1)

    kw = "N/A"
    if "Top_Keywords" in cluster_df.columns:
        kw = ", ".join(
            cluster_df["Top_Keywords"].astype(str).str.split(",").explode()
            .str.strip().value_counts().head(8).index.tolist()
        )

    if top_emotion in ("Fear", "Sadness") or avg_stress > 0.75:
        psych     = ("Dreams here suggest high anxiety or emotional tension — "
                     "overthinking, suppressed fears, or unresolved stress.")
        state     = "Anxious / Stressed"
        treatment = ("Consider speaking with a counsellor or therapist if these feelings persist. "
                     "CBT and structured stress management are effective.")
        self_help = ("Mindful breathing before bed, limiting screen time at night, "
                     "and keeping a dream journal safely process emotions.")
    elif top_emotion in ("Joy", "Calm") and avg_sentiment > 0.3:
        psych     = ("These dreams reflect a balanced, optimistic mindset. "
                     "The subconscious appears peaceful and emotionally integrated.")
        state     = "Mentally Stable / Positive"
        treatment = ("No clinical intervention required. "
                     "Continue maintaining healthy mental and emotional habits.")
        self_help = ("Gratitude practices, social connection, and mindfulness. "
                     "Positive reflection before sleep amplifies this effect.")
    elif top_emotion == "Neutral":
        psych     = ("Routine cognitive processing — the mind sorting everyday thoughts "
                     "with no strong emotional charge.")
        state     = "Calm / Reflective"
        treatment = ("No medical intervention needed unless accompanied by "
                     "emotional numbness or insomnia.")
        self_help = ("Light journaling, a consistent sleep schedule, and brief "
                     "relaxation exercises can maintain this calm state.")
    else:
        psych     = ("Mixed or transitional emotions — possibly identity exploration "
                     "or an ongoing life change.")
        state     = "Transitional / Unsettled"
        treatment = ("Monitor emotional fluctuations; brief therapy can help "
                     "if uncertainty or stress increases.")
        self_help = ("Creative activities, time outdoors, and guided relaxation "
                     "recordings provide stability during transitions.")

    return {
        "top_emotion":          top_emotion,
        "top_activity":         top_activity,
        "avg_stress":           avg_stress,
        "avg_sentiment":        avg_sentiment,
        "common_keywords":      kw,
        "count":                int(len(cluster_df)),
        "lucid_pct":            lucid_pct,
        "psychological_aspect": psych,
        "mental_state":         state,
        "treatment_advice":     treatment,
        "self_help":            self_help,
    }


_build_cluster_cache()


# ─── Cache tsne_data.json ────────────────────────────────────────────────────

_tsne_data = None

def _load_tsne_data():
    global _tsne_data
    if os.path.exists(TSNE_PATH):
        with open(TSNE_PATH) as f:
            _tsne_data = json.load(f)
        logger.info("Loaded tsne_data.json (%d points)", len(_tsne_data))
    else:
        logger.warning("tsne_data.json not found — /cluster_map will be unavailable")


_load_tsne_data()


# ─── AI clients (singletons) ────────────────────────────────────────────────

_oai_client_cached  = "UNSET"   # sentinel — None means "tried, no key"
_anth_client_cached = "UNSET"


def _get_openai_client():
    global _oai_client_cached
    if _oai_client_cached != "UNSET":
        return _oai_client_cached

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        _oai_client_cached = None
        return None
    try:
        from openai import OpenAI
        _oai_client_cached = OpenAI(api_key=key)
        logger.info("OpenAI client initialized")
        return _oai_client_cached
    except Exception as e:
        logger.warning("OpenAI client init failed: %s", e)
        _oai_client_cached = None
        return None


def _get_anthropic_client():
    global _anth_client_cached
    if _anth_client_cached != "UNSET":
        return _anth_client_cached

    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        _anth_client_cached = None
        return None
    try:
        import anthropic
        _anth_client_cached = anthropic.Anthropic(api_key=key)
        logger.info("Anthropic client initialized")
        return _anth_client_cached
    except Exception as e:
        logger.warning("Anthropic client init failed: %s", e)
        _anth_client_cached = None
        return None


# ─── Input parsing ────────────────────────────────────────────────────────────

def parse_form(form: dict) -> dict:
    """Validate and build a feature row from raw POST data."""
    dream_text = clean_dream_text(form.get("Dream_Text", "").strip())
    if not dream_text:
        raise ValueError("Dream description cannot be empty.")

    sentiment  = compute_sentiment_vader(dream_text)
    word_count = count_words(dream_text)

    try:
        stress = float(form.get("Stress_Before_Sleep") or 0.5)
        stress = max(0.0, min(1.0, stress))
    except (ValueError, TypeError):
        stress = 0.5

    return {
        "Sentiment":           sentiment,
        "Word_Count":          float(word_count),
        "Stress_Before_Sleep": stress,
        "Emotion":             safe_value(form.get("Emotion"),           VALID_EMOTIONS,   "Neutral"),
        "Lucid":               safe_value(form.get("Lucid"),             VALID_LUCID,      "No"),
        "Dominant_Activity":   safe_value(form.get("Dominant_Activity"), VALID_ACTIVITIES, "Other"),
        "Season":              safe_value(form.get("Season"),            VALID_SEASONS,    "Summer"),
        "Sleep_Stage":         safe_value(form.get("Sleep_Stage"),       VALID_STAGES,     "REM"),
        "Dream_Text":          dream_text,
        "_auto_sentiment":     sentiment,
        "_auto_word_count":    word_count,
    }


# ─── Rule-based fallback explanation ─────────────────────────────────────────

def _rule_based_explanation(dream_text: str, cluster_id: int) -> str:
    """Intelligent fallback when no AI key is set."""
    cache = _cluster_cache.get(cluster_id, {})
    top_emotion = cache.get("top_emotion", "Neutral")
    avg_stress  = cache.get("avg_stress", 0.5)

    word_count = count_words(dream_text)
    length_note = (
        "The richness of detail suggests active emotional processing during sleep."
        if word_count > 150
        else "The concise recall hints at a vivid but quickly passing dream state."
    )

    profiles = {
        "Fear":    ("anxiety or unresolved stress",
                    "grounding techniques and journaling may help process underlying fears"),
        "Sadness": ("grief or unmet emotional needs",
                    "connecting with supportive people and self-compassion can ease this"),
        "Joy":     ("positive emotional integration",
                    "your mind is reinforcing optimistic experiences — nurture that"),
        "Calm":    ("restful cognitive processing",
                    "your sleep quality appears healthy; maintain your routines"),
        "Neutral": ("routine information sorting by the subconscious",
                    "light journaling can help surface any hidden meaning"),
    }
    aspect, advice = profiles.get(top_emotion, profiles["Neutral"])
    stress_note = (
        "The elevated pre-sleep stress in this cluster suggests the mind is working through tension."
        if avg_stress > 0.65
        else "Moderate stress levels suggest everyday emotional processing."
    )

    return (
        f"This dream appears to reflect themes of {aspect}. "
        f"{length_note} {stress_note} "
        f"To support your wellbeing, {advice}. "
        "Dreams are a natural part of emotional regulation — they rarely predict the future "
        "but often mirror our inner world."
    )


# ─── Common template context ─────────────────────────────────────────────────

def _base_template_vars():
    """Variables passed to every index.html render to avoid repetition."""
    return {
        "valid_emotions":    sorted(VALID_EMOTIONS),
        "valid_activities":  sorted(VALID_ACTIVITIES),
        "valid_seasons":     sorted(VALID_SEASONS),
        "valid_stages":      sorted(VALID_STAGES),
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    result          = None
    cluster         = None
    cluster_info    = None
    similar_preview = None
    auto_info       = {}
    error_msg       = None
    confidence      = None

    if request.method == "POST":
        t0 = time.time()
        try:
            row = parse_form(request.form)
        except ValueError as e:
            error_msg = str(e)
            return render_template(
                "index.html",
                error_msg=error_msg,
                cluster_info=get_cluster_name(None),
                **_base_template_vars(),
            )

        auto_info = {
            "sentiment":  row.pop("_auto_sentiment"),
            "word_count": row.pop("_auto_word_count"),
        }

        X           = pd.DataFrame([row], columns=FEATURE_COLS)
        cluster     = int(pipeline.predict(X)[0])
        cluster_info = get_cluster_name(cluster, cluster_meta)
        result      = cluster_info["name"]

        # ── Confidence score ──────────────────────────────────────────────
        x_vec = pipeline.named_steps["preprocessor"].transform(X)
        centers = pipeline.named_steps["cluster"].cluster_centers_
        dist_to_own = np.linalg.norm(x_vec - centers[cluster])
        all_dists = [np.linalg.norm(x_vec - c) for c in centers]
        max_dist = max(all_dists) if all_dists else 1.0
        confidence = round((1.0 - dist_to_own / max_dist) * 100, 1) if max_dist > 0 else 50.0
        confidence = max(10.0, min(99.0, confidence))  # clamp to reasonable range

        # ── Similar dreams (optimized: use cached limited vectors) ────────
        cache = _cluster_cache.get(cluster)
        if cache and len(cache["vectors_limited"]) > 0:
            sims      = cosine_similarity(x_vec, cache["vectors_limited"])[0]
            top_local = np.argsort(sims)[::-1][:5]
            top_idx   = cache["indices_limited"][top_local]
            similar_preview = (
                df.iloc[top_idx][["Dream_ID", "Dream_Text", "KMeans_Cluster", "Emotion"]]
                .to_dict("records")
            )
        else:
            similar_preview = []

        elapsed = time.time() - t0
        logger.info("POST / → cluster=%d confidence=%.1f%% similar=%d  (%.2fs)",
                     cluster, confidence, len(similar_preview), elapsed)

    dream_text_value = request.form.get("Dream_Text", "") if request.method == "POST" else ""

    return render_template(
        "index.html",
        result=result,
        cluster=cluster,
        cluster_info=cluster_info or get_cluster_name(None),
        preview=similar_preview,
        dream_text_value=dream_text_value,
        auto_info=auto_info,
        error_msg=error_msg,
        confidence=confidence,
        **_base_template_vars(),
    )


@app.route("/explain", methods=["POST"])
def explain():
    data       = request.get_json(force=True)
    dream_text = clean_dream_text((data.get("dream_text") or "").strip())
    cluster_id = int(data.get("cluster_id", 0))

    if not dream_text:
        return jsonify({"explanation": "No dream text provided.", "source": "error"}), 400

    # Use cached cluster summary instead of recomputing
    cache = _cluster_cache.get(cluster_id, {})
    cluster_summary = cache.get("summary_text", f"Cluster {cluster_id}: insufficient data.")

    system_prompt = (
        "You are a compassionate dream analyst with expertise in Jungian psychology "
        "and cognitive neuroscience. Interpret dreams in 3-4 warm, insightful sentences. "
        "Use plain language. Do not make medical diagnoses. Focus on emotional meaning."
    )
    user_prompt = (
        f'A user described this dream:\n"{dream_text}"\n\n'
        f"AI cluster context:\n{cluster_summary}\n\n"
        "Provide a psychological interpretation focusing on what emotions or "
        "life experiences this dream might reflect."
    )

    # Try OpenAI (singleton client)
    oai = _get_openai_client()
    if oai:
        try:
            resp = oai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user",   "content": user_prompt}],
                max_tokens=250, temperature=0.7,
            )
            return jsonify({"explanation": resp.choices[0].message.content.strip(), "source": "gpt"})
        except Exception as e:
            logger.error("OpenAI API error: %s", e)

    # Try Anthropic (singleton client)
    anth = _get_anthropic_client()
    if anth:
        try:
            resp = anth.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=250,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return jsonify({"explanation": resp.content[0].text.strip(), "source": "claude"})
        except Exception as e:
            logger.error("Anthropic API error: %s", e)

    # Fallback
    return jsonify({
        "explanation": _rule_based_explanation(dream_text, cluster_id),
        "source": "fallback",
    })


@app.route("/similar/<int:cluster_id>")
def similar(cluster_id):
    if cluster_id not in _valid_cluster_ids:
        logger.warning("Invalid cluster_id requested: %d", cluster_id)
        return render_template(
            "index.html",
            error_msg=f"Cluster {cluster_id} does not exist.",
            cluster_info=get_cluster_name(None),
            **_base_template_vars(),
        ), 404

    subset = (
        df[df["KMeans_Cluster"] == cluster_id]
        [["Dream_ID", "Dream_Text", "KMeans_Cluster", "Emotion", "Stress_Before_Sleep"]]
        .copy().sort_values("Dream_ID")
    )
    # Convert to list of dicts in Python (not in Jinja template)
    dreams_list = subset.to_dict("records")
    cluster_info = get_cluster_name(cluster_id, cluster_meta)
    return render_template("similar.html", cluster_id=cluster_id,
                           dreams=dreams_list, cluster_info=cluster_info)


@app.route("/cluster_profile/<int:cluster_id>")
def cluster_profile(cluster_id):
    if cluster_id not in _valid_cluster_ids:
        logger.warning("Invalid cluster_id requested: %d", cluster_id)
        return render_template(
            "index.html",
            error_msg=f"Cluster {cluster_id} does not exist.",
            cluster_info=get_cluster_name(None),
            **_base_template_vars(),
        ), 404

    cluster_info_val = get_cluster_name(cluster_id, cluster_meta)

    # Use cached profile instead of recomputing
    cache = _cluster_cache.get(cluster_id)
    if cache:
        summary = cache["profile"]
    else:
        summary = {
            "top_emotion": "—", "top_activity": "—",
            "avg_stress": "—", "avg_sentiment": "—",
            "common_keywords": "—", "count": 0, "lucid_pct": 0,
            "psychological_aspect": "No data available for this cluster.",
            "mental_state": "N/A", "treatment_advice": "N/A", "self_help": "N/A",
        }

    return render_template("cluster_profile.html", cluster_id=cluster_id,
                           summary=summary, cluster_info=cluster_info_val)


@app.route("/cluster_map")
def cluster_map():
    if _tsne_data is None:
        return "❌ Run visualize_clusters.py first to generate tsne_data.json", 404
    return render_template("cluster_map.html", map_data=_tsne_data)


@app.route("/api/stats")
def api_stats():
    if _api_stats_cache:
        return jsonify(_api_stats_cache)
    return jsonify({"clusters": [], "total_dreams": 0})


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": pipeline is not None,
        "total_dreams": int(len(df)),
        "clusters": len(_valid_cluster_ids),
    })


# ─── Error handlers ──────────────────────────────────────────────────────────

@app.errorhandler(404)
def page_not_found(e):
    return render_template(
        "index.html",
        error_msg="Page not found.",
        cluster_info=get_cluster_name(None),
        **_base_template_vars(),
    ), 404


@app.errorhandler(500)
def internal_error(e):
    logger.exception("Internal server error")
    return render_template(
        "index.html",
        error_msg="Something went wrong. Please try again.",
        cluster_info=get_cluster_name(None),
        **_base_template_vars(),
    ), 500


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("🌙  AI Dream Analysis System  —  Starting …")
    logger.info("    Dataset : %s dreams", f"{len(df):,}")
    logger.info("    Clusters: %d", len(_valid_cluster_ids))
    logger.info("    OpenAI  : %s", "✓ set" if os.environ.get("OPENAI_API_KEY") else "✗ fallback active")
    logger.info("    Anthropic: %s", "✓ set" if os.environ.get("ANTHROPIC_API_KEY") else "✗ not set")
    logger.info("    → http://127.0.0.1:5000")
    logger.info("    → Cluster Map: http://127.0.0.1:5000/cluster_map")
    app.run(debug=True)
