"""
visualize_clusters.py  –  Full Detailed Version + Web Map Support
Agents: Product Manager (1) + Backend Coder (2) + Tester (3) + UI/UX (4) + Quality (5)

This file:
- Loads data_vectors.npy and scored_dreams.csv
- Loads cluster_meta.json
- Creates rich cluster labels
- Generates BOTH PCA and t-SNE matplotlib plots
- Saves tsne_data.json for the interactive Flask cluster map
- No shortcuts — everything is written out fully
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -------------------------------
# Load Data
# -------------------------------
print("📂 [1/5] Loading data files...")

X = np.load("data_vectors.npy")
df = pd.read_csv("scored_dreams.csv")
labels = df["KMeans_Cluster"].values

print(f"      Data shape: {X.shape}")
print(f"      Number of dreams: {len(df):,}")
print(f"      Clusters found: {len(set(labels))}")

# -------------------------------
# Load cluster_meta.json and create rich labels
# -------------------------------
print("📋 [2/5] Loading cluster metadata and creating nice labels...")

cluster_info = {}
try:
    with open("cluster_meta.json") as f:
        meta = json.load(f)
    
    for cid_str, info in meta.items():
        cid = int(cid_str)
        top_emotion = info.get("top_emotion", "Neutral")
        avg_stress = info.get("avg_stress", 0.5)
        avg_sent = info.get("avg_sentiment", 0.0)
        count = info.get("count", 0)
        lucid_pct = info.get("lucid_pct", 0)
        
        # Create meaningful label
        if top_emotion in ["Fear", "Sadness"] or avg_stress > 0.65:
            label = f"{top_emotion} / Anxiety Dreams"
        elif top_emotion in ["Joy", "Calm"] and avg_sent > 0.3:
            label = f"{top_emotion} / Positive & Calm Dreams"
        elif top_emotion == "Neutral":
            label = "Neutral / Routine Processing"
        else:
            label = f"{top_emotion} Dreams"
        
        # Add stress level
        if avg_stress > 0.7:
            label += " (High Stress)"
        elif avg_stress < 0.3:
            label += " (Low Stress)"
        
        # Add extra info
        cluster_info[cid] = {
            "label": label,
            "count": count,
            "top_emotion": top_emotion,
            "avg_stress": avg_stress,
            "avg_sentiment": avg_sent,
            "lucid_pct": lucid_pct
        }
        
    print(f"      Successfully loaded metadata for {len(cluster_info)} clusters")
except Exception as e:
    print(f"⚠️  Could not load cluster_meta.json: {e}")
    print("    Using plain cluster numbers instead")
    cluster_info = {}

# -------------------------------
# Optional: Subsample for faster plotting
# -------------------------------
MAX_SAMPLES = 3000
if len(X) > MAX_SAMPLES:
    print(f"⚡ [3/5] Using subset of {MAX_SAMPLES} samples for faster visualization...")
    idx = np.random.choice(len(X), MAX_SAMPLES, replace=False)
    X_sub = X[idx]
    labels_sub = labels[idx]
    df_sub = df.iloc[idx].reset_index(drop=True)
else:
    X_sub = X
    labels_sub = labels
    df_sub = df.reset_index(drop=True)

unique_clusters = sorted(set(labels))

# -------------------------------
# Helper: Create legend with rich labels
# -------------------------------
def create_legend(scatter, unique_clusters):
    handles = []
    labels_text = []
    for cid in unique_clusters:
        color = scatter.cmap(scatter.norm(cid))
        handles.append(
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color, markersize=10, markeredgewidth=1)
        )
        if cid in cluster_info:
            label_name = cluster_info[cid]["label"]
        else:
            label_name = f"Cluster {cid}"
        labels_text.append(label_name)
    return handles, labels_text

# -------------------------------
# PCA Visualization
# -------------------------------
print("📊 [4/5] Running PCA visualization...")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_sub)

plt.figure(figsize=(14, 10))
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=labels_sub,
    cmap='tab10',
    s=25,
    alpha=0.75,
    edgecolors='w',
    linewidth=0.3
)

plt.title("PCA Visualization of Dream Clusters", fontsize=18, pad=20)
plt.xlabel("Principal Component 1", fontsize=14)
plt.ylabel("Principal Component 2", fontsize=14)

handles, labels_text = create_legend(scatter, unique_clusters)
plt.legend(handles, labels_text, title="Clusters", bbox_to_anchor=(1.05, 1),
           fontsize=11, title_fontsize=13)

plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("pca_clusters.png", dpi=300, bbox_inches='tight')
plt.show()

print("      ✅ PCA plot saved as pca_clusters.png")

# -------------------------------
# t-SNE Visualization
# -------------------------------
print("🔥 [5/5] Running t-SNE visualization...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42,
    max_iter=1000      # ←←← CHANGED FROM n_iter
)

X_tsne = tsne.fit_transform(X_sub)



plt.figure(figsize=(14, 10))
scatter = plt.scatter(
    X_tsne[:, 0], X_tsne[:, 1],
    c=labels_sub,
    cmap='tab10',
    s=25,
    alpha=0.75,
    edgecolors='w',
    linewidth=0.3
)

plt.title("t-SNE Visualization of Dream Clusters", fontsize=18, pad=20)
plt.xlabel("t-SNE Dimension 1", fontsize=14)
plt.ylabel("t-SNE Dimension 2", fontsize=14)

handles, labels_text = create_legend(scatter, unique_clusters)
plt.legend(handles, labels_text, title="Clusters", bbox_to_anchor=(1.05, 1),
           fontsize=11, title_fontsize=13)

plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("tsne_clusters.png", dpi=300, bbox_inches='tight')
plt.show()

print("      ✅ t-SNE plot saved as tsne_clusters.png")

# -------------------------------
# Save data for interactive Flask map
# -------------------------------
print("🌐 Saving tsne_data.json for interactive web map...")

map_data = []
for i in range(len(X_tsne)):
    row = df_sub.iloc[i]
    cluster_id = int(labels_sub[i])
    dream_text_preview = row["Dream_Text"][:280]
    if len(row["Dream_Text"]) > 280:
        dream_text_preview += "..."
    
    map_data.append({
        "x": float(X_tsne[i, 0]),
        "y": float(X_tsne[i, 1]),
        "dream_id": row["Dream_ID"],
        "dream_text": dream_text_preview,
        "cluster": cluster_id,
        "emotion": row["Emotion"]
    })

with open("tsne_data.json", "w") as f:
    json.dump(map_data, f)

print("      ✅ tsne_data.json saved (used by /cluster_map)")

# -------------------------------
# Final Summary Statistics
# -------------------------------
print("\n📈 Cluster Distribution:")
print(df["KMeans_Cluster"].value_counts().sort_index())

print("\n🧠 Detailed Cluster Meanings:")
for cid in unique_clusters:
    if cid in cluster_info:
        info = cluster_info[cid]
        print(f"   {cid} → {info['label']}")
        print(f"       Dreams: {info['count']} | "
              f"Emotion: {info['top_emotion']} | "
              f"Stress: {info['avg_stress']:.3f} | "
              f"Sentiment: {info['avg_sentiment']:.3f} | "
              f"Lucid: {info['lucid_pct']}%")
    else:
        print(f"   {cid} → Cluster {cid}")

print("\n✅ Done! Two plots + web map data saved:")
print("   - pca_clusters.png")
print("   - tsne_clusters.png")
print("   - tsne_data.json")