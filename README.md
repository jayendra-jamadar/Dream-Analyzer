# 🌙 AI-Powered Dream Analysis System

An intelligent web application that analyzes dreams using **Machine Learning + NLP (BERT)** to uncover emotional patterns, cluster similar experiences, and generate meaningful psychological insights.

---

## 🧠 Overview

Dreams are complex reflections of our subconscious. This project uses **Natural Language Processing and Unsupervised Learning** to:

* Understand dream narratives semantically
* Group similar dreams into meaningful clusters
* Provide psychological interpretations
* Recommend similar dream patterns

---

## 🚀 Key Features

### 🔍 Dream Analysis

* Input your dream description
* System processes it using **BERT embeddings**
* Predicts the most relevant **dream cluster**

---

### 🧠 Cluster-Based Interpretation

* Dreams are grouped into clusters such as:

  * Fear / Anxiety Dreams
  * Calm / Neutral Processing
  * Positive / Joyful Dreams
* Each cluster reflects a psychological theme

---

### 🤖 AI-Powered Explanation

* Uses AI (OpenAI / fallback logic) to:

  * Interpret emotions
  * Suggest subconscious meaning
  * Provide actionable insights

---

### 🔗 Similar Dreams Recommendation

* Finds and displays **semantically similar dreams**
* Helps identify patterns and shared experiences

---

### 📊 Visualization

* PCA & t-SNE visualizations
* Understand how dreams are grouped in embedding space

---

### 🌐 Web Interface

* Built using Flask
* Clean and interactive UI
* Real-time dream analysis

---

## 🏗️ Tech Stack

### Backend

* Python
* Flask
* Scikit-learn

### Machine Learning

* Sentence Transformers (BERT)
* KMeans Clustering
* Cosine Similarity

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* t-SNE / PCA

### AI Integration

* OpenAI API (optional)
* Anthropic API (optional)

---

## ⚙️ How It Works

1. User inputs a dream
2. Text is converted into embeddings using BERT
3. KMeans predicts the cluster
4. System retrieves:

   * cluster meaning
   * similar dreams
5. AI generates explanation
6. Results are displayed in UI

---

## 📁 Project Structure

```bash
dream-analysis/
│
├── app.py                  # Flask application
├── helpers.py              # ML + preprocessing utilities
├── train_model.py          # Model training script
├── visualize_clusters.py   # PCA & t-SNE visualization
│
├── model.pkl               # Trained ML pipeline
├── data_vectors.npy        # Embeddings
├── scored_dreams.csv       # Processed dataset
├── cluster_meta.json       # Cluster metadata
├── cluster_labels.json     # Human-readable labels
│
├── templates/              # HTML templates
├── static/                 # CSS/JS (if any)
│
└── requirements.txt        # Dependencies
```

---

## 🛠️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/dream-analysis.git
cd dream-analysis
```

---

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Train model (optional)

```bash
python train_model.py
```

---

### 5. Run the app

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 🔑 API Setup (Optional)

### OpenAI

```bash
setx OPENAI_API_KEY "your_api_key"
```

### Anthropic

```bash
setx ANTHROPIC_API_KEY "your_api_key"
```

---

## 🌍 Deployment

Recommended platform:

* Render (for full backend support)

---

## ⚡ Performance Notes

* Uses optimized clustering and similarity search
* Model is loaded once for efficiency
* Works on CPU (GPU optional during training)

---

## 🧪 Future Improvements

* Personalized dream tracking
* Better clustering (HDBSCAN)
* Interactive dashboard
* Real-time analytics
* Mobile-friendly UI

---

## 🎯 Use Cases

* Psychological pattern analysis
* Dream journaling tools
* Mental wellness platforms
* NLP + ML academic projects

---

## 🤝 Contribution

Contributions are welcome!
Feel free to open issues or submit pull requests.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## ⭐ Acknowledgements

* Sentence Transformers
* Scikit-learn
* OpenAI / Anthropic APIs

---

## 💡 Final Note

This project demonstrates how **AI can bridge subconscious patterns and real-world insights**, combining ML, NLP, and human-centered design into a meaningful application.
