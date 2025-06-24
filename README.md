# 🛍️ Shopper Spectrum: Customer Segmentation & Product Recommendations

**Shopper Spectrum** is a real-time e-commerce analytics app that helps businesses:
- Segment customers using RFM analysis & clustering
- Recommend similar products using collaborative filtering

Built with `Streamlit`, this app turns raw transactional data into actionable business insights.

---

## 🚀 Features

### 📊 Dashboard (EDA)
- Top countries by transactions
- Best-selling products
- Monthly sales trend

### 🎯 Product Recommendation
- Input: Product **name**
- Output: 5 similar products based on cosine similarity

### 🔍 Customer Segmentation
- Input: Recency, Frequency, Monetary
- Output: Predicted customer segment:
  - High-Value
  - Regular
  - Occasional
  - At-Risk

---

## 🧰 Tech Stack

- `Python`, `Pandas`, `NumPy`
- `scikit-learn` for clustering
- `Streamlit` for web app UI
- `matplotlib`, `seaborn` for data visualization
- `pickle` for model storage

---

## 📦 Files

| File                         | Purpose                                 |
|------------------------------|-----------------------------------------|
| `app.py` | Main Streamlit app                    |
| `online_retail.csv`         | Transaction dataset                     |
| `kmeans_model.pkl`          | Pretrained clustering model             |
| `scaler.pkl`                | Pre-fitted scaler for RFM normalization |
| `item_similarity.pkl`       | Product similarity matrix               |
| `product_map.pkl`           | StockCode → Product name mapping        |
| `requirements.txt`          | Required Python libraries               |

---

## ▶️ How to Run

1. Clone/download the repo
2. Lauch the app:

```bash
streamlit run app.py

