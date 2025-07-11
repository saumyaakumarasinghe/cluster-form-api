# Cluster-Form API

An NLP-Based Clustering Solution for Summarizing Google Form Responses

---

## Overview

**Cluster-Form** is a RESTful API that leverages Natural Language Processing (NLP) and KMeans clustering to automatically group and summarize open-ended responses from Google Forms or Google Spreadsheets. This helps users quickly identify main themes and patterns in large sets of feedback or survey data.

---

## Features
- **Accepts Google Spreadsheet links** as input with specific column targeting
- **Extracts and preprocesses** text responses using advanced NLP techniques
- **Applies KMeans clustering** with automatic optimal cluster detection
- **Supports both TF-IDF and semantic embeddings** for different clustering approaches
- **Returns comprehensive analytics** including cluster summaries, quality metrics, and visualizations
- **Interactive API documentation** with Swagger/OpenAPI
- **Robust error handling** and data validation

---

## Technologies Used
- **Python 3**
- **Flask** & **Flask-RESTX** (API framework & documentation)
- **scikit-learn** (KMeans clustering, metrics)
- **sentence-transformers** (semantic embeddings)
- **pandas, numpy** (data manipulation)
- **Google API Client** (Sheets integration)
- **matplotlib, seaborn** (visualization)
- **Pillow** (image processing)
- **NLTK** (text preprocessing)

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/saumyaakumarasinghe/cluster-form-api.git
   cd cluster-form-api
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Google Service Account:**
   - Place your `keys.json` (Google service account credentials) in the project root.
   - Share your target Google Sheets with the service account email found in `keys.json`.

4. **Run the API server:**
   ```bash
   flask run
   # or
   python main.py run
   ```

---

## Common Commands

- **Run the API server (development):**
  ```bash
  flask run
  # or
  python main.py run
  ```

- **Format code with Black:**
  ```bash
  black .
  ```

- **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```

- **Check for outdated packages:**
  ```bash
  pip list --outdated
  ```

- **Export requirements (if you add new packages):**
  ```bash
  pip freeze > requirements.txt
  ```

---

## API Endpoints

### 1. Health Check
- **GET** `/api/health`
- **Response:** `{ "status": "OK", "message": "Server is healthy" }`

### 2. Form Column Clustering
- **POST** `/api/form/clustering`
- **Body:**
  ```json
  {
    "link": "<Google Sheet URL>",
    "column": "<Column Name or Letter>"
  }
  ```
- **Response:**
  ```json
  {
    "message": "Clustering operation completed successfully.",
    "optimal_clusters": 3,
    "link": "https://docs.google.com/spreadsheets/d/your-id/edit",
    "visualization": "data:image/png;base64,iVBOR...",
    "analytics": {
      "total_records": 61,
      "clusters": [
        {
          "cluster": 0,
          "count": 58,
          "percentage": 95.1,
          "samples": ["Great service", "Excellent support"]
        }
      ]
    },
    "metrics": {
      "silhouette_score": 0.234,
      "calinski_harabasz_score": 45.67,
      "davies_bouldin_score": 1.23
    }
  }
  ```

---

## Example Usage

1. **Send a POST request to `/api/form/clustering`**
   ```bash
   curl -X POST http://localhost:5000/api/form/clustering \
     -H "Content-Type: application/json" \
     -d '{ "link": "https://docs.google.com/spreadsheets/d/your-id", "column": "B" }'
   ```

2. **View the results:**
   - The response includes cluster count, updated sheet link, visualization image, and detailed analytics
   - Clustering quality metrics help evaluate the results
   - Sample responses from each cluster provide insights

---

## Clustering Features

### Text Preprocessing
- **Lemmatization** for better word roots (more natural than stemming)
- **Stopword removal** and punctuation cleaning
- **Case normalization** and whitespace handling
- **Data validation** and error handling

### Clustering Methods
- **TF-IDF Vectorization** with customizable parameters
- **Semantic Embeddings** using sentence-transformers (optional)
- **Automatic optimal cluster detection** using silhouette score
- **Quality metrics** including silhouette, Calinski-Harabasz, and Davies-Bouldin scores

### Visualization
- **Pie chart** showing cluster distribution
- **Quality metrics** displayed on the visualization
- **Base64 encoded** for easy frontend integration
- **Optimized compression** for web display

---

## Notes
- **KMeans clustering** with automatic optimal cluster detection
- **Google Sheets integration** requires proper service account setup
- **Large datasets** may take a few seconds to process
- **Semantic clustering** requires additional model download on first use
- **Visualization** includes all clustering quality metrics for academic presentations

---

## License
MIT
