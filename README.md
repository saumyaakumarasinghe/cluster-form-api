# Cluster-Form API

An NLP-Based Clustering Solution for Summarizing Google Form Responses

---

## Overview

**Cluster-Form** is a RESTful API that leverages Natural Language Processing (NLP) and KMeans clustering to automatically group and summarize open-ended responses from Google Forms or Google Spreadsheets. This helps users quickly identify main themes and patterns in large sets of feedback or survey data.

---

## Features
- **Accepts Google Spreadsheet or Form links** as input.
- **Extracts and preprocesses** text responses from the provided Google Sheet.
- **Applies KMeans clustering** to group similar responses.
- **Returns cluster summaries, analytics, and visualizations** to help users understand the main topics in their data.
- **API endpoints** for clustering, health checks, and more.

---

## Technologies Used
- **Python 3**
- **Flask** & **Flask-RESTX** (API framework & documentation)
- **scikit-learn** (KMeans clustering, metrics)
- **pandas, numpy** (data manipulation)
- **Google API Client** (Sheets & Forms integration)
- **Pillow** (image processing)

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

- **Set environment variables (if needed):**
  ```bash
  # On Windows (cmd)
  set FLASK_APP=main.py
  set FLASK_ENV=development

  # On Unix/Mac
  export FLASK_APP=main.py
  export FLASK_ENV=development
  ```

---

## API Endpoints

### 1. Health Check
- **GET** `/api/health`
- **Response:** `{ "status": "OK", "message": "Server is healthy" }`

### 2. Clustering (General)
- **POST** `/api/cluster/`
- **Body:**
  ```json
  {
    "spreadsheetLink": "<Google Sheet URL>",
    "formLink": "<Google Form URL, optional>"
  }
  ```
- **Response:**
  - `optimal_clusters`: Number of clusters found
  - `link`: Link to the updated Google Sheet with cluster labels
  - `visualization`: (base64 image of clusters)

### 3. Clustering (Form Column)
- **POST** `/api/form/clustering`
- **Body:**
  ```json
  {
    "link": "<Google Sheet URL>",
    "column": "<Column Name or Letter>"
  }
  ```
- **Response:**
  - Same as above, but for a specific column

---

## Example Usage

1. **Send a POST request to `/api/form/clustering`**
   ```bash
   curl -X POST http://localhost:5000/api/form/clustering \
     -H "Content-Type: application/json" \
     -d '{ "link": "https://docs.google.com/spreadsheets/d/your-id", "column": "B" }'
   ```
2. **View the results:**
   - The response will include the number of clusters, a link to the updated sheet, and a visualization image (base64).

---

## Notes
- Only **KMeans clustering** is used (no DBSCAN).
- Make sure your Google Sheet is shared with the service account.
- For large datasets, processing may take a few seconds.

---

## License
MIT
