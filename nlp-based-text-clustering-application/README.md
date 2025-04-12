# Text Clustering Web Application

This application is a Flask-based web service that performs text clustering on datasets. It uses natural language processing techniques and machine learning to group similar text documents together.

## Features

- Upload CSV or Excel datasets containing text data
- Automatic text preprocessing (stemming, stopword removal)
- TF-IDF vectorization for better text representation
- K-means clustering with automatic cluster number detection
- Interactive web interface for easy data uploading
- Comprehensive output including:
  - Clustered data
  - Top keywords for each cluster
  - Cluster statistics
  - Visualizations (pie chart, column chart)
  - PCA visualization for cluster analysis
  - Elbow plot for optimal cluster selection

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Download NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage

1. Run the application:

```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`
3. Upload your dataset (CSV or Excel file)
4. Specify the column containing text data
5. Choose the number of clusters or let the application determine it automatically
6. Download and review the results

## Sample Data

The application provides sample data that you can download and use for testing. Click the "Download Sample Data" button on the web interface.

## API Usage

You can also use the application programmatically by sending a POST request to the `/cluster` endpoint:

```python
import requests

url = 'http://localhost:5000/cluster'
files = {'dataset': open('your_data.csv', 'rb')}
data = {'col': 'text_column', 'no_of_clusters': 'auto'}

response = requests.post(url, files=files, data=data)

with open('cluster_results.zip', 'wb') as f:
    f.write(response.content)
```

## Output

The application generates a ZIP file containing:

1. Excel file with multiple sheets:
   - Original data with cluster assignments
   - Top keywords for each cluster
   - Cluster statistics and charts
   - PCA visualization (if available)
2. Elbow plot image for determining the optimal number of clusters
3. README text file explaining the results

## Requirements

- Python 3.7+
- Flask
- pandas
- numpy
- scikit-learn
- matplotlib
- nltk
- xlsxwriter
- openpyxl
