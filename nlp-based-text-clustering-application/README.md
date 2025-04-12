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
files = {'dataset': open('sample_text_data.csv', 'rb')}
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

## Docker Setup

If you prefer to run the application in a Docker container, follow these steps:

### 1. Build the Docker Image

First, make sure you have **Docker** installed on your system. You can follow the installation guide on the [official Docker website](https://docs.docker.com/get-docker/).

To build the Docker image, navigate to the root directory of your project (where the `Dockerfile` is located) and run the following command:

```bash
docker build -t text-clustering-app .
```

This will build the Docker image named `text-clustering-app`.

### 2. Run the Docker Container

Once the image is built, you can run the application inside a Docker container with this command:

```bash
docker run -d -p 5000:5000 --name text-clustering-container text-clustering-app
```

This will:

- Start the container in detached mode (`-d`).
- Map port 5000 of the container to port 5000 on your host machine (`-p 5000:5000`).
- Name the container `text-clustering-container`.

### 3. Access the Application

Once the container is running, you can access the web application in your browser at:

```
http://localhost:5000
```

The application will be fully functional within the Docker container.

### 4. Stopping and Removing the Container

If you want to stop the container, run:

```bash
docker stop text-clustering-container
```

To remove the container:

```bash
docker rm text-clustering-container
```
