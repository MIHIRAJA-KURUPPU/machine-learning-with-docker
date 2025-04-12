#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import re
from flask import Flask, request, make_response, send_file, render_template
from io import BytesIO
import time
import zipfile
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

def cleanse_text(text):
    """
    Preprocess text by removing special characters, lowercasing,
    and removing common English stopwords.
    
    This version doesn't require NLTK resources.
    """
    if not isinstance(text, str) or not text:
        return ""
    
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Simple tokenization by splitting on whitespace
    tokens = text.split()
    
    # Simple list of English stopwords
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 
                 'while', 'of', 'to', 'in', 'for', 'on', 'by', 'with', 'about', 'against', 
                 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 
                 'from', 'up', 'down', 'is', 'are', 'were', 'was', 'am', 'been', 'being', 
                 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'i', 'me', 
                 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
                 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 
                 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 
                 'themselves', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 
                 'would', 'should', 'could', 'ought', 'im', 'youre', 'hes', 'shes', 'were', 
                 'theyre', 'ive', 'youve', 'weve', 'theyve', 'cant', 'dont', 'wont', 'not'}
    
    # Filter out stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Simple stemming (just cutting off common endings)
    stemmed_tokens = []
    for word in tokens:
        if len(word) > 3:
            if word.endswith('ing'):
                word = word[:-3]
            elif word.endswith('ed'):
                word = word[:-2]
            elif word.endswith('s') and not word.endswith('ss'):
                word = word[:-1]
            elif word.endswith('ly'):
                word = word[:-2]
            elif word.endswith('ment'):
                word = word[:-4]
        stemmed_tokens.append(word)
    
    return ' '.join(stemmed_tokens)

def determine_optimal_clusters(counts, max_clusters=10):
    """
    Use the elbow method to determine the optimal number of clusters.
    Returns the suggested number of clusters.
    """
    inertia_values = []
    for k in range(1, min(max_clusters + 1, counts.shape[0])):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(counts)
        inertia_values.append(kmeans.inertia_)
    
    # Simple elbow detection - find the point of maximum curvature
    diffs = np.diff(inertia_values)
    second_diffs = np.diff(diffs)
    
    # Return the point where the rate of change slows down the most
    # Default to 2 if we can't determine
    if len(second_diffs) > 0:
        return np.argmax(second_diffs) + 2
    return 2

def create_elbow_plot(counts, max_clusters=10):
    """Create an elbow plot for determining optimal number of clusters"""
    inertia_values = []
    k_values = range(1, min(max_clusters + 1, counts.shape[0]))
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(counts)
        inertia_values.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertia_values, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.grid(True)
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cluster', methods=['POST'])
def cluster():
    try:
        if 'dataset' not in request.files:
            return make_response("No file provided", 400)
        
        file = request.files['dataset']
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file)
        else:
            return make_response("Unsupported file format. Please use CSV or Excel.", 400)
        
        # Get column containing text to cluster
        unstructured_col = request.form.get('col', 'text')
        if unstructured_col not in data.columns:
            return make_response(f"Column '{unstructured_col}' not found in dataset", 400)
        
        # Fill missing values
        data = data.fillna("NULL")
        
        # Apply text cleansing
        logger.info("Cleansing text data...")
        data['clean_text'] = data[unstructured_col].apply(cleanse_text)
        
        # Remove rows with empty text after cleansing
        data = data[data['clean_text'].str.strip() != '']
        if data.empty:
            return make_response("No valid text data to cluster after preprocessing", 400)
        
        # Vectorize text data using TF-IDF
        logger.info("Vectorizing text data...")
        vectorizer = TfidfVectorizer(max_features=1000)
        text_features = vectorizer.fit_transform(data['clean_text'])
        
        # Determine number of clusters if auto is selected
        no_of_clusters = request.form.get('no_of_clusters', '')
        max_clusters = min(10, len(data) - 1)  # Don't exceed n-1 clusters
        
        if no_of_clusters == 'auto':
            no_of_clusters = determine_optimal_clusters(text_features, max_clusters)
            logger.info(f"Auto-determined optimal clusters: {no_of_clusters}")
        else:
            try:
                no_of_clusters = int(no_of_clusters)
                if no_of_clusters < 2:
                    no_of_clusters = 2
                elif no_of_clusters > max_clusters:
                    no_of_clusters = max_clusters
            except (ValueError, TypeError):
                no_of_clusters = 2
        
        # Perform KMeans clustering
        logger.info(f"Performing KMeans clustering with {no_of_clusters} clusters...")
        kmeans = KMeans(n_clusters=no_of_clusters, random_state=42, n_init=10)
        data['cluster_num'] = kmeans.fit_predict(text_features)
        
        # Create the Excel file
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Original data with clusters
            data_to_save = data.drop(['clean_text'], axis=1)
            data_to_save.to_excel(writer, sheet_name='Clusters', index=False)
            
            # Extract top keywords for each cluster
            keywords_sheet = []
            feature_names = np.array(vectorizer.get_feature_names_out())
            
            for i in range(no_of_clusters):
                # Get top keywords by TF-IDF score
                cluster_center = kmeans.cluster_centers_[i]
                top_indices = cluster_center.argsort()[-15:][::-1]
                top_keywords = feature_names[top_indices]
                top_weights = cluster_center[top_indices]
                
                cluster_keywords = pd.DataFrame({
                    'cluster': i,
                    'keyword': top_keywords,
                    'weight': top_weights
                })
                keywords_sheet.append(cluster_keywords)
            
            # Save top keywords
            pd.concat(keywords_sheet).to_excel(writer, sheet_name='Top_Keywords', index=False)
            
            # Cluster statistics
            cluster_stats = data.groupby(['cluster_num']).agg({
                unstructured_col: 'count'
            }).reset_index()
            cluster_stats.columns = ['Cluster', 'Count']
            cluster_stats['Percentage'] = cluster_stats['Count'] / cluster_stats['Count'].sum() * 100
            cluster_stats.to_excel(writer, sheet_name='Cluster_Report', index=False)
            
            # Add charts
            workbook = writer.book
            worksheet = writer.sheets['Cluster_Report']
            
            # Add pie chart
            pie_chart = workbook.add_chart({'type': 'pie'})
            pie_chart.add_series({
                'name': 'Cluster Distribution',
                'categories': f'=Cluster_Report!$A$2:$A${no_of_clusters+1}',
                'values': f'=Cluster_Report!$B$2:$B${no_of_clusters+1}',
                'data_labels': {'percentage': True}
            })
            worksheet.insert_chart('E2', pie_chart, {'x_scale': 1.5, 'y_scale': 1.5})
            
            # Add column chart
            column_chart = workbook.add_chart({'type': 'column'})
            column_chart.add_series({
                'name': 'Count per Cluster',
                'categories': f'=Cluster_Report!$A$2:$A${no_of_clusters+1}',
                'values': f'=Cluster_Report!$B$2:$B${no_of_clusters+1}'
            })
            column_chart.set_title({'name': 'Documents per Cluster'})
            column_chart.set_x_axis({'name': 'Cluster'})
            column_chart.set_y_axis({'name': 'Count'})
            worksheet.insert_chart('E18', column_chart, {'x_scale': 1.5, 'y_scale': 1.5})
            
            # Create PCA visualization if enough data
            if text_features.shape[0] > 2:
                try:
                    # Reduce to 2D for visualization
                    pca = PCA(n_components=2)
                    reduced_features = pca.fit_transform(text_features.toarray())
                    
                    # Create dataframe with results
                    viz_df = pd.DataFrame(reduced_features, columns=['x', 'y'])
                    viz_df['cluster'] = data['cluster_num']
                    
                    # Save to Excel
                    viz_df.to_excel(writer, sheet_name='PCA_Visualization', index=False)
                    
                    # Create scatter plot
                    scatter_chart = workbook.add_chart({'type': 'scatter'})
                    for i in range(no_of_clusters):
                        cluster_data = viz_df[viz_df['cluster'] == i]
                        row_start = len(viz_df) + 5 + i*len(cluster_data)
                        row_end = row_start + len(cluster_data) - 1
                        
                        # Write cluster data
                        cluster_data.to_excel(writer, sheet_name='PCA_Visualization', 
                                              startrow=row_start, index=False)
                        
                        # Add series to chart
                        scatter_chart.add_series({
                            'name': f'Cluster {i}',
                            'categories': f'=PCA_Visualization!$A${row_start+1}:$A${row_end+1}',
                            'values': f'=PCA_Visualization!$B${row_start+1}:$B${row_end+1}'
                        })
                    
                    scatter_chart.set_title({'name': 'PCA Cluster Visualization'})
                    scatter_chart.set_x_axis({'name': 'Component 1'})
                    scatter_chart.set_y_axis({'name': 'Component 2'})
                    worksheet = writer.sheets['PCA_Visualization']
                    worksheet.insert_chart('D2', scatter_chart, {'x_scale': 1.5, 'y_scale': 1.5})
                    
                except Exception as e:
                    logger.error(f"Error creating PCA visualization: {e}")
        
        # Create elbow plot
        elbow_plot = create_elbow_plot(text_features)
        
        # Create zip file with all outputs
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            # Add Excel file
            excel_info = zipfile.ZipInfo('cluster_results.xlsx')
            excel_info.date_time = time.localtime(time.time())[:6]
            excel_info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(excel_info, output.getvalue())
            
            # Add elbow plot
            elbow_info = zipfile.ZipInfo('elbow_plot.png')
            elbow_info.date_time = time.localtime(time.time())[:6]
            elbow_info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(elbow_info, elbow_plot.getvalue())
            
            # Add README
            readme_text = f"""
            Text Clustering Results
            ----------------------
            
            This archive contains the results of text clustering performed on column '{unstructured_col}'.
            Number of clusters: {no_of_clusters}
            Total documents: {len(data)}
            
            Files included:
            - cluster_results.xlsx: Excel file with clustering results and visualizations
            - elbow_plot.png: Plot showing inertia vs. number of clusters
            
            The Excel file contains the following sheets:
            - Clusters: Original data with assigned cluster numbers
            - Top_Keywords: Top keywords for each cluster based on TF-IDF weights
            - Cluster_Report: Statistics about each cluster with charts
            - PCA_Visualization: 2D visualization of clusters (if available)
            """
            
            readme_info = zipfile.ZipInfo('README.txt')
            readme_info.date_time = time.localtime(time.time())[:6]
            readme_info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(readme_info, readme_text)
        
        memory_file.seek(0)
        
        response = make_response(send_file(
            memory_file,
            mimetype='application/zip',
            download_name='cluster_results.zip',
            as_attachment=True
        ))
        
        # Set CORS headers
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        
        return response
        
    except Exception as e:
        logger.error(f"Error in clustering: {e}", exc_info=True)
        return make_response(f"Error processing request: {str(e)}", 500)

@app.route('/sample', methods=['GET'])
def get_sample():
    """Provide a sample dataset for testing"""
    # Create sample data
    data = {
        'text': [
            "Machine learning is a subfield of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
            "Deep learning models consist of multiple layers of neural networks that can recognize patterns in vast amounts of data.",
            "Neural networks are inspired by the human brain and consist of interconnected nodes that process information.",
            "Reinforcement learning is a type of machine learning where agents learn by interacting with an environment and receiving rewards.",
            "Natural language processing combines computational linguistics and machine learning to process and analyze text data.",
            "Python is one of the most popular programming languages used in data science and machine learning.",
            "JavaScript is primarily used for web development but is expanding into machine learning with libraries like TensorFlow.js.",
            "R programming language is widely used for statistical analysis and data visualization in research.",
            "SQL is essential for data scientists to extract and manipulate data from relational databases.",
            "Rust provides memory safety without garbage collection, making it efficient for system programming.",
            "Data mining techniques help discover patterns and extract useful information from large datasets.",
            "Feature engineering is the process of selecting and transforming variables when creating a predictive model.",
            "Regression analysis helps understand relationships between dependent and independent variables in datasets.",
            "Cluster analysis groups objects based on their similarity, useful for market segmentation.",
            "Principal component analysis reduces the dimensionality of data while preserving important information.",
            "Climate change is accelerating with rising global temperatures affecting ecosystems worldwide.",
            "Renewable energy sources like solar and wind are becoming increasingly cost-competitive with fossil fuels.",
            "Biodiversity loss threatens ecosystem stability and human well-being across the planet.",
            "Water scarcity affects billions of people and is worsening due to climate change and population growth.",
            "Sustainable agriculture practices can reduce environmental impact while maintaining crop yields.",
            "The healthcare industry is implementing AI for early disease detection and personalized treatment plans.",
            "Telemedicine has expanded rapidly, allowing patients to consult with healthcare providers remotely.",
            "Electronic health records improve patient care coordination but raise privacy concerns.",
            "Preventive healthcare focuses on disease prevention rather than treatment, reducing overall costs.",
            "Genomic medicine uses information about a person's genetic makeup to tailor medical treatments.",
            "E-commerce has transformed retail by allowing businesses to reach customers globally online.",
            "Supply chain management optimizes the flow of goods and services from raw materials to finished products.",
            "Digital marketing uses online channels to connect with customers and promote products or services.",
            "Remote work has increased productivity for many companies while reducing overhead costs.",
            "Blockchain technology offers transparent and secure transactions for businesses across industries.",
            "Virtual reality creates immersive experiences for gaming, education, and professional training.",
            "Quantum computing promises to solve complex problems that are beyond the capabilities of classical computers.",
            "Internet of Things connects everyday devices to the internet, enabling data collection and remote control.",
            "Cybersecurity measures protect systems and networks from digital attacks and unauthorized access.",
            "Cloud computing provides on-demand access to computing resources without direct active management.",
            "Social media platforms have changed how people communicate and share information globally.",
            "Digital literacy is increasingly important as technology becomes integrated into daily life.",
            "Visual communication using graphics and imagery can convey complex ideas more effectively than text alone.",
            "Intercultural communication skills are essential in the globalized business environment.",
            "Nonverbal communication includes facial expressions, gestures, and body language that convey meaning."
        ],
        'category': [
            "AI", "AI", "AI", "AI", "AI",
            "Programming", "Programming", "Programming", "Programming", "Programming",
            "Data Science", "Data Science", "Data Science", "Data Science", "Data Science",
            "Environment", "Environment", "Environment", "Environment", "Environment",
            "Healthcare", "Healthcare", "Healthcare", "Healthcare", "Healthcare",
            "Business", "Business", "Business", "Business", "Business",
            "Technology", "Technology", "Technology", "Technology", "Technology",
            "Communication", "Communication", "Communication", "Communication", "Communication"
        ],
        'source': [
            "Tech Blog", "Research Paper", "Textbook", "Course Materials", "Conference",
            "Developer Forum", "Tech Article", "Academic Source", "Tutorial", "Documentation",
            "Research Journal", "Online Course", "Statistics Textbook", "Business Report", "Analytics Blog",
            "Scientific Journal", "News Article", "Conservation Report", "UN Publication", "Agricultural Study",
            "Medical Journal", "Health Magazine", "Policy Brief", "Public Health Report", "Research Paper",
            "Business Review", "Logistics Journal", "Marketing Guide", "Workplace Study", "Financial Report",
            "Tech Magazine", "Scientific American", "Engineering Journal", "Security Report", "IT Publication",
            "Social Science Journal", "Education Report", "Design Publication", "International Relations", "Psychology Text"
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Create CSV file
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(
        output,
        mimetype='text/csv',
        download_name='sample_text_data.csv',
        as_attachment=True
    )

if __name__=='__main__':
    # Make sure the templates directory exists
    os.makedirs('templates', exist_ok=True)
    
    # Create the HTML template file if it doesn't exist
    template_path = os.path.join('templates', 'index.html')
    if not os.path.exists(template_path):
        with open(template_path, 'w') as f:
            f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text Clustering Tool</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.2.3/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .header {
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #eee;
        }
        .card {
            margin-bottom: 1.5rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        }
        .steps {
            counter-reset: step-counter;
            margin-bottom: 2rem;
        }
        .step {
            position: relative;
            counter-increment: step-counter;
            padding-left: 3rem;
            margin-bottom: 1.5rem;
        }
        .step:before {
            content: counter(step-counter);
            position: absolute;
            left: 0;
            top: 0;
            width: 2.5rem;
            height: 2.5rem;
            line-height: 2.5rem;
            border-radius: 50%;
            background-color: #007bff;
            color: white;
            text-align: center;
            font-weight: bold;
        }
        .loading {
            display: none;
        }
        .progress {
            height: 1.5rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header text-center">
            <h1>Text Clustering Tool</h1>
            <p class="lead">Upload your dataset to discover natural clusters in your text data</p>
        </div>
        
        <div class="row">
            <div class="col-md-8 offset-md-2">
                <div class="card">
                    <div class="card-body">
                        <h3 class="card-title">How it works</h3>
                        <div class="steps">
                            <div class="step">
                                <h5>Upload your data</h5>
                                <p>Select a CSV or Excel file with text data you want to cluster</p>
                            </div>
                            <div class="step">
                                <h5>Configure settings</h5>
                                <p>Select the text column and the number of clusters</p>
                            </div>
                            <div class="step">
                                <h5>Get results</h5>
                                <p>Download a ZIP file with clustering results and visualizations</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-body">
                        <h3 class="card-title">Upload Dataset</h3>
                        <form id="clusterForm" enctype="multipart/form-data">
                            <div class="mb-3">
                                <label for="dataset" class="form-label">Dataset File (CSV or Excel)</label>
                                <input type="file" class="form-control" id="dataset" name="dataset" accept=".csv,.xls,.xlsx" required>
                                <div class="form-text">Your file should contain a column with text to be clustered</div>
                            </div>
                            
                            <div class="mb-3">
                                <label for="col" class="form-label">Text Column Name</label>
                                <input type="text" class="form-control" id="col" name="col" value="text" required>
                                <div class="form-text">The column containing text data to cluster</div>
                            </div>
                            
                            <div class="mb-3">
                                <label for="no_of_clusters" class="form-label">Number of Clusters</label>
                                <select class="form-select" id="no_of_clusters" name="no_of_clusters">
                                    <option value="auto">Auto-detect (recommended)</option>
                                    <option value="2">2</option>
                                    <option value="3">3</option>
                                    <option value="4">4</option>
                                    <option value="5">5</option>
                                    <option value="6">6</option>
                                    <option value="7">7</option>
                                    <option value="8">8</option>
                                    <option value="9">9</option>
                                    <option value="10">10</option>
                                </select>
                            </div>
                            
                            <div class="d-grid gap-2">
                                <button type="submit" class="btn btn-primary">Generate Clusters</button>
                                <button type="button" id="sampleBtn" class="btn btn-outline-secondary">Download Sample Data</button>
                            </div>
                        </form>
                    </div>
                </div>
                
                <div class="loading card">
                    <div class="card-body text-center">
                        <h4>Processing your data...</h4>
                        <div class="progress mb-3">
                            <div class="progress-bar progress-bar-striped progress-bar-animated" style="width: 100%"></div>
                        </div>
                        <p>This may take a few moments depending on the size of your dataset</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.2.3/js/bootstrap.bundle.min.js"></script>
    <script>
        $(document).ready(function() {
            $('#clusterForm').on('submit', function(e) {
                e.preventDefault();
                
                // Show loading
                $('.loading').show();
                
                // Create form data
                var formData = new FormData(this);
                
                // Submit form
                $.ajax({
                    url: '/cluster',
                    type: 'POST',
                    data: formData,
                    processData: false,
                    contentType: false,
                    xhrFields: {
                        responseType: 'blob'
                    },
                    success: function(blob) {
                        // Create download link
                        var url = window.URL.createObjectURL(blob);
                        var a = document.createElement('a');
                        a.href = url;
                        a.download = 'cluster_results.zip';
                        document.body.appendChild(a);
                        a.click();
                        window.URL.revokeObjectURL(url);
                        
                        // Hide loading
                        $('.loading').hide();
                    },
                    error: function(xhr) {
                        // Show error
                        $('.loading').hide();
                        
                        let errorMsg = 'An error occurred while processing your request.';
                        if (xhr.responseText) {
                            errorMsg = xhr.responseText;
                        }
                        
                        alert('Error: ' + errorMsg);
                    }
                });
            });
            
            // Sample data download
            $('#sampleBtn').on('click', function() {
                window.location.href = '/sample';
            });
        });
    </script>
</body>
</html>''')
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)