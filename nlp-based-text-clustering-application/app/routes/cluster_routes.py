#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Blueprint, request, make_response, send_file, render_template
from io import BytesIO
import logging
import time
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from flask import request, make_response, send_file
from io import BytesIO
from app.utils.cleanse_text import cleanse_text
from app.utils.create_elbow_plot import create_elbow_plot
from app.utils.determine_optimal_clusters import determine_optimal_clusters


logger = logging.getLogger(__name__)

# Create blueprint
cluster_bp = Blueprint('cluster', __name__, url_prefix='/cluster')

@cluster_bp.route('', methods=['POST'])
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
