import os
import pandas as pd
import numpy as np
from django.conf import settings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

class AirQualityClustering:
    def __init__(self, processed_data_path=None):
        """
        Initialize clustering analysis
        
        Parameters:
        - processed_data_path: Path to preprocessed data
        """
        if processed_data_path is None:
            processed_data_path = os.path.join(settings.BASE_DIR, 'data', 'processed_air_quality_data.csv')
        
        self.data = pd.read_csv(processed_data_path)
        self.clustering_features = [
            'AQI Value', 'CO AQI Value', 'Ozone AQI Value', 
            'NO2 AQI Value', 'PM2.5 AQI Value', 
            'Pollution_Complexity_Score', 'Country_Code'
        ]
    
    def kmeans_clustering(self, n_clusters=3):
    
        X = self.data[self.clustering_features]
        
        # Perform K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.data['Cluster'] = kmeans.fit_predict(X)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X, self.data['Cluster'])
        
        return {
            'clustered_data': self.data,
            'centroids': kmeans.cluster_centers_,
            'silhouette_score': silhouette_avg
        }
    
    def visualize_clusters(self, output_path):
       
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=self.data, 
            x='AQI Value', 
            y='Pollution_Complexity_Score', 
            hue='Cluster', 
            palette='deep'
        )
        plt.title('Air Quality City Clusters')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()  # Explicitly close the figure
    
    def cluster_summary(self):
        """
        Generate cluster summary statistics
        """
        # Convert to dictionary to avoid pandas DataFrame iteration issues
        summary = {}
        grouped = self.data.groupby('Cluster')
        
        for cluster, group in grouped:
            summary[cluster] = {
                'City': len(group),
                'AQI_Value': {
                    'mean': group['AQI Value'].mean(),
                    'std': group['AQI Value'].std()
                },
                'Country': list(group['Country'].unique())
            }
        
        return summary

def perform_air_quality_clustering(input_path=None, visualization_path=None):
    """
    Main clustering function
    """
    try:
        if input_path is None:
            input_path = os.path.join(settings.BASE_DIR, 'data', 'processed_air_quality_data.csv')
        
        if visualization_path is None:
            visualization_path = os.path.join(settings.BASE_DIR, 'web_interface', 'static', 'cluster_visualization.png')
        
        clustering = AirQualityClustering(input_path)
        results = clustering.kmeans_clustering(n_clusters=4)
        clustering.visualize_clusters(visualization_path)
        
        cluster_summary = clustering.cluster_summary()
        
        return {
            'cluster_summary': cluster_summary,
            'silhouette_score': results['silhouette_score']
        }
    except Exception as e:
        logger.error(f"Clustering error: {e}")
        logger.error(traceback.format_exc())
        raise