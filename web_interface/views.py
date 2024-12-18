from django.shortcuts import render
from django.conf import settings
import os
import traceback
import logging

logger = logging.getLogger(__name__)

def home_view(request):
    return render(request, 'home.html')

def clustering_results_view(request):
    """
    Perform clustering and display results
    """
    try:
        from data.preprocessor import preprocess_air_quality_data
        from analysis.clustering import perform_air_quality_clustering

        # Preprocess data (using default path)
        processed_data = preprocess_air_quality_data()
        
        # Perform clustering
        results = perform_air_quality_clustering()
        
        context = {
            'error': None,
            'results': results
        }
    except Exception as e:
        logger.error(f"Clustering view error: {e}")
        logger.error(traceback.format_exc())
        context = {
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    
    return render(request, 'results.html', context)