# data/preprocessor.py
import os
import pandas as pd
import numpy as np
from django.conf import settings
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class AirQualityPreprocessor:
    def __init__(self, file_path=None):
        """
        Initialize preprocessor with air quality dataset
        
        Parameters:
        - file_path: Path to the air quality CSV file
        """
        # If no path provided, use a default path
        if file_path is None:
            file_path = os.path.join(settings.BASE_DIR, 'data', 'air_quality_data.csv')
        
        # Check if file exists
        if not os.path.exists(file_path):
            # Create sample data if file doesn't exist
            self.raw_data = self.create_sample_data()
        else:
            self.raw_data = pd.read_csv(file_path)
        
        self.processed_data = None
    
    def create_sample_data(self):
        """
        Create a sample dataset if no file exists
        """
        sample_data = pd.DataFrame({
            'Country': ['USA', 'China', 'India', 'Russia', 'Brazil'],
            'City': ['New York', 'Beijing', 'Mumbai', 'Moscow', 'Sao Paulo'],
            'AQI Value': [55, 150, 120, 80, 70],
            'CO AQI Value': [20, 80, 60, 40, 35],
            'Ozone AQI Value': [15, 50, 40, 20, 25],
            'NO2 AQI Value': [20, 20, 20, 20, 10],
            'PM2.5 AQI Value': [15, 100, 60, 30, 25]
        })
        return sample_data
    
    def clean_data(self):
        """
        Clean and preprocess the air quality dataset
        """
        # Remove any duplicate entries
        self.raw_data.drop_duplicates(inplace=True)
        
        # Handle missing values
        numeric_columns = [
            'AQI Value', 'CO AQI Value', 'Ozone AQI Value', 
            'NO2 AQI Value', 'PM2.5 AQI Value'
        ]
        
        # Impute missing numeric values with median
        imputer = SimpleImputer(strategy='median')
        self.raw_data[numeric_columns] = imputer.fit_transform(
            self.raw_data[numeric_columns]
        )
        
        return self
    
    def engineer_features(self):
        """
        Create additional features for clustering
        """
        # Create aggregate pollution score
        self.raw_data['Pollution_Complexity_Score'] = (
            self.raw_data['CO AQI Value'] +
            self.raw_data['Ozone AQI Value'] +
            self.raw_data['NO2 AQI Value'] +
            self.raw_data['PM2.5 AQI Value']
        ) / 4
        
        # Encode categorical variables
        self.raw_data['Country_Code'] = pd.Categorical(
            self.raw_data['Country']
        ).codes
        
        return self
    
    def prepare_for_clustering(self):
        """
        Prepare dataset for clustering analysis
        """
        # Select features for clustering
        clustering_features = [
            'AQI Value', 'CO AQI Value', 'Ozone AQI Value', 
            'NO2 AQI Value', 'PM2.5 AQI Value', 
            'Pollution_Complexity_Score', 'Country_Code'
        ]
        
        # Scale the features
        scaler = StandardScaler()
        self.processed_data = pd.DataFrame(
            scaler.fit_transform(self.raw_data[clustering_features]),
            columns=clustering_features,
            index=self.raw_data.index
        )
        
        # Attach city and country information
        self.processed_data['City'] = self.raw_data['City']
        self.processed_data['Country'] = self.raw_data['Country']
        
        return self
    
    def save_processed_data(self, output_path):
        """
        Save processed data to a CSV file
        
        Parameters:
        - output_path: Path to save processed data
        """
        if self.processed_data is not None:
            self.processed_data.to_csv(output_path, index=False)
        return self

def preprocess_air_quality_data(input_path=None, output_path=None):
    """
    Main preprocessing function
    """
    if input_path is None:
        input_path = os.path.join(settings.BASE_DIR, 'data', 'air_quality_data.csv')
    
    if output_path is None:
        output_path = os.path.join(settings.BASE_DIR, 'data', 'processed_air_quality_data.csv')
    
    preprocessor = AirQualityPreprocessor(input_path)
    preprocessor = (preprocessor
        .clean_data()
        .engineer_features()
        .prepare_for_clustering())
    
    preprocessor.save_processed_data(output_path)
    
    return preprocessor.processed_data