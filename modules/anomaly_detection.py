"""
Isolation Forest Module
========================
Implements anomaly detection to identify unusual patterns in enrollment data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class AnomalyDetector:
    """Detects anomalies in Aadhaar enrollment and service data using Isolation Forest."""
    
    def __init__(self, contamination=0.1, random_state=42):
        """
        Initialize anomaly detector.
        
        Parameters:
        -----------
        contamination : float
            Expected proportion of anomalies in the dataset (0.0 to 0.5)
        random_state : int
            Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.anomalies = None
        
    def prepare_features(self, timeseries_df):
        """
        Prepare features for anomaly detection.
        
        Parameters:
        -----------
        timeseries_df : pd.DataFrame
            Time series enrollment data
            
        Returns:
        --------
        pd.DataFrame : Feature matrix for anomaly detection
        """
        # Create a copy
        df = timeseries_df.copy()
        
        # Calculate rate of change features
        df = df.sort_values(['district_id', 'date'])
        
        # Period-over-period changes
        for col in ['new_enrollments', 'biometric_updates', 'rejection_rate', 'saturation_level']:
            df[f'{col}_change'] = df.groupby('district_id')[col].pct_change()
        
        # Moving averages
        for col in ['new_enrollments', 'biometric_updates', 'rejection_rate']:
            df[f'{col}_ma3'] = df.groupby('district_id')[col].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
            df[f'{col}_deviation'] = (df[col] - df[f'{col}_ma3']) / (df[f'{col}_ma3'] + 1)
        
        # Fill NaN values from pct_change
        df = df.fillna(0)
        
        return df
    
    def fit_predict(self, features_df):
        """
        Fit Isolation Forest and predict anomalies.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature matrix
            
        Returns:
        --------
        pd.DataFrame : Data with anomaly scores and labels
        """
        # Select features for anomaly detection
        anomaly_features = [
            'new_enrollments', 'biometric_updates', 'rejection_rate', 'saturation_level',
            'new_enrollments_change', 'biometric_updates_change', 'rejection_rate_change',
            'new_enrollments_deviation', 'biometric_updates_deviation', 'rejection_rate_deviation'
        ]
        
        # Filter to features that exist
        self.feature_names = [f for f in anomaly_features if f in features_df.columns]
        
        X = features_df[self.feature_names].fillna(0)
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Standardize
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Isolation Forest
        print(f"Training Isolation Forest with contamination={self.contamination}...")
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )
        
        # Predict: -1 for anomalies, 1 for normal
        predictions = self.model.fit_predict(X_scaled)
        
        # Get anomaly scores (lower = more anomalous)
        scores = self.model.score_samples(X_scaled)
        
        # Add to dataframe
        features_df = features_df.copy()
        features_df['anomaly_score'] = scores
        features_df['is_anomaly'] = (predictions == -1).astype(int)
        
        # Identify anomaly details
        self.anomalies = features_df[features_df['is_anomaly'] == 1].copy()
        
        print(f"✓ Detected {len(self.anomalies)} anomalies ({len(self.anomalies)/len(features_df):.1%} of data)")
        
        return features_df
    
    def categorize_anomalies(self, anomalies_df):
        """
        Categorize anomalies by type.
        
        Parameters:
        -----------
        anomalies_df : pd.DataFrame
            Detected anomalies
            
        Returns:
        --------
        pd.DataFrame : Anomalies with categories
        """
        df = anomalies_df.copy()
        
        categories = []
        severities = []
        
        for _, row in df.iterrows():
            anomaly_types = []
            severity_score = 0
            
            # Enrollment spike
            if row.get('new_enrollments_change', 0) > 2.0:
                anomaly_types.append('Enrollment Spike')
                severity_score += 2
            
            # Enrollment drop
            if row.get('new_enrollments_change', 0) < -0.5:
                anomaly_types.append('Enrollment Drop')
                severity_score += 1
            
            # High rejection rate
            if row.get('rejection_rate', 0) > 0.10:
                anomaly_types.append('High Rejection Rate')
                severity_score += 3
            
            # Rejection spike
            if row.get('rejection_rate_change', 0) > 1.0:
                anomaly_types.append('Rejection Spike')
                severity_score += 3
            
            # Unusual biometric update pattern
            if abs(row.get('biometric_updates_deviation', 0)) > 2.0:
                anomaly_types.append('Unusual Update Pattern')
                severity_score += 1
            
            # Saturation anomaly
            if row.get('saturation_level', 0) < 0.5 or row.get('saturation_level_change', 0) < -0.05:
                anomaly_types.append('Saturation Issue')
                severity_score += 2
            
            # Default category
            if not anomaly_types:
                anomaly_types.append('General Anomaly')
                severity_score = 1
            
            categories.append('; '.join(anomaly_types))
            
            # Classify severity
            if severity_score >= 5:
                severity = 'Critical'
            elif severity_score >= 3:
                severity = 'High'
            elif severity_score >= 2:
                severity = 'Medium'
            else:
                severity = 'Low'
            
            severities.append(severity)
        
        df['anomaly_category'] = categories
        df['severity'] = severities
        
        return df
    
    def get_anomaly_summary(self):
        """Generate summary of detected anomalies."""
        if self.anomalies is None or len(self.anomalies) == 0:
            return "No anomalies detected."
        
        # Categorize anomalies
        categorized = self.categorize_anomalies(self.anomalies)
        
        # Summary statistics
        summary = {
            'total_anomalies': len(categorized),
            'critical_anomalies': len(categorized[categorized['severity'] == 'Critical']),
            'high_anomalies': len(categorized[categorized['severity'] == 'High']),
            'medium_anomalies': len(categorized[categorized['severity'] == 'Medium']),
            'low_anomalies': len(categorized[categorized['severity'] == 'Low']),
            'affected_districts': categorized['district_id'].nunique(),
            'category_distribution': categorized['anomaly_category'].value_counts().to_dict()
        }
        
        return summary
    
    def visualize_anomalies(self, features_df, output_path='output/anomaly_detection.png'):
        """
        Create visualization of anomalies.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Data with anomaly predictions
        output_path : str
            Path to save visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Anomaly score distribution
        axes[0, 0].hist(features_df['anomaly_score'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].axvline(features_df[features_df['is_anomaly'] == 1]['anomaly_score'].max(), 
                          color='red', linestyle='--', linewidth=2, label='Anomaly Threshold')
        axes[0, 0].set_xlabel('Anomaly Score', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Anomalies over time
        time_anomalies = features_df.groupby('year_month')['is_anomaly'].agg(['sum', 'count'])
        time_anomalies['anomaly_rate'] = time_anomalies['sum'] / time_anomalies['count']
        
        axes[0, 1].plot(range(len(time_anomalies)), time_anomalies['anomaly_rate'], 
                       marker='o', linewidth=2, markersize=6, color='crimson')
        axes[0, 1].fill_between(range(len(time_anomalies)), time_anomalies['anomaly_rate'], 
                               alpha=0.3, color='crimson')
        axes[0, 1].set_xlabel('Month Index', fontsize=12)
        axes[0, 1].set_ylabel('Anomaly Rate', fontsize=12)
        axes[0, 1].set_title('Anomaly Rate Over Time', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature importance (based on anomaly correlation)
        normal_data = features_df[features_df['is_anomaly'] == 0][self.feature_names].mean()
        anomaly_data = features_df[features_df['is_anomaly'] == 1][self.feature_names].mean()
        
        feature_diff = ((anomaly_data - normal_data) / (normal_data + 1)).abs().sort_values(ascending=True)
        top_features = feature_diff.tail(10)
        
        axes[1, 0].barh(range(len(top_features)), top_features.values, color='teal', edgecolor='black', alpha=0.7)
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels([f.replace('_', ' ').title() for f in top_features.index], fontsize=10)
        axes[1, 0].set_xlabel('Relative Difference', fontsize=12)
        axes[1, 0].set_title('Top Anomaly Indicators', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # 4. Anomaly severity heatmap
        if len(self.anomalies) > 0:
            categorized = self.categorize_anomalies(self.anomalies)
            severity_counts = categorized.groupby(['year_month', 'severity']).size().unstack(fill_value=0)
            
            # Ensure all severity levels are present
            for sev in ['Critical', 'High', 'Medium', 'Low']:
                if sev not in severity_counts.columns:
                    severity_counts[sev] = 0
            
            severity_counts = severity_counts[['Critical', 'High', 'Medium', 'Low']]
            
            sns.heatmap(severity_counts.T, annot=True, fmt='d', cmap='YlOrRd', 
                       cbar_kws={'label': 'Count'}, ax=axes[1, 1], linewidths=0.5)
            axes[1, 1].set_xlabel('Month', fontsize=12)
            axes[1, 1].set_ylabel('Severity', fontsize=12)
            axes[1, 1].set_title('Anomaly Severity Over Time', fontsize=14, fontweight='bold')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Anomalies with\nSeverity Data', 
                          ha='center', va='center', fontsize=14)
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Anomaly visualization saved to {output_path}")
    
    def get_top_anomalies(self, features_df, n=20):
        """
        Get top N most anomalous records.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Data with anomaly scores
        n : int
            Number of top anomalies to return
            
        Returns:
        --------
        pd.DataFrame : Top anomalies
        """
        anomalies = features_df[features_df['is_anomaly'] == 1].copy()
        anomalies = self.categorize_anomalies(anomalies)
        
        # Sort by anomaly score (most negative = most anomalous)
        top_anomalies = anomalies.nsmallest(n, 'anomaly_score')
        
        return top_anomalies[['district_id', 'date', 'anomaly_score', 'severity', 
                             'anomaly_category', 'new_enrollments', 'rejection_rate',
                             'biometric_updates']]


if __name__ == "__main__":
    print("Isolation Forest Module - Anomaly Detection")
    print("Use this module to identify unusual enrollment patterns")
