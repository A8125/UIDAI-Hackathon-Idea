"""
K-Means Clustering Module
==========================
Implements pattern detection to group districts into service profiles.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


class DistrictClusterer:
    """Groups districts into service profiles using K-Means clustering."""
    
    def __init__(self, n_clusters=5, random_state=42):
        """
        Initialize clusterer.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to identify
        random_state : int
            Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.cluster_profiles = None
        
    def prepare_features(self, timeseries_df, districts_df):
        """
        Aggregate features for clustering from time series data.
        
        Parameters:
        -----------
        timeseries_df : pd.DataFrame
            Time series enrollment data
        districts_df : pd.DataFrame
            District master data
            
        Returns:
        --------
        pd.DataFrame : Aggregated features per district
        """
        # Aggregate metrics per district
        features = timeseries_df.groupby('district_id').agg({
            'new_enrollments': ['mean', 'std', 'sum'],
            'biometric_updates': ['mean', 'std', 'sum'],
            'rejection_rate': ['mean', 'max'],
            'saturation_level': 'last',
            'active_centers': 'mean',
            'male_enrollments': 'sum',
            'female_enrollments': 'sum'
        }).reset_index()
        
        # Flatten column names
        features.columns = ['_'.join(col).strip('_') for col in features.columns.values]
        features.rename(columns={'district_id': 'district_id'}, inplace=True)
        
        # Calculate additional derived features
        features['enrollment_volatility'] = features['new_enrollments_std'] / (features['new_enrollments_mean'] + 1)
        features['update_to_enrollment_ratio'] = features['biometric_updates_sum'] / (features['new_enrollments_sum'] + 1)
        features['female_enrollment_pct'] = features['female_enrollments_sum'] / (
            features['female_enrollments_sum'] + features['male_enrollments_sum'])
        
        # Merge with district metadata
        features = features.merge(districts_df[['district_id', 'district_type', 'population_lakhs']], 
                                 on='district_id', how='left')
        
        # Encode categorical features
        features['is_urban'] = (features['district_type'] == 'Urban').astype(int)
        features['is_rural'] = (features['district_type'] == 'Rural').astype(int)
        
        return features
    
    def select_optimal_k(self, features_df, max_k=10):
        """
        Use elbow method to find optimal number of clusters.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature matrix
        max_k : int
            Maximum k to test
            
        Returns:
        --------
        dict : Inertia values for each k
        """
        # Select numeric features only
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        X = features_df[numeric_cols].fillna(0)
        
        # Standardize
        X_scaled = self.scaler.fit_transform(X)
        
        inertias = {}
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X_scaled)
            inertias[k] = kmeans.inertia_
        
        return inertias
    
    def fit(self, features_df):
        """
        Fit K-Means clustering model.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Aggregated features
            
        Returns:
        --------
        pd.DataFrame : Features with cluster assignments
        """
        # Select numeric features for clustering
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['district_id']
        self.feature_names = [col for col in numeric_cols if col not in exclude_cols]
        
        X = features_df[self.feature_names].fillna(0)
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit K-Means
        print(f"Fitting K-Means with {self.n_clusters} clusters...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, 
                            random_state=self.random_state, 
                            n_init=10,
                            max_iter=300)
        
        features_df['cluster'] = self.kmeans.fit_predict(X_scaled)
        
        # Generate cluster profiles
        self._generate_cluster_profiles(features_df)
        
        print(f"✓ Clustering complete. Identified {self.n_clusters} district profiles")
        
        return features_df
    
    def _generate_cluster_profiles(self, features_df):
        """Generate interpretable cluster profiles."""
        profiles = []
        
        for cluster_id in range(self.n_clusters):
            cluster_data = features_df[features_df['cluster'] == cluster_id]
            
            # Calculate cluster statistics
            profile = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'avg_saturation': cluster_data['saturation_level_last'].mean(),
                'avg_enrollments': cluster_data['new_enrollments_mean'].mean(),
                'avg_updates': cluster_data['biometric_updates_mean'].mean(),
                'avg_rejection_rate': cluster_data['rejection_rate_mean'].mean(),
                'urban_pct': cluster_data['is_urban'].mean(),
                'rural_pct': cluster_data['is_rural'].mean(),
                'female_enrollment_pct': cluster_data['female_enrollment_pct'].mean()
            }
            
            # Generate profile label based on characteristics
            profile['label'] = self._generate_cluster_label(profile)
            
            profiles.append(profile)
        
        self.cluster_profiles = pd.DataFrame(profiles)
    
    def _generate_cluster_label(self, profile):
        """Generate interpretable label for cluster."""
        labels = []
        
        # Geographic type
        if profile['urban_pct'] > 0.6:
            labels.append("Urban")
        elif profile['rural_pct'] > 0.6:
            labels.append("Rural")
        else:
            labels.append("Mixed")
        
        # Saturation level
        if profile['avg_saturation'] > 0.90:
            labels.append("High Saturation")
        elif profile['avg_saturation'] > 0.80:
            labels.append("Moderate Saturation")
        else:
            labels.append("Low Saturation")
        
        # Activity type
        update_ratio = profile['avg_updates'] / (profile['avg_enrollments'] + 1)
        if update_ratio > 1.5:
            labels.append("Update-Heavy")
        elif profile['avg_enrollments'] > profile['avg_updates']:
            labels.append("Enrollment-Focused")
        else:
            labels.append("Balanced")
        
        return " | ".join(labels)
    
    def visualize_clusters(self, features_df, output_path='output/cluster_analysis.png'):
        """
        Create visualization of clusters using PCA.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Features with cluster assignments
        output_path : str
            Path to save visualization
        """
        # Prepare data
        X = features_df[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Apply PCA for visualization (2 components)
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scatter plot
        scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], 
                                 c=features_df['cluster'], 
                                 cmap='viridis', 
                                 s=100, 
                                 alpha=0.6,
                                 edgecolors='black')
        axes[0].set_xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        axes[0].set_ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
        axes[0].set_title('District Clusters (PCA Projection)', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0], label='Cluster ID')
        
        # Cluster size bar chart
        cluster_sizes = features_df['cluster'].value_counts().sort_index()
        axes[1].bar(cluster_sizes.index, cluster_sizes.values, 
                   color=plt.cm.viridis(np.linspace(0, 1, self.n_clusters)),
                   edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Cluster ID', fontsize=12)
        axes[1].set_ylabel('Number of Districts', fontsize=12)
        axes[1].set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Cluster visualization saved to {output_path}")
        
        return X_pca
    
    def get_cluster_profiles(self):
        """Return cluster profiles DataFrame."""
        return self.cluster_profiles
    
    def get_cluster_summary_report(self):
        """Generate text summary of cluster analysis."""
        if self.cluster_profiles is None:
            return "No cluster profiles available. Run fit() first."
        
        report = ["=" * 80]
        report.append("DISTRICT CLUSTER ANALYSIS - PATTERN DETECTION")
        report.append("=" * 80)
        report.append("")
        
        for _, profile in self.cluster_profiles.iterrows():
            report.append(f"CLUSTER {profile['cluster_id']}: {profile['label']}")
            report.append("-" * 80)
            report.append(f"  Districts: {profile['size']}")
            report.append(f"  Avg Saturation: {profile['avg_saturation']:.2%}")
            report.append(f"  Avg Monthly Enrollments: {profile['avg_enrollments']:.0f}")
            report.append(f"  Avg Monthly Updates: {profile['avg_updates']:.0f}")
            report.append(f"  Avg Rejection Rate: {profile['avg_rejection_rate']:.2%}")
            report.append(f"  Urban Districts: {profile['urban_pct']:.1%}")
            report.append(f"  Female Enrollment: {profile['female_enrollment_pct']:.1%}")
            report.append("")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    print("K-Means Clustering Module - Pattern Detection")
    print("Use this module to identify district service profiles")
