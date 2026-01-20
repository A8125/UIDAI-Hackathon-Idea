"""
Visualization Module
====================
Creates comprehensive visualizations and heatmaps for decision-making.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ServiceVisualizer:
    """Creates visualizations for Aadhaar service optimization."""
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """Initialize visualizer with plot style."""
        plt.style.use('default')
        sns.set_palette("husl")
        self.fig_count = 0
        
    def create_service_stress_heatmap(self, features_df, output_path='output/service_stress_heatmap.png'):
        """
        Create heatmap showing service stress points across districts.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            District features with cluster assignments
        output_path : str
            Path to save visualization
        """
        # Create pivot for heatmap
        # Select key stress indicators
        stress_metrics = features_df[['district_id', 'rejection_rate_mean', 'saturation_level_last', 
                                      'active_centers_mean', 'new_enrollments_mean']].copy()
        
        # Normalize metrics to 0-1 scale for comparison
        for col in ['rejection_rate_mean', 'saturation_level_last', 'active_centers_mean', 'new_enrollments_mean']:
            if col in stress_metrics.columns:
                max_val = stress_metrics[col].max()
                min_val = stress_metrics[col].min()
                if max_val != min_val:
                    stress_metrics[f'{col}_normalized'] = (stress_metrics[col] - min_val) / (max_val - min_val)
                else:
                    stress_metrics[f'{col}_normalized'] = 0.5
        
        # Calculate composite stress score
        # High rejection = high stress, Low saturation = high stress (still growing), 
        # Low centers = high stress, High enrollments with low centers = high stress
        stress_metrics['stress_score'] = (
            stress_metrics['rejection_rate_mean_normalized'] * 0.4 +
            (1 - stress_metrics['saturation_level_last_normalized']) * 0.2 +
            (1 - stress_metrics['active_centers_mean_normalized']) * 0.2 +
            stress_metrics['new_enrollments_mean_normalized'] * 0.2
        )
        
        # Sort by stress score
        stress_metrics = stress_metrics.sort_values('stress_score', ascending=False)
        
        # Select top 30 for visualization
        top_districts = stress_metrics.head(30)
        
        # Prepare heatmap data
        heatmap_data = top_districts[[
            'rejection_rate_mean_normalized',
            'saturation_level_last_normalized', 
            'active_centers_mean_normalized',
            'new_enrollments_mean_normalized',
            'stress_score'
        ]].T
        
        heatmap_data.columns = [f"D{i+1}" for i in range(len(top_districts))]
        heatmap_data.index = ['Rejection Rate', 'Saturation Level', 'Center Availability', 
                              'Enrollment Volume', 'Overall Stress']
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(16, 6))
        
        sns.heatmap(heatmap_data, annot=False, cmap='YlOrRd', 
                   cbar_kws={'label': 'Normalized Score (0=Low, 1=High)'}, 
                   linewidths=0.5, ax=ax)
        
        ax.set_title('Service Stress Points - Top 30 High-Priority Districts', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('District (Ranked by Stress)', fontsize=12)
        ax.set_ylabel('Stress Indicators', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Service stress heatmap saved to {output_path}")
        
        return stress_metrics
    
    def create_geographic_distribution(self, districts_df, features_df, output_path='output/geographic_distribution.png'):
        """
        Visualize geographic distribution of metrics.
        
        Parameters:
        -----------
        districts_df : pd.DataFrame
            District metadata
        features_df : pd.DataFrame
            District features
        output_path : str
            Path to save visualization
        """
        # Merge data
        geo_data = districts_df.merge(
            features_df[['district_id', 'cluster', 'saturation_level_last', 'rejection_rate_mean']], 
            on='district_id', 
            how='left'
        )
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. District type distribution
        type_counts = geo_data['district_type'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        axes[0, 0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                      colors=colors, startangle=90, explode=[0.05] * len(type_counts))
        axes[0, 0].set_title('Distribution by District Type', fontsize=14, fontweight='bold')
        
        # 2. Saturation by state
        state_sat = geo_data.groupby('state')['saturation_level_last'].mean().sort_values(ascending=False).head(10)
        axes[0, 1].barh(range(len(state_sat)), state_sat.values, color='skyblue', edgecolor='black')
        axes[0, 1].set_yticks(range(len(state_sat)))
        axes[0, 1].set_yticklabels(state_sat.index, fontsize=10)
        axes[0, 1].set_xlabel('Average Saturation Level', fontsize=12)
        axes[0, 1].set_title('Top 10 States by Saturation', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        axes[0, 1].axvline(x=0.90, color='red', linestyle='--', linewidth=2, alpha=0.7, label='90% Target')
        axes[0, 1].legend()
        
        # 3. Cluster distribution by state
        cluster_state = pd.crosstab(geo_data['state'], geo_data['cluster'])
        top_states = geo_data['state'].value_counts().head(8).index
        cluster_state_top = cluster_state.loc[top_states]
        
        cluster_state_top.plot(kind='bar', stacked=True, ax=axes[1, 0], 
                              colormap='viridis', edgecolor='black')
        axes[1, 0].set_xlabel('State', fontsize=12)
        axes[1, 0].set_ylabel('Number of Districts', fontsize=12)
        axes[1, 0].set_title('District Clusters by State', fontsize=14, fontweight='bold')
        axes[1, 0].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Population vs Saturation scatter
        axes[1, 1].scatter(geo_data['population_lakhs'], geo_data['saturation_level_last'], 
                          c=geo_data['cluster'], cmap='viridis', s=100, alpha=0.6, edgecolors='black')
        axes[1, 1].set_xlabel('Population (Lakhs)', fontsize=12)
        axes[1, 1].set_ylabel('Saturation Level', fontsize=12)
        axes[1, 1].set_title('Population vs Saturation by Cluster', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0.90, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Geographic distribution saved to {output_path}")
    
    def create_comprehensive_dashboard(self, timeseries_df, features_df, 
                                      output_path='output/comprehensive_dashboard.png'):
        """
        Create comprehensive dashboard with multiple metrics.
        
        Parameters:
        -----------
        timeseries_df : pd.DataFrame
            Time series data
        features_df : pd.DataFrame
            District features
        output_path : str
            Path to save visualization
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Enrollment trends over time
        ax1 = fig.add_subplot(gs[0, :])
        monthly_enrollments = timeseries_df.groupby('year_month')['new_enrollments'].sum()
        monthly_revisions = timeseries_df.groupby('year_month')['biometric_revisions'].sum()
        
        x_pos = range(len(monthly_enrollments))
        ax1.plot(x_pos, monthly_enrollments.values, marker='o', linewidth=2, 
                markersize=6, label='New Enrollments', color='#2E86AB')
        ax1.plot(x_pos, monthly_revisions.values, marker='s', linewidth=2, 
                markersize=6, label='Biometric Revisions', color='#A23B72')
        ax1.fill_between(x_pos, monthly_enrollments.values, alpha=0.3, color='#2E86AB')
        ax1.fill_between(x_pos, monthly_revisions.values, alpha=0.3, color='#A23B72')
        ax1.set_xlabel('Month', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('National Enrollment Trends', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 2. Saturation distribution
        ax2 = fig.add_subplot(gs[1, 0])
        saturation_bins = [0, 0.70, 0.80, 0.90, 0.95, 1.0]
        saturation_labels = ['<70%', '70-80%', '80-90%', '90-95%', '95%+']
        features_df['saturation_bin'] = pd.cut(features_df['saturation_level_last'], 
                                               bins=saturation_bins, labels=saturation_labels)
        sat_dist = features_df['saturation_bin'].value_counts().sort_index()
        
        colors_sat = ['#FF6B6B', '#FFA06B', '#FFE66D', '#95E1D3', '#38A3A5']
        ax2.bar(range(len(sat_dist)), sat_dist.values, color=colors_sat, edgecolor='black', alpha=0.8)
        ax2.set_xticks(range(len(sat_dist)))
        ax2.set_xticklabels(sat_dist.index, rotation=45, ha='right')
        ax2.set_ylabel('Number of Districts', fontsize=12)
        ax2.set_title('Saturation Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Rejection rate distribution
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(features_df['rejection_rate_mean'], bins=30, edgecolor='black', 
                color='coral', alpha=0.7)
        ax3.axvline(features_df['rejection_rate_mean'].mean(), color='red', 
                   linestyle='--', linewidth=2, label='Mean')
        ax3.axvline(features_df['rejection_rate_mean'].median(), color='blue', 
                   linestyle='--', linewidth=2, label='Median')
        ax3.set_xlabel('Average Rejection Rate', fontsize=12)
        ax3.set_ylabel('Number of Districts', fontsize=12)
        ax3.set_title('Rejection Rate Distribution', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Gender distribution
        ax4 = fig.add_subplot(gs[1, 2])
        total_male = features_df['male_enrollments_sum'].sum()
        total_female = features_df['female_enrollments_sum'].sum()
        
        ax4.pie([total_male, total_female], labels=['Male', 'Female'], 
               autopct='%1.1f%%', colors=['#6C5CE7', '#FD79A8'], startangle=90,
               explode=[0.05, 0.05])
        ax4.set_title('Gender Distribution in Enrollments', fontsize=14, fontweight='bold')
        
        # 5. Cluster characteristics
        ax5 = fig.add_subplot(gs[2, :])
        cluster_metrics = features_df.groupby('cluster').agg({
            'new_enrollments_mean': 'mean',
            'biometric_revisions_mean': 'mean',
            'rejection_rate_mean': 'mean',
            'saturation_level_last': 'mean'
        })
        
        x = np.arange(len(cluster_metrics))
        width = 0.2
        
        ax5.bar(x - 1.5*width, cluster_metrics['new_enrollments_mean'], width, 
               label='Avg Enrollments', color='#6C5CE7', edgecolor='black')
        ax5.bar(x - 0.5*width, cluster_metrics['biometric_revisions_mean'], width, 
               label='Avg Revisions', color='#A29BFE', edgecolor='black')
        ax5_twin = ax5.twinx()
        ax5_twin.bar(x + 0.5*width, cluster_metrics['rejection_rate_mean'] * 100, width, 
                    label='Rejection Rate (%)', color='#FD79A8', edgecolor='black', alpha=0.7)
        ax5_twin.bar(x + 1.5*width, cluster_metrics['saturation_level_last'] * 100, width, 
                    label='Saturation (%)', color='#FDCB6E', edgecolor='black', alpha=0.7)
        
        ax5.set_xlabel('Cluster ID', fontsize=12)
        ax5.set_ylabel('Average Count', fontsize=12)
        ax5_twin.set_ylabel('Percentage (%)', fontsize=12)
        ax5.set_title('Cluster Characteristics Comparison', fontsize=16, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels([f'Cluster {i}' for i in cluster_metrics.index])
        ax5.legend(loc='upper left', fontsize=10)
        ax5_twin.legend(loc='upper right', fontsize=10)
        ax5.grid(True, alpha=0.3, axis='y')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comprehensive dashboard saved to {output_path}")


if __name__ == "__main__":
    print("Visualization Module - Service Analytics Dashboard")
    print("Use this module to create comprehensive visualizations")
