#!/usr/bin/env python3
"""
Aadhaar Seva Optimizer - Main Orchestration Script
===================================================

This script orchestrates the complete data-to-decision pipeline:
1. Data Generation/Loading
2. Pattern Detection (K-Means Clustering)
3. Anomaly Detection (Isolation Forest)
4. Forecasting (Prophet)
5. Geographic Verification (Web Scraping)
6. Visualization and Reporting

Author: UIDAI Analytics Team
Version: 1.0.0
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.data_generator import AadhaarDataGenerator
from modules.pattern_detection import DistrictClusterer
from modules.anomaly_detection import AnomalyDetector
from modules.forecasting import SaturationForecaster
from modules.web_scraping import GeographicVerifier, ProximityAnalyzer
from modules.visualization import ServiceVisualizer


class AadhaarSevaOptimizer:
    """Main orchestrator for Aadhaar service optimization pipeline."""
    
    def __init__(self, data_dir='data', output_dir='output'):
        """
        Initialize optimizer.
        
        Parameters:
        -----------
        data_dir : str
            Directory for data files
        output_dir : str
            Directory for output files
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.data_generator = AadhaarDataGenerator()
        self.clusterer = DistrictClusterer(n_clusters=5)
        self.anomaly_detector = AnomalyDetector(contamination=0.1)
        self.forecaster = SaturationForecaster()
        self.geo_verifier = GeographicVerifier()
        self.proximity_analyzer = ProximityAnalyzer()
        self.visualizer = ServiceVisualizer()
        
        # Data containers
        self.districts_df = None
        self.timeseries_df = None
        self.centers_df = None
        self.features_df = None
        self.anomalies_df = None
        
        print("=" * 80)
        print("AADHAAR SEVA OPTIMIZER - Data-to-Decision Pipeline")
        print("=" * 80)
        print()
    
    def step1_load_or_generate_data(self, generate_new=True, n_districts=100, n_months=24):
        """
        Step 1: Load existing data or generate new anonymized datasets.
        
        Parameters:
        -----------
        generate_new : bool
            Whether to generate new data or load existing
        n_districts : int
            Number of districts (if generating new)
        n_months : int
            Months of historical data (if generating new)
        """
        print("\n[STEP 1] DATA INGESTION")
        print("-" * 80)
        
        if generate_new:
            print("Generating new anonymized datasets...")
            datasets = self.data_generator.generate_complete_dataset(n_districts, n_months)
            self.data_generator.save_datasets(datasets, self.data_dir)
            
            self.districts_df = datasets['districts']
            self.timeseries_df = datasets['timeseries']
            self.centers_df = datasets['centers']
        else:
            print("Loading existing datasets...")
            self.districts_df = pd.read_csv(f'{self.data_dir}/districts.csv')
            self.timeseries_df = pd.read_csv(f'{self.data_dir}/timeseries.csv')
            self.centers_df = pd.read_csv(f'{self.data_dir}/centers.csv')
            print(f"✓ Loaded {len(self.districts_df)} districts")
            print(f"✓ Loaded {len(self.timeseries_df)} time series records")
            print(f"✓ Loaded {len(self.centers_df)} enrollment centers")
    
    def step2_pattern_detection(self):
        """Step 2: Identify district service profiles using K-Means clustering."""
        print("\n[STEP 2] PATTERN DETECTION - K-Means Clustering")
        print("-" * 80)
        
        # Prepare features
        self.features_df = self.clusterer.prepare_features(self.timeseries_df, self.districts_df)
        print(f"✓ Prepared {len(self.features_df.columns)} features for clustering")
        
        # Fit clustering model
        self.features_df = self.clusterer.fit(self.features_df)
        
        # Visualize
        self.clusterer.visualize_clusters(self.features_df, 
                                         f'{self.output_dir}/cluster_analysis.png')
        
        # Print cluster summary
        print("\n" + self.clusterer.get_cluster_summary_report())
        
        # Save cluster profiles
        cluster_profiles = self.clusterer.get_cluster_profiles()
        cluster_profiles.to_csv(f'{self.output_dir}/cluster_profiles.csv', index=False)
        print(f"\n✓ Cluster profiles saved to {self.output_dir}/cluster_profiles.csv")
    
    def step3_anomaly_detection(self):
        """Step 3: Detect anomalies using Isolation Forest."""
        print("\n[STEP 3] ANOMALY DETECTION - Isolation Forest")
        print("-" * 80)
        
        # Prepare features
        timeseries_prepared = self.anomaly_detector.prepare_features(self.timeseries_df)
        
        # Detect anomalies
        self.anomalies_df = self.anomaly_detector.fit_predict(timeseries_prepared)
        
        # Get summary
        anomaly_summary = self.anomaly_detector.get_anomaly_summary()
        print(f"\nAnomaly Detection Summary:")
        print(f"  Total Anomalies: {anomaly_summary['total_anomalies']}")
        print(f"  Critical: {anomaly_summary['critical_anomalies']}")
        print(f"  High: {anomaly_summary['high_anomalies']}")
        print(f"  Medium: {anomaly_summary['medium_anomalies']}")
        print(f"  Low: {anomaly_summary['low_anomalies']}")
        print(f"  Affected Districts: {anomaly_summary['affected_districts']}")
        
        # Visualize
        self.anomaly_detector.visualize_anomalies(self.anomalies_df, 
                                                 f'{self.output_dir}/anomaly_detection.png')
        
        # Get top anomalies
        top_anomalies = self.anomaly_detector.get_top_anomalies(self.anomalies_df, n=20)
        top_anomalies.to_csv(f'{self.output_dir}/top_anomalies.csv', index=False)
        print(f"\n✓ Top anomalies saved to {self.output_dir}/top_anomalies.csv")
    
    def step4_forecasting(self, n_sample_districts=10, forecast_periods=12):
        """
        Step 4: Forecast saturation levels using Prophet.
        
        Parameters:
        -----------
        n_sample_districts : int
            Number of districts to forecast (for demonstration)
        forecast_periods : int
            Months to forecast ahead
        """
        print("\n[STEP 4] PREDICTIVE FORECASTING - Prophet")
        print("-" * 80)
        
        # Generate saturation report
        saturation_report = self.forecaster.generate_saturation_report(
            self.timeseries_df, 
            self.districts_df, 
            sample_size=n_sample_districts
        )
        
        print(f"\nSaturation Forecast Summary:")
        if len(saturation_report) > 0:
            print(f"  Districts analyzed: {len(saturation_report)}")
            will_reach = saturation_report['will_reach_99pct'].sum()
            print(f"  Will reach 99% saturation: {will_reach}/{len(saturation_report)}")
        
            # Save report
            saturation_report.to_csv(f'{self.output_dir}/saturation_forecast.csv', index=False)
            print(f"\n✓ Saturation forecast saved to {self.output_dir}/saturation_forecast.csv")
            
            # Visualize sample forecast
            sample_district = saturation_report.iloc[0]['district_id']
            self.forecaster.visualize_forecast(sample_district, self.districts_df,
                                             f'{self.output_dir}/forecast_sample.png')
        else:
            print("  Note: No successful forecasts generated (this can happen with synthetic data)")
            print("  Skipping forecast visualizations")
    
    def step5_geographic_verification(self, use_mock=True):
        """
        Step 5: Verify center locations and analyze service gaps.
        
        Parameters:
        -----------
        use_mock : bool
            Use mock data instead of real API calls (recommended for demo)
        """
        print("\n[STEP 5] GEOGRAPHIC VERIFICATION - Web Scraping")
        print("-" * 80)
        
        if use_mock:
            print("Using mock geographic data for demonstration...")
            centers_verified = self.geo_verifier.generate_mock_geographic_data(self.centers_df)
        else:
            print("Verifying center locations via public API...")
            centers_verified = self.geo_verifier.verify_center_locations(self.centers_df)
        
        # Calculate center density
        density_df = self.geo_verifier.calculate_center_density(centers_verified, self.districts_df)
        
        print(f"\nCenter Density Summary:")
        print(f"  Avg centers per lakh: {density_df['centers_per_lakh_population'].mean():.2f}")
        print(f"  Underserved districts: {density_df['is_underserved'].sum()}")
        
        # Identify service gaps
        service_gaps = self.geo_verifier.identify_service_gaps(density_df)
        
        print(f"\nTop 5 Service Gap Districts:")
        for i, row in service_gaps.head(5).iterrows():
            print(f"  {row['district_id']}: {row['center_shortage']:.0f} centers needed " +
                  f"(Priority: {row['priority_score']:.1f})")
        
        # Save results
        density_df.to_csv(f'{self.output_dir}/center_density.csv', index=False)
        service_gaps.to_csv(f'{self.output_dir}/service_gaps.csv', index=False)
        print(f"\n✓ Geographic analysis saved to output directory")
    
    def step6_visualization(self):
        """Step 6: Create comprehensive visualizations."""
        print("\n[STEP 6] VISUALIZATION - Heatmaps and Dashboards")
        print("-" * 80)
        
        # Service stress heatmap
        stress_metrics = self.visualizer.create_service_stress_heatmap(
            self.features_df,
            f'{self.output_dir}/service_stress_heatmap.png'
        )
        
        # Geographic distribution
        self.visualizer.create_geographic_distribution(
            self.districts_df,
            self.features_df,
            f'{self.output_dir}/geographic_distribution.png'
        )
        
        # Comprehensive dashboard
        self.visualizer.create_comprehensive_dashboard(
            self.timeseries_df,
            self.features_df,
            f'{self.output_dir}/comprehensive_dashboard.png'
        )
        
        print(f"\n✓ All visualizations saved to {self.output_dir}/ directory")
    
    def step7_generate_recommendations(self):
        """Step 7: Generate actionable recommendations."""
        print("\n[STEP 7] DECISION SUPPORT - Recommendations")
        print("-" * 80)
        
        recommendations = []
        
        # Load service gaps
        if os.path.exists(f'{self.output_dir}/service_gaps.csv'):
            service_gaps = pd.read_csv(f'{self.output_dir}/service_gaps.csv')
            
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Infrastructure Deployment',
                'action': f"Deploy Mobile Enrollment Vans to {len(service_gaps)} underserved districts",
                'districts': service_gaps.head(5)['district_id'].tolist(),
                'impact': f"Improve coverage for {service_gaps['population_lakhs'].sum():.0f} lakh population"
            })
        
        # Anomaly-based recommendations
        if os.path.exists(f'{self.output_dir}/top_anomalies.csv'):
            top_anomalies = pd.read_csv(f'{self.output_dir}/top_anomalies.csv')
            high_rejection = top_anomalies[top_anomalies['severity'] == 'Critical']
            
            if len(high_rejection) > 0:
                recommendations.append({
                    'priority': 'CRITICAL',
                    'category': 'Quality Assurance',
                    'action': f"Investigate high rejection rates in {high_rejection['district_id'].nunique()} districts",
                    'districts': high_rejection['district_id'].unique().tolist()[:5],
                    'impact': "Reduce enrollment failures and improve citizen satisfaction"
                })
        
        # Saturation-based recommendations
        low_saturation = self.features_df[self.features_df['saturation_level_last'] < 0.80]
        if len(low_saturation) > 0:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Enrollment Campaign',
                'action': f"Launch awareness campaigns in {len(low_saturation)} low-saturation districts",
                'districts': low_saturation['district_id'].head(5).tolist(),
                'impact': "Accelerate path to universal Aadhaar coverage"
            })
        
        # Print recommendations
        print("\nACTIONABLE RECOMMENDATIONS:")
        print("=" * 80)
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. [{rec['priority']}] {rec['category']}")
            print(f"   Action: {rec['action']}")
            print(f"   Sample Districts: {', '.join(rec['districts'][:3])}")
            print(f"   Impact: {rec['impact']}")
        
        # Save recommendations
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df.to_json(f'{self.output_dir}/recommendations.json', 
                                   orient='records', indent=2)
        print(f"\n✓ Recommendations saved to {self.output_dir}/recommendations.json")
    
    def run_complete_pipeline(self, generate_new_data=True):
        """Execute the complete data-to-decision pipeline."""
        start_time = datetime.now()
        
        try:
            # Execute all steps
            self.step1_load_or_generate_data(generate_new=generate_new_data)
            self.step2_pattern_detection()
            self.step3_anomaly_detection()
            self.step4_forecasting()
            self.step5_geographic_verification(use_mock=True)
            self.step6_visualization()
            self.step7_generate_recommendations()
            
            # Completion summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print("\n" + "=" * 80)
            print("PIPELINE EXECUTION COMPLETE")
            print("=" * 80)
            print(f"Execution Time: {duration:.2f} seconds")
            print(f"Output Directory: {os.path.abspath(self.output_dir)}")
            print("\nGenerated Files:")
            for filename in sorted(os.listdir(self.output_dir)):
                filepath = os.path.join(self.output_dir, filename)
                if os.path.isfile(filepath):
                    size_kb = os.path.getsize(filepath) / 1024
                    print(f"  - {filename} ({size_kb:.1f} KB)")
            
            print("\n" + "=" * 80)
            print("Thank you for using Aadhaar Seva Optimizer!")
            print("=" * 80)
            
        except Exception as e:
            print(f"\n❌ ERROR: Pipeline execution failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║         AADHAAR SEVA OPTIMIZER - ML-Powered Analytics             ║")
    print("║                                                                    ║")
    print("║  Optimizing Aadhaar Service Delivery through:                     ║")
    print("║  • K-Means Clustering (Pattern Detection)                         ║")
    print("║  • Isolation Forest (Anomaly Detection)                           ║")
    print("║  • Prophet (Time-Series Forecasting)                              ║")
    print("║  • Web Scraping (Geographic Verification)                         ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print("\n")
    
    # Initialize and run optimizer
    optimizer = AadhaarSevaOptimizer(data_dir='data', output_dir='output')
    optimizer.run_complete_pipeline(generate_new_data=True)


if __name__ == "__main__":
    main()
