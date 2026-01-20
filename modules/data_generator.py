"""
Data Generator Module
=====================
Creates anonymized, aggregated sample datasets mimicking UIDAI data structure.
Uses no actual personal identifiable information.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class AadhaarDataGenerator:
    """Generates realistic anonymized Aadhaar enrollment and service data."""
    
    def __init__(self, seed=42):
        """Initialize generator with random seed for reproducibility."""
        np.random.seed(seed)
        self.seed = seed
    
    def generate_district_data(self, n_districts=100, n_months=24):
        """
        Generate district-level aggregated enrollment and service data.
        
        Parameters:
        -----------
        n_districts : int
            Number of districts to generate data for
        n_months : int
            Number of months of historical data
            
        Returns:
        --------
        pd.DataFrame : Aggregated district-level metrics
        """
        districts = []
        states = ['Maharashtra', 'Uttar Pradesh', 'Bihar', 'West Bengal', 
                  'Madhya Pradesh', 'Tamil Nadu', 'Rajasthan', 'Karnataka',
                  'Gujarat', 'Andhra Pradesh', 'Odisha', 'Telangana']
        
        urban_rural = ['Urban', 'Rural', 'Semi-Urban']
        
        for i in range(n_districts):
            state = np.random.choice(states)
            district_name = f"{state}_District_{i % 20 + 1}"
            district_type = np.random.choice(urban_rural, p=[0.3, 0.5, 0.2])
            
            # Base population (in lakhs)
            if district_type == 'Urban':
                base_pop = np.random.uniform(5, 20)
            elif district_type == 'Rural':
                base_pop = np.random.uniform(2, 10)
            else:
                base_pop = np.random.uniform(3, 12)
            
            districts.append({
                'district_id': f'D{i:04d}',
                'district_name': district_name,
                'state': state,
                'district_type': district_type,
                'population_lakhs': round(base_pop, 2)
            })
        
        return pd.DataFrame(districts)
    
    def generate_enrollment_timeseries(self, districts_df, n_months=24):
        """
        Generate monthly enrollment time series data.
        
        Parameters:
        -----------
        districts_df : pd.DataFrame
            District master data
        n_months : int
            Number of months of data
            
        Returns:
        --------
        pd.DataFrame : Time series enrollment data
        """
        records = []
        start_date = datetime.now() - timedelta(days=30*n_months)
        
        for _, district in districts_df.iterrows():
            district_id = district['district_id']
            district_type = district['district_type']
            population = district['population_lakhs'] * 100000
            
            # Saturation level affects enrollment rate
            current_saturation = np.random.uniform(0.70, 0.95)
            
            for month in range(n_months):
                date = start_date + timedelta(days=30*month)
                
                # Seasonal effects (school admission in April-May, Jan-Feb)
                seasonal_factor = 1.0
                if date.month in [4, 5, 1, 2]:
                    seasonal_factor = 1.3
                elif date.month in [6, 7, 11, 12]:
                    seasonal_factor = 1.1
                
                # Base enrollment rate depends on type and saturation
                if district_type == 'Urban':
                    base_rate = 0.002 * (1 - current_saturation)
                elif district_type == 'Rural':
                    base_rate = 0.003 * (1 - current_saturation)
                else:
                    base_rate = 0.0025 * (1 - current_saturation)
                
                # Add noise
                noise = np.random.normal(1.0, 0.15)
                
                # Calculate enrollments
                new_enrollments = int(population * base_rate * seasonal_factor * noise)
                biometric_revisions = int(new_enrollments * np.random.uniform(0.8, 1.2))
                
                # Update saturation
                current_saturation += (new_enrollments / population) * 0.5
                current_saturation = min(current_saturation, 0.99)
                
                # Service metrics
                total_requests = new_enrollments + biometric_revisions
                rejections = int(total_requests * np.random.uniform(0.01, 0.05))
                successful = total_requests - rejections
                
                # Gender distribution
                male_pct = np.random.uniform(0.48, 0.58)
                female_enrollments = int(new_enrollments * (1 - male_pct))
                male_enrollments = new_enrollments - female_enrollments
                
                records.append({
                    'district_id': district_id,
                    'date': date.strftime('%Y-%m-%d'),
                    'year_month': date.strftime('%Y-%m'),
                    'new_enrollments': new_enrollments,
                    'male_enrollments': male_enrollments,
                    'female_enrollments': female_enrollments,
                    'biometric_revisions': biometric_revisions,
                    'total_requests': total_requests,
                    'successful_requests': successful,
                    'rejections': rejections,
                    'rejection_rate': round(rejections / total_requests if total_requests > 0 else 0, 4),
                    'saturation_level': round(current_saturation, 4),
                    'active_centers': int(np.random.uniform(2, 15))
                })
        
        return pd.DataFrame(records)
    
    def generate_center_data(self, districts_df):
        """
        Generate enrollment center data with geographic information.
        
        Parameters:
        -----------
        districts_df : pd.DataFrame
            District master data
            
        Returns:
        --------
        pd.DataFrame : Enrollment center data
        """
        centers = []
        center_id = 1
        
        for _, district in districts_df.iterrows():
            district_id = district['district_id']
            district_type = district['district_type']
            
            # Number of centers based on type
            if district_type == 'Urban':
                n_centers = np.random.randint(5, 15)
            elif district_type == 'Rural':
                n_centers = np.random.randint(2, 8)
            else:
                n_centers = np.random.randint(3, 10)
            
            for _ in range(n_centers):
                # Generate fake coordinates (India's approximate bounds)
                latitude = np.random.uniform(8.0, 35.0)
                longitude = np.random.uniform(68.0, 97.0)
                
                # Center performance metrics
                avg_daily_capacity = np.random.randint(50, 300)
                avg_utilization = np.random.uniform(0.5, 0.95)
                
                centers.append({
                    'center_id': f'C{center_id:05d}',
                    'district_id': district_id,
                    'center_type': np.random.choice(['Government', 'Private', 'Post Office', 'Bank'], 
                                                    p=[0.4, 0.3, 0.2, 0.1]),
                    'latitude': round(latitude, 6),
                    'longitude': round(longitude, 6),
                    'pincode': np.random.randint(100000, 999999),
                    'avg_daily_capacity': avg_daily_capacity,
                    'avg_utilization_rate': round(avg_utilization, 3),
                    'operational_days_per_month': np.random.randint(20, 30)
                })
                center_id += 1
        
        return pd.DataFrame(centers)
    
    def generate_complete_dataset(self, n_districts=100, n_months=24):
        """
        Generate complete dataset with all components.
        
        Returns:
        --------
        dict : Dictionary containing all generated dataframes
        """
        print(f"Generating anonymized Aadhaar data...")
        print(f"Districts: {n_districts}, Months: {n_months}")
        
        districts_df = self.generate_district_data(n_districts, n_months)
        print(f"✓ Generated {len(districts_df)} districts")
        
        timeseries_df = self.generate_enrollment_timeseries(districts_df, n_months)
        print(f"✓ Generated {len(timeseries_df)} time series records")
        
        centers_df = self.generate_center_data(districts_df)
        print(f"✓ Generated {len(centers_df)} enrollment centers")
        
        return {
            'districts': districts_df,
            'timeseries': timeseries_df,
            'centers': centers_df
        }
    
    def save_datasets(self, datasets, output_dir='data'):
        """Save generated datasets to CSV files."""
        datasets['districts'].to_csv(f'{output_dir}/districts.csv', index=False)
        datasets['timeseries'].to_csv(f'{output_dir}/timeseries.csv', index=False)
        datasets['centers'].to_csv(f'{output_dir}/centers.csv', index=False)
        print(f"\n✓ All datasets saved to '{output_dir}/' directory")


if __name__ == "__main__":
    # Generate sample data
    generator = AadhaarDataGenerator(seed=42)
    datasets = generator.generate_complete_dataset(n_districts=100, n_months=24)
    generator.save_datasets(datasets, output_dir='../data')
