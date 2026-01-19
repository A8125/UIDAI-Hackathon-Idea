"""
Web Scraping Module
====================
Implements contextual web scraping for geographic verification.
Uses public geographic records to verify center proximity.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
from urllib.parse import urlencode
import warnings
warnings.filterwarnings('ignore')


class GeographicVerifier:
    """Verifies enrollment center locations using public geographic data."""
    
    def __init__(self, user_agent=None):
        """
        Initialize geographic verifier.
        
        Parameters:
        -----------
        user_agent : str
            Custom user agent string
        """
        self.user_agent = user_agent or 'AadhaarSevaOptimizer/1.0 (Research Purpose)'
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
        
    def get_pincode_info(self, pincode):
        """
        Fetch geographic information for a pincode from public API.
        
        Parameters:
        -----------
        pincode : str or int
            Indian postal code
            
        Returns:
        --------
        dict : Geographic information
        """
        # Use India Post Office API (publicly available)
        url = f"https://api.postalpincode.in/pincode/{pincode}"
        
        try:
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data and len(data) > 0 and data[0]['Status'] == 'Success':
                    post_office = data[0]['PostOffice'][0]
                    
                    return {
                        'pincode': pincode,
                        'district': post_office.get('District', 'Unknown'),
                        'state': post_office.get('State', 'Unknown'),
                        'region': post_office.get('Region', 'Unknown'),
                        'division': post_office.get('Division', 'Unknown'),
                        'status': 'verified'
                    }
                else:
                    return {'pincode': pincode, 'status': 'not_found'}
            else:
                return {'pincode': pincode, 'status': 'error'}
                
        except Exception as e:
            return {'pincode': pincode, 'status': 'error', 'error': str(e)}
    
    def verify_center_locations(self, centers_df, delay=0.5):
        """
        Verify enrollment center locations using pincode data.
        
        Parameters:
        -----------
        centers_df : pd.DataFrame
            Enrollment centers with pincode information
        delay : float
            Delay between requests to be respectful to API
            
        Returns:
        --------
        pd.DataFrame : Centers with verified geographic information
        """
        verified_data = []
        unique_pincodes = centers_df['pincode'].unique()
        
        print(f"Verifying {len(unique_pincodes)} unique pincodes...")
        
        # Create lookup dictionary
        pincode_lookup = {}
        
        for i, pincode in enumerate(unique_pincodes):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(unique_pincodes)} pincodes")
            
            info = self.get_pincode_info(pincode)
            pincode_lookup[pincode] = info
            
            # Be respectful to API
            time.sleep(delay)
        
        # Merge with centers
        centers_verified = centers_df.copy()
        centers_verified['verified_district'] = centers_verified['pincode'].map(
            lambda x: pincode_lookup.get(x, {}).get('district', 'Unknown')
        )
        centers_verified['verified_state'] = centers_verified['pincode'].map(
            lambda x: pincode_lookup.get(x, {}).get('state', 'Unknown')
        )
        centers_verified['verification_status'] = centers_verified['pincode'].map(
            lambda x: pincode_lookup.get(x, {}).get('status', 'unknown')
        )
        
        verified_count = (centers_verified['verification_status'] == 'verified').sum()
        print(f"✓ Verified {verified_count}/{len(centers_df)} centers")
        
        return centers_verified
    
    def calculate_center_density(self, centers_df, districts_df):
        """
        Calculate enrollment center density by district.
        
        Parameters:
        -----------
        centers_df : pd.DataFrame
            Enrollment center data
        districts_df : pd.DataFrame
            District data with population
            
        Returns:
        --------
        pd.DataFrame : Center density metrics
        """
        # Count centers per district
        center_counts = centers_df.groupby('district_id').agg({
            'center_id': 'count',
            'avg_daily_capacity': 'sum',
            'avg_utilization_rate': 'mean'
        }).reset_index()
        
        center_counts.rename(columns={
            'center_id': 'total_centers',
            'avg_daily_capacity': 'total_daily_capacity',
            'avg_utilization_rate': 'avg_utilization'
        }, inplace=True)
        
        # Merge with population data
        density = center_counts.merge(
            districts_df[['district_id', 'population_lakhs', 'district_type']], 
            on='district_id', 
            how='left'
        )
        
        # Calculate density metrics
        density['centers_per_lakh_population'] = density['total_centers'] / density['population_lakhs']
        density['capacity_per_lakh_population'] = density['total_daily_capacity'] / density['population_lakhs']
        
        # Identify underserved areas
        # Benchmark: at least 1 center per lakh population
        density['is_underserved'] = density['centers_per_lakh_population'] < 1.0
        
        return density
    
    def identify_service_gaps(self, density_df, threshold_centers=1.0):
        """
        Identify districts with inadequate enrollment center coverage.
        
        Parameters:
        -----------
        density_df : pd.DataFrame
            Center density data
        threshold_centers : float
            Minimum centers per lakh population
            
        Returns:
        --------
        pd.DataFrame : Service gap analysis
        """
        gaps = density_df[density_df['centers_per_lakh_population'] < threshold_centers].copy()
        
        # Calculate gap size
        gaps['center_shortage'] = (
            threshold_centers * gaps['population_lakhs'] - gaps['total_centers']
        ).apply(lambda x: max(0, round(x)))
        
        # Prioritize by gap size and utilization
        gaps['priority_score'] = (
            gaps['center_shortage'] * 0.5 + 
            gaps['avg_utilization'] * 100 * 0.3 +
            gaps['population_lakhs'] * 0.2
        )
        
        gaps = gaps.sort_values('priority_score', ascending=False)
        
        return gaps[['district_id', 'district_type', 'population_lakhs', 
                    'total_centers', 'centers_per_lakh_population', 
                    'center_shortage', 'avg_utilization', 'priority_score']]
    
    def generate_mock_geographic_data(self, centers_df):
        """
        Generate mock geographic verification data for demonstration.
        (Used when API is unavailable or for testing)
        
        Parameters:
        -----------
        centers_df : pd.DataFrame
            Enrollment centers
            
        Returns:
        --------
        pd.DataFrame : Centers with mock verified data
        """
        centers_verified = centers_df.copy()
        
        # Simulate verification
        centers_verified['verification_status'] = 'verified'
        centers_verified['verified_district'] = 'District_' + centers_verified['district_id']
        centers_verified['verified_state'] = 'State_' + centers_verified['district_id'].str[:2]
        
        print(f"✓ Generated mock geographic verification for {len(centers_verified)} centers")
        
        return centers_verified


class ProximityAnalyzer:
    """Analyzes proximity and accessibility of enrollment centers."""
    
    def __init__(self):
        """Initialize proximity analyzer."""
        pass
    
    def calculate_haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate distance between two points using Haversine formula.
        
        Parameters:
        -----------
        lat1, lon1 : float
            Coordinates of point 1
        lat2, lon2 : float
            Coordinates of point 2
            
        Returns:
        --------
        float : Distance in kilometers
        """
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        distance = R * c
        
        return distance
    
    def find_nearest_centers(self, target_lat, target_lon, centers_df, n=5):
        """
        Find nearest enrollment centers to a location.
        
        Parameters:
        -----------
        target_lat, target_lon : float
            Target coordinates
        centers_df : pd.DataFrame
            Enrollment centers with coordinates
        n : int
            Number of nearest centers to return
            
        Returns:
        --------
        pd.DataFrame : Nearest centers with distances
        """
        centers = centers_df.copy()
        
        # Calculate distances
        centers['distance_km'] = centers.apply(
            lambda row: self.calculate_haversine_distance(
                target_lat, target_lon, row['latitude'], row['longitude']
            ),
            axis=1
        )
        
        # Sort and return top N
        nearest = centers.nsmallest(n, 'distance_km')
        
        return nearest[['center_id', 'district_id', 'center_type', 
                       'latitude', 'longitude', 'distance_km']]
    
    def analyze_center_clustering(self, centers_df):
        """
        Analyze how centers are clustered geographically.
        
        Parameters:
        -----------
        centers_df : pd.DataFrame
            Enrollment centers with coordinates
            
        Returns:
        --------
        dict : Clustering statistics
        """
        stats = {}
        
        for district_id in centers_df['district_id'].unique():
            district_centers = centers_df[centers_df['district_id'] == district_id]
            
            if len(district_centers) < 2:
                continue
            
            # Calculate all pairwise distances
            distances = []
            for i, row1 in district_centers.iterrows():
                for j, row2 in district_centers.iterrows():
                    if i < j:
                        dist = self.calculate_haversine_distance(
                            row1['latitude'], row1['longitude'],
                            row2['latitude'], row2['longitude']
                        )
                        distances.append(dist)
            
            stats[district_id] = {
                'avg_distance': np.mean(distances) if distances else 0,
                'min_distance': np.min(distances) if distances else 0,
                'max_distance': np.max(distances) if distances else 0
            }
        
        return stats


if __name__ == "__main__":
    print("Web Scraping Module - Geographic Verification")
    print("Use this module to verify enrollment center locations")
