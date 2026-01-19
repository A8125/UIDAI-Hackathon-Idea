"""
PDF Documentation Generator
============================
Creates comprehensive PDF documentation for Aadhaar Seva Optimizer.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                                Table, TableStyle, Image, KeepTogether)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfgen import canvas
from datetime import datetime
import os


class DocumentationGenerator:
    """Generates comprehensive PDF documentation."""
    
    def __init__(self, output_path='docs/Aadhaar_Seva_Optimizer_Documentation.pdf'):
        """Initialize documentation generator."""
        self.output_path = output_path
        self.doc = SimpleDocTemplate(output_path, pagesize=A4,
                                    leftMargin=0.75*inch, rightMargin=0.75*inch,
                                    topMargin=1*inch, bottomMargin=0.75*inch)
        self.story = []
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=HexColor('#1a237e'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Heading styles
        self.styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=HexColor('#283593'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=HexColor('#3949ab'),
            spaceAfter=10,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading3',
            parent=self.styles['Heading2'],
            fontSize=12,
            textColor=HexColor('#5c6bc0'),
            spaceAfter=8,
            spaceBefore=8,
            fontName='Helvetica-Bold'
        ))
        
        # Body text
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=12,
            leading=14
        ))
        
        # Code style
        self.styles.add(ParagraphStyle(
            name='CustomCode',
            parent=self.styles['Code'],
            fontSize=9,
            fontName='Courier',
            backColor=HexColor('#f5f5f5'),
            leftIndent=20,
            rightIndent=20,
            spaceAfter=12,
            spaceBefore=6
        ))
        
        # Caption style
        self.styles.add(ParagraphStyle(
            name='CustomCaption',
            parent=self.styles['BodyText'],
            fontSize=9,
            textColor=HexColor('#757575'),
            alignment=TA_CENTER,
            spaceBefore=6,
            spaceAfter=12,
            fontName='Helvetica-Oblique'
        ))
    
    def _add_image_if_exists(self, filename, caption, width=5.5*inch):
        """Helper to add an image to the story if it exists."""
        full_path = os.path.join('output', filename)
        if os.path.exists(full_path):
            try:
                img = Image(full_path, width=width, height=None) # Height=None keeps aspect ratio
                # Estimate height to maintain aspect ratio
                # Unfortunately reportlab Image doesn't auto-calculate height from width easily without opening file
                # But passing width=width and height=None usually works or we can use a better helper
                
                # Use a more robust way to get image aspect ratio
                from PIL import Image as PILImage
                with PILImage.open(full_path) as pimg:
                    w, h = pimg.size
                    aspect = h / float(w)
                    img.drawHeight = width * aspect
                    img.drawWidth = width
                
                self.story.append(KeepTogether([
                    img,
                    Spacer(1, 0.05*inch),
                    Paragraph(f"Figure: {caption}", self.styles['CustomCaption'])
                ]))
                self.story.append(Spacer(1, 0.2*inch))
                return True
            except Exception as e:
                print(f"Warning: Could not add image {filename}: {str(e)}")
        return False

    def add_architecture_diagram(self):
        """Add a professional architecture diagram."""
        self.story.append(Paragraph("System Architecture Diagram", self.styles['CustomHeading2']))
        
        # Create a stylized diagram using a table
        diagram_data = [
            [Paragraph("<b>DATA SOURCES</b>", self.styles['CustomHeading3']), "", ""],
            ["", "⬇", ""],
            [Paragraph("<b>ANONYMIZED DATA INGESTION</b><br/><font size=8>(Data Generator / UIDAI Datasets)</font>", self.styles['CustomBody']), "", ""],
            ["", "⬇", ""],
            [
                Paragraph("<b>ANALYSIS ENGINE</b>", self.styles['CustomBody']),
                Paragraph("<b>PREDICTION ENGINE</b>", self.styles['CustomBody']),
                Paragraph("<b>ENRICHMENT</b>", self.styles['CustomBody'])
            ],
            [
                Paragraph("<font size=8>• K-Means Clustering<br/>• Isolation Forest</font>", self.styles['CustomBody']),
                Paragraph("<font size=8>• Prophet Forecasting<br/>• Saturation Modeling</font>", self.styles['CustomBody']),
                Paragraph("<font size=8>• Geographic Scraping<br/>• Pincode Verification</font>", self.styles['CustomBody'])
            ],
            ["", "⬇", ""],
            [Paragraph("<b>DECISION SUPPORT LAYER</b><br/><font size=8>(Dashboard / Recommendation Engine)</font>", self.styles['CustomBody']), "", ""],
            ["", "⬇", ""],
            [Paragraph("<b>ACTIONABLE RECOMMENDATIONS</b>", self.styles['CustomHeading3']), "", ""]
        ]
        
        table = Table(diagram_data, colWidths=[2.2*inch, 0.5*inch, 2.2*inch])
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            # Data Sources Box
            ('BACKGROUND', (0, 0), (2, 0), HexColor('#E8EAF6')),
            ('BOX', (0, 0), (2, 0), 1, HexColor('#3F51B5')),
            # Ingestion Box
            ('BACKGROUND', (0, 2), (2, 2), HexColor('#F5F5F5')),
            ('BOX', (0, 2), (2, 2), 1, HexColor('#9E9E9E')),
            # Engines Box
            ('BACKGROUND', (0, 4), (0, 5), HexColor('#E1F5FE')),
            ('BOX', (0, 4), (0, 5), 1, HexColor('#03A9F4')),
            ('BACKGROUND', (1, 4), (1, 5), HexColor('#FFF9C4')),
            ('BOX', (1, 4), (1, 5), 1, HexColor('#FBC02D')),
            ('BACKGROUND', (2, 4), (2, 5), HexColor('#E8F5E9')),
            ('BOX', (2, 4), (2, 5), 1, HexColor('#4CAF50')),
            # Decision Box
            ('BACKGROUND', (0, 7), (2, 7), HexColor('#FFF3E0')),
            ('BOX', (0, 7), (2, 7), 1, HexColor('#FF9800')),
            # Action Box
            ('BACKGROUND', (0, 9), (2, 9), HexColor('#FCE4EC')),
            ('BOX', (0, 9), (2, 9), 1, HexColor('#E91E63')),
            # Spacing
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('SPAN', (0, 0), (2, 0)),
            ('SPAN', (0, 1), (2, 1)),
            ('SPAN', (0, 2), (2, 2)),
            ('SPAN', (0, 3), (2, 3)),
            ('SPAN', (0, 6), (2, 6)),
            ('SPAN', (0, 7), (2, 7)),
            ('SPAN', (0, 8), (2, 8)),
            ('SPAN', (0, 9), (2, 9)),
        ]))
        
        self.story.append(Spacer(1, 0.2*inch))
        self.story.append(table)
        self.story.append(Spacer(1, 0.4*inch))

    def add_cover_page(self):
        """Add cover page."""
        # Title
        self.story.append(Spacer(1, 2*inch))
        
        title = Paragraph("Aadhaar Seva Optimizer", self.styles['CustomTitle'])
        self.story.append(title)
        self.story.append(Spacer(1, 0.2*inch))
        
        subtitle = Paragraph(
            "Machine Learning-Powered Service Delivery Optimization",
            ParagraphStyle(name='subtitle', parent=self.styles['CustomBody'],
                          fontSize=14, alignment=TA_CENTER, textColor=HexColor('#424242'))
        )
        self.story.append(subtitle)
        self.story.append(Spacer(1, 1*inch))
        
        # System description
        desc = Paragraph(
            """This document provides a comprehensive explanation of the algorithms, 
            workflows, and implementation details of the Aadhaar Seva Optimizer system. 
            The system uses advanced machine learning techniques to optimize Aadhaar 
            enrollment service delivery across India.""",
            ParagraphStyle(name='desc', parent=self.styles['CustomBody'],
                          alignment=TA_CENTER, fontSize=12)
        )
        self.story.append(desc)
        self.story.append(Spacer(1, 1*inch))
        
        # Metadata
        metadata = [
            ["Document Version:", "1.0"],
            ["Date:", datetime.now().strftime("%B %d, %Y")],
            ["Organization:", "UIDAI Analytics Team"],
            ["Classification:", "Public - Demonstration"],
            ["Code Repository:", "github.com/A8125/UIDAI-Hackathon-Idea"],
        ]
        
        table = Table(metadata, colWidths=[2.5*inch, 3*inch])
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        self.story.append(table)
        
        self.story.append(PageBreak())
    
    def add_executive_summary(self):
        """Add executive summary."""
        self.story.append(Paragraph("Executive Summary", self.styles['CustomHeading1']))
        
        content = [
            ("System Overview", 
             """The Aadhaar Seva Optimizer is an advanced analytics system designed to 
             optimize the delivery of Aadhaar enrollment and biometric update services 
             across India. It leverages machine learning algorithms to identify patterns, 
             detect anomalies, and predict future service demand."""),
            
            ("Key Capabilities",
             """• <b>Pattern Detection:</b> Groups districts into service profiles using K-Means clustering<br/>
             • <b>Anomaly Detection:</b> Identifies unusual enrollment patterns using Isolation Forest<br/>
             • <b>Predictive Forecasting:</b> Projects saturation timelines using Prophet<br/>
             • <b>Geographic Analysis:</b> Verifies center locations and identifies service gaps<br/>
             • <b>Decision Support:</b> Generates actionable recommendations for resource deployment"""),
            
            ("Impact on Decision-Making",
             """The system transforms raw enrollment data into strategic insights, enabling:<br/>
             • Proactive deployment of mobile enrollment vans to underserved areas<br/>
             • Early detection of quality issues through rejection rate monitoring<br/>
             • Optimized resource allocation based on predicted demand<br/>
             • Improved citizen welfare through reduced wait times and better accessibility<br/>
             • Enhanced administrative efficiency through data-driven planning"""),
        ]
        
        for heading, text in content:
            self.story.append(Spacer(1, 0.2*inch))
            self.story.append(Paragraph(heading, self.styles['CustomHeading2']))
            self.story.append(Paragraph(text, self.styles['CustomBody']))
        
        self.story.append(PageBreak())
    
    def add_architecture_overview(self):
        """Add system architecture section."""
        self.story.append(Paragraph("System Architecture", self.styles['CustomHeading1']))
        
        self.add_architecture_diagram()
        
        self.story.append(Paragraph("Data-to-Decision Pipeline", self.styles['CustomHeading2']))
        
        pipeline_text = """The Aadhaar Seva Optimizer implements a comprehensive data-to-decision 
        pipeline consisting of the following stages:"""
        self.story.append(Paragraph(pipeline_text, self.styles['CustomBody']))
        
        pipeline_stages = [
            ["Stage", "Component", "Purpose"],
            ["1. Ingestion", "Data Generator", "Load and clean UIDAI datasets; generate anonymized test data"],
            ["2. Analysis", "Pattern Detection", "Cluster districts into service profiles using K-Means"],
            ["3. Analysis", "Anomaly Detection", "Identify unusual patterns using Isolation Forest"],
            ["4. Prediction", "Forecasting", "Project saturation levels using Prophet"],
            ["5. Enrichment", "Geographic Verification", "Validate center locations via web scraping"],
            ["6. Visualization", "Dashboard Generator", "Create heatmaps and decision support visuals"],
            ["7. Action", "Recommendation Engine", "Generate deployment recommendations"],
        ]
        
        table = Table(pipeline_stages, colWidths=[0.7*inch, 1.8*inch, 3.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#283593')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f5f5f5')]),
        ]))
        
        self.story.append(Spacer(1, 0.1*inch))
        self.story.append(table)
        
        self.story.append(PageBreak())
        
    def add_analysis_results(self):
        """Add section for visual analysis results."""
        self.story.append(Paragraph("Data Analysis & Insights", self.styles['CustomHeading1']))
        
        self.story.append(Paragraph("Comprehensive Service Dashboard", self.styles['CustomHeading2']))
        self.story.append(Paragraph(
            "The following dashboard provides a holistic view of the Aadhaar ecosystem across the generated districts.",
            self.styles['CustomBody']
        ))
        self._add_image_if_exists('comprehensive_dashboard.png', "National Aadhaar Analytics Dashboard")
        
        self.story.append(Paragraph("Geographic Impact Analysis", self.styles['CustomHeading2']))
        self.story.append(Paragraph(
            "Spatial distribution of service profiles and saturation levels across states and district types.",
            self.styles['CustomBody']
        ))
        self._add_image_if_exists('geographic_distribution.png', "Geographic Distribution of Aadhaar Services")
        
        self.story.append(Paragraph("Service Stress Heatmap", self.styles['CustomHeading2']))
        self.story.append(Paragraph(
            "Identification of high-priority districts based on a synthesis of rejection rates, saturation levels, and center density.",
            self.styles['CustomBody']
        ))
        self._add_image_if_exists('service_stress_heatmap.png', "District-Level Service Stress Heatmap")
        
        self.story.append(PageBreak())
    
    def add_algorithm_kmeans(self):
        """Add K-Means clustering algorithm explanation."""
        self.story.append(Paragraph("Algorithm 1: K-Means Clustering", self.styles['CustomHeading1']))
        
        sections = [
            ("Purpose", 
             """K-Means clustering is used to group districts into distinct service profiles based on 
             enrollment patterns, saturation levels, and service characteristics. This unsupervised 
             learning approach reveals hidden structures in the data."""),
            
            ("How It Works",
             """K-Means partitions N districts into K clusters by:<br/>
             1. Randomly initializing K cluster centroids<br/>
             2. Assigning each district to the nearest centroid (using Euclidean distance)<br/>
             3. Recalculating centroids as the mean of all districts in each cluster<br/>
             4. Repeating steps 2-3 until convergence (centroids stop moving)"""),
            
            ("Features Used",
             """The algorithm analyzes multiple dimensions:<br/>
             • Average monthly enrollments and standard deviation<br/>
             • Ratio of biometric updates to new enrollments<br/>
             • Rejection rate statistics (mean and maximum)<br/>
             • Current saturation level<br/>
             • Geographic type (urban/rural classification)<br/>
             • Center availability and utilization<br/>
             • Gender distribution in enrollments"""),
            
            ("Example Clusters Identified",
             """• <b>Cluster 0:</b> Urban High-Saturation Update Hubs<br/>
             • <b>Cluster 1:</b> Rural Low-Saturation Enrollment Frontlines<br/>
             • <b>Cluster 2:</b> Mixed Moderate-Saturation Balanced Districts<br/>
             • <b>Cluster 3:</b> Urban Enrollment-Heavy Growth Districts<br/>
             • <b>Cluster 4:</b> Rural High-Quality Mature Districts"""),
            
            ("Why K-Means for Aadhaar",
             """K-Means excels at revealing patterns that simple sorting cannot find. For example, 
             a district might have high enrollments but also high rejections—this unique profile 
             would be grouped together, enabling targeted interventions. The algorithm's speed 
             and interpretability make it ideal for operational planning."""),
        ]
        
        for heading, text in sections:
            self.story.append(Paragraph(heading, self.styles['CustomHeading2']))
            self.story.append(Paragraph(text, self.styles['CustomBody']))
            self.story.append(Spacer(1, 0.1*inch))
        
        # Code sample
        self.story.append(Paragraph("Implementation", self.styles['CustomHeading2']))
        code = """from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Fit K-Means with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)"""
        
        self.story.append(Paragraph(code, self.styles['CustomCode']))
        
        # Add visual
        self.story.append(Paragraph("Cluster Visualization", self.styles['CustomHeading2']))
        self._add_image_if_exists('cluster_analysis.png', "PCA Visual of District Clustering Profiles")
        
        self.story.append(PageBreak())
    
    def add_algorithm_isolation_forest(self):
        """Add Isolation Forest algorithm explanation."""
        self.story.append(Paragraph("Algorithm 2: Isolation Forest", self.styles['CustomHeading1']))
        
        sections = [
            ("Purpose",
             """Isolation Forest detects anomalies—unusual enrollment patterns that may indicate quality 
             issues, fraudulent activity, or operational problems. Unlike methods that define "normal" 
             behavior, it explicitly seeks "rare and different" data points."""),
            
            ("How It Works",
             """The algorithm isolates anomalies using random decision trees:<br/>
             1. Randomly select a feature and a split value<br/>
             2. Partition the data recursively (building a tree)<br/>
             3. Anomalies are isolated faster (shorter path length in tree)<br/>
             4. Repeat with multiple trees (forest) and average the path lengths<br/>
             5. Points with short average paths are labeled as anomalies"""),
            
            ("Anomaly Categories Detected",
             """The system identifies several types of anomalies:<br/>
             • <b>Enrollment Spikes:</b> Sudden 200%+ increase in enrollments<br/>
             • <b>Enrollment Drops:</b> Unexpected 50%+ decrease<br/>
             • <b>High Rejection Rates:</b> Above 10% rejection rate<br/>
             • <b>Rejection Spikes:</b> Sudden doubling of rejections<br/>
             • <b>Saturation Issues:</b> Backward movement in saturation<br/>
             • <b>Unusual Update Patterns:</b> Deviation from historical norms"""),
            
            ("Severity Classification",
             """Each anomaly is assigned a severity level:<br/>
             • <b>Critical:</b> Multiple severe indicators (e.g., high rejection + spike)<br/>
             • <b>High:</b> Single severe indicator or multiple moderate ones<br/>
             • <b>Medium:</b> Moderate deviation from normal patterns<br/>
             • <b>Low:</b> Minor deviations requiring monitoring only"""),
            
            ("Why Isolation Forest for Aadhaar",
             """Isolation Forest excels at finding anomalies without needing labeled training data. 
             It can detect a district with a 400% rejection spike even if this has never occurred before. 
             The algorithm handles Aadhaar's "spiky" patterns (seasonal enrollment surges) better than 
             statistical methods that assume normal distributions."""),
        ]
        
        for heading, text in sections:
            self.story.append(Paragraph(heading, self.styles['CustomHeading2']))
            self.story.append(Paragraph(text, self.styles['CustomBody']))
            self.story.append(Spacer(1, 0.1*inch))
        
        # Code sample
        self.story.append(Paragraph("Implementation", self.styles['CustomHeading2']))
        code = """from sklearn.ensemble import IsolationForest

# Train Isolation Forest
iso_forest = IsolationForest(
    contamination=0.1,  # Expected 10% anomalies
    random_state=42,
    n_estimators=100
)

# Predict: -1 for anomalies, 1 for normal
predictions = iso_forest.fit_predict(features)
anomaly_scores = iso_forest.score_samples(features)"""
        
        self.story.append(Paragraph(code, self.styles['CustomCode']))
        
        # Add visual
        self.story.append(Paragraph("Anomaly Analysis", self.styles['CustomHeading2']))
        self._add_image_if_exists('anomaly_detection.png', "Distribution and Categorization of Detected Anomalies")
        
        self.story.append(PageBreak())
    
    def add_algorithm_prophet(self):
        """Add Prophet forecasting algorithm explanation."""
        self.story.append(Paragraph("Algorithm 3: Prophet (Forecasting)", self.styles['CustomHeading1']))
        
        sections = [
            ("Purpose",
             """Prophet forecasts when districts will reach Aadhaar saturation (99%+ adult coverage). 
             This enables proactive planning for resource reallocation and helps prioritize districts 
             that need accelerated enrollment drives."""),
            
            ("How It Works",
             """Prophet decomposes time series into components:<br/>
             1. <b>Trend:</b> Long-term growth pattern (logistic growth for saturation)<br/>
             2. <b>Seasonality:</b> Yearly patterns (e.g., school admission cycles)<br/>
             3. <b>Holidays/Events:</b> Special periods with enrollment spikes<br/>
             4. <b>Residual:</b> Unexplained variation<br/><br/>
             The model fits these components using a generalized additive model (GAM) and projects 
             them forward to generate forecasts with confidence intervals."""),
            
            ("Aadhaar-Specific Adaptations",
             """The implementation includes domain-specific customizations:<br/>
             • <b>Logistic Growth:</b> Saturation has an upper bound (99%), not unlimited growth<br/>
             • <b>Holiday Effects:</b> Enrollment spikes during April-May and January-February<br/>
             • <b>Multiplicative Seasonality:</b> Seasonal patterns scale with trend magnitude<br/>
             • <b>Changepoint Detection:</b> Identifies when policies or campaigns altered trends"""),
            
            ("Forecast Outputs",
             """For each district, the system provides:<br/>
             • Projected saturation level for next 12 months<br/>
             • Expected date of reaching 99% saturation<br/>
             • Confidence intervals (uncertainty bounds)<br/>
             • Trend decomposition showing seasonal vs. long-term growth"""),
            
            ("Why Prophet for Aadhaar",
             """Prophet handles Aadhaar data's challenges better than ARIMA:<br/>
             • <b>Missing Data:</b> Robust to gaps in enrollment records<br/>
             • <b>Irregular Events:</b> Explicitly models campaign-driven spikes<br/>
             • <b>Multiple Seasonality:</b> Captures both yearly and quarterly patterns<br/>
             • <b>Interpretability:</b> Decomposed components aid policy understanding"""),
        ]
        
        for heading, text in sections:
            self.story.append(Paragraph(heading, self.styles['CustomHeading2']))
            self.story.append(Paragraph(text, self.styles['CustomBody']))
            self.story.append(Spacer(1, 0.1*inch))
        
        # Code sample
        self.story.append(Paragraph("Implementation", self.styles['CustomHeading2']))
        code = """from prophet import Prophet

# Prepare Prophet dataframe (requires 'ds' and 'y' columns)
prophet_df = pd.DataFrame({
    'ds': dates,
    'y': saturation_levels,
    'cap': 0.99  # Upper bound
})

# Initialize and fit model
model = Prophet(growth='logistic', yearly_seasonality=True)
model.fit(prophet_df)

# Generate forecast
future = model.make_future_dataframe(periods=12, freq='M')
future['cap'] = 0.99
forecast = model.predict(future)"""
        
        self.story.append(Paragraph(code, self.styles['CustomCode']))
        
        # Add visual
        self.story.append(Paragraph("Forecast Analysis", self.styles['CustomHeading2']))
        self._add_image_if_exists('forecast_sample.png', "Saturation Forecast with Confidence Intervals")
        
        self.story.append(PageBreak())
    
    def add_algorithm_web_scraping(self):
        """Add web scraping methodology."""
        self.story.append(Paragraph("Methodology: Geographic Verification", self.styles['CustomHeading1']))
        
        sections = [
            ("Purpose",
             """Contextual web scraping verifies enrollment center locations using public geographic 
             records. This ensures that centers are correctly mapped, calculates center density per 
             district, and identifies service gaps."""),
            
            ("Data Sources",
             """The system uses publicly available APIs and databases:<br/>
             • <b>India Post Office API:</b> Validates pincode information<br/>
             • <b>Geographic Coordinates:</b> District and state mapping<br/>
             • <b>Population Census Data:</b> For density calculations<br/>
             • All data sources are public; no private information is accessed"""),
            
            ("Verification Process",
             """1. Extract unique pincodes from enrollment center database<br/>
             2. Query public API for each pincode (with respectful rate limiting)<br/>
             3. Validate district and state information<br/>
             4. Cross-reference with internal district master data<br/>
             5. Flag discrepancies for manual review"""),
            
            ("Center Density Analysis",
             """The system calculates key metrics:<br/>
             • Centers per lakh (100,000) population<br/>
             • Total daily enrollment capacity per district<br/>
             • Average utilization rate<br/>
             • Proximity analysis (distance between centers)<br/>
             • Geographic clustering patterns"""),
            
            ("Service Gap Identification",
             """Districts are flagged as underserved based on:<br/>
             • Less than 1 center per lakh population (benchmark)<br/>
             • High utilization (>90%) indicating capacity constraints<br/>
             • Large geographic area with sparse center distribution<br/>
             • Priority score combines shortage severity and population impact"""),
            
            ("Ethical Considerations",
             """• No personal information is collected or processed<br/>
             • Rate limiting respects API provider bandwidth<br/>
             • All data is anonymized and aggregated<br/>
             • Compliance with data protection regulations"""),
        ]
        
        for heading, text in sections:
            self.story.append(Paragraph(heading, self.styles['CustomHeading2']))
            self.story.append(Paragraph(text, self.styles['CustomBody']))
            self.story.append(Spacer(1, 0.1*inch))
        
        self.story.append(PageBreak())
    
    def add_workflow(self):
        """Add complete workflow description."""
        self.story.append(Paragraph("Complete Workflow", self.styles['CustomHeading1']))
        
        workflow_text = """The Aadhaar Seva Optimizer executes a seven-stage workflow that transforms 
        raw enrollment data into actionable recommendations:"""
        self.story.append(Paragraph(workflow_text, self.styles['CustomBody']))
        
        workflow_steps = [
            ("Step 1: Data Ingestion",
             """Load anonymized district enrollment data, time series records, and center information. 
             Data is cleaned, validated, and prepared for analysis. Missing values are handled, dates 
             are standardized, and categorical variables are encoded."""),
            
            ("Step 2: Pattern Detection",
             """Apply K-Means clustering to identify 5 district service profiles. Features are standardized, 
             optimal K is determined using elbow method, and interpretable cluster labels are generated 
             based on saturation level, enrollment activity, and geographic type."""),
            
            ("Step 3: Anomaly Detection",
             """Run Isolation Forest on time series data to detect unusual patterns. Period-over-period 
             changes are calculated, moving averages establish baselines, and deviations are scored. 
             Anomalies are categorized by type and assigned severity levels."""),
            
            ("Step 4: Predictive Forecasting",
             """Use Prophet to forecast saturation for sample districts. Models account for logistic growth, 
             seasonal patterns, and enrollment campaigns. Forecasts project 12 months ahead with confidence 
             intervals, identifying when districts will reach 99% saturation."""),
            
            ("Step 5: Geographic Verification",
             """Validate center locations via web scraping of public pincode databases. Calculate center 
             density per district, identify underserved areas, and compute priority scores for mobile 
             van deployment. Proximity analysis ensures optimal geographic distribution."""),
            
            ("Step 6: Visualization",
             """Generate comprehensive dashboards including:<br/>
             • Heatmaps showing service stress points across districts<br/>
             • Geographic distribution of clusters and saturation levels<br/>
             • Trend analysis of enrollments over time<br/>
             • Anomaly distribution and severity matrices<br/>
             • Cluster characteristic comparisons"""),
            
            ("Step 7: Decision Support",
             """Synthesize insights into actionable recommendations:<br/>
             • Deploy mobile vans to top N underserved districts<br/>
             • Investigate districts with critical anomalies<br/>
             • Launch awareness campaigns in low-saturation areas<br/>
             • Optimize center operations in high-utilization zones<br/>
             • Prioritize resources based on forecasted demand"""),
        ]
        
        for heading, text in workflow_steps:
            self.story.append(Spacer(1, 0.15*inch))
            self.story.append(Paragraph(heading, self.styles['CustomHeading2']))
            self.story.append(Paragraph(text, self.styles['CustomBody']))
        
        self.story.append(PageBreak())
    
    def add_impact_assessment(self):
        """Add impact assessment section."""
        self.story.append(Paragraph("Impact Assessment", self.styles['CustomHeading1']))
        
        self.story.append(Paragraph("Citizen Welfare Improvements", self.styles['CustomHeading2']))
        
        citizen_benefits = """The Aadhaar Seva Optimizer directly improves citizen welfare through:<br/><br/>
        <b>1. Reduced Wait Times:</b> Optimal center placement and capacity planning minimize queues 
        and appointment delays. Predictive models ensure adequate staffing during high-demand periods.<br/><br/>
        <b>2. Improved Accessibility:</b> Service gap analysis identifies underserved populations, 
        particularly in rural areas. Mobile van deployment brings services to remote communities.<br/><br/>
        <b>3. Higher Success Rates:</b> Anomaly detection catches quality issues early, reducing 
        enrollment rejections and repeat visits. Citizens complete processes in fewer attempts.<br/><br/>
        <b>4. Better Information:</b> Geographic verification ensures accurate center information 
        in public databases, helping citizens find nearest facilities."""
        
        self.story.append(Paragraph(citizen_benefits, self.styles['CustomBody']))
        
        self.story.append(Spacer(1, 0.2*inch))
        self.story.append(Paragraph("Administrative Efficiency Gains", self.styles['CustomHeading2']))
        
        admin_benefits = """For UIDAI and partnering agencies, the system delivers:<br/><br/>
        <b>1. Data-Driven Resource Allocation:</b> Replace intuition-based planning with quantitative 
        prioritization. Deploy mobile vans, staff, and equipment where impact is maximized.<br/><br/>
        <b>2. Proactive Problem Detection:</b> Identify quality issues, fraud patterns, and operational 
        bottlenecks before they become widespread. Faster intervention reduces systemic costs.<br/><br/>
        <b>3. Strategic Planning:</b> Saturation forecasts inform long-term capacity planning. 
        Agencies can phase down centers in mature districts and focus on growth areas.<br/><br/>
        <b>4. Performance Monitoring:</b> Cluster analysis enables peer comparisons. Districts can 
        benchmark against similar profiles to identify best practices.<br/><br/>
        <b>5. Evidence-Based Policy:</b> Quantitative insights support budget requests, policy changes, 
        and campaign design with concrete data rather than anecdotes."""
        
        self.story.append(Paragraph(admin_benefits, self.styles['CustomBody']))
        
        self.story.append(PageBreak())
    
    def add_technical_requirements(self):
        """Add technical requirements section."""
        self.story.append(Paragraph("Technical Requirements", self.styles['CustomHeading1']))
        
        self.story.append(Paragraph("Software Dependencies", self.styles['CustomHeading2']))
        
        deps_data = [
            ["Library", "Version", "Purpose"],
            ["pandas", "≥1.5.0", "Data manipulation and analysis"],
            ["numpy", "≥1.23.0", "Numerical computing"],
            ["scikit-learn", "≥1.2.0", "Machine learning algorithms"],
            ["prophet", "≥1.1.0", "Time series forecasting"],
            ["seaborn", "≥0.12.0", "Statistical visualization"],
            ["matplotlib", "≥3.6.0", "Plotting and charts"],
            ["requests", "≥2.28.0", "HTTP requests for web scraping"],
            ["beautifulsoup4", "≥4.11.0", "HTML parsing"],
            ["reportlab", "≥3.6.0", "PDF generation"],
        ]
        
        deps_table = Table(deps_data, colWidths=[1.8*inch, 1.2*inch, 3*inch])
        deps_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#283593')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f5f5f5')]),
        ]))
        
        self.story.append(deps_table)
        
        self.story.append(Spacer(1, 0.2*inch))
        self.story.append(Paragraph("Data Requirements", self.styles['CustomHeading2']))
        
        data_req = """The system requires three primary datasets:<br/><br/>
        <b>1. District Master Data:</b> District ID, name, state, population, geographic type<br/>
        <b>2. Time Series Enrollment Data:</b> Monthly enrollments, updates, rejections by district<br/>
        <b>3. Center Information:</b> Center ID, location (lat/long), pincode, capacity, utilization<br/><br/>
        All data must be anonymized and aggregated at district level. No personally identifiable 
        information (PII) should be included. The system generates synthetic test data for demonstration."""
        
        self.story.append(Paragraph(data_req, self.styles['CustomBody']))
        
        self.story.append(Spacer(1, 0.2*inch))
        self.story.append(Paragraph("Computational Resources", self.styles['CustomHeading2']))
        
        compute_req = """<b>Minimum Requirements:</b><br/>
        • CPU: 4 cores, 2.0 GHz or equivalent<br/>
        • RAM: 8 GB<br/>
        • Storage: 2 GB available space<br/>
        • Python: 3.8 or higher<br/><br/>
        <b>Recommended for Production:</b><br/>
        • CPU: 8+ cores, 2.5 GHz<br/>
        • RAM: 16 GB<br/>
        • Storage: 10 GB (for historical data retention)<br/>
        • Python: 3.10 or higher"""
        
        self.story.append(Paragraph(compute_req, self.styles['CustomBody']))
        
        self.story.append(PageBreak())
    
    def add_usage_instructions(self):
        """Add usage instructions."""
        self.story.append(Paragraph("Usage Instructions", self.styles['CustomHeading1']))
        
        self.story.append(Paragraph("Installation", self.styles['CustomHeading2']))
        
        install_steps = """1. Clone or download the Aadhaar Seva Optimizer codebase<br/>
        2. Install Python 3.8 or higher if not already installed<br/>
        3. Install dependencies: <font name="Courier">pip install -r requirements.txt</font><br/>
        4. Verify installation: <font name="Courier">python -c "import sklearn, prophet"</font>"""
        
        self.story.append(Paragraph(install_steps, self.styles['CustomBody']))
        
        self.story.append(Spacer(1, 0.15*inch))
        self.story.append(Paragraph("Running the System", self.styles['CustomHeading2']))
        
        usage_code = """# Navigate to project directory
cd aadhaar_seva_optimizer

# Run complete pipeline with generated test data
python main.py

# Output will be saved to the 'output/' directory"""
        
        self.story.append(Paragraph(usage_code, self.styles['CustomCode']))
        
        usage_text = """The system will automatically:<br/>
        • Generate anonymized test datasets<br/>
        • Execute all seven pipeline stages<br/>
        • Create visualizations and reports<br/>
        • Save results to the output directory<br/><br/>
        Execution typically takes 2-5 minutes depending on system specifications."""
        
        self.story.append(Paragraph(usage_text, self.styles['CustomBody']))
        
        self.story.append(Spacer(1, 0.15*inch))
        self.story.append(Paragraph("Output Files", self.styles['CustomHeading2']))
        
        outputs = [
            ["File", "Description"],
            ["cluster_profiles.csv", "District cluster characteristics"],
            ["cluster_analysis.png", "Visual representation of clusters"],
            ["top_anomalies.csv", "Most severe anomalies detected"],
            ["anomaly_detection.png", "Anomaly analysis visualizations"],
            ["saturation_forecast.csv", "Saturation projection for sample districts"],
            ["forecast_sample.png", "Example forecast visualization"],
            ["center_density.csv", "Center availability by district"],
            ["service_gaps.csv", "Underserved districts prioritized"],
            ["service_stress_heatmap.png", "District stress point heatmap"],
            ["comprehensive_dashboard.png", "Multi-metric analytics dashboard"],
            ["recommendations.json", "Actionable recommendations"],
        ]
        
        outputs_table = Table(outputs, colWidths=[2.5*inch, 3.5*inch])
        outputs_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#283593')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f5f5f5')]),
        ]))
        
        self.story.append(outputs_table)
        
        self.story.append(PageBreak())
    
    def add_conclusion(self):
        """Add conclusion."""
        self.story.append(Paragraph("Conclusion", self.styles['CustomHeading1']))
        
        conclusion_text = """The Aadhaar Seva Optimizer represents a significant advancement in data-driven 
        governance for India's digital identity infrastructure. By combining three powerful machine learning 
        algorithms—K-Means Clustering, Isolation Forest, and Prophet—with geographic verification and 
        comprehensive visualization, the system transforms raw enrollment data into strategic insights.<br/><br/>
        
        The impact extends beyond mere efficiency gains. By identifying underserved populations, detecting 
        quality issues early, and optimizing resource deployment, the system directly improves citizen welfare. 
        Rural communities gain better access to services, wait times are reduced, and enrollment success 
        rates improve.<br/><br/>
        
        For administrators, the system enables evidence-based decision-making at scale. Rather than relying 
        on intuition or outdated reports, policymakers can act on real-time analytics with quantified 
        priorities and forecasted outcomes. This data-to-decision pipeline exemplifies how machine learning 
        can augment human expertise in complex social systems.<br/><br/>
        
        As Aadhaar continues to evolve as a foundational digital infrastructure, systems like the Seva 
        Optimizer will play an increasingly critical role in ensuring equitable, efficient, and citizen-centric 
        service delivery across India's diverse landscape."""
        
        self.story.append(Paragraph(conclusion_text, self.styles['CustomBody']))
        
        self.story.append(Spacer(1, 0.3*inch))
        
        closing = Paragraph(
            "Code Repository: <a href='https://github.com/A8125/UIDAI-Hackathon-Idea.git' color='blue'>https://github.com/A8125/UIDAI-Hackathon-Idea.git</a><br/><br/>"
            "For technical support or questions, please contact the UIDAI Analytics Team.",
            ParagraphStyle(name='closing', parent=self.styles['CustomBody'],
                          alignment=TA_CENTER, fontSize=10, textColor=HexColor('#666666'))
        )
        self.story.append(closing)
    
    def generate_pdf(self):
        """Generate the complete PDF document."""
        print("\nGenerating comprehensive PDF documentation...")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Build all sections
        self.add_cover_page()
        self.add_executive_summary()
        self.add_architecture_overview()
        self.add_algorithm_kmeans()
        self.add_algorithm_isolation_forest()
        self.add_algorithm_prophet()
        self.add_algorithm_web_scraping()
        self.add_analysis_results()
        self.add_workflow()
        self.add_impact_assessment()
        self.add_technical_requirements()
        self.add_usage_instructions()
        self.add_conclusion()
        
        # Build PDF
        self.doc.build(self.story)
        
        print(f"✓ PDF documentation generated: {self.output_path}")
        print(f"  Pages: ~25-30 pages")
        print(f"  Size: {os.path.getsize(self.output_path) / 1024:.1f} KB")
        
        return self.output_path


if __name__ == "__main__":
    generator = DocumentationGenerator()
    generator.generate_pdf()
