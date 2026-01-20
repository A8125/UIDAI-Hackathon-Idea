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
            [
                Paragraph("<b>IDENTIFYING NEEDS</b>", self.styles['CustomHeading3']), "", ""],
            ["", "⬇", ""],
            [Paragraph("<b>SECURE DATA REVIEWS</b><br/><font size=8>(District Statistics / Broad Records)</font>", self.styles['CustomBody']), "", ""],
            ["", "⬇", ""],
            [
                Paragraph("<b>MANAGEMENT TOOLS</b>", self.styles['CustomBody']),
                Paragraph("<b>FORECAST TOOLS</b>", self.styles['CustomBody']),
                Paragraph("<b>LOCAL FINDINGS</b>", self.styles['CustomBody'])
            ],
            [
                Paragraph("<font size=8>• Grouping Data<br/>• Finding Differences</font>", self.styles['CustomBody']),
                Paragraph("<font size=8>• Future Estimates<br/>• Growth Levels</font>", self.styles['CustomBody']),
                Paragraph("<font size=8>• Location Checks<br/>• Local Area Mapping</font>", self.styles['CustomBody'])
            ],
            ["", "⬇", ""],
            [Paragraph("<b>ADMINISTRATIVE SUPPORT</b><br/><font size=8>(Visual Board / Planning Tips)</font>", self.styles['CustomBody']), "", ""],
            ["", "⬇", ""],
            [Paragraph("<b>HELPFUL NEXT STEPS</b>", self.styles['CustomHeading3']), "", ""]
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
            "Helping Reach Every Citizen for Better Service Delivery",
            ParagraphStyle(name='subtitle', parent=self.styles['CustomBody'],
                          fontSize=14, alignment=TA_CENTER, textColor=HexColor('#424242'))
        )
        self.story.append(subtitle)
        self.story.append(Spacer(1, 1*inch))
        
        # System description
        desc = Paragraph(
            """This document provides a full explanation of how we help improve center management 
            across India. The project uses smart planning methods to help make services 
            available to more people in more places.""",
            ParagraphStyle(name='desc', parent=self.styles['CustomBody'],
                          alignment=TA_CENTER, fontSize=12)
        )
        self.story.append(desc)
        self.story.append(Spacer(1, 1*inch))
        
        # Metadata
        metadata = [
            ["Document Version:", "1.0"],
            ["Date:", datetime.now().strftime("%B %d, %Y")],
            ["Organization:", "Citizen Outreach Team"],
            ["Classification:", "Public - Community Help"],
            ["Project Focus:", "Universal Reach and Support"],
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
             """The Aadhaar Seva Assistant is a project designed to 
             help make identity card services better in every district. 
             It finds where people need help and helps plan for more 
             centers to reach everyone."""
),
            
            ("Key Capabilities",
             """• <b>Finding Patterns:</b> Groups similar areas together using smart logic<br/>
             • <b>Watching for Changes:</b> Finds areas where enrollment needs are shifting<br/>
             • <b>Predicting Growth:</b> Sees when an area will reach its goals<br/>
             • <b>Location Support:</b> Checks local area maps to find gaps in services<br/>
             • <b>Planning Support:</b> Offers tips for where to send extra help"""
),
            
            ("Impact on Decision-Making",
             """The project helps leaders make better choices for citizens:<br/>
             • Sending mobile help vans to areas that need them most<br/>
             • Watching center work to make sure it is high quality<br/>
             • Preparing for future needs based on growth trends<br/>
             • Making it easier for people to get services nearby<br/>
             • Using clear data to help plan for every district"""
),
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
            ["Stage", "Area", "Goal"],
            ["1. Gathering", "Broad Stats", "Load and review district records"],
            ["2. Review", "Area Profiles", "Group districts with similar needs"],
            ["3. Review", "Change Checks", "Find unexpected shifts in needs"],
            ["4. Future", "Growth Plans", "Estimate when goals will be met"],
            ["5. Local", "Map Support", "Check locations for better reach"],
            ["6. Visuals", "Review Board", "Create clear maps and charts"],
            ["7. Support", "Next Steps", "Suggest best ways to help people"],
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
    
    def add_frontend_prototype(self):
        """Add interactive prototype section."""
        self.story.append(Paragraph("Interactive Frontend Prototype", self.styles['CustomHeading1']))
        
        self.story.append(Paragraph(
            "The system includes a premium interactive dashboard for visualizing ML insights. "
            "This prototype allows administrators to explore clustering results, track anomalies, "
            "and view saturation forecasts in a modern, single-page application.",
            self.styles['CustomBody']
        ))
        
        features = [
            ["Feature", "Description"],
            ["Overview Dashboard", "High-level KPIs and service stress heatmap"],
            ["District Clusters", "Interactive grouping of districts by service profile"],
            ["Anomaly Tracker", "Real-time list of detected enrollment and rejection spikes"],
            ["Forecast Explorer", "Visualization of projected saturation timelines"],
            ["Deployment Tips", "Evidence-based recommendations for mobile van deployment"],
        ]
        
        table = Table(features, colWidths=[2*inch, 4*inch])
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
        
        self.story.append(table)
        self.story.append(Spacer(1, 0.2*inch))
        
        self.story.append(Paragraph("System Requirements & Setup", self.styles['CustomHeading2']))
        setup_text = """The dashboard is built using standard web technologies (HTML5, CSS3, JavaScript) 
        and requires no external dependencies or Node.js environment. It can be served using 
        Python's built-in HTTP server:"""
        self.story.append(Paragraph(setup_text, self.styles['CustomBody']))
        
        self.story.append(Paragraph(
            "# Navigate to frontend directory\ncd frontend\n\n# Start server\npython3 -m http.server 8000",
            self.styles['CustomCode']
        ))
        
        self.story.append(PageBreak())
    
    def add_method_area_grouping(self):
        """Add area grouping explanation."""
        self.story.append(Paragraph("Method 1: Grouping Similar Areas", self.styles['CustomHeading1']))
        
        sections = [
            ("Purpose", 
             """Grouping data is used to find districts with similar needs for identity services. 
             This helps leaders understand which areas need more help and what kind of 
             support they need most."""),
            
            ("How It Works",
             """The grouping method sorts districts into sets by:<br/>
             1. Looking at the main needs of each area<br/>
             2. Finding districts that have close to the same numbers<br/>
             3. Creating a few main groups that are easy to manage<br/>
             4. Giving each group a clear name based on what they need"""),
            
            ("Info Used",
             """The method looks at several things:<br/>
             • Average monthly card sign-ups<br/>
             • How often people ask to fix their records<br/>
             • Number of people waiting for help<br/>
             • How many people already have their cards<br/>
             • Whether the area is a city or a village<br/>
             • How many centers are available nearby"""),
            
            ("Example Groups Found",
             """• <b>Group 1:</b> Busy City Centers with many Record Fixes<br/>
             • <b>Group 2:</b> Rural Areas with many New Sign-ups<br/>
             • <b>Group 3:</b> Balanced Districts with moderate needs<br/>
             • <b>Group 4:</b> Growing Areas with high future needs<br/>
             • <b>Group 5:</b> High-Quality Mature Districts"""),
            
            ("Why We Use Grouping",
             """Grouping helps us see patterns that we might miss. For example, some districts 
             might have many people but very few help centers. We can group these together 
             to make sure we send help vans to all of them at once."""),
        ]
        
        for heading, text in sections:
            self.story.append(Paragraph(heading, self.styles['CustomHeading2']))
            self.story.append(Paragraph(text, self.styles['CustomBody']))
            self.story.append(Spacer(1, 0.1*inch))
        
        self.story.append(PageBreak())
    
    def add_method_change_checking(self):
        """Add change checking explanation."""
        self.story.append(Paragraph("Method 2: Watching for Sudden Changes", self.styles['CustomHeading1']))
        
        sections = [
            ("Purpose",
             """Watching for changes helps find unusual patterns in sign-ups or fixes. 
             This helps us see if a center is having trouble or if many more people 
             suddenly need help in one area."""),
            
            ("How It Works",
             """The tool finds unusual days or weeks by:<br/>
             1. Looking at what is normal for each area<br/>
             2. Checking if any day is far outside the normal range<br/>
             3. Flagging those areas so we can send more support staff<br/>
             4. Helping us find and fix problems early before they grow"""),
            
            ("Changes We Watch For",
             """We look for several types of events:<br/>
             • <b>Unexpected Spikes:</b> Sudden large increases in people visiting<br/>
             • <b>Unusual Drops:</b> Sudden decreases that might mean a center is closed<br/>
             • <b>High Fix Needs:</b> When many people need to correct their records at once<br/>
             • <b>Unusual Service Gaps:</b> Areas where service has slowed down unexpectedly"""),
            
            ("Priority Levels",
             """Each event is given a priority:<br/>
             • <b>High Priority:</b> Many issues happening at once in one spot<br/>
             • <b>Medium Priority:</b> A clear shift that needs more help soon<br/>
             • <b>Low Priority:</b> Small changes that we should just keep an eye on"""),
            
            ("Why We Use This",
             """This tool helps find problems even if they have never happened before. 
             It ensures that citizens always get good service and that centers are 
             ready for any sudden changes in the number of visitors."""),
        ]
        
        for heading, text in sections:
            self.story.append(Paragraph(heading, self.styles['CustomHeading2']))
            self.story.append(Paragraph(text, self.styles['CustomBody']))
            self.story.append(Spacer(1, 0.1*inch))
        
        self.story.append(PageBreak())
    
    def add_method_growth_forecasting(self):
        """Add growth forecasting explanation."""
        self.story.append(Paragraph("Method 3: Growth Planning", self.styles['CustomHeading1']))
        
        sections = [
            ("Purpose",
             """Growth planning helps see when districts will reach their goals for card 
             coverage. This helps us know when to move resources to other areas that 
             still need more help."""),
            
            ("How It Works",
             """The model looks at several pieces for planning:<br/>
             1. <b>Trends:</b> Long-term growth goals for each area<br/>
             2. <b>Seasons:</b> Yearly times when many more people need help (like school start times)<br/>
             3. <b>Events:</b> Special camps or campaigns that bring in more people<br/><br/>
             By looking at these, the tool gives a clear view of how much work is left 
             to do in each district."""),
            
            ("Planning Support",
             """For each district, the tool provides:<br/>
             • Estimated card coverage for the next year<br/>
             • Expected month when goals will be met<br/>
             • Range of possible outcomes to help plan for different needs<br/>
             • A clear map of growth over time"""),
            
            ("Why We Use This",
             """This tool helps center managers plan ahead for the next 12 months. 
             It ensures that we don't have too many centers in areas that are already 
             well-covered, so we can focus on areas that are still growing."""),
        ]
        
        for heading, text in sections:
            self.story.append(Paragraph(heading, self.styles['CustomHeading2']))
            self.story.append(Paragraph(text, self.styles['CustomBody']))
            self.story.append(Spacer(1, 0.1*inch))
        
        self.story.append(PageBreak())
    
    def add_method_map_support(self):
        """Add map support methodology."""
        self.story.append(Paragraph("Method 4: Map Support", self.styles['CustomHeading1']))
        
        sections = [
            ("Purpose",
             """Map support checks center locations using public records. This ensures that 
             all centers are in the right places and helps us find areas that do not 
             have any centers nearby."""),
            
            ("Info Sources",
             """The project looks at public records and maps:<br/>
             • <b>Public Post Office Lists:</b> Checks pincode information<br/>
             • <b>Public Location Records:</b> Checks district and state names<br/>
             • <b>Broad Population Records:</b> For density checks<br/>
             • All info is from public sources and is kept anonymous"""),
            
            ("Check Process",
             """1. Review pincodes from center lists<br/>
             2. Check public records for each area<br/>
             3. Validate district and state names<br/>
             4. Flag any differences for review<br/>
             5. Update maps with correct center info"""),
            
            ("Service Reach Check",
             """The tool looks at several things:<br/>
             • Centers per lakh (100,000) population<br/>
             • How many people a center can help each day<br/>
             • How busy each center is on average<br/>
             • Distance between centers in one district"""),
            
            ("Finding Gaps",
             """Areas are flagged for more help if:<br/>
             • There are too few centers for the number of people<br/>
             • Centers are too busy to help everyone quickly<br/>
             • Large areas have no centers nearby"""),
            
            ("Safety and Privacy",
             """• No personal info is used at any time<br/>
             • All data is grouped by district for privacy<br/>
             • We follow all rules for data use and citizen safety"""),
        ]
        
        for heading, text in sections:
            self.story.append(Paragraph(heading, self.styles['CustomHeading2']))
            self.story.append(Paragraph(text, self.styles['CustomBody']))
            self.story.append(Spacer(1, 0.1*inch))
        
        self.story.append(PageBreak())
    
    def add_workflow(self):
        """Add complete workflow description."""
        self.story.append(Paragraph("Complete Workflow", self.styles['CustomHeading1']))
        
        workflow_text = """The project follows a simple seven-step process to help citizens 
        and managers reach their goals:"""
        self.story.append(Paragraph(workflow_text, self.styles['CustomBody']))
        
        workflow_steps = [
            ("Step 1: Gathering Info",
             """We look at broad district records and sign-up numbers. All data is 
             cleaned and prepared to find the best ways to help."""),
            
            ("Step 2: Grouping Areas",
             """We group districts with similar needs. This helps us see which areas 
             need city-style support and which need rural-style help."""),
            
            ("Step 3: Checking for Changes",
             """We watch for any sudden shifts in sign-ups or record fixes. This 
             helps us catch small problems before they grow into large ones."""),
            
            ("Step 4: Looking Ahead",
             """We estimate when each area will meet its goals for card reach. This 
             helps us plan where to move staff in the coming months."""),
            
            ("Step 5: Local Support",
             """We check center locations against local maps. This helps us find gaps 
             and plan where to send mobile help vans."""),
            
            ("Step 6: Clear Visuals",
             """We create clear charts and maps for managers. These visual boards 
             make it easy to see where help is needed most at a glance."""),
            
            ("Step 7: Planning Tips",
             """We offer clear tips for localized support. Leaders can use these 
             to send help where it will make the most difference for citizens."""),
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
        
        admin_benefits = """For leaders and managers, the project offers:<br/><br/>
        <b>1. Better Use of Resources:</b> Place help and staff where they have the 
        most impact based on real needs.<br/><br/>
        <b>2. Early Problem Finding:</b> Find bottlenecks or service slowdowns early. 
        Taking action quickly saves time and helps more people.<br/><br/>
        <b>3. Planning for Growth:</b> See which areas are almost meeting their goals 
        and which need more support to grow.<br/><br/>
        <b>4. Balanced Support:</b> Compare similar districts to see which ones are 
        doing well and why, so others can learn from them.<br/><br/>
        <b>5. Fact-Based Planning:</b> Use clear records and growth trends to help 
        plan for budgets and future needs."""
        
        self.story.append(Paragraph(admin_benefits, self.styles['CustomBody']))
        
        self.story.append(PageBreak())
    
    def add_project_summary(self):
        """Add a simple project summary."""
        self.story.append(Paragraph("Project Summary", self.styles['CustomHeading1']))
        
        summary_text = """The Aadhaar Seva Assistant is a helpful project for identity 
        services in India. By using simple grouping, change checking, and growth 
        estimates, we can help people get their cards faster and fix their records 
        more easily.<br/><br/>
        
        This project helps both citizens and managers. Citizens get better reach and 
        shorter wait times. Managers get better planning and support tools to reach 
        every corner of their district. By working together with clear data, we can 
        make sure no one is left behind.<br/><br/>
        
        As the project grows, we hope it will continue to help reach every citizen 
        with the services they need to succeed. Our goal is a helpful, efficient, 
        and citizen-first approach to service delivery."""
        
        self.story.append(Paragraph(summary_text, self.styles['CustomBody']))
        
        self.story.append(Spacer(1, 0.3*inch))
        
        closing = Paragraph(
            "For technical support or questions, please contact the Outreach Team.",
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
        self.add_method_area_grouping()
        self.add_method_change_checking()
        self.add_method_growth_forecasting()
        self.add_method_map_support()
        self.add_analysis_results()
        self.add_frontend_prototype()
        self.add_workflow()
        self.add_impact_assessment()
        self.add_project_summary()
        
        # Build PDF
        self.doc.build(self.story)
        
        print(f"✓ PDF documentation generated: {self.output_path}")
        print(f"  Pages: ~25-30 pages")
        print(f"  Size: {os.path.getsize(self.output_path) / 1024:.1f} KB")
        
        return self.output_path


if __name__ == "__main__":
    generator = DocumentationGenerator()
    generator.generate_pdf()
