"""
AI Business Case Development

This module provides methods and frameworks for developing compelling
business cases for AI projects.
"""

class BusinessCaseComponent:
    """Represents a component of an AI business case"""
    
    def __init__(self, name, description, key_elements):
        self.name = name
        self.description = description
        self.key_elements = key_elements
    
    def __str__(self):
        return f"{self.name}: {self.description}"

class AIBusinessCaseFramework:
    """Framework for developing AI business cases"""
    
    def __init__(self):
        self.components = self._initialize_components()
    
    def _initialize_components(self):
        """Initialize business case components"""
        return [
            BusinessCaseComponent(
                "Problem Statement",
                "Clear articulation of the business challenge",
                [
                    "Current state analysis",
                    "Impact quantification",
                    "Root cause identification",
                    "Stakeholder pain points"
                ]
            ),
            BusinessCaseComponent(
                "Solution Overview",
                "Description of the proposed AI solution",
                [
                    "AI approach and methodology",
                    "Technology requirements",
                    "Integration considerations",
                    "Innovation aspects"
                ]
            ),
            BusinessCaseComponent(
                "Market Analysis",
                "Industry context and competitive positioning",
                [
                    "Market size and growth trends",
                    "Competitive landscape",
                    "Best practices and benchmarks",
                    "Technology adoption trends"
                ]
            ),
            BusinessCaseComponent(
                "Financial Analysis",
                "Cost-benefit analysis and ROI projections",
                [
                    "Investment requirements",
                    "Benefit quantification",
                    "ROI and payback calculations",
                    "Sensitivity analysis"
                ]
            ),
            BusinessCaseComponent(
                "Implementation Plan",
                "Approach and timeline for execution",
                [
                    "Project phases and milestones",
                    "Resource requirements",
                    "Risk assessment",
                    "Success criteria"
                ]
            ),
            BusinessCaseComponent(
                "Strategic Alignment",
                "Connection to organizational objectives",
                [
                    "Strategic initiative linkage",
                    "Value creation mechanisms",
                    "Competitive advantage",
                    "Long-term implications"
                ]
            )
        ]
    
    def get_component(self, component_name):
        """Get a specific component by name"""
        for component in self.components:
            if component.name.lower() == component_name.lower():
                return component
        return None
    
    def list_all_components(self):
        """List all business case components"""
        return [component.name for component in self.components]
    
    def calculate_roi(self, investment, annual_benefits, project_lifespan):
        """
        Calculate ROI for AI project
        
        Args:
            investment (float): Total project investment
            annual_benefits (float): Annual benefits generated
            project_lifespan (int): Project lifespan in years
            
        Returns:
            dict: ROI metrics
        """
        total_benefits = annual_benefits * project_lifespan
        net_benefits = total_benefits - investment
        roi = (net_benefits / investment) * 100 if investment > 0 else 0
        payback_period = investment / annual_benefits if annual_benefits > 0 else float('inf')
        
        return {
            "total_investment": investment,
            "total_benefits": total_benefits,
            "net_benefits": net_benefits,
            "roi_percentage": round(roi, 2),
            "payback_period": round(payback_period, 2)
        }
    
    def benefit_categories(self):
        """Categorize potential AI project benefits"""
        return {
            "cost_reduction": [
                "Labor cost savings",
                "Operational efficiency gains",
                "Reduced error correction costs",
                "Lower processing times"
            ],
            "revenue_enhancement": [
                "Increased sales conversion",
                "Higher customer lifetime value",
                "New product/service opportunities",
                "Premium pricing potential"
            ],
            "risk_mitigation": [
                "Fraud prevention savings",
                "Compliance cost reduction",
                "Operational risk reduction",
                "Reputational risk management"
            ],
            "quality_improvement": [
                "Enhanced customer satisfaction",
                "Improved decision accuracy",
                "Better regulatory compliance",
                "Increased employee productivity"
            ]
        }
    
    def create_business_case_template(self, project_name):
        """Create a business case template"""
        return {
            "project_name": project_name,
            "executive_summary": "",
            "problem_statement": "",
            "solution_overview": "",
            "market_analysis": "",
            "financial_analysis": {
                "investment": 0,
                "annual_benefits": 0,
                "lifespan_years": 0,
                "roi_metrics": {}
            },
            "implementation_plan": {
                "timeline": "",
                "resources": [],
                "risks": []
            },
            "strategic_alignment": ""
        }
    
    def sensitivity_analysis(self, base_investment, base_benefits, variance_range=0.2):
        """
        Perform sensitivity analysis on ROI calculations
        
        Args:
            base_investment (float): Base investment amount
            base_benefits (float): Base annual benefits
            variance_range (float): Variance range as decimal (0.2 = 20%)
            
        Returns:
            dict: Sensitivity analysis results
        """
        scenarios = {
            "pessimistic": {
                "investment": base_investment * (1 + variance_range),
                "benefits": base_benefits * (1 - variance_range)
            },
            "base_case": {
                "investment": base_investment,
                "benefits": base_benefits
            },
            "optimistic": {
                "investment": base_investment * (1 - variance_range),
                "benefits": base_benefits * (1 + variance_range)
            }
        }
        
        results = {}
        for scenario, values in scenarios.items():
            roi_metrics = self.calculate_roi(
                values["investment"], 
                values["benefits"], 
                3  # Assuming 3-year lifespan
            )
            results[scenario] = roi_metrics
        
        return results

# Example usage and demonstration
if __name__ == "__main__":
    # Create the business case framework
    framework = AIBusinessCaseFramework()
    
    print("AI Business Case Development Framework:")
    print("=" * 45)
    
    # Display all components
    print("Business Case Components:")
    for i, component_name in enumerate(framework.list_all_components(), 1):
        component = framework.get_component(component_name)
        print(f"{i}. {component.name}")
        print(f"   Description: {component.description}")
        print("   Key Elements:")
        for element in component.key_elements:
            print(f"     - {element}")
        print()
    
    # Demonstrate ROI calculation
    print("ROI Calculation Example:")
    investment = 500000
    annual_benefits = 200000
    lifespan = 3
    
    roi_metrics = framework.calculate_roi(investment, annual_benefits, lifespan)
    print(f"  Investment: ${investment:,}")
    print(f"  Annual Benefits: ${annual_benefits:,}")
    print(f"  Lifespan: {lifespan} years")
    print(f"  ROI: {roi_metrics['roi_percentage']}%")
    print(f"  Payback Period: {roi_metrics['payback_period']} years")
    
    # Show benefit categories
    print("\nBenefit Categories:")
    categories = framework.benefit_categories()
    for category, benefits in categories.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for benefit in benefits:
            print(f"  - {benefit}")
    
    # Demonstrate sensitivity analysis
    print("\nSensitivity Analysis:")
    sensitivity = framework.sensitivity_analysis(investment, annual_benefits)
    for scenario, metrics in sensitivity.items():
        print(f"\n{scenario.replace('_', ' ').title()} Scenario:")
        print(f"  ROI: {metrics['roi_percentage']}%")
        print(f"  Payback: {metrics['payback_period']} years")