"""
AI Implementation ROI Evaluation

This module provides frameworks and methods for evaluating the return on investment
of AI implementations in business environments.
"""

class ROIComponent:
    """Represents a component of AI implementation ROI"""
    
    def __init__(self, name, description, measurement_approach, typical_timeframe):
        self.name = name
        self.description = description
        self.measurement_approach = measurement_approach
        self.typical_timeframe = typical_timeframe  # In months
    
    def __str__(self):
        return f"{self.name}: {self.description}"

class AIROIFramework:
    """Framework for evaluating AI implementation return on investment"""
    
    def __init__(self):
        self.components = self._initialize_components()
        self.benefit_categories = self._initialize_benefit_categories()
    
    def _initialize_components(self):
        """Initialize ROI components"""
        return [
            ROIComponent(
                "Cost Reduction",
                "Direct savings from automation and efficiency improvements",
                "Measure reduction in labor costs, operational expenses, and processing time",
                6
            ),
            ROIComponent(
                "Revenue Enhancement",
                "Increased revenue through improved customer experience and new capabilities",
                "Track sales growth, customer acquisition, and upsell rates",
                12
            ),
            ROIComponent(
                "Quality Improvement",
                "Enhanced product or service quality leading to customer satisfaction",
                "Monitor defect rates, customer satisfaction scores, and complaint volumes",
                6
            ),
            ROIComponent(
                "Risk Mitigation",
                "Reduced business risks through better decision-making and compliance",
                "Quantify reduction in fraud losses, regulatory fines, and operational risks",
                12
            ),
            ROIComponent(
                "Speed and Agility",
                "Faster decision-making and response times improving business agility",
                "Measure process cycle times, time-to-market, and response rates",
                3
            )
        ]
    
    def _initialize_benefit_categories(self):
        """Initialize benefit categories"""
        return {
            "quantitative": [
                "Labor cost savings",
                "Increased productivity",
                "Reduced error rates",
                "Faster processing times",
                "Higher conversion rates",
                "Improved accuracy",
                "Reduced operational costs"
            ],
            "qualitative": [
                "Enhanced customer experience",
                "Improved decision-making",
                "Increased employee satisfaction",
                "Better regulatory compliance",
                "Enhanced brand reputation",
                "Competitive advantage",
                "Innovation enablement"
            ]
        }
    
    def calculate_simple_roi(self, investment, returns):
        """
        Calculate simple ROI
        
        Args:
            investment (float): Total investment amount
            returns (float): Total returns generated
            
        Returns:
            float: ROI as percentage
        """
        if investment == 0:
            return 0
        return ((returns - investment) / investment) * 100
    
    def calculate_npv(self, cash_flows, discount_rate):
        """
        Calculate Net Present Value
        
        Args:
            cash_flows (list): List of cash flows for each period
            discount_rate (float): Discount rate as decimal (e.g., 0.1 for 10%)
            
        Returns:
            float: NPV value
        """
        npv = 0
        for i, cash_flow in enumerate(cash_flows):
            npv += cash_flow / ((1 + discount_rate) ** (i + 1))
        return npv
    
    def calculate_payback_period(self, initial_investment, annual_savings):
        """
        Calculate payback period
        
        Args:
            initial_investment (float): Initial investment amount
            annual_savings (float): Annual savings or returns
            
        Returns:
            float: Payback period in years
        """
        if annual_savings <= 0:
            return float('inf')
        return initial_investment / annual_savings
    
    def estimate_ai_investment(self, project_scope):
        """
        Estimate AI implementation investment based on project scope
        
        Args:
            project_scope (dict): Dictionary with scope parameters
            
        Returns:
            dict: Breakdown of estimated costs
        """
        # Base costs
        personnel_cost = project_scope.get("team_size", 5) * project_scope.get("avg_salary", 100000) * project_scope.get("duration_years", 1)
        technology_cost = project_scope.get("tech_complexity", 1) * 50000
        data_cost = project_scope.get("data_volume", 1) * 25000
        training_cost = project_scope.get("training_intensity", 1) * 20000
        integration_cost = project_scope.get("integration_complexity", 1) * 30000
        
        total_cost = personnel_cost + technology_cost + data_cost + training_cost + integration_cost
        
        return {
            "personnel": personnel_cost,
            "technology": technology_cost,
            "data": data_cost,
            "training": training_cost,
            "integration": integration_cost,
            "total": total_cost
        }
    
    def estimate_ai_benefits(self, business_context):
        """
        Estimate potential AI benefits based on business context
        
        Args:
            business_context (dict): Dictionary with business parameters
            
        Returns:
            dict: Estimated benefits by category
        """
        # Benefit multipliers based on business context
        volume_multiplier = business_context.get("transaction_volume", 10000) / 10000
        complexity_multiplier = business_context.get("process_complexity", 1)
        automation_potential = business_context.get("automation_potential", 0.5)
        
        # Estimated benefits
        cost_savings = volume_multiplier * complexity_multiplier * automation_potential * 100000
        revenue_increase = volume_multiplier * 0.1 * 50000  # 10% of cost savings
        quality_improvement = complexity_multiplier * 0.05 * 75000  # 5% of cost savings
        
        total_benefits = cost_savings + revenue_increase + quality_improvement
        
        return {
            "cost_savings": cost_savings,
            "revenue_increase": revenue_increase,
            "quality_improvement": quality_improvement,
            "total": total_benefits
        }
    
    def create_roi_report(self, investment_breakdown, benefit_estimates, timeframe_months=24):
        """
        Create a comprehensive ROI report
        
        Args:
            investment_breakdown (dict): Investment cost breakdown
            benefit_estimates (dict): Benefit estimates
            timeframe_months (int): Analysis timeframe in months
            
        Returns:
            dict: Comprehensive ROI analysis
        """
        total_investment = investment_breakdown["total"]
        total_benefits = benefit_estimates["total"]
        
        # Calculate key metrics
        simple_roi = self.calculate_simple_roi(total_investment, total_benefits)
        payback_period = self.calculate_payback_period(total_investment, total_benefits / (timeframe_months / 12))
        
        # Annualized benefits
        annual_benefits = total_benefits / (timeframe_months / 12)
        
        return {
            "investment_breakdown": investment_breakdown,
            "benefit_estimates": benefit_estimates,
            "key_metrics": {
                "total_investment": total_investment,
                "total_benefits": total_benefits,
                "simple_roi": round(simple_roi, 2),
                "payback_period": round(payback_period, 2),
                "annualized_benefits": round(annual_benefits, 2)
            },
            "recommendation": self._generate_recommendation(simple_roi, payback_period)
        }
    
    def _generate_recommendation(self, roi, payback_period):
        """Generate recommendation based on ROI metrics"""
        if roi > 50 and payback_period < 1.5:
            return "Strongly recommend proceeding with implementation"
        elif roi > 20 and payback_period < 2:
            return "Recommend proceeding with implementation"
        elif roi > 0 and payback_period < 3:
            return "Consider proceeding with implementation with risk mitigation"
        else:
            return "Recommend reevaluating project scope or deferring implementation"

# Example usage and demonstration
if __name__ == "__main__":
    # Create the ROI framework
    roi_framework = AIROIFramework()
    
    print("AI Implementation ROI Evaluation Framework:")
    print("=" * 50)
    
    # Display ROI components
    print("ROI Components:")
    for component in roi_framework.components:
        print(f"- {component.name}: {component.description}")
        print(f"  Measurement: {component.measurement_approach}")
        print(f"  Typical Timeframe: {component.typical_timeframe} months")
        print()
    
    print("Benefit Categories:")
    print("Quantitative Benefits:")
    for benefit in roi_framework.benefit_categories["quantitative"]:
        print(f"  - {benefit}")
    
    print("\nQualitative Benefits:")
    for benefit in roi_framework.benefit_categories["qualitative"]:
        print(f"  - {benefit}")
    
    print("\nROI Calculation Examples:")
    print("=" * 50)
    
    # Example 1: Simple ROI calculation
    investment = 200000
    returns = 350000
    simple_roi = roi_framework.calculate_simple_roi(investment, returns)
    print(f"Simple ROI Example:")
    print(f"  Investment: ${investment:,}")
    print(f"  Returns: ${returns:,}")
    print(f"  ROI: {simple_roi:.2f}%")
    
    # Example 2: Payback period calculation
    annual_savings = 100000
    payback_period = roi_framework.calculate_payback_period(investment, annual_savings)
    print(f"\nPayback Period Example:")
    print(f"  Initial Investment: ${investment:,}")
    print(f"  Annual Savings: ${annual_savings:,}")
    print(f"  Payback Period: {payback_period:.2f} years")
    
    # Example 3: Comprehensive ROI analysis
    print("\nComprehensive ROI Analysis Example:")
    print("=" * 50)
    
    # Sample project scope
    project_scope = {
        "team_size": 6,
        "avg_salary": 120000,
        "duration_years": 1.5,
        "tech_complexity": 2,
        "data_volume": 3,
        "training_intensity": 2,
        "integration_complexity": 3
    }
    
    # Sample business context
    business_context = {
        "transaction_volume": 50000,
        "process_complexity": 2,
        "automation_potential": 0.6
    }
    
    # Estimate investment and benefits
    investment_estimate = roi_framework.estimate_ai_investment(project_scope)
    benefit_estimate = roi_framework.estimate_ai_benefits(business_context)
    
    # Create ROI report
    roi_report = roi_framework.create_roi_report(investment_estimate, benefit_estimate, 24)
    
    print("Investment Breakdown:")
    for category, amount in roi_report["investment_breakdown"].items():
        print(f"  {category.title()}: ${amount:,.2f}")
    
    print("\nBenefit Estimates:")
    for category, amount in roi_report["benefit_estimates"].items():
        print(f"  {category.replace('_', ' ').title()}: ${amount:,.2f}")
    
    print("\nKey Metrics:")
    metrics = roi_report["key_metrics"]
    for metric, value in metrics.items():
        if metric == "simple_roi":
            print(f"  {metric.replace('_', ' ').title()}: {value}%")
        else:
            print(f"  {metric.replace('_', ' ').title()}: ${value:,.2f}" if isinstance(value, (int, float)) else f"  {metric.replace('_', ' ').title()}: {value:.2f} years")
    
    print(f"\nRecommendation: {roi_report['recommendation']}")