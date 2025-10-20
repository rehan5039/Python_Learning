"""
Practice Problems Solutions: AI Projects

This module contains example solutions for the practice problems in Chapter 8.
"""

# Import project modules
from ..projects.project_planning import AIProjectFramework
from ..projects.business_case import AIBusinessCaseFramework
from ..projects.implementation_plan import AIImplementationPlanner
from ..projects.success_evaluation import AIProjectEvaluator
from ..projects.lessons_learned import AILessonsLearnedManager

def problem_1_solution():
    """Solution for Problem 1: Project Planning and Timeline Development"""
    print("Problem 1: Project Planning and Timeline Development")
    print("=" * 50)
    
    # Create project framework
    framework = AIProjectFramework()
    
    print("Comprehensive Project Plan for Equipment Maintenance Prediction:")
    
    # List all phases
    print("\nProject Phases:")
    for i, (phase_name, timeline) in enumerate(framework.list_all_phases(), 1):
        print(f"{i}. {phase_name} ({timeline} weeks)")
    
    # Show total timeline
    timeline = framework.get_project_timeline()
    print(f"\nTotal Project Timeline: {timeline['total_weeks']} weeks ({timeline['total_months']} months)")
    
    # Identify key stakeholders
    print("\nKey Stakeholders and Roles:")
    stakeholders = framework.stakeholders
    for role, description in list(stakeholders.items())[:6]:  # Show first 6
        print(f"  {role.replace('_', ' ').title()}: {description}")
    
    # Risk assessment
    print("\nRisk Assessment:")
    risks = framework.risk_assessment_framework()
    for category, risk_list in list(risks.items())[:2]:  # Show first 2 categories
        print(f"\n{category.replace('_', ' ').title()}:")
        for risk in risk_list[:2]:  # Show first 2 risks per category
            print(f"  - {risk}")

def problem_2_solution():
    """Solution for Problem 2: Business Case Development"""
    print("\n\nProblem 2: Business Case Development")
    print("=" * 50)
    
    # Create business case framework
    framework = AIBusinessCaseFramework()
    
    print("Business Case for Retail Recommendation System:")
    
    # List components
    print("\nBusiness Case Components:")
    for i, component_name in enumerate(framework.list_all_components(), 1):
        component = framework.get_component(component_name)
        print(f"{i}. {component.name}: {component.description}")
    
    # Calculate ROI
    print("\nROI Calculation:")
    investment = 750000  # $750K investment
    annual_benefits = 300000  # $300K annual benefits
    lifespan = 3  # 3-year lifespan
    
    roi_metrics = framework.calculate_roi(investment, annual_benefits, lifespan)
    print(f"  Investment: ${investment:,}")
    print(f"  Annual Benefits: ${annual_benefits:,}")
    print(f"  Lifespan: {lifespan} years")
    print(f"  Total Benefits: ${roi_metrics['total_benefits']:,}")
    print(f"  Net Benefits: ${roi_metrics['net_benefits']:,}")
    print(f"  ROI: {roi_metrics['roi_percentage']}%")
    print(f"  Payback Period: {roi_metrics['payback_period']} years")
    
    # Benefit categories
    print("\nPotential Benefits:")
    categories = framework.benefit_categories()
    for category, benefits in list(categories.items())[:3]:  # Show first 3 categories
        print(f"\n{category.replace('_', ' ').title()}:")
        for benefit in benefits[:2]:  # Show first 2 benefits per category
            print(f"  - {benefit}")
    
    # Sensitivity analysis
    print("\nSensitivity Analysis:")
    sensitivity = framework.sensitivity_analysis(investment, annual_benefits)
    for scenario, metrics in sensitivity.items():
        print(f"\n{scenario.replace('_', ' ').title()} Scenario:")
        print(f"  ROI: {metrics['roi_percentage']}%")
        print(f"  Payback: {metrics['payback_period']} years")

def problem_3_solution():
    """Solution for Problem 3: Implementation Plan Creation"""
    print("\n\nProblem 3: Implementation Plan Creation")
    print("=" * 50)
    
    # Create implementation planner
    planner = AIImplementationPlanner()
    
    print("Implementation Plan for Financial Services Fraud Detection:")
    
    # List all phases
    print("\nImplementation Phases:")
    for i, (phase_name, timeline) in enumerate(planner.list_all_phases(), 1):
        print(f"{i}. {phase_name} ({timeline} weeks)")
    
    # Show total timeline
    timeline = planner.get_total_timeline()
    print(f"\nTotal Implementation Timeline: {timeline['total_weeks']} weeks ({timeline['total_months']} months)")
    
    # Resource requirements
    print("\nResource Requirements:")
    resources = planner.resource_categories
    for category, items in list(resources.items())[:3]:  # Show first 3 categories
        print(f"\n{category.replace('_', ' ').title()}:")
        for item in items[:3]:  # Show first 3 items per category
            print(f"  - {item}")
    
    # Key milestones
    print("\nKey Milestones:")
    milestones = planner.milestone_planner()
    for milestone in milestones[:5]:  # Show first 5 milestones
        print(f"  Week {milestone['target_week']} (Month {milestone['target_month']}): {milestone['milestone']}")
    
    # Risk management
    print("\nRisk Management Approach:")
    risks = planner.risk_management_plan()
    for category, risk_list in list(risks.items())[:2]:  # Show first 2 categories
        print(f"\n{category.replace('_', ' ').title()}:")
        for risk, mitigation in list(risk_list.items())[:2]:  # Show first 2 risks per category
            print(f"  Risk: {risk}")
            print(f"  Mitigation: {mitigation}")

def problem_4_solution():
    """Solution for Problem 4: Success Metrics and Evaluation"""
    print("\n\nProblem 4: Success Metrics and Evaluation")
    print("=" * 50)
    
    # Create project evaluator
    evaluator = AIProjectEvaluator()
    
    print("Success Evaluation Plan for Healthcare Diagnostic Support System:")
    
    # List metrics by category
    print("\nSuccess Metrics by Category:")
    metrics_by_category = evaluator.list_metrics_by_category()
    for category, metrics in list(metrics_by_category.items())[:4]:  # Show first 4 categories
        print(f"\n{category}:")
        for metric in metrics[:2]:  # Show first 2 metrics per category
            print(f"  - {metric['name']}: {metric['description']}")
            print(f"    Target: {metric['target']}")
    
    # Evaluation plan
    print("\nEvaluation Plan:")
    plan = evaluator.create_evaluation_plan("Healthcare Diagnostic Support System")
    print(f"Project: {plan['project_name']}")
    print("Reporting Schedule:")
    for frequency, activities in plan['reporting_schedule'].items():
        print(f"\n{frequency.replace('_', ' ').title()}:")
        for activity in activities[:2]:  # Show first 2 activities per frequency
            print(f"  - {activity}")
    
    # Project scoring example
    print("\nProject Success Scoring Example:")
    sample_metrics = {
        "roi": "35%",
        "business_impact": "100% of objectives met",
        "model_performance": "96% accuracy",
        "user_adoption": "92% engagement",
        "time_to_value": "3.5 months",
        "scalability": "Handles 3x load effectively"
    }
    
    score = evaluator.calculate_project_score(sample_metrics)
    print(f"Overall Success Score: {score['overall_score']}%")
    print(f"Metrics Achieved: {score['metrics_met']}/{score['total_metrics']}")
    
    # Benchmarking framework
    print("\nBenchmarking Framework:")
    benchmarks = evaluator.benchmarking_framework()
    for category, methods in list(benchmarks.items())[:2]:  # Show first 2 categories
        print(f"\n{category.replace('_', ' ').title()}:")
        for method in methods[:2]:  # Show first 2 methods per category
            print(f"  - {method}")

def problem_5_solution():
    """Solution for Problem 5: Lessons Learned Documentation"""
    print("\n\nProblem 5: Lessons Learned Documentation")
    print("=" * 50)
    
    # Create lessons learned manager
    manager = AILessonsLearnedManager()
    
    print("Lessons Learned for AI Customer Service Chatbot Project:")
    
    # List all categories
    print("\nLesson Categories:")
    for i, (category_name, description) in enumerate(manager.list_all_categories(), 1):
        print(f"{i}. {category_name}: {description}")
    
    # Create lessons learned template
    print("\nLessons Learned Template:")
    template = manager.create_lessons_learned_template("AI Customer Service Chatbot")
    print(f"Project: {template['project_name']}")
    print("Key Categories for Documentation:")
    for category_name in list(template['key_lessons_by_category'].keys())[:4]:
        print(f"  - {category_name}")
    
    # Sample lessons learned content
    print("\nSample Lessons Learned Content:")
    sample_lessons = {
        "Data Management": {
            "positive_outcomes": [
                "Implemented robust data governance practices",
                "Created comprehensive training dataset"
            ],
            "challenges_faced": [
                "Initial data quality issues affected model performance",
                "Privacy concerns required additional compliance measures"
            ],
            "recommendations": [
                "Conduct thorough data quality assessment early",
                "Establish clear data privacy protocols from start"
            ]
        },
        "Model Development": {
            "positive_outcomes": [
                "Started with simple rule-based approach before ML",
                "Used ensemble methods for improved accuracy"
            ],
            "challenges_faced": [
                "Balancing accuracy with response time",
                "Handling edge cases and unusual queries"
            ],
            "recommendations": [
                "Plan for iterative model improvements",
                "Include fallback mechanisms for complex queries"
            ]
        }
    }
    
    for category, content in list(sample_lessons.items())[:2]:
        print(f"\n{category}:")
        print("  Positive Outcomes:")
        for outcome in content['positive_outcomes']:
            print(f"    - {outcome}")
        print("  Challenges Faced:")
        for challenge in content['challenges_faced']:
            print(f"    - {challenge}")
        print("  Recommendations:")
        for recommendation in content['recommendations']:
            print(f"    - {recommendation}")
    
    # Best practices
    print("\nBest Practices from Successful Projects:")
    best_practices = manager.best_practices_catalog()
    for category, practices in list(best_practices.items())[:3]:  # Show first 3 categories
        print(f"\n{category.replace('_', ' ').title()}:")
        for practice in practices[:2]:  # Show first 2 practices per category
            print(f"  - {practice}")

# Run all solutions
if __name__ == "__main__":
    problem_1_solution()
    problem_2_solution()
    problem_3_solution()
    problem_4_solution()
    problem_5_solution()