"""
Practice Problems Solutions: AI Implementation

This module contains example solutions for the practice problems in Chapter 6.
"""

from ..implementation_phases import AIImplementationFramework
from ..success_factors import AISuccessFactorsFramework
from ..deployment_strategies import AIDeploymentFramework
from ..challenges_mitigation import AIChallengesFramework
from ..roi_evaluation import AIROIFramework

def problem_1_solution():
    """Solution for Problem 1: Implementation Phase Planning"""
    print("Problem 1: Implementation Phase Planning")
    print("=" * 40)
    
    framework = AIImplementationFramework()
    
    # Get details for each phase
    phases = [
        "Assessment and Planning",
        "Design and Development", 
        "Testing and Validation",
        "Deployment",
        "Monitoring and Optimization",
        "Scaling and Expansion"
    ]
    
    for phase_name in phases:
        details = framework.get_phase_details(phase_name)
        if details:
            print(f"\n{phase_name}:")
            print("Key Activities:")
            for activity in details['key_activities'][:3]:  # Show top 3 activities
                print(f"  - {activity}")
    
    # Success metrics for Testing and Validation
    validation_phase = framework.get_phase("Testing and Validation")
    print(f"\nTesting and Validation Success Metrics:")
    for metric in validation_phase.success_metrics:
        print(f"  - {metric}")
    
    # Deployment challenges and solutions
    challenges_framework = AIChallengesFramework()
    deployment_challenge = challenges_framework.get_challenge("Integration Complexity")
    print(f"\nDeployment Phase Challenge - {deployment_challenge.name}:")
    print(f"Description: {deployment_challenge.description}")
    print("Mitigation Strategies:")
    for strategy in deployment_challenge.mitigation_strategies[:3]:
        print(f"  - {strategy}")

def problem_2_solution():
    """Solution for Problem 2: Success Factor Assessment"""
    print("\n\nProblem 2: Success Factor Assessment")
    print("=" * 40)
    
    framework = AISuccessFactorsFramework()
    
    # Sample organization profile for chatbot implementation
    org_profile = {
        "Leadership Commitment": 7,
        "Data Quality and Availability": 6,
        "Skilled Talent": 5,
        "Change Management": 6,
        "Technology Infrastructure": 7,
        "Clear Business Objectives": 8,
        "Ethical Considerations": 6,
        "Integration Capabilities": 5,
        "Continuous Learning Culture": 4,
        "Measurement and Monitoring": 6
    }
    
    # Evaluate readiness
    evaluation = framework.evaluate_implementation_readiness(org_profile)
    
    print(f"Overall Readiness Score: {evaluation['overall_readiness']}%")
    print("\nTop 3 Factors Needing Attention:")
    
    # Sort factors by readiness score (ascending) to find lowest scores
    sorted_factors = sorted(
        evaluation['detailed_results'].items(), 
        key=lambda x: x[1]['readiness_score']
    )
    
    for factor_name, data in sorted_factors[:3]:
        print(f"  {factor_name}: {data['readiness_score']}/10")
        factor_obj = framework.get_factor(factor_name)
        if factor_obj:
            print(f"    Improvement Actions: {'; '.join(factor_obj.implementation_tips[:2])}")

def problem_3_solution():
    """Solution for Problem 3: Deployment Strategy Selection"""
    print("\n\nProblem 3: Deployment Strategy Selection")
    print("=" * 40)
    
    framework = AIDeploymentFramework()
    
    # Financial services context for fraud detection
    business_context = {
        "organization_size": "large",
        "risk_tolerance": "low",
        "system_complexity": "high",
        "budget_constraints": "low"
    }
    
    # Recommend strategy
    recommendation = framework.recommend_strategy(business_context)
    strategy_details = framework.get_strategy_details(recommendation)
    
    print(f"Recommended Strategy: {recommendation}")
    print(f"Description: {strategy_details['description']}")
    print("\nWhy This Strategy:")
    print("  - Low risk tolerance requires careful implementation")
    print("  - High system complexity needs phased approach")
    print("  - Large organization benefits from controlled rollout")
    print("\nKey Implementation Steps:")
    for step in strategy_details['implementation_steps'][:5]:
        print(f"  - {step}")

def problem_4_solution():
    """Solution for Problem 4: Challenge Mitigation Planning"""
    print("\n\nProblem 4: Challenge Mitigation Planning")
    print("=" * 40)
    
    framework = AIChallengesFramework()
    
    # Top challenges in regulated industry
    top_challenges = [
        "Regulatory Compliance",
        "Ethical and Bias Concerns",
        "Data Quality Issues",
        "Model Drift",
        "Integration Complexity"
    ]
    
    print("Top 5 Challenges and Mitigation Strategies:")
    for challenge_name in top_challenges:
        challenge = framework.get_challenge(challenge_name)
        if challenge:
            print(f"\n{challenge.name} (Impact: {challenge.impact_level}/10):")
            print(f"  Description: {challenge.description}")
            print("  Mitigation Strategies:")
            for strategy in challenge.mitigation_strategies[:2]:
                print(f"    - {strategy}")
    
    print("\nOngoing Risk Monitoring Plan:")
    print("  1. Monthly compliance audits")
    print("  2. Quarterly bias detection reviews")
    print("  3. Weekly data quality assessments")
    print("  4. Continuous model performance monitoring")
    print("  5. Annual integration architecture reviews")

def problem_5_solution():
    """Solution for Problem 5: ROI Calculation and Analysis"""
    print("\n\nProblem 5: ROI Calculation and Analysis")
    print("=" * 40)
    
    framework = AIROIFramework()
    
    # Given data
    initial_investment = 500000
    annual_benefits = 200000
    project_lifespan = 3
    discount_rate = 0.08
    
    # Calculate metrics
    simple_roi = framework.calculate_simple_roi(initial_investment, annual_benefits * project_lifespan)
    payback_period = framework.calculate_payback_period(initial_investment, annual_benefits)
    
    # Calculate NPV
    cash_flows = [annual_benefits] * project_lifespan
    npv = framework.calculate_npv(cash_flows, discount_rate) - initial_investment
    
    print("Financial Analysis:")
    print(f"  Initial Investment: ${initial_investment:,}")
    print(f"  Annual Benefits: ${annual_benefits:,}")
    print(f"  Project Lifespan: {project_lifespan} years")
    print(f"  Discount Rate: {discount_rate*100}%")
    print(f"\nCalculated Metrics:")
    print(f"  Simple ROI: {simple_roi:.2f}%")
    print(f"  Payback Period: {payback_period:.2f} years")
    print(f"  Net Present Value (NPV): ${npv:,.2f}")
    
    # Recommendation
    if simple_roi > 20 and payback_period < 3 and npv > 0:
        recommendation = "RECOMMEND PROCEEDING with the AI implementation project."
    else:
        recommendation = "RECOMMEND RECONSIDERING the project scope or timing."
    
    print(f"\nRecommendation: {recommendation}")
    print("\nJustification:")
    print("  - Positive ROI indicates value creation")
    print("  - Reasonable payback period under 3 years")
    print("  - Positive NPV suggests value exceeds cost of capital")

# Run all solutions
if __name__ == "__main__":
    problem_1_solution()
    problem_2_solution()
    problem_3_solution()
    problem_4_solution()
    problem_5_solution()