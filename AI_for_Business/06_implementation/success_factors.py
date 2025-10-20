"""
Key Success Factors for AI Implementation in Business

This module identifies and explains the critical factors that contribute
to successful AI implementation in business environments.
"""

class AISuccessFactor:
    """Represents a key success factor for AI implementation"""
    
    def __init__(self, name, description, importance_level, implementation_tips):
        self.name = name
        self.description = description
        self.importance_level = importance_level  # Scale of 1-10
        self.implementation_tips = implementation_tips
    
    def __str__(self):
        return f"{self.name} (Importance: {self.importance_level}/10)"

class AISuccessFactorsFramework:
    """Framework for evaluating and managing AI success factors"""
    
    def __init__(self):
        self.factors = self._initialize_success_factors()
    
    def _initialize_success_factors(self):
        """Initialize the key success factors for AI implementation"""
        return [
            AISuccessFactor(
                "Leadership Commitment",
                "Strong executive support and clear vision for AI adoption",
                9,
                [
                    "Secure C-suite sponsorship",
                    "Allocate dedicated budget and resources",
                    "Communicate AI vision consistently",
                    "Lead by example in AI adoption"
                ]
            ),
            AISuccessFactor(
                "Data Quality and Availability",
                "Access to high-quality, relevant data for training and operation",
                10,
                [
                    "Implement data governance policies",
                    "Invest in data cleaning and preprocessing",
                    "Ensure data privacy and security compliance",
                    "Establish data lineage and traceability"
                ]
            ),
            AISuccessFactor(
                "Skilled Talent",
                "Availability of personnel with AI/ML expertise and domain knowledge",
                8,
                [
                    "Hire AI specialists and data scientists",
                    "Upskill existing employees",
                    "Partner with academic institutions",
                    "Create cross-functional teams"
                ]
            ),
            AISuccessFactor(
                "Change Management",
                "Effective organizational transformation to embrace AI solutions",
                8,
                [
                    "Develop communication strategies",
                    "Provide comprehensive training programs",
                    "Address employee concerns and resistance",
                    "Create incentives for AI adoption"
                ]
            ),
            AISuccessFactor(
                "Technology Infrastructure",
                "Robust IT systems capable of supporting AI workloads",
                7,
                [
                    "Upgrade computing resources (cloud/on-premise)",
                    "Ensure scalable storage solutions",
                    "Implement API-first architecture",
                    "Establish MLOps pipelines"
                ]
            ),
            AISuccessFactor(
                "Clear Business Objectives",
                "Well-defined goals aligned with organizational strategy",
                9,
                [
                    "Link AI initiatives to business KPIs",
                    "Set measurable success criteria",
                    "Prioritize high-impact use cases",
                    "Establish realistic timelines"
                ]
            ),
            AISuccessFactor(
                "Ethical Considerations",
                "Responsible AI development with fairness and transparency",
                8,
                [
                    "Implement bias detection mechanisms",
                    "Ensure explainable AI decisions",
                    "Establish ethical review processes",
                    "Maintain regulatory compliance"
                ]
            ),
            AISuccessFactor(
                "Integration Capabilities",
                "Ability to connect AI solutions with existing business systems",
                7,
                [
                    "Design for system interoperability",
                    "Use standardized APIs and protocols",
                    "Plan for legacy system compatibility",
                    "Implement robust testing procedures"
                ]
            ),
            AISuccessFactor(
                "Continuous Learning Culture",
                "Organization-wide commitment to ongoing AI education",
                7,
                [
                    "Promote experimentation and innovation",
                    "Encourage knowledge sharing",
                    "Support continuous skill development",
                    "Celebrate AI successes and learn from failures"
                ]
            ),
            AISuccessFactor(
                "Measurement and Monitoring",
                "Systems to track performance and business impact",
                8,
                [
                    "Define relevant success metrics",
                    "Implement real-time monitoring dashboards",
                    "Conduct regular performance reviews",
                    "Adjust strategies based on insights"
                ]
            )
        ]
    
    def get_factor(self, factor_name):
        """Get a specific success factor by name"""
        for factor in self.factors:
            if factor.name.lower() == factor_name.lower():
                return factor
        return None
    
    def list_all_factors(self):
        """List all success factors"""
        return [(factor.name, factor.importance_level) for factor in self.factors]
    
    def get_top_factors(self, count=5):
        """Get the top N most important factors"""
        sorted_factors = sorted(self.factors, key=lambda x: x.importance_level, reverse=True)
        return sorted_factors[:count]
    
    def evaluate_implementation_readiness(self, organization_profile):
        """
        Evaluate an organization's readiness based on success factors
        
        Args:
            organization_profile (dict): Dictionary with factor names as keys and 
                                       readiness scores (1-10) as values
        """
        total_score = 0
        max_possible_score = 0
        factor_count = 0
        
        results = {}
        
        for factor in self.factors:
            factor_name = factor.name
            importance = factor.importance_level
            
            # Get organization's readiness score for this factor (default to 5 if not provided)
            org_score = organization_profile.get(factor_name, 5)
            
            # Calculate weighted score
            weighted_score = (org_score / 10) * importance
            max_weighted_score = importance
            
            total_score += weighted_score
            max_possible_score += max_weighted_score
            factor_count += 1
            
            results[factor_name] = {
                "importance": importance,
                "readiness_score": org_score,
                "weighted_contribution": weighted_score
            }
        
        # Calculate overall readiness percentage
        overall_readiness = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
        
        return {
            "overall_readiness": round(overall_readiness, 2),
            "detailed_results": results,
            "recommendations": self._generate_recommendations(results)
        }
    
    def _generate_recommendations(self, results):
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        # Find factors with low readiness scores
        for factor_name, data in results.items():
            if data["readiness_score"] <= 4:
                factor = self.get_factor(factor_name)
                if factor:
                    recommendations.append({
                        "factor": factor_name,
                        "issue": f"Low readiness score ({data['readiness_score']}/10)",
                        "suggestion": f"Focus on: {'; '.join(factor.implementation_tips[:2])}"
                    })
        
        # If no critical issues, suggest general improvements
        if not recommendations:
            # Find factors with moderate scores for improvement
            for factor_name, data in results.items():
                if 5 <= data["readiness_score"] <= 7:
                    factor = self.get_factor(factor_name)
                    if factor:
                        recommendations.append({
                            "factor": factor_name,
                            "issue": f"Moderate readiness score ({data['readiness_score']}/10)",
                            "suggestion": f"Consider: {factor.implementation_tips[0]}"
                        })
        
        return recommendations

# Example usage and demonstration
if __name__ == "__main__":
    # Create the success factors framework
    framework = AISuccessFactorsFramework()
    
    print("Key Success Factors for AI Implementation:")
    print("=" * 50)
    
    # Display all factors with their importance levels
    for factor_name, importance in framework.list_all_factors():
        print(f"{factor_name}: {importance}/10")
    
    print("\nTop 5 Most Important Factors:")
    print("=" * 50)
    
    # Show top factors
    top_factors = framework.get_top_factors(5)
    for i, factor in enumerate(top_factors, 1):
        print(f"{i}. {factor.name} (Importance: {factor.importance_level}/10)")
        print(f"   Description: {factor.description}")
    
    print("\nSample Implementation Readiness Evaluation:")
    print("=" * 50)
    
    # Sample organization profile (readiness scores for each factor)
    sample_org_profile = {
        "Leadership Commitment": 8,
        "Data Quality and Availability": 6,
        "Skilled Talent": 4,
        "Change Management": 7,
        "Technology Infrastructure": 5,
        "Clear Business Objectives": 9,
        "Ethical Considerations": 6,
        "Integration Capabilities": 4,
        "Continuous Learning Culture": 5,
        "Measurement and Monitoring": 7
    }
    
    # Evaluate readiness
    evaluation = framework.evaluate_implementation_readiness(sample_org_profile)
    
    print(f"Overall Readiness Score: {evaluation['overall_readiness']}%")
    print("\nDetailed Results:")
    for factor_name, data in evaluation['detailed_results'].items():
        print(f"  {factor_name}:")
        print(f"    Importance: {data['importance']}/10")
        print(f"    Readiness: {data['readiness_score']}/10")
        print(f"    Contribution: {data['weighted_contribution']:.1f}")
    
    print("\nRecommendations:")
    for rec in evaluation['recommendations']:
        print(f"  Factor: {rec['factor']}")
        print(f"    Issue: {rec['issue']}")
        print(f"    Suggestion: {rec['suggestion']}")