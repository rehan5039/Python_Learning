"""
AI Project Success Evaluation

This module provides methods for measuring and evaluating the success
of AI projects in business environments.
"""

class SuccessMetric:
    """Represents a success metric for AI projects"""
    
    def __init__(self, name, category, description, measurement_approach, target_value):
        self.name = name
        self.category = category
        self.description = description
        self.measurement_approach = measurement_approach
        self.target_value = target_value
    
    def __str__(self):
        return f"{self.name} ({self.category}): {self.description}"

class AIProjectEvaluator:
    """Evaluator for AI project success"""
    
    def __init__(self):
        self.metrics = self._initialize_metrics()
        self.evaluation_framework = self._initialize_evaluation_framework()
    
    def _initialize_metrics(self):
        """Initialize success metrics"""
        return [
            SuccessMetric(
                "ROI",
                "Financial",
                "Return on Investment from AI implementation",
                "Compare benefits to costs over project lifespan",
                ">20%"
            ),
            SuccessMetric(
                "Business Impact",
                "Strategic",
                "Achievement of stated business objectives",
                "Measure against predefined success criteria",
                "100% of key objectives met"
            ),
            SuccessMetric(
                "Model Performance",
                "Technical",
                "Accuracy and reliability of AI models",
                "Track precision, recall, F1-score, or domain-specific metrics",
                ">85% accuracy"
            ),
            SuccessMetric(
                "User Adoption",
                "Operational",
                "Rate of user acceptance and engagement",
                "Monitor usage statistics and user feedback",
                ">80% active user engagement"
            ),
            SuccessMetric(
                "Time to Value",
                "Project",
                "Speed of realizing business benefits",
                "Measure from project start to benefit realization",
                "<6 months"
            ),
            SuccessMetric(
                "Scalability",
                "Technical",
                "Ability to handle increased load or scope",
                "Monitor performance under varying conditions",
                "Linear performance scaling"
            )
        ]
    
    def _initialize_evaluation_framework(self):
        """Initialize evaluation framework"""
        return {
            "pre_evaluation": {
                "baseline_measurement": "Establish metrics before implementation",
                "stakeholder_alignment": "Ensure agreement on success criteria",
                "data_collection_setup": "Implement tracking mechanisms"
            },
            "ongoing_monitoring": {
                "regular_checkpoints": "Monthly progress reviews",
                "real_time_metrics": "Continuous performance tracking",
                "stakeholder_feedback": "Regular user surveys and interviews"
            },
            "post_evaluation": {
                "comprehensive_assessment": "Final project evaluation",
                "lessons_learned": "Document successes and challenges",
                "knowledge_transfer": "Share insights across organization"
            }
        }
    
    def get_metric(self, metric_name):
        """Get a specific metric by name"""
        for metric in self.metrics:
            if metric.name.lower() == metric_name.lower():
                return metric
        return None
    
    def list_metrics_by_category(self):
        """List metrics organized by category"""
        categories = {}
        for metric in self.metrics:
            if metric.category not in categories:
                categories[metric.category] = []
            categories[metric.category].append({
                "name": metric.name,
                "description": metric.description,
                "target": metric.target_value
            })
        return categories
    
    def create_evaluation_plan(self, project_name):
        """Create a comprehensive evaluation plan"""
        return {
            "project_name": project_name,
            "metrics": [
                {
                    "name": metric.name,
                    "category": metric.category,
                    "description": metric.description,
                    "measurement_approach": metric.measurement_approach,
                    "target_value": metric.target_value
                }
                for metric in self.metrics
            ],
            "evaluation_phases": self.evaluation_framework,
            "reporting_schedule": {
                "weekly": ["Real-time metrics monitoring"],
                "monthly": ["Progress reviews", "Stakeholder feedback"],
                "quarterly": ["Comprehensive performance assessment"],
                "post_project": ["Final evaluation", "Lessons learned session"]
            }
        }
    
    def calculate_project_score(self, actual_metrics):
        """
        Calculate overall project success score
        
        Args:
            actual_metrics (dict): Dictionary with actual metric values
            
        Returns:
            dict: Score calculation results
        """
        total_possible = len(self.metrics)
        met_targets = 0
        scores = {}
        
        for metric in self.metrics:
            actual_value = actual_metrics.get(metric.name.lower().replace(" ", "_"), "N/A")
            scores[metric.name] = {
                "target": metric.target_value,
                "actual": actual_value,
                "met": self._check_target_achievement(metric, actual_value)
            }
            if scores[metric.name]["met"]:
                met_targets += 1
        
        overall_score = (met_targets / total_possible) * 100 if total_possible > 0 else 0
        
        return {
            "overall_score": round(overall_score, 2),
            "metrics_met": met_targets,
            "total_metrics": total_possible,
            "detailed_scores": scores
        }
    
    def _check_target_achievement(self, metric, actual_value):
        """Check if a metric target has been achieved"""
        # This is a simplified implementation
        # In practice, this would need more sophisticated logic
        if isinstance(actual_value, str):
            return "N/A" not in actual_value
        elif isinstance(actual_value, (int, float)):
            # Simplified comparison - would need more logic for different target formats
            return actual_value > 0
        return False
    
    def benchmarking_framework(self):
        """Framework for benchmarking AI project success"""
        return {
            "internal_benchmarks": [
                "Compare against previous AI projects",
                "Measure against organizational averages",
                "Track improvement over time"
            ],
            "industry_benchmarks": [
                "Compare against industry standards",
                "Measure against competitor performance",
                "Benchmark against best practices"
            ],
            "historical_benchmarks": [
                "Compare pre and post implementation",
                "Measure improvement from baseline",
                "Track trend over multiple periods"
            ]
        }
    
    def feedback_collection_methods(self):
        """Methods for collecting project feedback"""
        return {
            "quantitative_methods": [
                "Usage analytics and metrics",
                "Performance monitoring dashboards",
                "Financial impact tracking"
            ],
            "qualitative_methods": [
                "User surveys and interviews",
                "Focus groups with stakeholders",
                "Expert reviews and assessments"
            ],
            "continuous_methods": [
                "Real-time user feedback systems",
                "Automated alert mechanisms",
                "Regular stakeholder check-ins"
            ]
        }

# Example usage and demonstration
if __name__ == "__main__":
    # Create the project evaluator
    evaluator = AIProjectEvaluator()
    
    print("AI Project Success Evaluation Framework:")
    print("=" * 40)
    
    # Display metrics by category
    print("Success Metrics by Category:")
    metrics_by_category = evaluator.list_metrics_by_category()
    for category, metrics in metrics_by_category.items():
        print(f"\n{category}:")
        for metric in metrics:
            print(f"  - {metric['name']}: {metric['description']}")
            print(f"    Target: {metric['target']}")
    
    # Create sample evaluation plan
    print("\nSample Evaluation Plan:")
    plan = evaluator.create_evaluation_plan("Customer Churn Prediction System")
    print(f"Project: {plan['project_name']}")
    print("Key Metrics:")
    for metric in plan['metrics'][:3]:  # Show first 3 metrics
        print(f"  - {metric['name']} ({metric['category']}): {metric['target_value']}")
    
    # Show evaluation phases
    print("\nEvaluation Phases:")
    for phase, activities in plan['evaluation_phases'].items():
        print(f"\n{phase.replace('_', ' ').title()}:")
        for activity in list(activities.values())[:2] if isinstance(activities, dict) else activities[:2]:
            print(f"  - {activity}")
    
    # Demonstrate project scoring
    print("\nProject Success Scoring Example:")
    sample_metrics = {
        "roi": "25%",
        "business_impact": "95% of objectives met",
        "model_performance": "92% accuracy",
        "user_adoption": "85% engagement",
        "time_to_value": "4.2 months",
        "scalability": "Handles 2x load effectively"
    }
    
    score = evaluator.calculate_project_score(sample_metrics)
    print(f"Overall Success Score: {score['overall_score']}%")
    print(f"Metrics Achieved: {score['metrics_met']}/{score['total_metrics']}")
    print("\nDetailed Metric Performance:")
    for metric_name, details in list(score['detailed_scores'].items())[:3]:
        print(f"  {metric_name}:")
        print(f"    Target: {details['target']}")
        print(f"    Actual: {details['actual']}")
        print(f"    Met Target: {'Yes' if details['met'] else 'No'}")
    
    # Show benchmarking framework
    print("\nBenchmarking Framework:")
    benchmarks = evaluator.benchmarking_framework()
    for category, methods in benchmarks.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for method in methods:
            print(f"  - {method}")