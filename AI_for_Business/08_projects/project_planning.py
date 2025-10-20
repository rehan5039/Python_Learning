"""
AI Project Planning Framework

This module provides a comprehensive framework for planning and managing
AI projects in business environments.
"""

class AIProjectPhase:
    """Represents a phase in the AI project lifecycle"""
    
    def __init__(self, name, description, key_activities, deliverables, timeline):
        self.name = name
        self.description = description
        self.key_activities = key_activities
        self.deliverables = deliverables
        self.timeline = timeline  # in weeks
    
    def __str__(self):
        return f"{self.name}: {self.description}"

class AIProjectFramework:
    """Framework for managing AI projects through their lifecycle"""
    
    def __init__(self):
        self.phases = self._initialize_phases()
        self.stakeholders = self._initialize_stakeholders()
    
    def _initialize_phases(self):
        """Initialize the standard phases of an AI project"""
        return [
            AIProjectPhase(
                "Problem Definition",
                "Clearly articulate the business challenge and success criteria",
                [
                    "Conduct stakeholder interviews",
                    "Define project scope and boundaries",
                    "Establish success metrics and KPIs",
                    "Identify constraints and assumptions"
                ],
                [
                    "Project charter document",
                    "Stakeholder analysis",
                    "Success criteria matrix",
                    "Initial risk assessment"
                ],
                2
            ),
            AIProjectPhase(
                "Opportunity Assessment",
                "Evaluate AI feasibility and potential business impact",
                [
                    "Analyze available data sources",
                    "Assess technical feasibility",
                    "Estimate business value and ROI",
                    "Identify potential risks and challenges"
                ],
                [
                    "Data assessment report",
                    "Feasibility analysis",
                    "Business case summary",
                    "Risk register"
                ],
                3
            ),
            AIProjectPhase(
                "Solution Design",
                "Define the AI approach and technical architecture",
                [
                    "Select appropriate algorithms and techniques",
                    "Design system architecture",
                    "Plan data collection and preprocessing",
                    "Define model evaluation criteria"
                ],
                [
                    "Technical design document",
                    "Architecture diagram",
                    "Data requirements specification",
                    "Model evaluation plan"
                ],
                4
            ),
            AIProjectPhase(
                "Development",
                "Build and train the AI models",
                [
                    "Collect and preprocess training data",
                    "Develop and train models",
                    "Conduct model validation and testing",
                    "Iterate based on results"
                ],
                [
                    "Trained models",
                    "Model performance reports",
                    "Code repository",
                    "Testing documentation"
                ],
                8
            ),
            AIProjectPhase(
                "Deployment",
                "Integrate the AI solution into business processes",
                [
                    "Set up production environment",
                    "Integrate with existing systems",
                    "Train end users",
                    "Execute deployment plan"
                ],
                [
                    "Deployed solution",
                    "Integration documentation",
                    "User training materials",
                    "Go-live report"
                ],
                4
            ),
            AIProjectPhase(
                "Monitoring & Optimization",
                "Track performance and continuously improve the solution",
                [
                    "Monitor model performance",
                    "Collect user feedback",
                    "Implement improvements",
                    "Plan for scaling"
                ],
                [
                    "Performance monitoring dashboard",
                    "Feedback analysis report",
                    "Improvement recommendations",
                    "Scaling plan"
                ],
                6
            )
        ]
    
    def _initialize_stakeholders(self):
        """Initialize key stakeholder roles"""
        return {
            "project_sponsor": "Executive sponsor with budget authority",
            "project_manager": "Day-to-day project leadership",
            "business_owner": "Subject matter expert for business process",
            "data_scientist": "AI/ML model development expertise",
            "data_engineer": "Data infrastructure and pipeline management",
            "it_architect": "Technical architecture and integration",
            "end_users": "Those who will use the AI solution",
            "compliance_officer": "Regulatory and ethical considerations"
        }
    
    def get_phase(self, phase_name):
        """Get a specific phase by name"""
        for phase in self.phases:
            if phase.name.lower() == phase_name.lower():
                return phase
        return None
    
    def list_all_phases(self):
        """List all project phases"""
        return [(phase.name, phase.timeline) for phase in self.phases]
    
    def get_project_timeline(self):
        """Calculate total project timeline"""
        total_weeks = sum(phase.timeline for phase in self.phases)
        return {
            "total_weeks": total_weeks,
            "total_months": round(total_weeks / 4, 1)
        }
    
    def create_project_charter(self, project_name, business_problem, expected_outcomes):
        """Create a project charter template"""
        return {
            "project_name": project_name,
            "business_problem": business_problem,
            "expected_outcomes": expected_outcomes,
            "phases": self.list_all_phases(),
            "total_timeline": self.get_project_timeline(),
            "key_stakeholders": list(self.stakeholders.keys())
        }
    
    def risk_assessment_framework(self):
        """Framework for assessing project risks"""
        return {
            "technical_risks": [
                "Data quality and availability",
                "Model performance and accuracy",
                "Integration complexity",
                "Scalability challenges"
            ],
            "business_risks": [
                "Changing business requirements",
                "User adoption resistance",
                "ROI realization delays",
                "Competitive response"
            ],
            "organizational_risks": [
                "Skill gaps and resource constraints",
                "Stakeholder alignment issues",
                "Budget and timeline pressures",
                "Governance and compliance"
            ]
        }

# Example usage and demonstration
if __name__ == "__main__":
    # Create the project framework
    framework = AIProjectFramework()
    
    print("AI Project Planning Framework:")
    print("=" * 40)
    
    # Display all phases
    print("Project Phases:")
    for i, (phase_name, timeline) in enumerate(framework.list_all_phases(), 1):
        print(f"{i}. {phase_name} ({timeline} weeks)")
    
    # Show total timeline
    timeline = framework.get_project_timeline()
    print(f"\nTotal Project Timeline: {timeline['total_weeks']} weeks ({timeline['total_months']} months)")
    
    # Create sample project charter
    print("\nSample Project Charter:")
    charter = framework.create_project_charter(
        "Customer Churn Prediction",
        "High customer churn rate impacting revenue and growth",
        "Reduce churn by 25% through early identification and intervention"
    )
    
    print(f"Project: {charter['project_name']}")
    print(f"Problem: {charter['business_problem']}")
    print(f"Outcome: {charter['expected_outcomes']}")
    print(f"Duration: {charter['total_timeline']['total_months']} months")
    
    # Show risk assessment framework
    print("\nRisk Assessment Framework:")
    risks = framework.risk_assessment_framework()
    for category, risk_list in risks.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for risk in risk_list:
            print(f"  - {risk}")