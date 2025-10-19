"""
Data Analysis Tool

This application demonstrates data analysis techniques using pandas, matplotlib,
and statistical computing. It provides tools for loading, cleaning, analyzing,
and visualizing datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataManager:
    """Manages dataset loading and storage"""
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.file_path: Optional[str] = None
        self.data_info: Dict = {}
    
    def load_dataset(self, file_path: str) -> bool:
        """Load dataset from file"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False
            
            # Determine file type and load accordingly
            _, ext = os.path.splitext(file_path)
            
            if ext.lower() == '.csv':
                self.data = pd.read_csv(file_path)
            elif ext.lower() == '.json':
                self.data = pd.read_json(file_path)
            elif ext.lower() in ['.xlsx', '.xls']:
                self.data = pd.read_excel(file_path)
            else:
                logger.error(f"Unsupported file format: {ext}")
                return False
            
            self.file_path = file_path
            self._update_data_info()
            logger.info(f"Dataset loaded successfully: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return False
    
    def _update_data_info(self):
        """Update data information"""
        if self.data is not None:
            self.data_info = {
                'rows': len(self.data),
                'columns': len(self.data.columns),
                'column_names': list(self.data.columns),
                'data_types': self.data.dtypes.to_dict(),
                'memory_usage': self.data.memory_usage(deep=True).sum()
            }
    
    def get_info(self) -> Dict:
        """Get dataset information"""
        return self.data_info
    
    def get_sample(self, n: int = 5) -> Optional[pd.DataFrame]:
        """Get sample of data"""
        if self.data is not None:
            return self.data.head(n)
        return None
    
    def get_column_names(self) -> List[str]:
        """Get column names"""
        if self.data is not None:
            return list(self.data.columns)
        return []


class Analyzer:
    """Performs statistical analysis on datasets"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
    
    def descriptive_statistics(self) -> Optional[pd.DataFrame]:
        """Calculate descriptive statistics"""
        data = self.data_manager.data
        if data is None:
            return None
        
        try:
            # Numeric columns only
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                logger.warning("No numeric columns found for statistics")
                return None
            
            stats = numeric_data.describe()
            return stats
        except Exception as e:
            logger.error(f"Error calculating descriptive statistics: {e}")
            return None
    
    def correlation_matrix(self) -> Optional[pd.DataFrame]:
        """Calculate correlation matrix"""
        data = self.data_manager.data
        if data is None:
            return None
        
        try:
            # Numeric columns only
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                logger.warning("No numeric columns found for correlation")
                return None
            
            correlation = numeric_data.corr()
            return correlation
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return None
    
    def missing_values(self) -> Optional[pd.Series]:
        """Count missing values per column"""
        data = self.data_manager.data
        if data is None:
            return None
        
        try:
            missing = data.isnull().sum()
            return missing[missing > 0]
        except Exception as e:
            logger.error(f"Error counting missing values: {e}")
            return None
    
    def duplicates(self) -> int:
        """Count duplicate rows"""
        data = self.data_manager.data
        if data is None:
            return 0
        
        try:
            return data.duplicated().sum()
        except Exception as e:
            logger.error(f"Error counting duplicates: {e}")
            return 0


class Visualizer:
    """Creates data visualizations"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.output_dir = 'results/charts'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def histogram(self, column: str, bins: int = 30) -> bool:
        """Create histogram for a column"""
        data = self.data_manager.data
        if data is None or column not in data.columns:
            return False
        
        try:
            plt.figure(figsize=(10, 6))
            plt.hist(data[column].dropna(), bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # Save plot
            filename = f"histogram_{column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Histogram saved: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error creating histogram: {e}")
            return False
    
    def scatter_plot(self, x_col: str, y_col: str) -> bool:
        """Create scatter plot between two columns"""
        data = self.data_manager.data
        if data is None or x_col not in data.columns or y_col not in data.columns:
            return False
        
        try:
            plt.figure(figsize=(10, 6))
            plt.scatter(data[x_col], data[y_col], alpha=0.6, color='coral')
            plt.title(f'{y_col} vs {x_col}')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.grid(True, alpha=0.3)
            
            # Calculate correlation for display
            corr = data[x_col].corr(data[y_col])
            plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Save plot
            filename = f"scatter_{x_col}_vs_{y_col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Scatter plot saved: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error creating scatter plot: {e}")
            return False
    
    def correlation_heatmap(self) -> bool:
        """Create correlation heatmap"""
        data = self.data_manager.data
        if data is None:
            return False
        
        try:
            # Numeric columns only
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                return False
            
            correlation = numeric_data.corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5)
            plt.title('Correlation Matrix Heatmap')
            
            # Save plot
            filename = f"correlation_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Correlation heatmap saved: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            return False


class Reporter:
    """Generates analysis reports"""
    
    def __init__(self, data_manager: DataManager, analyzer: Analyzer, visualizer: Visualizer):
        self.data_manager = data_manager
        self.analyzer = analyzer
        self.visualizer = visualizer
        self.output_dir = 'results/reports'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_summary_report(self) -> bool:
        """Generate summary analysis report"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Get data info
            data_info = self.data_manager.get_info()
            if not data_info:
                logger.error("No data available for report")
                return False
            
            # Get statistics
            stats = self.analyzer.descriptive_statistics()
            correlation = self.analyzer.correlation_matrix()
            missing = self.analyzer.missing_values()
            duplicates = self.analyzer.duplicates()
            
            # Create report content
            report_content = []
            report_content.append("=" * 60)
            report_content.append("DATA ANALYSIS SUMMARY REPORT")
            report_content.append("=" * 60)
            report_content.append(f"Generated: {timestamp}")
            report_content.append(f"Source: {self.data_manager.file_path or 'No data loaded'}")
            report_content.append("")
            
            # Dataset Information
            report_content.append("DATASET INFORMATION")
            report_content.append("-" * 20)
            report_content.append(f"Rows: {data_info.get('rows', 'N/A')}")
            report_content.append(f"Columns: {data_info.get('columns', 'N/A')}")
            report_content.append(f"Memory Usage: {data_info.get('memory_usage', 0) / 1024:.2f} KB")
            report_content.append("")
            
            # Column Information
            report_content.append("COLUMNS")
            report_content.append("-" * 8)
            for col in data_info.get('column_names', []):
                dtype = data_info.get('data_types', {}).get(col, 'Unknown')
                report_content.append(f"  {col}: {dtype}")
            report_content.append("")
            
            # Missing Values
            if missing is not None and not missing.empty:
                report_content.append("MISSING VALUES")
                report_content.append("-" * 14)
                for col, count in missing.items():
                    report_content.append(f"  {col}: {count}")
                report_content.append("")
            
            # Duplicates
            if duplicates > 0:
                report_content.append("DUPLICATES")
                report_content.append("-" * 9)
                report_content.append(f"  Duplicate rows: {duplicates}")
                report_content.append("")
            
            # Descriptive Statistics
            if stats is not None:
                report_content.append("DESCRIPTIVE STATISTICS")
                report_content.append("-" * 21)
                report_content.append(stats.to_string())
                report_content.append("")
            
            # Correlation
            if correlation is not None:
                report_content.append("CORRELATION MATRIX")
                report_content.append("-" * 18)
                report_content.append(correlation.to_string())
                report_content.append("")
            
            # Save report
            filename = f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w') as file:
                file.write('\n'.join(report_content))
            
            logger.info(f"Summary report saved: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            return False


class DataAnalysisTool:
    """Main data analysis application"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.analyzer = Analyzer(self.data_manager)
        self.visualizer = Visualizer(self.data_manager)
        self.reporter = Reporter(self.data_manager, self.analyzer, self.visualizer)
        
        # Create required directories
        for directory in ['logs', 'results/charts', 'results/reports', 'datasets']:
            os.makedirs(directory, exist_ok=True)
    
    def load_dataset(self, file_path: str) -> bool:
        """Load dataset"""
        return self.data_manager.load_dataset(file_path)
    
    def show_data_info(self):
        """Display dataset information"""
        info = self.data_manager.get_info()
        if not info:
            print("No dataset loaded")
            return
        
        print(f"\nDataset Information:")
        print(f"  File: {self.data_manager.file_path or 'N/A'}")
        print(f"  Rows: {info.get('rows', 'N/A')}")
        print(f"  Columns: {info.get('columns', 'N/A')}")
        print(f"  Memory Usage: {info.get('memory_usage', 0) / 1024:.2f} KB")
        print(f"\nColumns:")
        for col in info.get('column_names', []):
            dtype = info.get('data_types', {}).get(col, 'Unknown')
            print(f"  {col}: {dtype}")
    
    def show_sample_data(self, n: int = 5):
        """Display sample data"""
        sample = self.data_manager.get_sample(n)
        if sample is not None:
            print(f"\nFirst {n} rows:")
            print(sample.to_string())
        else:
            print("No dataset loaded")
    
    def analyze_data(self):
        """Perform data analysis"""
        print("\nPerforming data analysis...")
        
        # Descriptive statistics
        stats = self.analyzer.descriptive_statistics()
        if stats is not None:
            print("\nDescriptive Statistics:")
            print(stats.to_string())
        
        # Missing values
        missing = self.analyzer.missing_values()
        if missing is not None and not missing.empty:
            print("\nMissing Values:")
            for col, count in missing.items():
                print(f"  {col}: {count}")
        
        # Duplicates
        duplicates = self.analyzer.duplicates()
        if duplicates > 0:
            print(f"\nDuplicate Rows: {duplicates}")
    
    def visualize_data(self):
        """Create visualizations"""
        if self.data_manager.data is None:
            print("No dataset loaded")
            return
        
        columns = self.data_manager.get_column_names()
        if not columns:
            print("No columns available for visualization")
            return
        
        print(f"\nAvailable columns: {', '.join(columns)}")
        
        print("\nVisualization Options:")
        print("1. Histogram")
        print("2. Scatter Plot")
        print("3. Correlation Heatmap")
        
        choice = input("Choose visualization type (1-3): ").strip()
        
        if choice == '1':
            col = input("Enter column name: ").strip()
            if col in columns:
                if self.visualizer.histogram(col):
                    print("Histogram created successfully!")
                else:
                    print("Failed to create histogram")
            else:
                print("Invalid column name")
        
        elif choice == '2':
            x_col = input("Enter X-axis column: ").strip()
            y_col = input("Enter Y-axis column: ").strip()
            if x_col in columns and y_col in columns:
                if self.visualizer.scatter_plot(x_col, y_col):
                    print("Scatter plot created successfully!")
                else:
                    print("Failed to create scatter plot")
            else:
                print("Invalid column names")
        
        elif choice == '3':
            if self.visualizer.correlation_heatmap():
                print("Correlation heatmap created successfully!")
            else:
                print("Failed to create correlation heatmap")
        
        else:
            print("Invalid choice")
    
    def generate_report(self):
        """Generate analysis report"""
        print("Generating analysis report...")
        if self.reporter.generate_summary_report():
            print("Report generated successfully!")
        else:
            print("Failed to generate report")
    
    def export_results(self):
        """Export results"""
        if self.data_manager.data is None:
            print("No dataset loaded")
            return
        
        print("\nExport Options:")
        print("1. Export current dataset")
        print("2. Export analysis results")
        
        choice = input("Choose export option (1-2): ").strip()
        
        if choice == '1':
            filename = input("Enter filename (with extension .csv/.json/.xlsx): ").strip()
            try:
                _, ext = os.path.splitext(filename)
                if ext.lower() == '.csv':
                    self.data_manager.data.to_csv(filename, index=False)
                elif ext.lower() == '.json':
                    self.data_manager.data.to_json(filename, indent=2)
                elif ext.lower() in ['.xlsx', '.xls']:
                    self.data_manager.data.to_excel(filename, index=False)
                else:
                    print("Unsupported file format")
                    return
                print(f"Dataset exported to: {filename}")
            except Exception as e:
                print(f"Error exporting dataset: {e}")
        
        elif choice == '2':
            print("Analysis results export would be implemented here")
        else:
            print("Invalid choice")


def main():
    """Main application loop"""
    tool = DataAnalysisTool()
    
    # Create sample dataset if it doesn't exist
    sample_file = 'datasets/sales.csv'
    if not os.path.exists(sample_file):
        create_sample_dataset(sample_file)
    
    while True:
        print("\n=== Data Analysis Tool ===")
        print("1. Load Dataset")
        print("2. View Data Info")
        print("3. View Sample Data")
        print("4. Clean Data")
        print("5. Analyze Data")
        print("6. Visualize Data")
        print("7. Generate Report")
        print("8. Export Results")
        print("9. Settings")
        print("10. Exit")
        
        choice = input("\nEnter your choice (1-10): ").strip()
        
        if choice == '1':
            file_path = input("Enter dataset path: ").strip()
            if file_path:
                if tool.load_dataset(file_path):
                    print("Dataset loaded successfully!")
                else:
                    print("Failed to load dataset")
            else:
                print("Invalid file path")
        
        elif choice == '2':
            tool.show_data_info()
        
        elif choice == '3':
            n_str = input("Number of rows to show (default 5): ").strip()
            n = int(n_str) if n_str.isdigit() else 5
            tool.show_sample_data(n)
        
        elif choice == '4':
            print("Data cleaning would be implemented here")
            # This would handle missing values, duplicates, etc.
        
        elif choice == '5':
            tool.analyze_data()
        
        elif choice == '6':
            tool.visualize_data()
        
        elif choice == '7':
            tool.generate_report()
        
        elif choice == '8':
            tool.export_results()
        
        elif choice == '9':
            print("Settings would be implemented here")
            # This would handle configuration options
        
        elif choice == '10':
            print("Thank you for using Data Analysis Tool!")
            break
        
        else:
            print("Invalid choice. Please try again.")


def create_sample_dataset(filename: str):
    """Create a sample dataset for demonstration"""
    try:
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
            'sales': np.random.normal(1000, 200, n_samples),
            'quantity': np.random.poisson(15, n_samples),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_samples),
            'customer_age': np.random.randint(18, 80, n_samples),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'discount': np.random.uniform(0, 0.3, n_samples)
        }
        
        # Ensure sales and quantity are positive
        data['sales'] = np.abs(data['sales'])
        data['quantity'] = np.abs(data['quantity']) + 1
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"Sample dataset created: {filename}")
    except Exception as e:
        logger.error(f"Error creating sample dataset: {e}")


if __name__ == "__main__":
    main()