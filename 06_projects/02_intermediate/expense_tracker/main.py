"""
Expense Tracker Application

This application allows users to track their personal expenses, categorize spending,
set budgets, and generate reports. It demonstrates file I/O, data management,
and object-oriented programming concepts.
"""

import csv
import json
import os
from datetime import datetime
from typing import List, Dict, Optional


class Expense:
    """Represents a single expense record"""
    
    def __init__(self, date: str, amount: float, category: str, description: str):
        self.date = date
        self.amount = amount
        self.category = category
        self.description = description
    
    def to_dict(self) -> Dict:
        """Convert expense to dictionary for serialization"""
        return {
            'date': self.date,
            'amount': self.amount,
            'category': self.category,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create expense from dictionary"""
        return cls(
            data['date'],
            float(data['amount']),
            data['category'],
            data['description']
        )
    
    def __str__(self):
        return f"{self.date} - ${self.amount:.2f} - {self.category} - {self.description}"


class Category:
    """Represents a spending category"""
    
    def __init__(self, name: str, budget: float = 0.0):
        self.name = name
        self.budget = budget
    
    def to_dict(self) -> Dict:
        """Convert category to dictionary for serialization"""
        return {
            'name': self.name,
            'budget': self.budget
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create category from dictionary"""
        return cls(
            data['name'],
            float(data.get('budget', 0.0))
        )


class Budget:
    """Represents a monthly budget"""
    
    def __init__(self, month: str, year: str, amount: float):
        self.month = month
        self.year = year
        self.amount = amount
    
    def to_dict(self) -> Dict:
        """Convert budget to dictionary for serialization"""
        return {
            'month': self.month,
            'year': self.year,
            'amount': self.amount
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create budget from dictionary"""
        return cls(
            data['month'],
            data['year'],
            float(data['amount'])
        )


class ExpenseTracker:
    """Main expense tracker application"""
    
    def __init__(self):
        self.expenses: List[Expense] = []
        self.categories: List[Category] = []
        self.budgets: List[Budget] = []
        self.data_file = 'expenses.csv'
        self.categories_file = 'categories.txt'
        self.budgets_file = 'budgets.json'
        self.reports_dir = 'reports'
        
        # Create reports directory if it doesn't exist
        if not os.path.exists(self.reports_dir):
            os.makedirs(self.reports_dir)
        
        # Load existing data
        self.load_data()
    
    def load_data(self):
        """Load data from files"""
        self.load_expenses()
        self.load_categories()
        self.load_budgets()
    
    def load_expenses(self):
        """Load expenses from CSV file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', newline='') as file:
                    reader = csv.DictReader(file)
                    self.expenses = [Expense.from_dict(row) for row in reader]
            except Exception as e:
                print(f"Error loading expenses: {e}")
    
    def load_categories(self):
        """Load categories from text file"""
        if os.path.exists(self.categories_file):
            try:
                with open(self.categories_file, 'r') as file:
                    lines = file.readlines()
                    self.categories = []
                    for line in lines:
                        parts = line.strip().split('|')
                        if len(parts) >= 1:
                            name = parts[0]
                            budget = float(parts[1]) if len(parts) > 1 else 0.0
                            self.categories.append(Category(name, budget))
            except Exception as e:
                print(f"Error loading categories: {e}")
    
    def load_budgets(self):
        """Load budgets from JSON file"""
        if os.path.exists(self.budgets_file):
            try:
                with open(self.budgets_file, 'r') as file:
                    data = json.load(file)
                    self.budgets = [Budget.from_dict(item) for item in data]
            except Exception as e:
                print(f"Error loading budgets: {e}")
    
    def save_data(self):
        """Save all data to files"""
        self.save_expenses()
        self.save_categories()
        self.save_budgets()
    
    def save_expenses(self):
        """Save expenses to CSV file"""
        try:
            with open(self.data_file, 'w', newline='') as file:
                if self.expenses:
                    writer = csv.DictWriter(file, fieldnames=['date', 'amount', 'category', 'description'])
                    writer.writeheader()
                    for expense in self.expenses:
                        writer.writerow(expense.to_dict())
        except Exception as e:
            print(f"Error saving expenses: {e}")
    
    def save_categories(self):
        """Save categories to text file"""
        try:
            with open(self.categories_file, 'w') as file:
                for category in self.categories:
                    file.write(f"{category.name}|{category.budget}\n")
        except Exception as e:
            print(f"Error saving categories: {e}")
    
    def save_budgets(self):
        """Save budgets to JSON file"""
        try:
            with open(self.budgets_file, 'w') as file:
                data = [budget.to_dict() for budget in self.budgets]
                json.dump(data, file, indent=2)
        except Exception as e:
            print(f"Error saving budgets: {e}")
    
    def add_expense(self, date: str, amount: float, category: str, description: str):
        """Add a new expense"""
        # Validate date format
        try:
            datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD")
            return False
        
        # Validate amount
        if amount <= 0:
            print("Amount must be positive")
            return False
        
        # Check if category exists
        if not any(c.name.lower() == category.lower() for c in self.categories):
            print(f"Category '{category}' not found. Please add it first.")
            return False
        
        expense = Expense(date, amount, category, description)
        self.expenses.append(expense)
        self.save_expenses()
        print("Expense added successfully!")
        return True
    
    def add_category(self, name: str, budget: float = 0.0):
        """Add a new category"""
        if any(c.name.lower() == name.lower() for c in self.categories):
            print(f"Category '{name}' already exists")
            return False
        
        category = Category(name, budget)
        self.categories.append(category)
        self.save_categories()
        print(f"Category '{name}' added successfully!")
        return True
    
    def view_expenses(self, limit: int = 10):
        """View recent expenses"""
        if not self.expenses:
            print("No expenses recorded yet")
            return
        
        print(f"\nRecent Expenses (Last {limit}):")
        print("-" * 60)
        # Sort by date (newest first)
        sorted_expenses = sorted(self.expenses, key=lambda x: x.date, reverse=True)
        
        for expense in sorted_expenses[:limit]:
            print(expense)
    
    def view_categories(self):
        """View all categories"""
        if not self.categories:
            print("No categories defined yet")
            return
        
        print("\nCategories:")
        print("-" * 30)
        for category in self.categories:
            budget_info = f" (Budget: ${category.budget:.2f})" if category.budget > 0 else ""
            print(f"{category.name}{budget_info}")
    
    def generate_monthly_report(self, month: str, year: str):
        """Generate monthly expense report"""
        # Filter expenses for the specified month and year
        monthly_expenses = [
            e for e in self.expenses
            if e.date.startswith(f"{year}-{month.zfill(2)}")
        ]
        
        if not monthly_expenses:
            print(f"No expenses found for {month}/{year}")
            return
        
        # Group by category
        category_totals = {}
        for expense in monthly_expenses:
            if expense.category in category_totals:
                category_totals[expense.category] += expense.amount
            else:
                category_totals[expense.category] = expense.amount
        
        # Generate report
        report_file = f"{self.reports_dir}/monthly_report_{year}_{month}.txt"
        try:
            with open(report_file, 'w') as file:
                file.write(f"Monthly Expense Report - {month}/{year}\n")
                file.write("=" * 40 + "\n\n")
                
                total = sum(category_totals.values())
                file.write(f"Total Expenses: ${total:.2f}\n\n")
                
                file.write("Category Breakdown:\n")
                file.write("-" * 25 + "\n")
                for category, amount in sorted(category_totals.items()):
                    percentage = (amount / total) * 100 if total > 0 else 0
                    file.write(f"{category}: ${amount:.2f} ({percentage:.1f}%)\n")
                
                file.write("\nDetailed Expenses:\n")
                file.write("-" * 25 + "\n")
                for expense in monthly_expenses:
                    file.write(f"{expense}\n")
            
            print(f"Report generated: {report_file}")
            
            # Also display to console
            print(f"\nMonthly Report - {month}/{year}")
            print("=" * 30)
            print(f"Total Expenses: ${total:.2f}\n")
            
            print("Category Breakdown:")
            for category, amount in sorted(category_totals.items()):
                percentage = (amount / total) * 100 if total > 0 else 0
                print(f"  {category}: ${amount:.2f} ({percentage:.1f}%)")
                
        except Exception as e:
            print(f"Error generating report: {e}")
    
    def set_budget(self, category_name: str, amount: float):
        """Set budget for a category"""
        category = next((c for c in self.categories if c.name.lower() == category_name.lower()), None)
        if not category:
            print(f"Category '{category_name}' not found")
            return False
        
        category.budget = amount
        self.save_categories()
        print(f"Budget for '{category_name}' set to ${amount:.2f}")
        return True
    
    def check_budget_status(self):
        """Check budget status for all categories"""
        if not self.categories:
            print("No categories defined")
            return
        
        current_month = datetime.now().strftime('%m')
        current_year = datetime.now().strftime('%Y')
        
        print("\nBudget Status:")
        print("-" * 40)
        
        for category in self.categories:
            if category.budget <= 0:
                continue
            
            # Calculate spending for this category this month
            monthly_spending = sum(
                e.amount for e in self.expenses
                if e.category.lower() == category.name.lower()
                and e.date.startswith(f"{current_year}-{current_month}")
            )
            
            remaining = category.budget - monthly_spending
            percentage = (monthly_spending / category.budget) * 100 if category.budget > 0 else 0
            
            status = "✅ Under Budget" if remaining >= 0 else "⚠️ Over Budget"
            print(f"{category.name}:")
            print(f"  Budget: ${category.budget:.2f}")
            print(f"  Spent: ${monthly_spending:.2f}")
            print(f"  Remaining: ${remaining:.2f}")
            print(f"  Status: {status} ({percentage:.1f}% used)")
            print()


def get_valid_date() -> str:
    """Get a valid date from user input"""
    while True:
        date_str = input("Date (YYYY-MM-DD) [today]: ").strip()
        if not date_str:
            return datetime.now().strftime('%Y-%m-%d')
        
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return date_str
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD")


def get_valid_amount() -> float:
    """Get a valid amount from user input"""
    while True:
        try:
            amount_str = input("Amount: ").strip()
            amount = float(amount_str)
            if amount <= 0:
                print("Amount must be positive")
                continue
            return amount
        except ValueError:
            print("Please enter a valid number")


def main():
    """Main application loop"""
    tracker = ExpenseTracker()
    
    # Default categories
    default_categories = ['Food', 'Groceries', 'Transportation', 'Entertainment', 'Utilities', 'Other']
    for category in default_categories:
        if not any(c.name == category for c in tracker.categories):
            tracker.add_category(category)
    
    while True:
        print("\n=== Personal Expense Tracker ===")
        print("1. Add Expense")
        print("2. View Expenses")
        print("3. View Categories")
        print("4. Add Category")
        print("5. Generate Report")
        print("6. Set Budget")
        print("7. Check Budget Status")
        print("8. Export Data")
        print("9. Import Data")
        print("10. Search Expenses")
        print("11. Exit")
        
        choice = input("\nEnter your choice (1-11): ").strip()
        
        if choice == '1':
            date = get_valid_date()
            amount = get_valid_amount()
            tracker.view_categories()
            category = input("Category: ").strip()
            description = input("Description: ").strip()
            tracker.add_expense(date, amount, category, description)
        
        elif choice == '2':
            limit_str = input("Number of recent expenses to show (default 10): ").strip()
            limit = int(limit_str) if limit_str.isdigit() else 10
            tracker.view_expenses(limit)
        
        elif choice == '3':
            tracker.view_categories()
        
        elif choice == '4':
            name = input("Category name: ").strip()
            budget_str = input("Monthly budget (optional): ").strip()
            budget = float(budget_str) if budget_str and budget_str.replace('.', '').isdigit() else 0.0
            tracker.add_category(name, budget)
        
        elif choice == '5':
            month = input("Month (1-12): ").strip()
            year = input("Year (YYYY): ").strip()
            if month.isdigit() and year.isdigit():
                tracker.generate_monthly_report(month, year)
            else:
                print("Please enter valid month and year")
        
        elif choice == '6':
            tracker.view_categories()
            category = input("Category: ").strip()
            amount = get_valid_amount()
            tracker.set_budget(category, amount)
        
        elif choice == '7':
            tracker.check_budget_status()
        
        elif choice == '8':
            print("Export functionality would be implemented here")
            # This would typically export to CSV or other formats
        
        elif choice == '9':
            print("Import functionality would be implemented here")
            # This would typically import from CSV or other formats
        
        elif choice == '10':
            print("Search functionality would be implemented here")
            # This would allow searching expenses by various criteria
        
        elif choice == '11':
            print("Thank you for using Expense Tracker!")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()