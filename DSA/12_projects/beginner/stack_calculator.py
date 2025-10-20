"""
Beginner Project: Stack-Based Calculator

This project implements a calculator that uses stack data structures to evaluate
mathematical expressions in postfix notation (Reverse Polish Notation).

Concepts covered:
- Stack data structure implementation
- Expression parsing and evaluation
- Algorithm design and implementation
- Error handling and validation
"""

from typing import List, Union


class Stack:
    """
    Stack implementation using Python list as underlying data structure.
    
    Time Complexities:
    - Push: O(1) amortized
    - Pop: O(1)
    - Peek: O(1)
    - IsEmpty: O(1)
    """
    
    def __init__(self):
        self.items = []
    
    def push(self, item) -> None:
        """Push item onto stack."""
        self.items.append(item)
    
    def pop(self):
        """Pop item from stack."""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items.pop()
    
    def peek(self):
        """Peek at top item without removing it."""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items[-1]
    
    def is_empty(self) -> bool:
        """Check if stack is empty."""
        return len(self.items) == 0
    
    def size(self) -> int:
        """Get size of stack."""
        return len(self.items)
    
    def __str__(self) -> str:
        """String representation of stack."""
        return f"Stack({self.items})"


class StackCalculator:
    """
    Calculator that evaluates postfix expressions using stack data structure.
    """
    
    def __init__(self):
        self.stack = Stack()
    
    def is_operator(self, token: str) -> bool:
        """Check if token is an operator."""
        return token in ['+', '-', '*', '/', '^', '**']
    
    def is_number(self, token: str) -> bool:
        """Check if token is a number."""
        try:
            float(token)
            return True
        except ValueError:
            return False
    
    def apply_operator(self, operator: str, operand2: float, operand1: float) -> float:
        """
        Apply operator to two operands.
        
        Args:
            operator: Mathematical operator
            operand2: Second operand (popped first)
            operand1: First operand (popped second)
        """
        if operator == '+':
            return operand1 + operand2
        elif operator == '-':
            return operand1 - operand2
        elif operator == '*':
            return operand1 * operand2
        elif operator == '/':
            if operand2 == 0:
                raise ValueError("Division by zero")
            return operand1 / operand2
        elif operator in ['^', '**']:
            return operand1 ** operand2
        else:
            raise ValueError(f"Unknown operator: {operator}")
    
    def evaluate_postfix(self, expression: str) -> float:
        """
        Evaluate postfix expression using stack.
        
        Time Complexity: O(n) where n is number of tokens
        Space Complexity: O(n) for stack storage
        """
        # Clear stack
        self.stack = Stack()
        
        # Tokenize expression
        tokens = expression.split()
        
        for token in tokens:
            if self.is_number(token):
                # Push number onto stack
                self.stack.push(float(token))
            elif self.is_operator(token):
                # Check if we have enough operands
                if self.stack.size() < 2:
                    raise ValueError(f"Insufficient operands for operator '{token}'")
                
                # Pop two operands (note order!)
                operand2 = self.stack.pop()
                operand1 = self.stack.pop()
                
                # Apply operator and push result
                result = self.apply_operator(token, operand2, operand1)
                self.stack.push(result)
            else:
                raise ValueError(f"Invalid token: {token}")
        
        # Check if we have exactly one result
        if self.stack.size() != 1:
            raise ValueError("Invalid expression: incorrect number of operands")
        
        return self.stack.pop()
    
    def infix_to_postfix(self, expression: str) -> str:
        """
        Convert infix expression to postfix using Shunting Yard algorithm.
        
        Time Complexity: O(n) where n is expression length
        Space Complexity: O(n) for output and operator stack
        """
        # Operator precedence
        precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3, '**': 3}
        right_associative = {'^', '**'}
        
        output = []
        operator_stack = Stack()
        
        # Tokenize expression (simple tokenization)
        tokens = self.tokenize_infix(expression)
        
        for token in tokens:
            if self.is_number(token):
                output.append(token)
            elif token == '(':
                operator_stack.push(token)
            elif token == ')':
                # Pop operators until we find opening parenthesis
                while not operator_stack.is_empty() and operator_stack.peek() != '(':
                    output.append(operator_stack.pop())
                if operator_stack.is_empty():
                    raise ValueError("Mismatched parentheses")
                operator_stack.pop()  # Remove opening parenthesis
            elif self.is_operator(token):
                # Pop operators with higher or equal precedence
                while (not operator_stack.is_empty() and 
                       operator_stack.peek() != '(' and
                       operator_stack.peek() in precedence and
                       (precedence[operator_stack.peek()] > precedence[token] or
                        (precedence[operator_stack.peek()] == precedence[token] and 
                         token not in right_associative))):
                    output.append(operator_stack.pop())
                operator_stack.push(token)
            else:
                raise ValueError(f"Invalid token: {token}")
        
        # Pop remaining operators
        while not operator_stack.is_empty():
            if operator_stack.peek() in ['(', ')']:
                raise ValueError("Mismatched parentheses")
            output.append(operator_stack.pop())
        
        return ' '.join(output)
    
    def tokenize_infix(self, expression: str) -> List[str]:
        """Tokenize infix expression."""
        tokens = []
        i = 0
        while i < len(expression):
            if expression[i].isspace():
                i += 1
            elif expression[i].isdigit() or expression[i] == '.':
                # Parse number
                start = i
                while i < len(expression) and (expression[i].isdigit() or expression[i] == '.'):
                    i += 1
                tokens.append(expression[start:i])
            elif expression[i:i+2] == '**':
                tokens.append('**')
                i += 2
            elif expression[i] in '+-*/^()':
                tokens.append(expression[i])
                i += 1
            else:
                raise ValueError(f"Invalid character: {expression[i]}")
        return tokens
    
    def evaluate_infix(self, expression: str) -> float:
        """
        Evaluate infix expression by converting to postfix first.
        
        Time Complexity: O(n) where n is expression length
        Space Complexity: O(n)
        """
        postfix = self.infix_to_postfix(expression)
        return self.evaluate_postfix(postfix)


def demonstrate_calculator():
    """Demonstrate stack-based calculator functionality."""
    print("=== Stack-Based Calculator Demo ===\n")
    
    calc = StackCalculator()
    
    # Test postfix evaluation
    print("1. Postfix Expression Evaluation:")
    postfix_expressions = [
        "3 4 +",           # 3 + 4 = 7
        "2 3 4 + *",       # 2 * (3 + 4) = 14
        "15 7 1 1 + - / 3 * 2 1 1 + + -",  # Complex expression
        "5 1 2 + 4 * + 3 -"  # 5 + ((1 + 2) * 4) - 3 = 14
    ]
    
    for expr in postfix_expressions:
        try:
            result = calc.evaluate_postfix(expr)
            print(f"  {expr} = {result}")
        except Exception as e:
            print(f"  {expr} = Error: {e}")
    
    # Test infix to postfix conversion
    print("\n2. Infix to Postfix Conversion:")
    infix_expressions = [
        "3 + 4",
        "2 * (3 + 4)",
        "(15 / (7 - (1 + 1))) * 3 - (2 + (1 + 1))",
        "5 + (1 + 2) * 4 - 3"
    ]
    
    for expr in infix_expressions:
        try:
            postfix = calc.infix_to_postfix(expr)
            result = calc.evaluate_postfix(postfix)
            print(f"  {expr} -> {postfix} = {result}")
        except Exception as e:
            print(f"  {expr} -> Error: {e}")
    
    # Test direct infix evaluation
    print("\n3. Direct Infix Evaluation:")
    for expr in infix_expressions:
        try:
            result = calc.evaluate_infix(expr)
            print(f"  {expr} = {result}")
        except Exception as e:
            print(f"  {expr} = Error: {e}")
    
    # Test stack operations
    print("\n4. Stack Operations:")
    stack = Stack()
    operations = [1, 2, 3, '+', 4, '*']
    print(f"  Operations: {operations}")
    
    for op in operations:
        if isinstance(op, (int, float)):
            stack.push(op)
            print(f"    Push {op}: {stack}")
        elif op == '+':
            if stack.size() >= 2:
                b = stack.pop()
                a = stack.pop()
                result = a + b
                stack.push(result)
                print(f"    {a} + {b} = {result}: {stack}")
        elif op == '*':
            if stack.size() >= 2:
                b = stack.pop()
                a = stack.pop()
                result = a * b
                stack.push(result)
                print(f"    {a} * {b} = {result}: {stack}")


def performance_comparison():
    """Compare performance of different calculator operations."""
    import time
    
    print("\n=== Performance Comparison ===\n")
    
    calc = StackCalculator()
    
    # Test complex postfix expression
    complex_postfix = "1 2 + 3 4 + * 5 6 + 7 8 + * + 9 10 + 11 12 + * +"
    
    print("1. Postfix Evaluation Performance:")
    start_time = time.time()
    for _ in range(10000):
        result = calc.evaluate_postfix(complex_postfix)
    postfix_time = time.time() - start_time
    print(f"   Time for 10,000 evaluations: {postfix_time:.6f} seconds")
    print(f"   Average time per evaluation: {postfix_time/10000:.8f} seconds")
    
    # Test infix to postfix conversion
    complex_infix = "(1 + 2) * (3 + 4) + (5 + 6) * (7 + 8) + (9 + 10) * (11 + 12)"
    
    print("\n2. Infix to Postfix Conversion Performance:")
    start_time = time.time()
    for _ in range(10000):
        postfix = calc.infix_to_postfix(complex_infix)
    conversion_time = time.time() - start_time
    print(f"   Time for 10,000 conversions: {conversion_time:.6f} seconds")
    print(f"   Average time per conversion: {conversion_time/10000:.8f} seconds")
    
    # Test direct infix evaluation
    print("\n3. Direct Infix Evaluation Performance:")
    start_time = time.time()
    for _ in range(10000):
        result = calc.evaluate_infix(complex_infix)
    infix_time = time.time() - start_time
    print(f"   Time for 10,000 evaluations: {infix_time:.6f} seconds")
    print(f"   Average time per evaluation: {infix_time/10000:.8f} seconds")


if __name__ == "__main__":
    demonstrate_calculator()
    performance_comparison()