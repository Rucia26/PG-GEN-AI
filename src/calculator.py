class Calculator:
    """Simple calculator class providing basic arithmetic operations."""

    def add(self, a, b):
        return a + b

    def sub(self, a, b):
        return a - b

    def mul(self, a, b):
        return a * b

    def div(self, a, b):
        if b == 0:
            raise ValueError("Division by zero")
        return a / b

    def pow(self, a, b):
        return a ** b

    def mod(self, a, b):
        return a % b
