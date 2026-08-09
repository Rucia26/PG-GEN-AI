from services.rectangle import Rectangle
length = float(input("Enter the length of the rectangle: "))
width = float(input("Enter the width of the rectangle: "))
rect = Rectangle(length, width)
rect.area()
rect.perimeter()

from calculator import Calculator
calculator = Calculator()
print(calculator.add(5, 3))
print(calculator.sub(5, 3))
