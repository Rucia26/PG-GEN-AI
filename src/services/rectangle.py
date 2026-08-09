class Rectangle:
    def __init__(self,length,width):
        self.length=length
        self.width=width
    def area(self):
        print("Area of rectangle is:",self.length*self.width)
    def perimeter(self):
        print("Perimeter of rectangle is:",2*(self.length+self.width))
