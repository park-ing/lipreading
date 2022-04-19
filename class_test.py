class test():
    def __init__(self):
        self.b = 10
    
    def change(self, b):
        self.b = b
        return self.b

x = test()
print(x.change(9))

print()
print()
