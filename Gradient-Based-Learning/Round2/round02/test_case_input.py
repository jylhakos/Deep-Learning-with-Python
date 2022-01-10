
class TestCaseInput:
    def __init__(self, x, y, weight, lrate):
        self.x = x
        self.y = y
        self.lrate = lrate
        self.weight = weight

    def __str__(self):
        return f'''
 <p>x = {self.x}</p>
 <p>y = {self.y}</p>
 <p>lrate = {self.lrate}</p>
 <p>weight = {self.weight}</p>       
        '''
