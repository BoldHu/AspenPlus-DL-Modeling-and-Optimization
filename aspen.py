import random
class aspen_model(object):
    def __init__(self,aspen):
        self.aspen = aspen
        self.init_setting()
    
    def init_setting(self):
        # read the reactor temperature, path is '\Data\Blocks\HEAT1\Input\TEMP'
        self.reactor_temperature = []
        for i in range(1, 5):
            self.reactor_temperature.append(self.aspen.Tree.FindNode(r'\Data\Blocks\HEAT' + str(i) + '\Input\TEMP').Value)
        
        # read the reactor pressure, path is '\Data\Blocks\SEC1\Input\PRES'
        self.reactor_pressure = []
        for i in range(1, 5):
            self.reactor_pressure.append(self.aspen.Tree.FindNode(r'\Data\Blocks\SEC' + str(i) + '\Input\PRES').Value)
        
        # read the feed flow rate, path is '\Data\Streams\FEED-101\Input\TOTFLOW\MIXED'
        self.feed_flow_rate = self.aspen.Tree.FindNode(r'\Data\Streams\FEED-101\Input\TOTFLOW\MIXED').Value
        
        # read the split ratio, path is 'Data\Blocks\B3\Input\FRAC\14'
        self.split_ratio = self.aspen.Tree.FindNode(r'\Data\Blocks\B3\Input\FRAC\14').Value
        
    def change_parameter(self):
        # the function change the parameter simultaneously and randomly
        # change the reactor temperature, from 470 to 600
        for i in range(1, 5):
            self.aspen.Tree.FindNode(r'\Data\Blocks\HEAT' + str(i) + '\Input\TEMP').Value = random.randint(470, 600)
        
        # change the reactor pressure, from 0
        
        
        
        
                
