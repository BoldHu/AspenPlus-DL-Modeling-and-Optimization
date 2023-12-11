import random
from export import exporter
import os

class aspen_model(object):
    def __init__(self,aspen):
        self.aspen = aspen
        self.init_setting()
        self.exporter = exporter(self.aspen)
        self.aromatics_yeild = 0
    
    def init_setting(self):
        # read the reactor temperature, path is '\Data\Blocks\HEAT1\Input\TEMP'
        self.reactor_temperature = []
        self.reactor_temperature_init = []
        for i in range(1, 5):
            self.reactor_temperature.append(self.aspen.Tree.FindNode(r'\Data\Blocks\HEAT' + str(i) + '\Input\TEMP').Value)
            self.reactor_temperature_init.append(self.aspen.Tree.FindNode(r'\Data\Blocks\HEAT' + str(i) + '\Input\TEMP').Value)
        
        # read the reactor pressure, path is '\Data\Blocks\SEC1\Input\PRES'
        self.reactor_pressure = []
        self.reactor_pressure_init = []
        for i in range(1, 5):
            self.reactor_pressure.append(self.aspen.Tree.FindNode(r'\Data\Blocks\SEC' + str(i) + '\Input\PRES').Value)
            self.reactor_pressure_init.append(self.aspen.Tree.FindNode(r'\Data\Blocks\SEC' + str(i) + '\Input\PRES').Value)
        
        # read the feed flow rate, path is '\Data\Streams\FEED-101\Input\TOTFLOW\MIXED'
        self.feed_flow_rate = self.aspen.Tree.FindNode(r'\Data\Streams\FEED-101\Input\TOTFLOW\MIXED').Value
        self.feed_flow_rate_init = self.aspen.Tree.FindNode(r'\Data\Streams\FEED-101\Input\TOTFLOW\MIXED').Value
        
        # read the split ratio, path is 'Data\Blocks\B3\Input\FRAC\14'
        self.split_ratio = self.aspen.Tree.FindNode(r'\Data\Blocks\B3\Input\FRAC\14').Value
        self.split_ratio_init = self.aspen.Tree.FindNode(r'\Data\Blocks\B3\Input\FRAC\14').Value
    
    def initialize(self):
        # init the reactor temperature
        for i in range(1, 5):
            self.aspen.Tree.FindNode(r'\Data\Blocks\HEAT' + str(i) + '\Input\TEMP').Value = self.reactor_temperature_init[i-1]
        # init the reactor pressure
        for i in range(1, 5):
            self.aspen.Tree.FindNode(r'\Data\Blocks\SEC' + str(i) + '\Input\PRES').Value = self.reactor_pressure_init[i-1]
        # init the feed flow rate
        self.aspen.Tree.FindNode(r'\Data\Streams\FEED-101\Input\TOTFLOW\MIXED').Value = self.feed_flow_rate_init
        # init the split ratio
        self.aspen.Tree.FindNode(r'\Data\Blocks\B3\Input\FRAC\14').Value = self.split_ratio_init
        
    def change_parameter(self):
        # the function change the parameter simultaneously and randomly
        # change the reactor temperature, from 470 to 600
        for i in range(1, 5):
            self.aspen.Tree.FindNode(r'\Data\Blocks\HEAT' + str(i) + '\Input\TEMP').Value = random.randint(470, 600)
            self.reactor_temperature[i-1] = self.aspen.Tree.FindNode(r'\Data\Blocks\HEAT' + str(i) + '\Input\TEMP').Value
        
        # change the reactor pressure, from 0 to 1 MPa
        for i in range(1, 5):
            self.aspen.Tree.FindNode(r'\Data\Blocks\SEC' + str(i) + '\Input\PRES').Value = random.randint(0, 1000) / 1000.0
            self.reactor_pressure[i-1] = self.aspen.Tree.FindNode(r'\Data\Blocks\SEC' + str(i) + '\Input\PRES').Value
        
        # change the feed flow rate, from 900 to 1100 kmol/h
        self.aspen.Tree.FindNode(r'\Data\Streams\FEED-101\Input\TOTFLOW\MIXED').Value = random.randint(900, 1100)
        self.feed_flow_rate = self.aspen.Tree.FindNode(r'\Data\Streams\FEED-101\Input\TOTFLOW\MIXED').Value
        
        # change the split ratio, from 0.4 to 0.6
        self.aspen.Tree.FindNode(r'\Data\Blocks\B3\Input\FRAC\14').Value = random.randint(400, 600) / 1000.0
        self.split_ratio = self.aspen.Tree.FindNode(r'\Data\Blocks\B3\Input\FRAC\14').Value
        
    def run_simulation(self, inter_num=10):
        # init the setting
        self.initialize()
        # run the simulation for inter_num times
        for i in range(inter_num):
            self.change_parameter()
            self.aspen.Engine.Run2()
        self.aromatics_yeild = self.exporter.read_stream_result()
        self.write_result()
        
    def write_result(self):
        # write the result to the .csv file in data folder by format of 'reactor_temperature, reactor_pressure, feed_flow_rate, split_ratio, aromatics_yeild'
        # if there is no result.csv file in data folder, create one
        if not os.path.exists('/data/result.csv'):
            # create the file by format of SEC1 Temperature	SEC2 Temperature	SEC3 Temperature	SEC4 Temperature	SEC1 Pressure	SEC2 Pressure	SEC3 Pressure	SEC4 Pressure	Flow Rate	split ratio	 sum of aromatics
            with open('data.csv', 'w') as f:
                f.write('SEC1 Temperature, SEC2 Temperature, SEC3 Temperature, SEC4 Temperature, SEC1 Pressure, SEC2 Pressure, SEC3 Pressure, SEC4 Pressure, Flow Rate, split ratio, sum of aromatics\n')
        # write the result to the .csv file
        for i in range(4):
            with open('data.csv', 'a') as f:
                f.write(str(self.reactor_temperature[i]) + ', ')
                f.write(str(self.reactor_pressure[i]) + ', ')
        f.write(str(self.feed_flow_rate) + ', ')
        f.write(str(self.split_ratio) + ', ')
        f.write(str(self.aromatics_yeild) + '\n')
        

        
        
        
        
                
