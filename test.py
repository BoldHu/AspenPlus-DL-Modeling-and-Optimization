from export import export

class aspen_model():
    def __init__(self, aspen, export, change_list=['PRES', 'TEMP', 'FLOWRATE', 'SPLIT'], change_limit=0.50) -> None:
        self.aspen = aspen
        self.change_list = change_list
        self.change_limit = change_limit
        self.FLOWRATE_value = aspen.Tree.FindNode("\Data\Streams\FEED-101\Input\TOTFLOW\MIXED").Value
        self.FLOWRATE_value_initial = self.FLOWRATE_value
        self.split_fraction = aspen.Tree.FindNode(r"\Data\Blocks\B3\Input\FRAC\14").Value
        self.split_fraction_initial = 0.5
        self.setting_parameter = self.getting_parameter()
        self.initial_setting_parameter = self.initial_parameter()
        self.exporter = export
        self.stream_result = []

    def generate_data_list(self, value, change_limit=None):
        # generate a list which is from 95%~105% and 10 points
        value = float(value)
        if change_limit == None:
            difference = value * self.change_limit
            data_list = []
            if value == 0:
                # 0~5 20 points
                data_list = [i * 5 / 20 for i in range(1, 21)]
            else:
                for i in range(1, 11):
                    data_list.append(value - difference + i * difference / 5)
        else:
            difference = value * change_limit
            data_list = []
            if value == 0:
                # 0~5 20 points
                data_list = [i * 5 / 20 for i in range(1, 21)]
            else:
                for i in range(1, 21):
                    data_list.append(value - difference + i * difference / 10)
        return data_list
    
    def change_parameter_SEC(self, id_of_SEC, value):
        # change the setting parameter of SEC
        path_SEC = f"Data\Blocks\SEC{str(id_of_SEC)}\Input"
        node_SEC_PRES = self.aspen.Tree.FindNode(path_SEC + "\PRES")
        
        # change the value of SEC's PRES
        node_SEC_PRES.Value = value
    
    def change_parameter_HEAT(self, id_of_HEAT, value, change_operation):
        # change the setting parameter of HEAT
        path_HEAT = f"Data\Blocks\HEAT{str(id_of_HEAT)}\Input"
        node_HEAT_PRES = self.aspen.Tree.FindNode(path_HEAT + "\PRES")
        node_HEAT_TEMP = self.aspen.Tree.FindNode(path_HEAT + "\TEMP")
        
        # change the value of HEAT's PRES
        if change_operation == 'PRES':
            node_HEAT_PRES.Value = value
        elif change_operation == 'TEMP':
            node_HEAT_TEMP.Value = value
    
    def change_parameter_FLOWRATE(self, value):
        # change the setting parameter of FLOWRATE
        path_FLOWRATE = "\Data\Streams\FEED-101\Input\TOTFLOW\MIXED"
        node_FLOWRATE = self.aspen.Tree.FindNode(path_FLOWRATE)
        
        # change the value of FLOWRATE
        node_FLOWRATE.Value = value
        
    def change_paramter_split_fraction(self, value):
        # change the setting parameter of split fraction
        path_split_fraction = r"Data\Blocks\B3\Input\FRAC\14"
        node_split_fraction = self.aspen.Tree.FindNode(path_split_fraction)
        
        # change the value of split fraction
        node_split_fraction.Value = value
    
    def initial_parameter(self):
        self.initial_setting_parameter = []
        self.stream_result = []
        # initial FLOWRATE
        self.aspen.Tree.FindNode("\Data\Streams\FEED-101\Input\TOTFLOW\MIXED").Value = self.FLOWRATE_value_initial
        self.aspen.Tree.FindNode(r"\Data\Blocks\B3\Input\FRAC\14").Value = self.split_fraction_initial
        # read the initial value of the setting parameter in initial_setting_parameter.txt. The data is like: ['0', '549', '0.53']
        with open('initial_setting_parameter.txt', 'r') as f:
            for line in f:
                self.initial_setting_parameter.append(line.strip())
        for i in range(1, 5):
            path_SEC = f"Data\Blocks\SEC{str(i)}\Input"
            path_HEAT = f"Data\Blocks\HEAT{str(i)}\Input"
            
            node_SEC_PRES = self.aspen.Tree.FindNode(path_SEC + "\PRES")
            node_HEAT_PRES = self.aspen.Tree.FindNode(path_HEAT + "\PRES")
            node_HEAT_TEMP = self.aspen.Tree.FindNode(path_HEAT + "\TEMP")
            
            # reset the value of SEC and HEAT
            node_SEC_PRES.Value = float(self.initial_setting_parameter[3 * i - 3])
            node_HEAT_PRES.Value = float(self.initial_setting_parameter[3 * i + 1 - 3])
            node_HEAT_TEMP.Value = float(self.initial_setting_parameter[3 * i + 2 - 3])
        return self.initial_setting_parameter, self.FLOWRATE_value
            
    def run_change_simulation(self, id_of_SEC=None, id_of_HEAT=None, change_operation=None):
        # initial parameter
        self.initial_parameter()
        # change the setting parameter
        if id_of_SEC is not None:
            # change the setting parameter of SEC
            value_list = self.generate_data_list(self.initial_setting_parameter[3 * id_of_SEC - 3])
            for value in value_list:
                self.change_parameter_SEC(id_of_SEC, value)
                # run the simulation
                self.aspen.Engine.Run2()
                # save the result
                self.exporter.save_result(f'SEC{id_of_SEC}_value={value}')
                print(f'The process of SEC{id_of_SEC} has finished.')
            # reset the setting parameter
            self.initial_parameter()
        
        if id_of_HEAT is not None and change_operation == 'PRES':
            # change the setting parameter of HEAT of PRES
            value_list = self.generate_data_list(self.initial_setting_parameter[3 * id_of_HEAT - 2])
            for value in value_list:
                self.change_parameter_HEAT(id_of_HEAT, value, change_operation)
                # run the simulation
                self.aspen.Engine.Run2()
                # save the result
                self.exporter.save_result(f'HEAT_{change_operation}{id_of_HEAT}_value={value}')
                print(f'The process of HEAT_{change_operation}{id_of_HEAT} has finished.')
            # reset the setting parameter
            self.initial_parameter()
        
        if id_of_HEAT is not None and change_operation == 'TEMP':
            # change the setting parameter of HEAT of TEMP
            value_list = self.generate_data_list(self.initial_setting_parameter[3 * id_of_HEAT - 1])
            for value in value_list:
                self.change_parameter_HEAT(id_of_HEAT, value, change_operation)
                # run the simulation
                self.aspen.Engine.Run2()
                # save the result
                self.exporter.save_result(f'HEAT_{change_operation}{id_of_HEAT}_value={value}')
                print(f'The process of HEAT_{change_operation}_{id_of_HEAT} has finished.')
        
        if change_operation == 'FLOWRATE':
            # change the setting parameter of FLOWRATE
            value_list = self.generate_data_list(self.FLOWRATE_value, change_limit=0.5)
            for value in value_list:
                self.change_parameter_FLOWRATE(value)
                # run the simulation
                self.aspen.Engine.Run2()
                # save the result
                self.exporter.save_result(f'FLOWRATE_value={value}')
                print(f'The process of FLOWRATE has finished.')
        
        if change_operation == 'split_fraction':
            # change the setting parameter of split fraction, 0.1~0.9
            value_list = [0.1 * i for i in range(1, 10)]
            ratio_list = []
            for value in value_list:
                self.change_paramter_split_fraction(value)
                # run the simulation
                self.aspen.Engine.Run2()
                # save the result
                self.exporter.save_result(f'split_fraction_value={value}')
                print(f'The process of split_fraction has finished.')
                mass_hydrogen = self.aspen.Tree.FindNode(r"\Data\Streams\15\Output\MASSFLOW\MIXED\HYDROGEN").Value
                mass_all_flow = self.aspen.Tree.FindNode(r"\Data\Streams\30\Output\MASSFLMX\MIXED").Value
                hydrogen_ratio = mass_hydrogen / mass_all_flow
                ratio_list.append([value, mass_hydrogen, mass_all_flow, hydrogen_ratio])

            # save the hydrogen ratio to the data folder 
            self.exporter.save_hydrogen_ratio(ratio_list)        
            # get the input settiing parameters
            self.getting_parameter()
    
    def run_simulation(self):
        # run the simulation
        self.aspen.Engine.Run2()
        # save the result
        self.exporter.save_result('initial_result')
        print('The process of initial_result has finished.')
        # get the input settiing parameters
        self.getting_parameter()
    
    def getting_parameter(self):
        self.setting_parameter = []
        # get the input settiing parameters
        for i in range(1, 5):
            path_SEC = f"Data\Blocks\SEC{str(i)}\Input"
            path_HEAT = f"Data\Blocks\HEAT{str(i)}\Input"
            
            node_SEC_PRES = self.aspen.Tree.FindNode(path_SEC + "\PRES")
            node_HEAT_PRES = self.aspen.Tree.FindNode(path_HEAT + "\PRES")
            node_HEAT_TEMP = self.aspen.Tree.FindNode(path_HEAT + "\TEMP")
            
            # append to list with the name of SEC and HEAT and value of PRES and TEMP
            self.setting_parameter.append([f"SEC{i}_Pressure", node_SEC_PRES.Value])
            self.setting_parameter.append([f"HEAT{i}_Temperature", node_HEAT_PRES.Value])
            self.setting_parameter.append([f"HEAT{i}_Temperature", node_HEAT_TEMP.Value])
        self.setting_parameter.append(['FLOWRATE', self.FLOWRATE_value])
        self.setting_parameter.append(['split_fraction', self.split_fraction])
        return self.setting_parameter
    
    # run a simulation and get the temp of specific stream 28 C 2 1 3 4 5 7 8, path to node is like "\Data\Streams\28\Output\TEMP_OUT\MIXED"
    def get_stream_temp(self, stream_id=['28', 'C', '2', '1', '3', '4', '5', '7', '8']):
        # initial parameter
        self.initial_parameter()
        # run the simulation
        self.aspen.Engine.Run2()
        # get the temp of specific stream
        stream_temp = []
        for i in stream_id:
            path = f"Data\Streams\{i}\Output\TEMP_OUT\MIXED"
            node = self.aspen.Tree.FindNode(path)
            stream_temp.append([i, node.Value])
        
        self.exporter.save_stream_temp(stream_temp)
        print('The process of stream_temp has finished.')
        return stream_temp
    
    # run a simulation and get the maximum and minimum temp of SEC1-SEC4, from the node path like '\Data\Blocks\SEC1\Output\TMIN' and '\Data\Blocks\SEC1\Output\TMAX'
    def get_SEC_react_temp(self):
        # initial parameter
        self.initial_parameter()
        # run the simulation
        self.aspen.Engine.Run2()
        # get the temp of specific stream, by the format of [SEC1, 0.0, 10.0]
        SEC_react_temp = []
        for i in range(1, 5):
            path_min = f"Data\Blocks\SEC{str(i)}\Output\TMIN"
            path_max = f"Data\Blocks\SEC{str(i)}\Output\TMAX"
            node_min = self.aspen.Tree.FindNode(path_min)
            node_max = self.aspen.Tree.FindNode(path_max)
            SEC_react_temp.append([f'SEC{i}', node_min.Value, node_max.Value])
        self.exporter.save_SEC_temp(SEC_react_temp)
        print('The process of SEC_react_temp has finished.')
        return SEC_react_temp
        
        
