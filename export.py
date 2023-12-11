class exporter(object):
    def __init__(self, aspen) -> None:
        self.aspen = aspen
        self.read_result_material_list()
    
    def read_result_material_list(self):
        # read the .txt file of properties_material_list to the list
        with open('result_material_list.txt', 'r') as f:
            for line in f:
                self.result_material_list.append(line.strip())
        return self.result_material_list
    
    def read_stream_result(self):
        self.total_result = 0
        stream_id = 10  # Set the stream_id to 10
        for material in self.result_material_list:
            try:
                path = f"Data\\Streams\\{stream_id}\\Output\\MASSFLOW3\\{material}"
                node = self.aspen.Tree.FindNode(path)
                if node is None:
                    raise ValueError(f"Could not find node at path {path}")
                aromatics_component = node.Value
                self.total_result += aromatics_component
            except Exception as e:
                print(f"Could not retrieve data for stream {stream_id}. Error: {e}")
        return self.total_result