import yaml

class Param_manager():
        
    def get_data(self):
        return self.data
    
    def set_data(self, path):
        with open(path, 'r') as f:
            self.data = yaml.load(f, Loader=yaml.SafeLoader)

        # with open(self.data["sim_conf"], "r") as s, \
        #      open(self.data["agent_conf"], "r") as a, \
        #      open(self.data["model_conf"], "r") as m:
            
        #     file_list = [s, a, m]
            
        #     for i in file_list:
        #         self.data.update(yaml.load(i, Loader=yaml.SafeLoader))
            
            