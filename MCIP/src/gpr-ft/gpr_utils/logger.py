import os
import pickle
import json 

class ExpLogger():
    
    def __init__(self, param_dict, save_path="../logs/gpr-ft"):
        self.param_dict = param_dict
        self.save_path  = save_path
        self.logger = {}
        self.logger["n_updates"] = 0

        self._create_dirs()

    def _create_dirs(self):
        n = self.get_name()
        dir_name = f"{self.save_path}/{n}"
        self.dir_name = dir_name
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        
        plot_dir_name = f"{dir_name}/plots"
        self.plot_dir_name = plot_dir_name
        if not os.path.isdir(plot_dir_name):
            os.makedirs(plot_dir_name)
    
    def get_name(self):
        
        output_dim = ""
        if "OD" in self.param_dict: output_dim = f"OD={self.param_dict.OD}"
        name_suffix = self.param_dict.name_sufix if "name_sufix" in self.param_dict else ""
        return f"model={self.param_dict.model}{name_suffix}_loss={self.param_dict.loss}_opt={self.param_dict.optimizer}_blr={self.param_dict.base_lr}" + output_dim
    
    def save(self):
        with open(os.path.join(self.dir_name, f"{self.get_name()}.log"), 'wb') as handle:
            pickle.dump(self.logger, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def update(self, metric_dict, step_size=500):
        
        self.logger["n_updates"] += step_size
        
        for k,v in metric_dict.items():
          
            if k in self.logger:
                self.logger[k] += [v]
            else:
                self.logger[k] = [v]
        
        self.save()


class EvalLogger():
    
    def __init__(self, exp_logger, check_point_type, use_l2=True, write_logs=True):
        self.param_dict = exp_logger.param_dict
        self.save_path  = exp_logger.save_path
        self.logger = {}
        self.check_point_type = check_point_type
        self.dir_name = exp_logger.dir_name if use_l2 else os.path.join(exp_logger.dir_name, "noL2")
        self._create_dirs()
        self.write_logs = write_logs
    
    def get_name(self):
        #return "_".join(["{}={}".format(it[0], it[1]) for it in self.param_dict.items()]) 
        return f"model={self.param_dict.model}_loss={self.param_dict.loss}_opt={self.param_dict.optimizer}_blr={self.param_dict.base_lr}_data={self.param_dict.data}"

    def _create_dirs(self):
        n = self.get_name()
        
        plot_dir_name = f"{self.dir_name}/{self.check_point_type}_eval_plots"
        self.plot_dir_name = plot_dir_name
        if not os.path.isdir(plot_dir_name):
            os.makedirs(plot_dir_name)
    
    def save(self):
        if self.write_logs:
            with open(os.path.join(self.dir_name, f"eval_{self.check_point_type}.json"), 'w') as handle:
                json.dump(self.logger, handle)
    
    def update(self, metric_dict):
        
        for k,v in metric_dict.items():
          
            if k in self.logger:
                self.logger[k] += [v]
            else:
                self.logger[k] = [v]
        
        self.save()