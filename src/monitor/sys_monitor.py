from hardware.sys_utils import training_latency
import os.path as osp
import os

class Sys_Monitor:
    """
    collect the validation loss of each client
    """
    def __init__(self,clients,log_path=None):
        self.clients=clients
        self.chosen_clients = []
        self.chosen_devices = []
        self.runtime_apps = []
        self.avail_perfs = []
        self.avail_mems = []
        # training latency
        self.training_time = []
        self.total_time = 0

        self.epochs = []
        
        # create log file
        self.log_path = log_path
        if self.log_path is not None:
            if not osp.exists(self.log_path):
                os.makedirs(self.log_path)
            self.tsv_file = osp.join(self.log_path, 'sys.log.tsv')
            self.pkl_file = osp.join(self.log_path, 'sys.pkl')
            with open(self.tsv_file, 'w') as wf:
                columns = ['epoch', 'training_time', 'total_time']
                wf.write('\t'.join(columns) + '\n')
        
    def collect(self,model,epoch=None,chosen_idxs=None,batch_size=None,iteration = None,log=False):
        if batch_size is not None and iteration is not None:
            # collect training latency
            train_time = []
            for n,c in enumerate(self.clients):
                if isinstance(model,list):
                    pass
    


    


    
    
    
        


    