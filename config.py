import argparse
import os
import pandas as pd
import numpy as np

class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, config, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)


        for k, v in config.items():
            self[k] = v

        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        try:
            if attr in self:
                return self[attr]
            raise AttributeError
        except Exception as e :
            raise
        return None 

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]

class Config (Map):
    def __init__ (self, *args, **kwargs):
        config = get_args(argparse.ArgumentParser(description='')).parse_args().__dict__
        super(Map, self).__init__(config, *args, **kwargs)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpus
        print(self.gpus)
        print(self.data_path)
        self.num_gpus = len(self.gpus.split(','))
        self.gpus = [x for x in range(self.num_gpus)]
        
        # data split
        self.intv_data = SplitData(info_path=self.data_path, splt_mode = self.mode, oversample_ratio = self.oversample_ratio)
        self.extv_data = SplitData(info_path=self.ext_path, splt_mode = 'test') if self.ext_path is not None else None

######## data settting ###########
class SplitData(object):
    
    def __init__(self, info_path, splt_mode = 'intv', oversample_ratio = 0.5, random_state = 0):
        self.info_path = info_path
        self.data_info = pd.read_csv(info_path, dtype={'fname' : 'str'}, index_col=0).reset_index(drop=False)#.iloc[:10000] 
        self.splt_mode = splt_mode
        print('splt_mode: ',splt_mode)
        self.oversample_ratio=oversample_ratio
        self.random_state = random_state
        self.train_case, self.valid_case, self.test_case = None,None,None
        self.get()
        
    def get(self):
        print(self.info_path)
        if 'ptb-xl' in self.data_info['site'].unique():
            self._split_index_PTBXL()
            print('split PTBXL')
        if 'physionet' in self.data_info['site'].unique():
            self._split_index()
            print('split PHYSIONET')
        if ((self.splt_mode != 'test') & (self.oversample_ratio !=0)): 
            print('== Original dataset {} =='.format(len(self.data_info)))
            self._oversampling()

        print('=={}, Train cases: {}, Validation cases: {}, Test cases: {}==\n'.format(len(self.data_info),len(self.train_case),len(self.valid_case),len(self.test_case)))

    ## PTBXL용 _split_index(self): 만들기
    def _split_index_PTBXL(self):
        all_cases = np.array(self.data_info.index)    
        self.count_label = dict(self.data_info['label'].value_counts())        
        train_idx, val_idx, test_idx = [],[],[]

        if self.splt_mode == 'test':      
            test_idx = all_cases

        elif self.splt_mode == 'train': ## train only
            #ratio = [int(0.8*len(self.data_info))]
            train_idx = np.array(self.data_info.loc[self.data_info.strat_fold<=8].index)
            val_idx = np.array(self.data_info.loc[self.data_info.strat_fold>=9].index)
            test_idx = val_idx

        else: # == 'intv'
            train_idx = np.array(self.data_info.loc[self.data_info.strat_fold<=8].index)
            val_idx = np.array(self.data_info.loc[self.data_info.strat_fold==9].index)
            test_idx = np.array(self.data_info.loc[self.data_info.strat_fold==10].index)

        self.train_case= self.data_info.loc[train_idx]
        self.valid_case = self.data_info.loc[val_idx]
        self.test_case = self.data_info.loc[test_idx]

    def _split_index(self):
        all_cases = np.array(self.data_info.index)    
        self.count_label = dict(self.data_info['label'].value_counts())        
        train_idx, val_idx, test_idx = [],[],[]

        if self.splt_mode == 'test':      
            test_idx = all_cases

        elif self.splt_mode == 'train': ## train only
            ratio = [int(0.8*len(self.data_info))]
            train_idx, val_idx = np.split(all_cases,ratio)
            test_idx = val_idx

        else: # == 'intv'
            ratio = [int(0.7*len(self.data_info)),int(0.8*len(self.data_info))]
            train_idx, val_idx, test_idx = np.split(all_cases,ratio)

        print(train_idx, val_idx, test_idx)
        
        self.train_case= self.data_info.loc[train_idx]
        self.valid_case = self.data_info.loc[val_idx]
        self.test_case = self.data_info.loc[test_idx]

    def _oversampling(self):
        ## train over sampling ##
        labels = dict(self.train_case['label'].value_counts())
        major_label = max(labels, key = lambda x:labels[x])       
        
        minor_label_list = list(labels.keys())
        minor_label_list.remove(major_label)

        major_case = self.train_case[self.train_case['label']==major_label]
        train_augmented = major_case.copy()

        for label in minor_label_list:
            minor_case = self.train_case[self.train_case['label']==label]
            unbalance_rate = int((len(major_case)/len(minor_case))*self.oversample_ratio)
            if unbalance_rate ==0: 
                unbalance_rate = 1
                self.oversample_ratio = None
            train_augmented = pd.concat([train_augmented]+[minor_case]*unbalance_rate) # 단순 몇번 더 뽑는 oversampling

        self.train_case= train_augmented.sample(frac=1,random_state =self.random_state).reset_index(drop=True) 


############## args ##############
def get_args (parser):   

    ## model ##
    parser.add_argument('--checkpoint', type=str, help ='checkpoint', default=None)
    parser.add_argument('--inplanes', type=int, help ='inplanes', default=None)
    parser.add_argument('--kernel_size', type=int, help ='the size of kernel', default=None)
    parser.add_argument('--num_blocks', type=int, help ='num_blocks', default=None) 
    parser.add_argument('--model_output', type = int, help='output for model',default=None)  
    parser.add_argument('--loss_lambda', type = float, help='loss_lambda',default=None)  
    parser.add_argument('--num_layers', type = int, help='number of layers',default=None)
    parser.add_argument('--diminution', type = int, help='diminution',default=None) 

    parser.add_argument('--n_channel', type=int, help ='number of channels', default=None)
    #parser.add_argument('--n_dim', type=int, help ='n_dim', default=None)
    parser.add_argument('--n_split', type=int, help ='number of splits', default=None)

    ## model - common ##
    parser.add_argument('--hidden_size', type = int, help='hidden',default=None)
    parser.add_argument('--dropout', type = float, help='drop_out',default=None)

    ## common - in/output ##
    parser.add_argument('--num_epochs', type = int, help='number of epochs', default=100)
    parser.add_argument('--oversample_ratio', type=float, help ='oversample ratio', default=0)
    parser.add_argument('--data_path', type=str, help='data information path', default = None)
    parser.add_argument('--ext_path', type=str, help='data information path', default = None)
    parser.add_argument('--lead_type', type=str, help='the type of leads', default = 'original')
    parser.add_argument('--mode', type = str, help='options: train=train only, test=test only, intv=internal validation', default='intv')
    parser.add_argument('--num_workers', type = int, help='number of data loaders', default=8)
    parser.add_argument('--gpus', type = str, help='used gpu (example:"0,1")', default='0,1')
    parser.add_argument('--num_classes', type = int, help='number of classes', default=2)
    parser.add_argument('--comment', type = str, help='note',default='') 
    parser.add_argument('--lr', type = float, help='learning rate',default=1e-4)
    parser.add_argument('--batch_size', type = int, help='batch size',default=256) 
    parser.add_argument('--gamma', type = float, help='focalloss_gamma',default=2) 
    parser.add_argument('--alpha', type = float, help='focalloss_alpha',default=0) 
    parser.add_argument('--calibration', type = int, help='calibration mode',default=1) 
    
    

    ## pretrined mode ##
    parser.add_argument('--pre_trained', type = str, default=None)

    return parser
