from models import * # MINA 모델을 적용
from data_loader import MINA_TwelveLeads, TwelveLeads, MINA_SingleLead,SingleLead# TwelveLeads #DualLeadsIdx 
from solver import Solver
from config import Config
from utils import *

def main():
    
    ### Not Test
    config = Config()
    print(config)

    
    if config.checkpoint is not None:
        # Pretrainmode를 사용
        checkpoint = torch.load(config.checkpoint)
        
        setattr(config,'pre_trained',True)
        setattr(config,'inplanes',checkpoint.inplanes)
        setattr(config,'kernel_size',checkpoint.kernel_size)
        setattr(config,'num_blocks',checkpoint.num_blocks)
        setattr(config,'model_output',checkpoint.model_output)
        setattr(config,'loss_lambda',checkpoint.loss_lambda)
        setattr(config,'num_layers',checkpoint.num_layers)
        setattr(config,'diminution',checkpoint.diminution)
        setattr(config,'hidden_size',checkpoint.hidden_size)
        setattr(config,'weight',checkpoint.weight)
        setattr(config,'temp',checkpoint.temp)
        setattr(config,'optim',checkpoint.optim)
        setattr(config,'n_split',checkpoint.n_split)

        print('NEW CONFIG SET')
        



    # data setup for internal validation
    dataset = MINA_SingleLead
    train_set = dataset(config.intv_data.train_case, mode = 'train')
    val_set = dataset(config.intv_data.valid_case,  mode = 'train')
    test_set = dataset(config.intv_data.test_case,  mode = 'test')
    input_shape = [x.shape for x in test_set[0]] if len(test_set[0]) >1 else test_set[0][0].shape
    setattr(config,'input_shape',input_shape[0])

    # set n_dim
    n_dim = (config.n_split*(config.input_shape[1]-1))/(2*config.n_split+1)
    setattr(config,'n_dim',n_dim)

    
    print('input_shapes = ', input_shape) 
    print('input_shape0 = ', input_shape[0])
    
    model = MINANet(config)
    #model = TwelveLeadsMINANET(config)
    #model = MINANet(config)
    # run
    print('begin solver')
    solver = Solver(model, config) 
    report = solver.fit(train_set = train_set, test_set = test_set, valid_set = val_set)
    
    # data setup for external validation
    if config.extv_data is not None: 
        ext_test_set = dataset(config.extv_data.test_case, mode = 'test')
        report = solver.fit(test_set = ext_test_set)

if __name__ == '__main__' :
    main()