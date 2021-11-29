from utils import *
import inspect
from torch.utils.data import DataLoader
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
# from tensorboardX import SummaryWriter

class Solver(object): 
    def __init__(self, model, config):    
        self.config = config

        if config.pre_trained:
            model.load_state_dict(config.weight)
        
        ## set model on GPU
        self.m = model_on_GPU(model, self.config.gpus, self.config.input_shape)
        
        self.best_model = copy.deepcopy(self.m)
        self.criterion = self.config.criterion if hasattr(self.config,'criterion') else FocalLoss()
        self.optimizer = self.config.optimizer if hasattr(self.config,'optimizer') else torch.optim.Adam(self.m.parameters(), lr=self.config.lr, weight_decay=1e-4,)  

        self.model_name = model._get_name()  

            
        ## calibration
        self.calib = True if self.config.calibration ==1 else False
        if self.calib==True: 
            self.temperature = nn.Parameter(to_GPU(self.config.temp)) if hasattr(self.config,'temp') else nn.Parameter(to_GPU(torch.ones(1)))

        ## scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.num_epochs, eta_min=self.config.lr)
        self.path = os.path.join('./result', datetime.today().strftime("%Y%m%d"))
        
        ## tensorboard
        # self.summary = SummaryWriter(os.path.join('./runs',model['arch']._get_name()+'_'+self.config.comment']))

    def feed_foward (self, x, y = None, mode = 'train'): 
        #print('Solver feed_forward')
        x, y = to_GPU(x), to_GPU(y)
        if mode in ['test', 'calib']:
            y_prob,att_dic = self.best_model.forward(x)
        else:
            y_prob,att_dic = self.m.forward(x)

        loss = self.criterion(y_prob, y.long())
        return  y, y_prob, loss, att_dic


    def fit (self, train_set=None, test_set=None, valid_set=None, save=True): 
        pre_trained = True
        if train_set:
            pre_trained = False
            print('~'*20,'Training','~'*20) 
            self._training(train_set, valid_set)   
            if self.calib==True:
                self._calibration(valid_set)

        # self.summary.close() ## tensor board close
        if test_set:                
            print('~'*20,'Testing','~'*20)   
            report, att_dic = self._test(test_set)
            if save:
                self._save(report, test_set, pre_trained)
            return report
    
    
    def _training(self, train_set, valid_set):  

        batch_size=train_set.len if self.config.batch_size>train_set.len else self.config.batch_size
        early_stop = EarlyStopping(patience=5) ## stop earlier 
        loss_prev = float('inf') 
        best_perform = 0

        for ep in range(self.config.num_epochs):  
            self.m.train()
            
            ## random batch 
            if self.config.batch_size ==1:
                batch_size = int(2**(6+np.random.randint(0,5,1)[0])) # 12~22 중 정함
                if train_set.len//batch_size%self.config.num_gpus ==1:
                    batch_size = int(2**(6+np.random.randint(0,5,1)[0]))

            print('+++batch size {:} in {:} epoch+++'.format(batch_size,ep))

            train_loader = DataLoader(dataset=train_set, batch_size = batch_size, \
                                shuffle=False, num_workers =self.config.num_workers, drop_last=False,
                                sampler=train_set.sampler) 
            total_batch = train_set.len//batch_size
            
            prog_checker = ProgressChecker(total_batch, mode= 'train', init_time=time.process_time(), output=self.config.num_classes)

            for curr_batch, (x, y) in enumerate(train_loader):    
                ## feed foward 
                y, y_prob, loss, _ = self.feed_foward(x, y)
                ## backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step() 
                if self.scheduler is not None:
                    self.scheduler.step()

                prog_checker(ep, curr_batch, y, y_prob, loss.item())
             
            ## validation 
            if valid_set is not None:
                val_eval,att_dic = self._test(valid_set, mode='val')
                loss = val_eval.loss_epoch
                # if loss < loss_prev: 
                #     loss_prev = loss
                #     self.best_model = copy.deepcopy(self.m)
                #     print('keep model')

                ep_perform = val_eval.full_metrics['AUROC'] ## roc는 1에 가까울수록 좋은 성능                
                if best_perform < ep_perform: 
                    best_perform = ep_perform 
                    self.best_model = copy.deepcopy(self.m)
                    print('keep model')
                
                # early stopping
                elif early_stop.validate(loss):  
                    break  
                

    def _test(self, test_set, mode='test', att_show=False):
        
        batch_size = 16 if self.config.batch_size>test_set.len else self.config.batch_size
        test_loader = DataLoader(dataset=test_set, batch_size = batch_size, \
                                    num_workers = self.config.num_workers, drop_last=False)     

        self.m.eval()
        self.best_model.eval()
        prog_checker = ProgressChecker(test_set.len//batch_size, init_time=time.process_time(), mode= mode, output=self.config.num_classes)  ## check progress
 
        with torch.no_grad():
            for curr_batch, (x, y) in enumerate(test_loader):
                ## feed foward 
                y, y_prob, loss, att_dic = self.feed_foward(x, y, mode)
                # if curr_batch==1:
                #     print(att_dic['alpha_0'].shape)
                    
                if self.calib == True:
                    y_prob = self.__temperature_scale(y_prob)
                    sm = torch.nn.Softmax(dim=1)
                    y_prob = sm(y_prob)
                    
                if att_show:
                    pass

                prog_checker(mode, curr_batch, y, y_prob, loss.item())
            
            return prog_checker, att_dic

    def __temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def _calibration(self, calib_set): 
        prog,att_dic = self._test(calib_set, mode='calib')
        nll_criterion = torch.nn.CrossEntropyLoss()
        ece_criterion = ECELoss()

        logits = to_GPU(prog.y_prob)
        labels = to_GPU(prog.y_label).long()
        labels = labels.squeeze()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: {:.3f}, ECE: {:.3f}'.format(before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval():
            loss = nll_criterion(self.__temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.__temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.__temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: {:.3f}, ECE: {:.3f}'.format(after_temperature_nll, after_temperature_ece))


    def _save (self, report, test_set, pre_trained):  
        print('Start Save')
        path = dict()
        if pre_trained: self.config.comment+='_p'
        file_name = self.model_name+'_'+self.config.comment

        ## mkdirs
        for dir_name in ['model','prob','ref','perform']:
            print(os.path.join(self.path,dir_name))
            path[dir_name] = os.path.join(self.path,dir_name) 
            os.makedirs(path[dir_name], exist_ok=True)  #while making leaf directory if any intermediate-level directory is missing, os.makedirs() method will create them all.

        if not pre_trained:
            ### save model and solver information
            setattr(self.config, 'weight', self.best_model.module.state_dict() if self.config.num_gpus>1 else self.best_model.state_dict())
            setattr(self.config, 'temp', self.temperature.item())
            setattr(self.config, 'optim', self.optimizer.state_dict())
            print(path['model'],file_name+'.pth')
            torch.save(self.config, os.path.join(path['model'],file_name+'.pth')) 

        ### save predction result (.txt file)
        np.savetxt (os.path.join(path['prob'],file_name+'.txt'),\
                                np.column_stack((report.y_label, report.y_prob)), fmt='%.11f')

        ### performance (performan and calibration) 
        attr_dict = {}
        for k, v in inspect.getmembers(self.config):   
            if ((not k.startswith('_'))  and  (k not in ['weight', 'model','optim'])):  
                if not inspect.ismethod(v):   
                    attr_dict[k] = v 

        param_str = '\n'+pd.DataFrame.from_dict([attr_dict]).T.to_string()

        draw_reliability_graph(report.y_label[:,0], report.y_prob[:,1],\
             perform=report.contents+param_str, fname=os.path.join(path['perform'],file_name))

