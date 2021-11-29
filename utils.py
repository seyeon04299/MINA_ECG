import os
import numpy as np
import pandas as pd
import time
from datetime import datetime
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F 
import collections
import warnings
from torch.utils.data import WeightedRandomSampler
from metrics import task_eval, draw_reliability_graph




############### Progress check ###############
class ProgressChecker(object): 
    def __init__(self, total_batch, mode= 'train', init_time=None, output=2): ## [0,1], interval 0.1
        self.total_batch = total_batch
        self.chk_point = list(np.round(np.arange(0.1,1,0.1),1))
        self.loss_epoch =0
        self.loss_batch = 0
        self.y_label = np.empty((0,1))
        self.y_prob = np.empty((0,output))
        self.init_time = init_time
        self.mode = mode 
        self.output =output
        self.contents, self.full_metrics = None, None
        
    def __call__(self, ep, batch, y_label, y_prob, loss, others=None):
        self.y_label = np.concatenate((self.y_label,to_CPU(y_label).numpy()), axis=0)
        self.y_prob = np.concatenate((self.y_prob,to_CPU(y_prob).numpy()), axis=0)
        self.loss_batch += loss
        self.loss_epoch += loss

        rate = np.floor(batch/self.total_batch * 10)/10
        if self.mode == 'train':
            if rate in self.chk_point:
                self.chk_point.remove(rate)
                contents, _ = task_eval(self.y_label, self.y_prob)
                if others is None:
                    print('{:}% {:}th epoch| loss: {:.4f} | {}'.format(rate*100, ep, self.loss_batch, contents))
                else:
                    other_str = ''
                    if isinstance(others, dict):
                        for key, value in others.items():
                            other_str += '{}: {:.4f}'.format(key, value.item())
                    elif isinstance(others, list):
                        for value in others:
                            other_str += '{:.4f} '.format( value.item())
                    print('{:}% {:}th epoch| loss: {:.4f} | {} | {}'.format(rate*100, ep, self.loss_batch, other_str, contents))
                
                ### reset 
                self.y_label = np.empty((0,1))
                self.y_prob = np.empty((0,self.output))
                self.loss_batch = 0

        if batch == self.total_batch: ## last batch 
            if self.mode not in ['train','calib']:
                self.contents, self.full_metrics = task_eval(self.y_label, self.y_prob, mode='test')
                print(self.contents)

            self.chk_point = list(np.round(np.arange(0.1,1,0.1),1))

            print('{} Epoch total loss:{:.3f} -- Elapsed time: {:5.3f}\n'\
                    .format(ep, self.loss_epoch, time.process_time() - self.init_time))
            

########## Early stopping  ###########
class EarlyStopping():
    def __init__(self, patience=5, threshold = 0.01):
        self._step = 0 
        #self._zero = 0
        self._same = patience ## 거의 변화가 없는 경우
        self.threshold = threshold 
        self._loss = float('inf')
        self.patience  = patience
 
    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                print('Early stopped by the instability....')
                return True
        elif abs(loss - self._loss) < self._loss*self.threshold:
            self._same += 1 
            if self._same > self.patience:
                print('Early stopped by unchangeability...')
                return True
        else:
            self._step = 0
            self._same = 0
        self._loss = loss # loss update 
        return False

######### LOSS #######
class FocalLoss(nn.Module):
    
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([1 - alpha, alpha]) ## positive weight
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


########### Calibration #######
class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
##############

def make_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_pth(path=None,trg_fname=None):
    if trg_fname is not None:
        if trg_fname.endswith('pth'):
            pth_fname = trg_fname
    else: 
        for d,_,f in os.walk(path): 
            for fname in f:
                if fname.endswith('pth'):
                    if trg_fname is not None:
                        if fname.find(trg_fname)>-1:
                            pth_fname= (os.path.join(d,fname))
                    else:
                        pth_fname= (os.path.join(d,fname))

    print(pth_fname)
    return pth_fname

######### GPU realated methods ############
def to_GPU(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.tensor(tensor, dtype = torch.float32) 
    elif isinstance(tensor, float):
        tensor = torch.tensor([tensor], dtype = torch.float32) 
    if torch.cuda.is_available(): 
        return tensor.cuda()
    else: 
        return tensor
    
def to_CPU(tensor):
    if tensor.is_cuda:
        return tensor.cpu().detach()
    else:
        return tensor.detach()

# from torch.nn.parallel import DistributedDataParallel
from torchsummary import summary
def model_on_GPU (model, gpus, input_shape=None, show=True):   
    ### For Summary, input shape should be torch.Size([1, 4000]), not torch.Size([1, 1, 4000])
    # input_shape = input_shape[1:3] if input_shape is not None else None
    
    num_gpus = len(gpus)
    print(num_gpus, 'GPUs used!')  
    if num_gpus > 1:
        print('set Parallel GPUs')
        model = torch.nn.DataParallel(model,device_ids=gpus).cuda()
    elif num_gpus > 0:  
        model = model.cuda()     
    else:
        if input_shape is not None:
            summary(model,input_shape)
        print("Model on Host memory")
    
    if input_shape is not None:
        #summary(model,input_shape) --> LSTM 에서 사이즈 구하는데에서 계속 에러
        pass
    if show: print(model)

    return model

def criterion_on_GPU (criterion):
    if torch.cuda.device_count() > 10: ## 현재 적용 X
        print('Loss fucntion parallelized')
        # criterion = DataParallelCriterion(criterion)
        return criterion.cuda()
    return criterion

def initialize_weights(net):
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            try: m.bias.data.zero_()
            except: pass
        elif isinstance(m, torch.nn.Conv1d):
            m.weight.data.normal_(0, 0.02)
            try: m.bias.data.zero_()
            except: pass
        elif isinstance(m, torch.nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            try: m.bias.data.zero_()
            except: pass
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0, 0.02)
            try: m.bias.data.zero_()
            except: pass



##### visualize results in GITLAB ##
def save_result(learning_data, model, solver_param, report):
    #### README.md 저장 ####
    file_md = 'README.md' 

    if not os.path.isfile(file_md):
        with open(file_md,'wt') as f: 
            f.write(f"**Label File Path**\n\n{learning_data.info_path}\n\n")
            f.write("**CASE**\n\n")
            for label, cnt in learning_data.count_label.items():
                f.write("Count of '{}': {}  \n".format(label,cnt))
            f.write("train: {:5d},  validation: {:5d},  test: {:5d}\n\n".format(len(learning_data.train_case), len(learning_data.valid_case), len(learning_data.test_case)))     
            f.write(f"""|log file name|mode|auroc|auprc|specificity|sensitivity|accuracy|precision|f1-score|
|--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|{solver_param['comment']}|{learning_data.mode}|{report.full_metrics['AUROC']}|{report.full_metrics['AUPRC']}|{report.full_metrics['specificity']}|{report.full_metrics['sensitivity']}|{report.full_metrics['accuracy']}|{report.full_metrics['precision']}|{report.full_metrics['f1_score']}|

-----
""")

    else:
        contents = "{}".format(open(file_md).read())
        contents = contents.split('\n-----\n')
        with open(file_md,'wt') as f:
            f.write(contents[0] + \
                f"|{solver_param['comment']}|{learning_data.mode}|{report.full_metrics['AUROC']}|{report.full_metrics['AUPRC']}|{report.full_metrics['specificity']}|{report.full_metrics['sensitivity']}|{report.full_metrics['accuracy']}|{report.full_metrics['precision']}|{report.full_metrics['f1_score']}|\n" \
                + "\n-----\n" + contents[1])

    with open(file_md,'a') as f:
        f.write(f"""---

<details>
<summary>{solver_param['comment']}</summary>
<div markdown="1">

- (메모)
- (메모)

</div>

<details>
<summary>parameters</summary>
<div markdown="1">

**in/out**

|model name|mode|
|:--:|:--:|:--:|:--:|:--:|
|{model['arch']._get_name()}|{learning_data.mode}|

**model common**

|hidden_size|dropout|learning_rate|num_epochs|batch_size|
|:--:|:--:|:--:|:--:|:--:|
|{model['param']['hidden_size']}|{model['param']['do']}|{solver_param['lr']}|{solver_param['num_epochs']}|{solver_param['batch_size']}|
 

</div>
</details>

</details>


""")


    #### result.txt 저장 ####
    file_txt = 'result.txt'
    with open(file_txt,'a') as f: 
        f.write("#"*20 + f"  {solver_param['comment']}  "+ "#"*20 + "\n")
        f.write(f"""auroc = {report.full_metrics['AUROC']}
auprc = {report.full_metrics['AUPRC']}
specificity = {report.full_metrics['specificity']}
sensitivity = {report.full_metrics['sensitivity']}
accuracy = {report.full_metrics['accuracy']}
precision = {report.full_metrics['precision']}
f1-score = {report.full_metrics['f1_score']}

model name = {model['arch']._get_name()}
mode = {learning_data.mode}
oversample_ratio = {learning_data.oversample_ratio}
num_workers = {solver_param['num_workers']}

hidden_size = {model['param']['hidden_size']}
dropout = {model['param']['do']}
learning_rate = {solver_param['lr']}
num_epochs = {solver_param['num_epochs']}
batch_size = {solver_param['batch_size']}
          

""")