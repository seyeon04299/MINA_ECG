import pywt
from utils import *
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from ecg_feature_selection import * 
#from classification_master.data_loader import SingleLead

from scipy.signal import butter, lfilter, periodogram


DATA_PATH = {#'sj':'home/ubuntu/data/ecg/sj/12_lead/npz',
            #'chapman':'/data/ecg/chapman/12_lead/npz',
            'ptb-xl':'/ubuntu/data/ecg/ptb-xl/12_lead/npz',
            #'cpsc':'/data/ecg/cpsc/12_lead/npz',
            #'georgia':'/data/complete/ecg/georgia/12_lead/npz',
            #'ptb': '/data/complete/ecg/ptb_db/12_lead/500hz/npz',
            'physionet': '/ubuntu/data/ecg/physionet/npz/'
            
            }





class TwelveLeadsWithThings (Dataset):
    '''
    본 데이터 셋은 original version의 12리드 or filtering version의 12리드만 출력하는 버전입니다. 
    output shape (batch, 12 , 4000) : batch size, num_leads, length
    '''
    def __init__(self, label_info, option=None, lead_type = 'original', mode ='train', norm_method = 'z-norm', filter_method='medi_simple', task='cls'):
        self.path = DATA_PATH
        self.label_info = label_info  ## dataframe 
        self.mode = mode 
        self.norm_method = norm_method
        self.filter_method = filter_method
        self.lead_type = lead_type
        self.len = len(self.label_info)
        if (mode =='train') & (len(self.label_info)>0):
            self.sampler = make_sampler(self.label_info['label'].values)
 
    def __getitem__(self, index):   
        row = self.label_info.iloc[index]
        age = row['age']
        gender = row['gender']
        y = row['label']
        npz_path = self.path[row['site']]       
        x1 = None

        if self.lead_type =='original':
            ori_full_path = os.path.join(npz_path, str(row['fname'])+'.npz')
            ecg = np.load(ori_full_path)['arr_0'].astype(float) # read 
            x1 = ecg_crop_norm(ecg)
        else:
            filt_full_path = os.path.join(npz_path+'_denoise', str(row['fname'])+'.npz')
            try: 
                x1 = ecg_crop_norm(np.load(filt_full_path)['arr_0'].astype(float))
                # x1 = ecg_start_rpeak_heartpy(np.load(filt_full_path)['arr_0'].astype(float))
            except:
                x1 = ecg_clean(ecg, sampling_rate=500, method=self.filter_method)
                x1 = ecg_crop_norm(x1, norm=self.norm_method)

            
        idx_permutation=[6,7,8,9,10,11,4,0,3,1,5,2] 

        x1 = x1[idx_permutation,:]
        x1[8,:] *= (-1) # aVR * (-1)

        
        return (torch.tensor(np.expand_dims(x1, axis=0), dtype = torch.float32), torch.tensor([age, gender], dtype = torch.float32)), torch.tensor([y], dtype = torch.float32)  # x1 =[minibatch,1,12,4000], x2 = [minibatch,1,2], y = [minibatch,1]
        
    def __len__(self):
        return self.len


class TwelveLeads (Dataset): 
    def __init__(self, label_info, option=None, mode ='train'):
        self.path = DATA_PATH
        self.label_info = label_info  ## dataframe 
        self.mode = mode 
        self.option = option
        self.len = len(self.label_info)     
        if (mode =='train') & (len(self.label_info)>0):
            self.sampler = make_sampler(self.label_info['label'].values)
 
    def __getitem__(self, index):   
        row = self.label_info.iloc[index] 
        label = row['label']
        npz_path = self.path[row['site']]
        ecg = np.load(os.path.join(npz_path, row['fname']+'.npz'))['arr_0'].astype(float) # read        
        
        ecg_norm = ecg_crop_norm(ecg,norm='all-z_norm')
        if self.option in ['gan', 'GAN']:
            ecg_norm = ecg_norm[:,:2048]

        return torch.tensor(np.expand_dims(ecg_norm,0), dtype = torch.float32), torch.tensor([label],dtype = torch.float32)
        
    def __len__(self):
        return self.len


class UnSyncTwoLeads (Dataset): 
    def __init__(self, label_info, option=None, mode ='train'):
        self.path = DATA_PATH
        self.label_info = label_info  ## dataframe 
        self.mode = mode 
        self.option = option
        self.len = len(self.label_info)  
        if (mode =='train') & (len(self.label_info)>0):
            self.sampler = make_sampler(self.label_info['label'].values)
 
    def __getitem__(self, index):   
        row = self.label_info.iloc[index] 
        label = row['label']
        npz_path = self.path[row['site']]
        ecg = np.load(os.path.join(npz_path, row['fname']+'.npz'))['arr_0'].astype(float) # read        
        
        ecg_norm = ecg_crop_norm(ecg,norm='all-z_norm')
        if self.option in ['gan', 'GAN']:
            offset = int(0.5*500)
            ecg_norm_l1 = ecg_norm[0,:2048]
            ecg_norm_l2 = ecg_norm[1,offset:offset+2048] 
            ecg_norm = np.stack((ecg_norm_l1,ecg_norm_l2))

        return torch.tensor(ecg_norm, dtype = torch.float32), torch.tensor([label],dtype = torch.float32)
        
    def __len__(self):
        return self.len



class SixLeads (Dataset): 
    def __init__(self, label_info, option = 'limb', mode ='train'):
        self.path = DATA_PATH
        self.label_info = label_info  ## dataframe 
        self.option = option
        self.mode = mode 
        self.len = len(self.label_info)  
        if (mode =='train') & (len(self.label_info)>0):
            self.sampler = make_sampler(self.label_info['label'].values)
 
    def __getitem__(self, index):   
        row = self.label_info.iloc[index] 
        label = row['label']
        npz_path = self.path[row['site']]
        ecg = np.load(os.path.join(npz_path, row['fname']+'.npz'))['arr_0'].astype(float) # read        

        ecg_norm = ecg_crop_norm(ecg,norm='all-z_norm')

        if self.option =='limb':
            ecg_norm = ecg_norm[:6,:]
        elif self.option == 'chest':
            ecg_norm = ecg_norm[6:,:]
        else:
            raise TypeError('only [limb] and [chest]')

        return torch.tensor(np.expand_dims(ecg_norm,0), dtype = torch.float32), torch.tensor([label],dtype = torch.float32)
        
    def __len__(self):
        return self.len


class SingleLead (Dataset): 
    def __init__(self, label_info, option=0, mode ='train'):
        self.path = DATA_PATH
        self.option = option if isinstance (option, int) else 0
        self.label_info = label_info  ## dataframe 
        self.mode = mode
        self.len = len(self.label_info)
        if (mode =='train') & (len(self.label_info)>0):
            self.sampler = make_sampler(self.label_info['label'].values)
 
    def __getitem__(self, index):   
        row = self.label_info.iloc[index] 
        label = row['label']
        npz_path = self.path[row['site']]   # DATA_PATH['physionet']
        ecg = np.load(os.path.join(npz_path, row['fname']+'.npz'))['arr_0'].astype(float) # read        
        #print('----')
        #print(ecg.shape)
        ecg_norm = ecg_crop_norm(ecg,norm='all-z_norm')     # (5000,)->(1,5000)->(1,4000)->normalize
        ecg_norm = ecg_norm[self.option:self.option+1,:]
        #print(ecg_norm.shape)
        
        #Filter Channels
        ecg_filter = filter_channel(ecg_norm)
        #print(ecg_filter.shape)
        #x_0 = ecg_filter_tmp[0,:]
        #x_1 = ecg_filter_tmp[1,:]
        #x_2 = ecg_filter_tmp[2,:]
        #x_3 = ecg_filter_tmp[3,:]
        # ecg_filter = np.squeeze(ecg_filter_tmp,1)

        # Compute Beat
        ecg_beat = compute_beat(ecg_filter)
        #print(ecg_beat.shape)
        #k_beat_0 = ecg_beat[0,:]
        #k_beat_1 = ecg_beat[1,:]
        #k_beat_2 = ecg_beat[2,:]
        #k_beat_3 = ecg_beat[3,:]
        
        # Compute Rhythm
        ecg_rhythm = compute_rhythm(ecg_filter,n_split=50)
        #print(ecg_rhythm.shape)
        #k_rhythm_0 = ecg_rhythm[0,:]
        #k_rhythm_1 = ecg_rhythm[1,:]
        #k_rhythm_2 = ecg_rhythm[2,:]
        #k_rhythm_3 = ecg_rhythm[3,:]
        
        # Compute Frequency
        ecg_freq = compute_freq(ecg_filter)
        #print(ecg_freq.shape)
        
        ecg_input = np.concatenate((ecg_filter,ecg_beat,ecg_rhythm,ecg_freq),axis=1)
        #print(ecg_input.shape)
        #print('return ecg_input')
        #np.expand_dims(ecg_norm,0) ==> (1,4000)->(1,1,4000)
        #return torch.tensor(np.expand_dims(ecg_norm,0), dtype = torch.float32), torch.tensor([label],dtype = torch.float32)
        return torch.tensor(ecg_input, dtype = torch.float32), torch.tensor([label],dtype = torch.float32)
               #torch.tensor(x_0, dtype = torch.float32),
               #torch.tensor(x_1, dtype = torch.float32),
               #torch.tensor(x_2, dtype = torch.float32),
               #torch.tensor(x_3, dtype = torch.float32),
               #torch.tensor(k_beat_0, dtype = torch.float32),
               #torch.tensor(k_beat_1, dtype = torch.float32),
               #torch.tensor(k_beat_2, dtype = torch.float32),
               #torch.tensor(k_beat_3, dtype = torch.float32),
               #torch.tensor(k_rhythm_0, dtype = torch.float32),
               #torch.tensor(k_rhythm_1, dtype = torch.float32),
               #torch.tensor(k_rhythm_2, dtype = torch.float32),
               #torch.tensor(k_rhythm_3, dtype = torch.float32),
               #torch.tensor(ecg_freq, dtype = torch.float32))
        
    def __len__(self):
        return self.len

class DualTwelveLeads (Dataset):
    '''
    본 데이터 셋은 original version의 12리드와 filtering version의 12리드를 한번에 출력하는 버전입니다. 
    output shape (batch, 2, 12 , 4000) : batch size, ori+filter version, num_leads, length
    '''
    def __init__(self, label_info, mode ='train', norm_method = 'z-norm', filter_method='medi_simple'):
        self.path = DATA_PATH
        self.label_info = label_info ## dataframe 
        self.mode = mode 
        self.norm_method = norm_method
        self.filter_method = filter_method 
        self.len = len(self.label_info)  
        # self.sampler = make_sampler(self.label_info['label'].values)
 
    def __getitem__(self, index):   
        row = self.label_info.iloc[index] 
        y = row['label']
        npz_path = self.path[row['site']]
        ori_full_path = os.path.join(npz_path, row['fname']+'.npz')
        ecg = np.load(ori_full_path)['arr_0'].astype(float) # read 

        ## original
        x = ecg_crop_norm(ecg, norm=self.norm_method)
        x_filt = None

        ## filtering
        filt_full_path = os.path.join(npz_path+'_denoise', row['fname']+'.npz')
        try: 
            x_filt = ecg_crop_norm(np.load(filt_full_path)['arr_0'].astype(float),norm=self.norm_method)
        except:
            tmp = np.zeros((12,5000))
            for i in range(12):
                tmp[i] = ecg_clean(ecg[i,:], sampling_rate=500, method=self.filter_method)

            x_filt = ecg_crop_norm(tmp, norm=self.norm_method) 
        
        x = np.concatenate((np.expand_dims(x, axis=0),np.expand_dims(x_filt, axis=0)),axis=0)  # shape (batch, 2,12,4000)
        
        return torch.tensor(x, dtype = torch.float32), torch.tensor([y], dtype = torch.float32)
        
    def __len__(self):
        return self.len

class DualAnyLeadsIdx (Dataset):
    def __init__(self, label_info, sampling_rate = 500, lead_idx = 0, mode ='train', filter_method='medi_simple'):
        self.path = DATA_PATH
        self.label_info = label_info  ## dataframe 
        self.sampling_rate = sampling_rate
        self.lead_idx = lead_idx 
        self.mode = mode 
        self.filter = filter_method
        self.len = len(self.label_info)  
        if (mode =='train') & (len(self.label_info)>0):
            self.sampler = make_sampler(self.label_info['label'].values)
 
    def __getitem__(self, index):   
        row = self.label_info.iloc[index] 
        label = row['label']
        sel_lead = self.lead_idx
        npz_path = self.path[row['site']]
        ecg = np.load(os.path.join(npz_path, row['fname']+'.npz'))['arr_0'].astype(float) # read 

        if self.mode == 'train':
            np.random.seed(int(datetime.now().microsecond/1000))
            sel_lead = np.random.randint(12) ## random selection for any leads
            if ecg[sel_lead,:].sum() == 0.:
                sel_lead = 0
        
        idx_vec =np.zeros(12)                
        idx_vec[sel_lead]=1
        
        ecg_ori = ecg_crop_norm(ecg)[sel_lead,:] ## shape (4000)

        try: 
            ecg_filt = np.load(os.path.join(npz_path+'_denoise', row['fname']+'.npz'))['arr_0'].astype(float) # read 
        except:
            # print('No filtered ecg') 
            ecg_filt = ecg_clean(ecg[sel_lead,:], sampling_rate=self.sampling_rate, method=self.filter) # input: 4000
    
        ecg_filt = ecg_crop_norm(ecg_filt)[sel_lead,:] ## shape : (4000)

        ## merging leads
        ecg_ori, ecg_filt = np.concatenate((ecg_ori,idx_vec)), np.concatenate((ecg_filt,idx_vec))
        input = np.concatenate((ecg_ori.reshape(1,-1),ecg_filt.reshape(1,-1)),axis=0) 

        return torch.from_numpy(input).float(),torch.tensor([label,sel_lead],dtype = torch.float32)
        
    def __len__(self):
        return self.len

class DualLeadsIdx (Dataset): # only name changed
    def __init__(self, label_info, sampling_rate = 500, lead_idx = 0, num_classes =2, mode ='train', filter_method='medi_simple'):
        self.path = DATA_PATH
        self.label_info = label_info  ## dataframe 
        self.sampling_rate = sampling_rate
        self.lead_idx = lead_idx 
        self.mode = mode 
        self.num_classes = num_classes
        self.filter = filter_method
        self.len = len(self.label_info)  
        if (mode =='train') & (len(self.label_info)>0):
            self.sampler = make_sampler(self.label_info['label'].values)
 
    def __getitem__(self, index):   
        row = self.label_info.iloc[index] 
        
        label = row['label']

        sel_lead = self.lead_idx
        npz_path = self.path[row['site']]
        ecg = np.load(os.path.join(npz_path, row['fname']+'.npz'))['arr_0'].astype(float) # read 

        if self.mode == 'train':
            np.random.seed(int(datetime.now().microsecond/1000))
            sel_lead = np.random.randint(12) ## random selection for any leads
            if ecg[sel_lead,:].sum() == 0.:
                sel_lead = 0
        
        idx_vec =np.zeros(12)                
        idx_vec[sel_lead]=1
        
        ecg_ori = ecg_crop_norm(ecg)[sel_lead,:] ## shape (4000)

        try: 
            ecg_filt = np.load(os.path.join(npz_path+'_denoise', row['fname']+'.npz'))['arr_0'].astype(float) # read 
        except:
            # print('No filtered ecg') 
            ecg_filt = ecg_clean(ecg[sel_lead,:], sampling_rate=self.sampling_rate, method=self.filter) # input: 4000
    
        ecg_filt = ecg_crop_norm(ecg_filt)[sel_lead,:] ## shape : (4000)

        ## merging leads
        ecg_ori, ecg_filt = np.concatenate((ecg_ori,idx_vec)), np.concatenate((ecg_filt,idx_vec))
        input = np.concatenate((ecg_ori.reshape(1,-1),ecg_filt.reshape(1,-1)),axis=0) 

        return torch.from_numpy(input).float(),torch.tensor([label,sel_lead],dtype = torch.float32)
        
    def __len__(self):
        return self.len

class TriAnyLeadsIdx (Dataset):
    def __init__(self, label_info, lead_idx = 0, mode ='train', filter_method='medi_simple'):
        self.path = DATA_PATH
        self.label_info = label_info  ## dataframe 
        self.lead_idx = lead_idx 
        self.mode = mode 
        self.filter = filter_method
        self.len = len(self.label_info)  
        if (mode =='train') & (len(self.label_info)>0):
            self.sampler = make_sampler(self.label_info['label'].values)
 
    def __getitem__(self, index):   
        row = self.label_info.iloc[index] 
        label = row['label']
        sel_lead = self.lead_idx
        npz_path = self.path[row['site']]
        ecg = np.load(os.path.join(npz_path, row['fname']+'.npz'))['arr_0'].astype(float) # read 

        if self.mode == 'train':
            np.random.seed(int(datetime.now().microsecond/1000))
            sel_lead = np.random.randint(12) ## random selection for any leads
            if ecg[sel_lead,:].sum() == 0.:
                sel_lead = 0
        
        idx_vec =np.zeros(12)                
        idx_vec[sel_lead]=1
        
        ecg_ori = ecg_crop_norm(ecg)[sel_lead,:] ## shape (4000)

        try: 
            ecg_filt = np.load(os.path.join(npz_path+'_denoise', row['fname']+'.npz'))['arr_0'].astype(float) # read 
        except:
            # print('No filtered ecg') 
            ecg_filt = ecg_clean(ecg[sel_lead,:], sampling_rate=500, method=self.filter) # input: 4000
    
        ecg_filt = ecg_crop_norm(ecg_filt)[sel_lead,:] ## shape : (4000)

        ## merging leads
        ecg_res = ecg_ori-ecg_filt
        ecg_ori, ecg_filt, ecg_res = np.concatenate((ecg_ori,idx_vec)), np.concatenate((ecg_filt,idx_vec)), np.concatenate((ecg_res,idx_vec))
        input = np.concatenate((ecg_ori.reshape(1,-1),ecg_filt.reshape(1,-1),ecg_res.reshape(1,-1)),axis=0)

        return torch.from_numpy(input).float(),torch.tensor([label,sel_lead],dtype = torch.float32)
        
    def __len__(self):
        return self.len


class ArrDataset (Dataset):
    def __init__(self, label_info, mode ='train'):
        self.label_info = label_info
        self.len = len(self.label_info)     
        self.mode = mode
        self.label_info['label'] = self.label_info['label'].astype(int)
        if (mode =='train') & (len(self.label_info)>0):
            self.sampler = make_sampler(self.label_info['label'].values)
 
    def __getitem__(self, index):   

        label = self.label_info.iloc[index]['label'].astype(float)
        input_col = []
        col_list = self.label_info.columns
        for task in ['irreg','pwave','pm','svt','seq1','seq2','seq3','flutter']:
            task_cols = [c for c in col_list if c.find(task)>-1]
            task_cols.sort()
            input_col.extend(task_cols)

        input = self.label_info.iloc[index][input_col].fillna(0).values.astype(float) 
        return torch.tensor(input, dtype= torch.float32), torch.tensor([label],dtype = torch.long)
        
    def __len__(self):
        return self.len


######### ECG Beats ###########
class ECGBeats(Dataset):
    def __init__(self, label_info, norm_method = 'all-z_norm', filter_method='medi_simple', lead_type=None, mode='train'):
        self.path = DATA_PATH
        self.label_info = label_info  ## dataframe 
        self.norm_method = norm_method
        self.filter_method = filter_method
        self.lead_type = lead_type
        self.len = len(self.label_info)  
        if (mode =='train') & (len(self.label_info)>0):
            self.sampler = make_sampler(self.label_info['label'].values)

    def __getitem__(self, index):   
        row = self.label_info.iloc[index] 
        age = row['age']
        gender = row['gender']
        label = row['label']
        npz_path = self.path[row['site']]
        ori_full_path = os.path.join(npz_path, row['fname']+'.npz')
        ecg = np.load(ori_full_path)['arr_0'].astype(float) # read 
        
        if self.lead_type=='ori': 
            x = ecg_crop_norm(ecg, norm=self.norm_method)
        else:
            filt_full_path = os.path.join(npz_path+'_denoise', row['fname']+'.npz')
            try: 
                x = np.load(filt_full_path)['arr_0'].astype(float)
            except:
                x = np.array([ecg_clean(i, sampling_rate=500, method=self.filter_method) for i in ecg])
            x = ecg_crop_norm(x, norm=self.norm_method) 

        x[3,:] *= (-1)
        lead_order = [6,7,8,9,10,11,4,0,3,1,5,2]
        x = x[lead_order,:]
        
        try:
            rpeak = get_rpeak_list(x) # in utils.py
            rpeak = [r for r in rpeak if (r>150) & (r<3800)]
        except:
            rpeak = range(150,3800)
        np.random.seed(int(datetime.now().microsecond/1000))
        rpeak = int(np.random.choice(rpeak, 1))
        x = x[:,rpeak-125:rpeak+200]  

        return (torch.tensor(np.expand_dims(x, axis=0), dtype = torch.float32), torch.tensor([age, gender], dtype = torch.float32)), torch.tensor([label], dtype = torch.float32)

    def __len__(self):
        return self.len




class WaveletBeats(Dataset):
    def __init__(self, label_info, norm_method = 'all-z_norm', filter_method='medi_simple', wavelet='morl', mode='train'):
        self.path = DATA_PATH
        self.label_info = label_info  ## dataframe 
        self.norm_method = norm_method
        self.filter_method = filter_method
        self.len = len(self.label_info)  

        self.wavelet = wavelet
        if (mode =='train') & (len(self.label_info)>0):
            self.sampler = make_sampler(self.label_info['label'].values)
 
    def __getitem__(self, index):   
        row = self.label_info.iloc[index] 
        age = row['age']
        gender = row['gender']
        label = row['label']
        npz_path = self.path[row['site']]

        
        ori_full_path = os.path.join(npz_path, row['fname']+'.npz')
        ecg = np.load(ori_full_path)['arr_0'].astype(float) # read 
        ecg = ecg_crop_norm(ecg, crop=False, norm=None)
        
        filt_full_path = os.path.join(npz_path+'_denoise', row['fname']+'.npz')
        try:
            x_filt = np.load(filt_full_path)['arr_0'].astype(float) # read 
        except:
            x_filt = np.array([ecg_clean(x, method=self.filter_method) for x in ecg])
        x_filt = ecg_crop_norm(x_filt, norm=None)
        
        x_filt[3,:] *= (-1)
        lead_order = [6,7,8,9,10,11,4,0,3,1,5,2]  
        x_filt = x_filt[lead_order,:]

        rpeak = get_rpeak_list(x_filt) # in utils.py
        rpeak = [r for r in rpeak if (r>150) & (r<3800)]
        np.random.seed(int(datetime.now().microsecond/1000))
        rpeak = int(np.random.choice(rpeak, 1))
        x_filt = x_filt[:,rpeak-125:rpeak+200]

        assert self.wavelet in ['morl', 'mexh', 'gaus4'], 'set the mother wavelet to one of morl, mexh, gaus4'
        s = 0.6 if self.wavelet in ['morl', 'gaus4'] else 0.2
        scales = np.arange(s, 301*s, s)
        coefs = ecg_norm(np.array([np.abs(pywt.cwt(x, scales, self.wavelet, 1)[0]) for x in x_filt]))

        return (torch.tensor(coefs, dtype = torch.float32), torch.tensor([age, gender], dtype = torch.float32)), torch.tensor([label], dtype = torch.float32)

    def __len__(self):
        return self.len

def ecg_crop_norm(ecg, sampling_rate =500, crop=True, seconds = 8, norm='z_norm'):  
    ## shape: channel, wavefrom
    if ecg.shape[0] > ecg.shape[1]:
        ecg = np.transpose(ecg)
    
    if crop:
        ecg = ecg[:,sampling_rate:sampling_rate+sampling_rate*seconds] ## triming

    if norm is not None:
        return ecg_norm(ecg,norm)    ## (ch, length, lead) --> (leads, waveform)
    else:
        return ecg
    
### normalization ## 
def ecg_norm (ecg, norm='z_norm'):

    if norm=='minmax':
        for i in range(ecg.shape[0]):
            min_val = np.nanmin(ecg[i,:])
            max_val = np.nanmax(ecg[i,:])
            ecg[i,:] = (ecg[i,:]-min_val)/(max_val-min_val+1e-5)
    
    elif norm=='z_norm':
        for i in range(ecg.shape[0]):
            ecg[i,:] = (ecg[i,:]-np.nanmean(ecg[i,:]))/(np.nanstd(ecg[i,:])+1e-5)

    elif norm=='all-z_norm':
        ecg = (ecg-np.nanmean(ecg))/(np.nanstd(ecg)+1e-5)
    else:
        ecg = (ecg-np.nanmean(ecg))/(np.nanstd(ecg)+1e-5)

    return ecg




############### Filter Channel / Compute Beat, Rhythm, Frequency ###############


def filter_channel(x,site):
    ### candidate channels for ECG
    P_wave = (0.67,5)
    QRS_complex = (10,50)
    T_wave = (1,7)
    muscle = (5,50)
    resp = (0.12,0.5)
    ECG_preprocessed = (0.5, 50)
    wander = (0.001, 0.5)
    noise = 50
    
    ### use low (wander), middle (ECG_preprocessed) and high (noise) for example
    bandpass_list = [wander, ECG_preprocessed]
    highpass_list = [noise]
    
    if site=='physionet':
        signal_freq = 300
        nyquist_freq = 0.5 * signal_freq
        filter_order = 1
        ### out including original x
        out_list = [x]
        for bandpass in bandpass_list:
            low = bandpass[0] / nyquist_freq
            high = bandpass[1] / nyquist_freq
            b, a = butter(filter_order, [low, high], btype="band")
            y = lfilter(b, a, x)
            out_list.append(y)
        for highpass in highpass_list:
            high = highpass / nyquist_freq
            b, a = butter(filter_order, high, btype="high")
            y = lfilter(b, a, x)
            out_list.append(y)
        out = np.array(out_list)
        out = np.squeeze(out,1)
    
    if site=='ptb-xl':
        signal_freq = 400
        nyquist_freq = 0.5 * signal_freq
        filter_order = 1
        out_list = []
        for i in range(x.shape[0]):
            xx = x[i,:]
            out_list = out_list+[xx]
            for bandpass in bandpass_list:
                low = bandpass[0] / nyquist_freq
                high = bandpass[1] / nyquist_freq
                b, a = butter(filter_order, [low, high], btype="band")
                y = lfilter(b, a, xx)
                out_list.append(y)
            for highpass in highpass_list:
                high = highpass / nyquist_freq
                b, a = butter(filter_order, high, btype="high")
                y = lfilter(b, a, xx)
                out_list.append(y)
        out = np.array(out_list)
    # print('out: ', out.shape)
    
    return out


def compute_beat(X):
    out = np.zeros((X.shape[0], X.shape[1]))
    for i in range(out.shape[0]):
        out[i, :] = np.concatenate([[0], np.abs(np.diff(X[i,:]))])
    return out

'''
def compute_beat(X):
    out = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.concatenate([[0], np.abs(np.diff(X[i,j,:]))])
    return out
'''


def compute_rhythm(X, n_split):
    cnt_split = int(X.shape[1]/n_split)
    out = np.zeros((X.shape[0], cnt_split))
    for i in range(out.shape[0]):
        tmp_ts = X[i,:]
        tmp_ts_cut = np.split(tmp_ts, X.shape[1]/n_split)
        for k in range(cnt_split):
            out[i, k] = np.std(tmp_ts_cut[k])
    return out

'''
def compute_rhythm(X, n_split):
    cnt_split = int(X.shape[2]/n_split)
    out = np.zeros((X.shape[0], X.shape[1], cnt_split))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            tmp_ts = X[i,j,:]
            tmp_ts_cut = np.split(tmp_ts, X.shape[2]/n_split)
            for k in range(cnt_split):
                out[i, j, k] = np.std(tmp_ts_cut[k])
    return out
'''

def compute_freq(X,site):
    if site=='ptb-xl':
        fs = 400
    if site=='physionet':
        fs = 300
    out = np.zeros((X.shape[0], 1))
    for i in range(out.shape[0]):
        _, Pxx_den = periodogram(X[i,:], fs)
        out[i,0] = np.sum(Pxx_den)
    return out







class MINA_SingleLead(SingleLead): 
    def __init__(self, label_info, option=0, mode ='train'):
        super().__init__(label_info, option=0, mode ='train')
 
    def __getitem__(self, index):   
        row = self.label_info.iloc[index] 
        label = row['label']
        npz_path = self.path[row['site']]   # DATA_PATH['physionet']
        ecg = np.load(os.path.join(npz_path, row['fname']+'.npz'))['arr_0'].astype(float) # read        
        #print('----')
        #print(ecg.shape)
        ecg_norm = ecg_crop_norm(ecg,norm='all-z_norm')     # (5000,)->(1,5000)->(1,4000)->normalize
        ecg_norm = ecg_norm[self.option:self.option+1,:]
        #print(ecg_norm.shape)
        
        #Filter Channels
        ecg_filter = filter_channel(ecg_norm,row['site'])
        #print(ecg_filter.shape)

        # Compute Beat
        ecg_beat = compute_beat(ecg_filter)
        #print(ecg_beat.shape)
        
        # Compute Rhythm
        ecg_rhythm = compute_rhythm(ecg_filter,n_split=50)
        #print(ecg_rhythm.shape)
        
        # Compute Frequency
        ecg_freq = compute_freq(ecg_filter,row['site'])
        #print(ecg_freq.shape)
        
        ecg_input = np.concatenate((ecg_filter,ecg_beat,ecg_rhythm,ecg_freq),axis=1)
        #print(ecg_input.shape)
        #print('return ecg_input')
        #np.expand_dims(ecg_norm,0) ==> (1,4000)->(1,1,4000)
        #return torch.tensor(np.expand_dims(ecg_norm,0), dtype = torch.float32), torch.tensor([label],dtype = torch.float32)
        return torch.tensor(ecg_input, dtype = torch.float32), torch.tensor([label],dtype = torch.float32)



class MINA_TwelveLeads(TwelveLeads): 
    def __init__(self, label_info, option=0, mode ='train'):
        super().__init__(label_info, option=0, mode ='train')
 
    def __getitem__(self, index):   
        row = self.label_info.iloc[index] 
        label = row['label']
        # print(row['site'])
        npz_path = self.path[row['site']]   # DATA_PATH['physionet']
        ecg = np.load(os.path.join(npz_path, row['fname']+'.npz'))['arr_0'].astype(float) # read        
        # print('----')
        # print(ecg.shape)
        ecg_norm = ecg_crop_norm(ecg,sampling_rate =100,norm='all-z_norm')     # (5000,)->(1,5000)->(1,4000)->normalize
        #ecg_norm = ecg_norm[self.option:self.option+1,:]
        # print(ecg_norm.shape)
        
        #Filter Channels
        ecg_filter = filter_channel(ecg_norm,row['site'])
        # print(ecg_filter.shape)

        # Compute Beat
        ecg_beat = compute_beat(ecg_filter)
        # print(ecg_beat.shape)

        # Compute Rhythm
        ecg_rhythm = compute_rhythm(ecg_filter,n_split=40)
        # print(ecg_rhythm.shape)
        
        # Compute Frequency
        ecg_freq = compute_freq(ecg_filter,row['site'])
        # print(ecg_freq.shape)
        
        ecg_input = np.concatenate((ecg_filter,ecg_beat,ecg_rhythm,ecg_freq),axis=1)
        # print(ecg_input.shape)
        # print('return ecg_input')
        #np.expand_dims(ecg_norm,0) ==> (1,4000)->(1,1,4000)
        #return torch.tensor(np.expand_dims(ecg_norm,0), dtype = torch.float32), torch.tensor([label],dtype = torch.float32)
        return torch.tensor(ecg_input, dtype = torch.float32), torch.tensor([label],dtype = torch.float32)
    
    
    
