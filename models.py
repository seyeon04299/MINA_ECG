import sys
from utils import *

################################################################################################
# MINA
################################################################################################
class MINANet(nn.Module):
    def __init__(self, config, n_channel=4,n_dim=4000,n_split=50,num_classes=2):
        super(MINANet, self).__init__()
        
        self.config = config
        # self.n_channel = n_channel
        # self.n_dim = n_dim
        # self.n_split = n_split
        
        # self.num_classes = self.config.num_classes if hasattr(config,'num_classes') else 2
        self.num_classes = num_classes
        inplanes = self.config.inplanes if hasattr(config,'inplanes') else 64
        kernel_size = self.config.kernel_size if hasattr(config, 'kernel_size') else 15
        dropout = self.config.dropout if hasattr(config, 'dropout') else 0.5
        model_output = self.config.model_output if hasattr(config, 'model_output') else 64
        dropout = self.config.dropout if hasattr(config, 'dropout') else 0.5
        self.n_channel = self.config.n_channel if hasattr(config, 'n_channel') else 4
        self.n_dim = self.config.n_dim if hasattr(config, 'n_dim') else 800
        self.n_split = self.config.n_split if hasattr(config, 'n_split') else 40
        
        ###

        # __init__(self, n_dim, n_split, inplanes=64, kernel_size=3, stride=2, att_cnn_dim=8, att_rnn_dim=8):
        self.base_net_0 = BaseNet(n_dim, n_split, inplanes=inplanes, kernel_size=kernel_size, stride=2, att_cnn_dim=8, att_rnn_dim=8, dropout = dropout)
        self.base_net_1 = BaseNet(n_dim, n_split, inplanes=inplanes, kernel_size=kernel_size, stride=2, att_cnn_dim=8, att_rnn_dim=8, dropout = dropout)
        self.base_net_2 = BaseNet(n_dim, n_split, inplanes=inplanes, kernel_size=kernel_size, stride=2, att_cnn_dim=8, att_rnn_dim=8, dropout = dropout)
        self.base_net_3 = BaseNet(n_dim, n_split, inplanes=inplanes, kernel_size=kernel_size, stride=2, att_cnn_dim=8, att_rnn_dim=8, dropout = dropout)
            
        ### attention
        self.out_size = 16
        self.att_channel_dim = 2
        self.W_att_channel = nn.Parameter(torch.randn(self.out_size+1, self.att_channel_dim))
        self.v_att_channel = nn.Parameter(torch.randn(self.att_channel_dim, 1))
        
        ### fc
        self.fc = nn.Linear(self.out_size, self.num_classes)
        
    def forward(self, x):
                # x_0, x_1, x_2, x_3, 
                # k_beat_0, k_beat_1, k_beat_2, k_beat_3, 
                # k_rhythm_0, k_rhythm_1, k_rhythm_2, k_rhythm_3, 
                # k_freq):
        # print(x.shape)
        x_input = x[:,:,0:int(self.n_dim)]
        k_beats = x[:,:,int(self.n_dim):int(self.n_dim)*2]
        k_rhythms = x[:,:,2*int(self.n_dim):int(self.n_dim)*2+int(self.n_dim/self.n_split)]
        k_freq = x[:,:,int(self.n_dim)*2+int(self.n_dim/self.n_split)]
        k_freq = k_freq.unsqueeze(2)
        # print('x_input.shape : ', x_input.shape)
        # print('k_beats.shape : ', k_beats.shape)
        # print('k_rhythms.shape : ', k_rhythms.shape)
        # print('k_freq.shape : ', k_freq.shape)
        
        # x_input_tmp = np.expand_dims(x_input.cpu(), axis=1)
        # k_beats_tmp = np.expand_dims(k_beats.cpu(), axis=1)
        # k_rhythms_tmp = np.expand_dims(k_rhythms.cpu(), axis=1)
        # k_freq_tmp = np.expand_dims(k_freq.cpu(), axis=1)
        x_0 = x_input[:,0,:]
        x_1 = x_input[:,1,:]
        x_2 = x_input[:,2,:]
        x_3 = x_input[:,3,:]
        k_beat_0, k_beat_1 = k_beats[:,0,:], k_beats[:,1,:]
        k_beat_2, k_beat_3 = k_beats[:,2,:], k_beats[:,3,:]
        k_rhythm_0, k_rhythm_1 = k_rhythms[:,0,:], k_rhythms[:,1,:]
        k_rhythm_2, k_rhythm_3 = k_rhythms[:,2,:], k_rhythms[:,3,:]
        # print('x_0.shape : ', x_0.shape)
        # print('x_1.shape : ', x_1.shape)
        # print('x_2.shape : ', x_2.shape)
        # print('x_3.shape : ', x_3.shape)
        # print('k_beat_0.shape : ', k_beat_0.shape)
        # print('k_beat_1.shape : ', k_beat_1.shape)
        # print('k_beat_2.shape : ', k_beat_2.shape)
        # print('k_beat_3.shape : ', k_beat_3.shape)
        # print('k_rhythm_0.shape : ', k_rhythm_0.shape)
        # print('k_rhythm_1.shape : ', k_rhythm_1.shape)
        # print('k_rhythm_2.shape : ', k_rhythm_2.shape)
        # print('k_rhythm_3.shape : ', k_rhythm_3.shape)
        
        
        # x_0 = torch.tensor(x_input[0,:], dtype = torch.float32)
        # x_1 = torch.tensor(x_input[1,:], dtype = torch.float32)
        # x_2 = torch.tensor(x_input[2,:], dtype = torch.float32)
        # x_3 = torch.tensor(x_input[3,:], dtype = torch.float32)
        # k_beat_0, k_beat_1 = torch.tensor(k_beats[0,:], dtype = torch.float32), torch.tensor(k_beats[1,:], dtype = torch.float32)
        # k_beat_2, k_beat_3 = torch.tensor(k_beats[2,:], dtype = torch.float32), torch.tensor(k_beats[3,:], dtype = torch.float32)
        # k_rhythm_0, k_rhythm_1 = torch.tensor(k_rhythms[0,:], dtype = torch.float32), torch.tensor(k_rhythms[1,:], dtype = torch.float32)
        # k_rhythm_2, k_rhythm_3 = torch.tensor(k_rhythms[2,:], dtype = torch.float32), torch.tensor(k_rhythms[3,:], dtype = torch.float32)

        x_0, alpha_0, beta_0 = self.base_net_0(x_0, k_beat_0, k_rhythm_0)
        x_1, alpha_1, beta_1 = self.base_net_1(x_1, k_beat_1, k_rhythm_1)
        x_2, alpha_2, beta_2 = self.base_net_2(x_2, k_beat_2, k_rhythm_2)
        x_3, alpha_3, beta_3 = self.base_net_3(x_3, k_beat_3, k_rhythm_3)
        
        x = torch.stack([x_0, x_1, x_2, x_3], 1)
        # print('x.shape : ', x.shape)
        # print('k_freq.shape: ', k_freq.shape)
        # ### attention on channel
        #k_freq = k_freq.permute(1, 0, 2)
        
        tmp_x = torch.cat((x, k_freq), dim=-1)
        # print('tmp_x.shape: ', tmp_x.shape)
        e = torch.matmul(tmp_x, self.W_att_channel)
        # print('e.shape: ',e.shape)
        e = torch.matmul(torch.tanh(e), self.v_att_channel)
        # print('e.shape: ',e.shape)
        n1 = torch.exp(e)
        n2 = torch.sum(torch.exp(e), 1, keepdim=True)
        gama = torch.div(n1, n2)
        x = torch.sum(torch.mul(gama, x), 1)
        # print('x.shape 2 : ', x.shape)
        
        ### fc
        #x = self.fc(x)
        #x = F.softmax(self.fc(x))
        x = F.softmax(self.fc(x), 1)
        # print('x.shape 3 : ', x.shape)
        ### return 
        
        att_dic = {"alpha_0":alpha_0, "beta_0":beta_0, 
                  "alpha_1":alpha_1, "beta_1":beta_1, 
                  "alpha_2":alpha_2, "beta_2":beta_2, 
                  "alpha_3":alpha_3, "beta_3":beta_3, 
                  "gama":gama}
        
        return x, att_dic

class BaseNet(nn.Module):
    def __init__(self, n_dim, n_split, inplanes=64, kernel_size=3, stride=2, att_cnn_dim=8, att_rnn_dim=8, dropout = 0.5):
        super(BaseNet, self).__init__()
        
        self.n_dim = n_dim
        self.n_split = n_split
        self.n_seg = int(n_dim/n_split)
        self.dropout = dropout
        
        ### Input: (batch size, number of channels, length of signal sequence)
        self.conv_out_channels = inplanes
        self.conv_kernel_size = kernel_size
        self.conv_stride = stride
        
        self.conv = nn.Conv1d(in_channels=1, 
                              out_channels=self.conv_out_channels, 
                              kernel_size=self.conv_kernel_size, 
                              stride=self.conv_stride)
        self.conv_k = nn.Conv1d(in_channels=1, 
                                out_channels=1, 
                                kernel_size=self.conv_kernel_size, 
                                stride=self.conv_stride)
        self.att_cnn_dim = att_cnn_dim
        self.W_att_cnn = nn.Parameter(torch.randn(self.conv_out_channels+1, self.att_cnn_dim))
        self.v_att_cnn = nn.Parameter(torch.randn(self.att_cnn_dim, 1))
        
        ### Input: (batch size, length of signal sequence, input_size)
        self.rnn_hidden_size = 32
        self.lstm = nn.LSTM(input_size=(self.conv_out_channels), 
                            hidden_size=self.rnn_hidden_size, 
                            num_layers=1, batch_first=True, bidirectional=True)
        self.att_rnn_dim = att_rnn_dim
        self.W_att_rnn = nn.Parameter(torch.randn(2*self.rnn_hidden_size+1, self.att_rnn_dim))
        self.v_att_rnn = nn.Parameter(torch.randn(self.att_rnn_dim, 1))
        
        ### fc
        self.do = nn.Dropout(p=self.dropout)
        self.out_size = 16
        self.fc = nn.Linear(2*self.rnn_hidden_size, self.out_size)
    
    def forward(self, x, k_beat, k_rhythm):
        self.lstm.flatten_parameters()
        self.batch_size = x.size()[0]

        ### reshape
        
        x = x.reshape(-1, self.n_split)
        x = x.unsqueeze(1)
        
        k_beat = k_beat.reshape(-1, self.n_split)
        k_beat = k_beat.unsqueeze(1)
        
        ### conv
        x = F.relu(self.conv(x))
        k_beat = F.relu(self.conv_k(k_beat))
        
        ### attention conv
        x = x.permute(0, 2, 1)
        k_beat = k_beat.permute(0, 2, 1)
        tmp_x = torch.cat((x, k_beat), dim=-1)
        e = torch.matmul(tmp_x, self.W_att_cnn)
        e = torch.matmul(torch.tanh(e), self.v_att_cnn)
        n1 = torch.exp(e)
        n2 = torch.sum(torch.exp(e), 1, keepdim=True)
        alpha = torch.div(n1, n2)
        x = torch.sum(torch.mul(alpha, x), 1)
        
        ### reshape for rnn
        x = x.view(self.batch_size, self.n_seg, -1)
    
        ### rnn        
        k_rhythm = k_rhythm.unsqueeze(-1)
        o, (ht, ct) = self.lstm(x)
        tmp_o = torch.cat((o, k_rhythm), dim=-1)
        e = torch.matmul(tmp_o, self.W_att_rnn)
        e = torch.matmul(torch.tanh(e), self.v_att_rnn)
        n1 = torch.exp(e)
        n2 = torch.sum(torch.exp(e), 1, keepdim=True)
        beta = torch.div(n1, n2)
        x = torch.sum(torch.mul(beta, o), 1)
        
        ### fc
        x = F.relu(self.fc(x))
        x = self.do(x)
        
        return x, alpha, beta


class TwelveLeadsMINANET (nn.Module): 

    def __init__(self, config):        
        super(TwelveLeadsMINANET, self).__init__()

        self.config = config

        self.n_class = self.config.num_classes if hasattr(config,'num_classes') else 2
        
        inplanes = self.config.inplanes if hasattr(config,'inplanes') else 64
        kernel_size = self.config.kernel_size if hasattr(config, 'kernel_size') else 15
        model_output = self.config.model_output if hasattr(config, 'model_output') else 64
        dropout = self.config.dropout if hasattr(config, 'dropout') else 0.5
        n_channel = self.config.n_channel if hasattr(config, 'n_channel') else 4
        n_dim = self.config.n_dim if hasattr(config, 'n_dim') else 800
        n_split = self.config.n_split if hasattr(config, 'n_split') else 40
        module = []        
        module += [MINANet(config, n_channel=n_channel,n_dim=n_dim,n_split=n_split,num_classes=model_output)]
        module += [MINANet(config, n_channel=n_channel,n_dim=n_dim,n_split=n_split,num_classes=model_output)]
        module += [MINANet(config, n_channel=n_channel,n_dim=n_dim,n_split=n_split,num_classes=model_output)]
        module += [MINANet(config, n_channel=n_channel,n_dim=n_dim,n_split=n_split,num_classes=model_output)]
        module += [MINANet(config, n_channel=n_channel,n_dim=n_dim,n_split=n_split,num_classes=model_output)]
        module += [MINANet(config, n_channel=n_channel,n_dim=n_dim,n_split=n_split,num_classes=model_output)]
        module += [MINANet(config, n_channel=n_channel,n_dim=n_dim,n_split=n_split,num_classes=model_output)]
        module += [MINANet(config, n_channel=n_channel,n_dim=n_dim,n_split=n_split,num_classes=model_output)]
        module += [MINANet(config, n_channel=n_channel,n_dim=n_dim,n_split=n_split,num_classes=model_output)]
        module += [MINANet(config, n_channel=n_channel,n_dim=n_dim,n_split=n_split,num_classes=model_output)]
        module += [MINANet(config, n_channel=n_channel,n_dim=n_dim,n_split=n_split,num_classes=model_output)]
        module += [MINANet(config, n_channel=n_channel,n_dim=n_dim,n_split=n_split,num_classes=model_output)]
        self.minanet = nn.Sequential(*module)

        ## fc

        num_classes = self.config.num_classes if hasattr(config,'num_classes') else 2
        hidden = self.config.hidden_size if hasattr(config,'hidden_size') else 32
        do = self.config.do if hasattr(config,'do') else 0.5

        self.fc_label = self.make_fc(model_output*12, num_classes, hidden, do)  
        self.fc_idx = self.make_fc(model_output*12, 12, hidden, do)  
        
        initialize_weights(self)

    def make_fc (self, input_size, out_size, hidden, do):
        return torch.nn.Sequential(
            nn.Linear(input_size, hidden), 
            # nn.BatchNorm1d(hidden),
            nn.Dropout(do),
            nn.ReLU(), 
            nn.Linear(hidden , out_size),
            nn.Softmax(dim=1)
        )

         
    def forward(self, inputs): 
        out = []
        att_list = []
        for i, net in enumerate (self.minanet):
            x, att_dic = net(inputs[:,4*i:4*(i+1),:])
            out += [x]
            att_list += [att_dic]
        
        out = torch.cat((out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],out[8],out[9],out[10],out[11]),1)
        
        y_pred = self.fc_label(out)
        print('y_pred.shape : ', y_pred.shape)
        #lead_idx = self.fc_idx(out)

        return y_pred, att_list #, lead_idx





################################################################################################
# ResNet
################################################################################################

class ResBlk1d(nn.Module):
    def __init__(self, blk_id, inplanes, outplanes, kernel_size=3, stride=1, dilation=1, norm_layer=nn.BatchNorm1d, downsample=None, pooling=True):
        super(ResBlk1d, self).__init__()

        padding = kernel_size//2+1 if (kernel_size//2)%2==0 else kernel_size//2
        self.conv1 = nn.Conv1d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, dilation=dilation)
        self.bn1 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(outplanes, outplanes, kernel_size=kernel_size, stride=1, padding=padding, bias=False, dilation=dilation)
        self.bn2 = norm_layer(outplanes)

        self.downsample = downsample if downsample is not None else None
        
        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=2, padding=padding) if pooling else None

    def forward(self, x):
        identity = x
        # print('identity = ', identity.shape)
        # print('X_shape = ', x.shape)
        out = self.conv1(x)
        # print(out.shape)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # print(out.shape)
        out = self.bn2(out)
        # print('downsample = ', self.downsample)
        if self.downsample is not None:
            identity = self.downsample(x)
        # print('identity2 = ', identity.shape)
        out += identity
        out = self.relu(out)

        if self.maxpool is not None:
            out = self.maxpool(out)

        return out


class ResNet1d(nn.Module):

    def __init__(self, block, inplanes= 16, kernel_size=3, stride=1, num_classes=2, norm_layer=nn.BatchNorm1d):
        super(ResNet1d, self).__init__() 
        self._norm_layer = norm_layer    #nn.BatchNorm1d
        self.inplanes = inplanes         #16
        self.dilation = 1
        # stem
        self.conv1 = nn.Conv1d(1, self.inplanes, kernel_size=kernel_size, stride=2, padding=kernel_size//2, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=2, padding=kernel_size//2)


        self.layer1 = self._make_layer(0, block, inplanes, kernel_size)
        self.layer2 = self._make_layer(1, block, inplanes, kernel_size, stride, norm_layer=norm_layer, pooling=True)
        self.layer3 = self._make_layer(2, block, inplanes*2, kernel_size, stride, norm_layer=norm_layer) #, pooling=True)
        self.layer4 = self._make_layer(3, block, inplanes*4, kernel_size, stride, norm_layer=norm_layer, pooling=True)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(inplanes*8*32+2, num_classes)

    def _make_layer(self, blk_id, block, inplanes, kernel_size=15, stride=2, norm_layer=nn.BatchNorm1d, residual=True, pooling=False):
        
        outplanes = inplanes*2 if blk_id >0 else inplanes
        downsample = nn.Conv1d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False) if residual else None
        # print('-------------------------------------------')
        # print('block id', blk_id)
        # print('kernel_size = ', kernel_size)
        # print('inplanes = ', inplanes)
        # print('outplanes = ', outplanes)
        # print('stride = ', stride)
        
        layers = []
        layers.append(block(blk_id, inplanes, outplanes, kernel_size, stride, norm_layer=norm_layer, downsample=downsample, pooling=pooling))
        layers.append(block(blk_id, outplanes, outplanes, kernel_size, 1, norm_layer=norm_layer, pooling=pooling))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = x[:,:,:-2]
        aux = torch.flatten(x[:,:,-2:],1)
        # print(x.shape, aux.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print('-------------------')
        # print('layer1',x.shape)
        x = self.layer1(x)
        # print('-------------------')
        # print('layer2',x.shape)
        x = self.layer2(x)
        # print('-------------------')
        # print('layer3',x.shape)
        x = self.layer3(x)
        # print('-------------------')
        # print('layer4',x.shape)
        x = self.layer4(x)
        # print('-------------------')
        # print('flatten',x.shape)
        x = torch.flatten(x, 1)
        # print('-------------------')
        # print('fc',x.shape)
        # print('aux',aux.shape)
        x = torch.cat((x,aux),1)
        # print('-------------------')
        # print('cat_x,aux',x.shape)
        x = self.fc(x)
        # print('final',x.shape)
        return x


class DualLeadsARR (nn.Module): 

    def __init__(self, config):        
        super(DualLeadsARR, self).__init__()
        self.config = config
        ## resnet
        inplanes = self.config.inplanes if hasattr(config,'inplanes') else 64
        kernel_size = self.config.kernel_size if hasattr(config, 'kernel_size') else 15
        model_output = self.config.model_output if hasattr(config, 'model_output') else 64

        module = []        
        module += [ResNet1d(ResBlk1d, inplanes= inplanes, kernel_size=kernel_size, stride=2, num_classes=model_output)]
        module += [ResNet1d(ResBlk1d, inplanes= inplanes, kernel_size=kernel_size, stride=2, num_classes=model_output)]
        self.resnet = nn.Sequential(*module)

        ## fc

        num_classes = self.config.num_classes if hasattr(config,'num_classes') else 2
        hidden = self.config.hidden_size if hasattr(config,'hidden_size') else 32
        do = self.config.do if hasattr(config,'do') else 0.5

        self.fc_label = self.make_fc(model_output*2, num_classes, hidden, do)  
        self.fc_idx = self.make_fc(model_output*2, 12, hidden, do)  
        
        initialize_weights(self)

    def make_fc (self, input_size, out_size, hidden, do):
        return torch.nn.Sequential(
            nn.Linear(input_size, hidden), 
            # nn.BatchNorm1d(hidden),
            nn.Dropout(do),
            nn.ReLU(), 
            nn.Linear(hidden , out_size),
            nn.Softmax(dim=1)
        )

         
    def forward(self, inputs): 
        out = []
        for i, net in enumerate (self.resnet):
            out += [net(inputs[:,i:i+1,:])]
        out = torch.cat((out[0],out[1]),1)
        y_pred = self.fc_label(out)
        lead_idx = self.fc_idx(out)

        return y_pred, lead_idx

