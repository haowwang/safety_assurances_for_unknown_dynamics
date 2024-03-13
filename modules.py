import torch 


class CAFCNet(torch.nn.Module):
    """
    Implements a fully connected control affine architecture f(x,u) = f1(x) + f2(x)u
    """
    def __init__(self, state_dim:int, control_dim:int, num_layers:int, num_neurons_per_layer:int, if_batch_norm:bool, inputs_mean, inputs_std, labels_mean, labels_std, if_gpu:bool):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.num_layers = num_layers # number of layers of f1(x) and f2(x)
        self.num_neurons_per_layer = num_neurons_per_layer
        
        assert type(if_batch_norm) is bool
        self.if_batch_norm = if_batch_norm
        
        assert type(if_gpu) is bool
        if if_gpu:
            if torch.cuda.is_available():
                self.gpu_device = 'cuda'
            else:
                self.gpu_device = torch.device("mps")
        else:
            self.gpu_device = 'cpu'

        self.inputs_mean = inputs_mean.to(self.gpu_device)
        self.inputs_std = inputs_std.to(self.gpu_device)
        self.labels_mean = labels_mean.to(self.gpu_device)
        self.labels_std = labels_std.to(self.gpu_device)

        self.nl = torch.nn.ReLU()
        self.f1_net_layers = []
        self.f2_net_layers = []
        for i in range(self.num_layers):
            if not i: # first layer
                self.f1_net_layers.append(torch.nn.Sequential(
                    torch.nn.Linear(self.state_dim, self.num_neurons_per_layer),
                    self.nl 
                ))
                self.f2_net_layers.append(torch.nn.Sequential(
                    torch.nn.Linear(self.state_dim, self.num_neurons_per_layer),
                    self.nl 
                ))
                if self.if_batch_norm:
                    self.f1_net_layers.append(torch.nn.Sequential(torch.nn.BatchNorm1d(num_features = self.num_neurons_per_layer)))
                    self.f2_net_layers.append(torch.nn.Sequential(torch.nn.BatchNorm1d(num_features = self.num_neurons_per_layer)))
            elif i == self.num_layers - 1: # last layer
                self.f1_net_layers.append(torch.nn.Sequential(
                    torch.nn.Linear(self.num_neurons_per_layer, self.state_dim)
                ))
                self.f2_net_layers.append(torch.nn.Sequential(
                    torch.nn.Linear(self.num_neurons_per_layer, self.state_dim * self.control_dim)
                ))
            else:
                self.f1_net_layers.append(torch.nn.Sequential(
                    torch.nn.Linear(self.num_neurons_per_layer, self.num_neurons_per_layer),
                    self.nl
                ))
                self.f2_net_layers.append(torch.nn.Sequential(
                    torch.nn.Linear(self.num_neurons_per_layer, self.num_neurons_per_layer),
                    self.nl
                ))
                if self.if_batch_norm:
                    self.f1_net_layers.append(torch.nn.Sequential(torch.nn.BatchNorm1d(num_features = self.num_neurons_per_layer)))
                    self.f2_net_layers.append(torch.nn.Sequential(torch.nn.BatchNorm1d(num_features = self.num_neurons_per_layer)))
        self.f1_net = torch.nn.Sequential(*self.f1_net_layers)
        self.f2_net = torch.nn.Sequential(*self.f2_net_layers)

    def forward(self, inputs):
        state_inputs = inputs[:, :self.state_dim]
        state_inputs = (state_inputs - self.inputs_mean) / self.inputs_std # state input normalization
        if self.control_dim == 1:
            control_inputs = inputs[:, self.state_dim].unsqueeze(-1).unsqueeze(-1)
        else:
            control_inputs = inputs[:, self.state_dim:].unsqueeze(-1)
        
        f1_net_output = self.f1_net(state_inputs)
        f2_net_output = self.f2_net(state_inputs).view(-1, self.state_dim, self.control_dim)
        net_output = f1_net_output + torch.matmul(f2_net_output, control_inputs).squeeze(-1) # f(x, u) = f1(x) + f2(x)u

        if self.training: # train mode
            return net_output
        else:  # eval mode
            unnormalized_net_output = net_output * self.labels_std + self.labels_mean
            return unnormalized_net_output, f1_net_output, f2_net_output
    

class DynDataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, inputs, labels):
        assert inputs.dtype == torch.float32 and labels.dtype == torch.float32 # model params are in torch.float32; so dataset must be in the same dtype
        self.list_IDs = list_IDs
        self.inputs = inputs
        self.labels = labels
        
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        X = self.inputs[ID, :]
        y = self.labels[ID, :]
        return X,y