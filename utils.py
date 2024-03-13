import configargparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import shutil
from datetime import datetime
import os, sys
import wandb
from torchinfo import summary
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
sys.path.append( os.path.dirname(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ))
sys.path.append( os.path.dirname(os.path.dirname(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )))

from modules import CAFCNet, DynDataset
import training
from scipy.io import loadmat, savemat



def config_parsing(config_file_fname):
    config_file_path = [config_file_fname]
    if not config_file_path:
        print('config file path is empty. exit')
        exit()

    p = configargparse.ArgumentParser(default_config_files = config_file_path)
    
    # training params
    p.add_argument('--gpu', type = str, required = True, help = 'gpu to train on.')
    p.add_argument('--if_gpu', type = str, required = True, choices = ["True", "False"], help = 'whether to use gpu')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, required = True, default=2e-5, help='learning rate. default=2e-5')
    p.add_argument('--num_epochs', type=int, required = True, default=100000, help='Number of epochs to train for.')
    p.add_argument('--num_models', type=int, required = True, default=2, help='Number of models in the ensemble.')
    p.add_argument('--num_layers', type=int, required = True, default=2, help='Number of layers of net.')
    p.add_argument('--num_neurons_per_layer', type=int, required = True, default=2, help='Number of neurons per layer.')
    p.add_argument('--batch_norm', type=str, required=False, choices = ["True", "False"], default='False')
    p.add_argument('--weight_decay', type=float, default=0, required=False, help='weight decay coeff for optimizer')
    p.add_argument('--l1_lambda', type = float, default = 0, required = False, help='coeff for l1 regularization')
    p.add_argument('--use_lbfgs', default=False, type=bool, help='use L-BFGS.')
    p.add_argument('--clip_grad', default=0.0, type=float, help='Clip gradient.')
    p.add_argument('--checkpoint_toload', type = int, default=0, help='Checkpoint from which to restart the training.')
    p.add_argument('--num_eval_samples', type = int, default = 10000, required = False, help = 'number of evaluation samples')

    # data logging params
    p.add_argument('--experiment_name', type=str, required=True,
                help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
    p.add_argument('--log_to_wandb', type=str, choices = ["True", "False"], required=False)
    p.add_argument('--wandb_project_name', type=str, required=False)
    p.add_argument('--epochs_til_ckpt', type=int, default=10,
               help='Number of epochs until checkpoint is saved.')
    p.add_argument('--steps_til_summary', type=int, default=1,
                help='Time interval in seconds until tensorboard summary is saved.')
    
    p.add_argument('--dataset_name', type = str, required = True)

    opt = p.parse_known_args()
    return opt


def ensemble_forward_initialize(state_dim, control_dim, dyn_model_idx_list, dyn_experiment_name, base_path, if_gpu, batch_size, model_checkpoint_epoch, mode:str):
    
    assert mode in ['eval_with_ctrl', 'eval_without_ctrl']
    dyn_experiment_summary_path = os.path.join(base_path, 'logs', dyn_experiment_name, 'summary_file.txt')
    opt = config_parsing(dyn_experiment_summary_path)[0]
    if if_gpu:
        if torch.cuda.is_available():
            gpu_device = 'cuda'
            print('Using GPU:CUDA', os.environ["CUDA_VISIBLE_DEVICES"])
        else:
            gpu_device = torch.device("mps")
            print('Using GPU:MPS')
    elif not if_gpu:
        gpu_device = 'cpu'
        print('Using CPU')
    
    if opt.batch_norm == "True":
        batch_norm = True
    elif opt.batch_norm == "False":
        batch_norm = False
    else:
        sys.exit("Unknown parameter for paparamter if_batch_norm")

    assert len(dyn_model_idx_list) > 0

    def ensemble_forward(dyn_forward_samples):
        # grid point loaded into torch dataset
        if mode == 'eval_with_ctrl':
            assert dyn_forward_samples.shape[1] == int(state_dim) + int(control_dim)
        elif mode == 'eval_without_ctrl':
            assert dyn_forward_samples.shape[1] == int(state_dim)
        dyn_forward_dataset = DynDataset(range(dyn_forward_samples.shape[0]), inputs = dyn_forward_samples, labels = dyn_forward_samples)
        dyn_forward_data_loader = torch.utils.data.DataLoader(dyn_forward_dataset, batch_size=batch_size, shuffle = False)

        dyn_models_x_dot_f1_all = None
        dyn_models_x_dot_f2_all = None
        dyn_models_x_dot_unnormed_all = None

        for i in dyn_model_idx_list:
            if type(model_checkpoint_epoch) is int and model_checkpoint_epoch < 100:
                dyn_model_state_dict_path = os.path.join(base_path, 'logs', dyn_experiment_name, 'checkpoints_model_idx'+str(i),'model_epoch_00' + str(model_checkpoint_epoch) + '.pth')
            elif type(model_checkpoint_epoch) is int and model_checkpoint_epoch > 100:
                dyn_model_state_dict_path = os.path.join(base_path, 'logs', dyn_experiment_name, 'checkpoints_model_idx'+str(i),'model_epoch_0' + str(model_checkpoint_epoch) + '.pth')
            elif model_checkpoint_epoch == 'final':
                dyn_model_state_dict_path = os.path.join(base_path, 'logs', dyn_experiment_name, 'checkpoints_model_idx'+str(i),'model_final.pth')
            dyn_model_checkpoint = torch.load(dyn_model_state_dict_path, map_location=torch.device('cpu'))
            dyn_model_training_dataset_stats = dyn_model_checkpoint['training_dataset_stats']
            inputs_mean = dyn_model_training_dataset_stats['inputs_mean'].to(gpu_device)
            inputs_std = dyn_model_training_dataset_stats['inputs_std'].to(gpu_device)
            labels_mean = dyn_model_training_dataset_stats['labels_mean'].to(gpu_device)
            labels_std = dyn_model_training_dataset_stats['labels_std'].to(gpu_device)
            dyn_model = CAFCNet(state_dim = state_dim, control_dim = control_dim, num_layers=opt.num_layers, num_neurons_per_layer=opt.num_neurons_per_layer, 
                                if_batch_norm=batch_norm, inputs_mean = inputs_mean, inputs_std = inputs_std, labels_mean = labels_mean, labels_std = labels_std, if_gpu=if_gpu)
            dyn_model.load_state_dict(dyn_model_checkpoint['model'])
            dyn_model = dyn_model.to(gpu_device)
            dyn_model.eval()
            dyn_model_x_dot_f1 = None
            dyn_model_x_dot_f2 = None
            dyn_model_x_dot_unnormed = None
            with torch.no_grad():
                for dyn_forward_batch, _ in dyn_forward_data_loader:
                    if mode == 'eval_without_ctrl':
                        dyn_forward_batch = torch.hstack((dyn_forward_batch, torch.zeros((dyn_forward_batch.shape[0], control_dim), dtype = torch.float32))) # hstack with zero control
                    dyn_forward_batch = dyn_forward_batch.to(gpu_device)
                    dyn_model_out_unnormed_x_dot, dyn_model_out_f1, dyn_model_out_f2 = dyn_model(dyn_forward_batch)
                    if dyn_model_x_dot_f1 is None:
                        if gpu_device != 'cpu':
                            dyn_model_x_dot_f1 = dyn_model_out_f1.detach().cpu()
                            dyn_model_x_dot_f2 = dyn_model_out_f2.detach().cpu()
                            dyn_model_x_dot_unnormed = dyn_model_out_unnormed_x_dot.detach().cpu()
                        else:
                            dyn_model_x_dot_f1 = dyn_model_out_f1.detach()
                            dyn_model_x_dot_f2 = dyn_model_out_f2.detach()
                            dyn_model_x_dot_unnormed = dyn_model_out_unnormed_x_dot.detach()
                    else:
                        if gpu_device != 'cpu':
                            dyn_model_x_dot_f1 = torch.vstack((dyn_model_x_dot_f1, dyn_model_out_f1.detach().cpu()))
                            dyn_model_x_dot_f2 = torch.cat((dyn_model_x_dot_f2, dyn_model_out_f2.detach().cpu()), dim = 0)
                            dyn_model_x_dot_unnormed = torch.vstack((dyn_model_x_dot_unnormed, dyn_model_out_unnormed_x_dot.detach().cpu()))
                        else:
                            dyn_model_x_dot_f1 = torch.vstack((dyn_model_x_dot_f1, dyn_model_out_f1.detach()))
                            dyn_model_x_dot_f2 = torch.cat((dyn_model_x_dot_f2, dyn_model_out_f2.detach()), dim = 0)
                            dyn_model_x_dot_unnormed = torch.vstack((dyn_model_x_dot_unnormed, dyn_model_out_unnormed_x_dot.detach()))
            if dyn_models_x_dot_f1_all is None:
                dyn_models_x_dot_f1_all = dyn_model_x_dot_f1.unsqueeze(0)
                dyn_models_x_dot_f2_all = dyn_model_x_dot_f2.unsqueeze(0)
                dyn_models_x_dot_unnormed_all = dyn_model_x_dot_unnormed.unsqueeze(0)
            else:
                dyn_models_x_dot_f1_all = torch.cat((dyn_models_x_dot_f1_all, dyn_model_x_dot_f1.unsqueeze(0)), dim = 0)
                dyn_models_x_dot_f2_all = torch.cat((dyn_models_x_dot_f2_all, dyn_model_x_dot_f2.unsqueeze(0)), dim = 0)
                dyn_models_x_dot_unnormed_all = torch.cat((dyn_models_x_dot_unnormed_all, dyn_model_x_dot_unnormed.unsqueeze(0)), dim = 0)

        dyn_models_x_dot_f1_all_mean = torch.mean(dyn_models_x_dot_f1_all, dim = 0)
        dyn_models_x_dot_f1_all_std = torch.std(dyn_models_x_dot_f1_all, dim = 0)
        dyn_models_x_dot_f2_all_mean = torch.mean(dyn_models_x_dot_f2_all, dim = 0)
        dyn_models_x_dot_f2_all_std = torch.std(dyn_models_x_dot_f2_all, dim = 0)
        dyn_models_x_dot_unnormed_all_mean = torch.mean(dyn_models_x_dot_unnormed_all, dim = 0)
        return {'x_dot_f1_mean':dyn_models_x_dot_f1_all_mean.numpy(), 'x_dot_f1_std':dyn_models_x_dot_f1_all_std.numpy(),
                'x_dot_f2_mean':dyn_models_x_dot_f2_all_mean.numpy(), 'x_dot_f2_std':dyn_models_x_dot_f2_all_std.numpy(),
                'x_dot_unnormed_mean':dyn_models_x_dot_unnormed_all_mean.numpy()}
    return ensemble_forward


def ensemble_dyn_training(state_dim, control_dim, dir_path, config_file_name):
    STATE_DIM = state_dim
    CONTROL_DIM = control_dim
    config_file_fname = os.path.join(dir_path, config_file_name) # absolute path of the config file
    opt = config_parsing(config_file_fname)[0]
    experiment_logging_path = os.path.join(dir_path, 'logs',opt.experiment_name) # absolute path of the current experiment logging directory
    if opt.if_gpu == 'True':
        if_gpu = True
        if torch.cuda.is_available():
            gpu_device = 'cuda'
            print('Using GPU:CUDA', os.environ["CUDA_VISIBLE_DEVICES"])
        else:
            gpu_device = torch.device("mps")
            print('Using GPU:MPS')
    elif opt.if_gpu == 'False':
        if_gpu = False
        gpu_device = 'cpu'
        print('Using CPU')

    if opt.batch_norm == "True":
        batch_norm = True
    elif opt.batch_norm == "False":
        batch_norm = False

    if opt.log_to_wandb == "True":
        wandb.init(
            entity='',
            project=opt.wandb_project_name,
            config = opt,
            name = opt.experiment_name
        )
        log_to_wandb = True
    elif opt.log_to_wandb == "False":
        log_to_wandb = False

    dyn_dataset_summary = loadmat(os.path.join(dir_path, 'dyn_datasets', opt.dataset_name + '_summary.mat'))
    inputs_mean = torch.tensor(dyn_dataset_summary['inputs_mean'], dtype=torch.float32)
    inputs_std = torch.tensor(dyn_dataset_summary['inputs_std'], dtype=torch.float32)
    labels_mean = torch.tensor(dyn_dataset_summary['labels_mean'], dtype=torch.float32)
    labels_std = torch.tensor(dyn_dataset_summary['labels_std'], dtype=torch.float32)
    
    models = [CAFCNet(state_dim=STATE_DIM, control_dim=CONTROL_DIM, num_layers=opt.num_layers, num_neurons_per_layer=opt.num_neurons_per_layer, if_batch_norm = batch_norm, 
            inputs_mean=inputs_mean, inputs_std=inputs_std, labels_mean=labels_mean, labels_std=labels_std, if_gpu=if_gpu) for i in range(opt.num_models)]
    
    # confirm experiment set up
    with open(config_file_fname, mode = 'r') as config_file:
        for lines in config_file:
            print(lines)
    # model_summary = summary(models[0], input_size=(opt.batch_size, int(STATE_DIM+CONTROL_DIM)), col_names = ("input_size", "output_size"),row_settings=["var_names"], verbose=1)
    val = input("Continue to training (y/n)?")
    if val != 'y':
        print('-------------Exit-----------')
        exit()

    if opt.checkpoint_toload == 0:
        if os.path.exists(experiment_logging_path):
            val = input("The experiment directory %s exists. Overwrite? (y/n)"%experiment_logging_path)
            if val == 'y':
                shutil.rmtree(experiment_logging_path)
        os.makedirs(experiment_logging_path)

    # sys.stdout = utils.Logger(os.path.join(experiment_logging_path, 'log.txt'))
    
    # write to experiment set up summary file
    summary_file_name = os.path.join(experiment_logging_path, 'summary_file.txt')
    if os.path.exists(summary_file_name):
            val = input("The summary file %s exists. Overwrite? (y/n)" % summary_file_name)
            if val == 'y':
                os.remove(summary_file_name)
            else:
                print("Summary file not removed. Exit training")
                exit()
    with open(summary_file_name, mode='a') as summary_file, open(config_file_fname, mode = 'r') as config_file:
        now = datetime.now()
        dt_string = now.strftime("%m/%d/%Y %H:%M:%S") + '\n'
        summary_file.write(dt_string)
        for lines in config_file:
            summary_file.write(lines)
        summary_file.write('\n')
    
    # load data
    dyn_dataset = loadmat(os.path.join(dir_path, 'dyn_datasets', opt.dataset_name + '.mat'))
    train_inputs = torch.tensor(dyn_dataset['train_inputs'], dtype=torch.float32)
    train_labels = torch.tensor(dyn_dataset['train_labels'], dtype=torch.float32)
    eval_inputs = torch.tensor(dyn_dataset['eval_inputs'], dtype=torch.float32)
    eval_labels = torch.tensor(dyn_dataset['eval_labels'], dtype=torch.float32)
    print('dataset size = ', train_inputs.shape[0])

    # only need to normalize the label; the input normalization is done within the model
    train_labels = (train_labels - labels_mean) / labels_std

    for key in dyn_dataset_summary.keys():
        if type(dyn_dataset_summary[key]) == np.ndarray:
            dyn_dataset_summary[key] = torch.tensor(dyn_dataset_summary[key], dtype = torch.float32).to(gpu_device)

    loss_fn = torch.nn.MSELoss(reduction = 'sum')
    # train ensemble
    for model_idx, model in enumerate(models):
        optimizer = torch.optim.Adam(lr=opt.lr, params=model.parameters(), weight_decay=opt.weight_decay)
        training_set = DynDataset(range(train_inputs.shape[0]), inputs = train_inputs, labels = train_labels)
        train_loader = DataLoader(training_set, batch_size=opt.batch_size, shuffle = True)
        eval_set = DynDataset(range(eval_inputs.shape[0]), inputs = eval_inputs, labels = eval_labels)
        eval_loader = DataLoader(eval_set, batch_size=opt.num_eval_samples, shuffle = True)
        if model_idx > 0: # for dyn training, only log one model training session
            log_to_wandb = False 
        trained_model = training.train(model=model, model_idx = model_idx, train_loader=train_loader, epochs=opt.num_epochs, optimizer = optimizer, state_dim = STATE_DIM,
                steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                model_dir=experiment_logging_path, loss_fn=loss_fn, clip_grad=opt.clip_grad,
                use_lbfgs=opt.use_lbfgs, eval_loader = eval_loader, validation_fn=None, start_epoch=opt.checkpoint_toload, 
                l1_lambda=opt.l1_lambda, num_train_samples = train_inputs.shape[0], 
                batch_size = opt.batch_size, dataset_func = None, sim_func=None, dataset_stats = dyn_dataset_summary, gpu_device = gpu_device, log_to_wandb = log_to_wandb)
    print('--------------------------------Training Complete------------------------------')
    print(opt)

