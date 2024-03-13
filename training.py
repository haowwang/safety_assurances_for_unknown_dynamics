'''Implements a generic training loop.
'''

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os, io
import copy
import wandb
import matplotlib.pyplot as plt
import plotly.express as px

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)



def train(model, model_idx, train_loader, epochs, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn, optimizer, state_dim, 
         summary_fn=None, eval_loader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None,
          validation_fn=None, start_epoch=0, l1_lambda=0, num_train_samples = 0, batch_size = 0, dataset_func = None, 
          sim_func = None, dataset_stats=None, gpu_device = 'cuda', log_to_wandb = False):

    model.to(gpu_device)
    dataset_states_cpu = {} # for storing in model checkpoint
    for key in dataset_stats.keys():
        if type(dataset_stats[key]) == torch.Tensor:
            dataset_states_cpu[key] = dataset_stats[key].cpu()

    # copy settings from Raissi et al. (2019) and here 
    # https://github.com/maziarraissi/PINNs
    # if use_lbfgs:
    #     optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
    #                               history_size=50, line_search_fn='strong_wolfe')

    # Load the checkpoint if required
    if start_epoch > 0:
        # Load the model and start training from that point onwards
        model_path = os.path.join(model_dir, 'checkpoints_model_idx'+str(model_idx), 'model_epoch_%04d.pth' % start_epoch)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        model.train()
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer.param_groups[0]['lr'] = optimizer.state_dict()['param_groups'][0]['lr']
        assert(start_epoch == checkpoint['epoch'])

    summaries_dir = os.path.join(model_dir, 'summaries_model_idx'+str(model_idx))
    cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints_model_idx'+str(model_idx))
    cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    with tqdm(total= (1 + int(num_train_samples / batch_size)) * epochs) as pbar:
        train_losses = []
        for epoch in range(start_epoch, epochs):
            # save model and stats
            if not epoch % epochs_til_checkpoint and epoch:
                checkpoint = { 
                    'epoch': epoch,
                    'model': copy.deepcopy(model).to('cpu').state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'training_dataset_stats':dataset_states_cpu}
                torch.save(checkpoint,
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))
                if validation_fn is not None:
                    validation_fn(model, checkpoints_dir, epoch)

            sum_epoch_mse_train_loss_pseudo = 0.
            sum_epoch_mse_train_loss_true = 0.
            start_time = time.time()
            sum_epoch_train_per_channel_l1_loss_true = None
            
            # train with mini batch
            for step, (model_input, gt) in enumerate(train_loader):
                model.train()
                model_input = model_input.to(gpu_device)
                gt = gt.to(gpu_device)
                model_output = model(model_input)
                sum_batch_mse_train_loss_pseudo = loss_fn(model_output, gt) # loss function applied sum reduction
                model_output_unnormed = model_output * dataset_stats['labels_std'] + dataset_stats['labels_mean']
                gt_unnormed = gt * dataset_stats['labels_std'] + dataset_stats['labels_mean']
                sum_batch_train_per_channel_l1_loss_true = torch.sum(torch.abs(gt_unnormed - model_output_unnormed), dim = 0)
                sum_batch_mse_train_loss_true = loss_fn(model_output_unnormed, gt_unnormed)

                sum_epoch_mse_train_loss_pseudo += sum_batch_mse_train_loss_pseudo
                sum_epoch_mse_train_loss_true += sum_batch_mse_train_loss_true

                if sum_epoch_train_per_channel_l1_loss_true is None:
                    sum_epoch_train_per_channel_l1_loss_true = sum_batch_train_per_channel_l1_loss_true
                else:
                    sum_epoch_train_per_channel_l1_loss_true =  sum_epoch_train_per_channel_l1_loss_true + sum_batch_train_per_channel_l1_loss_true

                # l1 regularization
                if l1_lambda > 0:
                    l1_penalty = torch.nn.L1Loss()
                    regularization_loss = 0.
                    for p in model.parameters():
                        if p.requires_grad:
                            regularization_loss += l1_penalty(p, torch.zeros_like(p))
                    regularization_loss = l1_lambda * regularization_loss
                else:
                    regularization_loss = 0.

                if not use_lbfgs:
                    model.zero_grad()
                    avg_batch_mse_train_loss_pseudo = sum_batch_mse_train_loss_pseudo / (gt.shape[0] * gt.shape[1]) # avg mse loss per sample
                    batch_total_loss = avg_batch_mse_train_loss_pseudo + regularization_loss
                    batch_total_loss.backward()

                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                    optimizer.step()

                pbar.update(1)
            eval_loss_quantile = 0.95
            if eval_loader is not None:
                avg_eval_mse_loss, avg_eval_per_channel_l1_loss = model_eval_analysis(model = model, data_loader = eval_loader, dataset_stats = dataset_stats, mode = 'eval', 
                                            gpu_device = gpu_device, eval_loss_quantile = eval_loss_quantile, epoch=epoch, log_to_wandb=log_to_wandb)
            avg_epoch_mse_train_loss_true = sum_epoch_mse_train_loss_true.detach().cpu().item() / (len(train_loader.dataset)*state_dim)
            avg_epoch_mse_train_loss_pseudo = sum_epoch_mse_train_loss_pseudo.detach().cpu().item() / (len(train_loader.dataset)*state_dim) # avg mse training loss per sample
            avg_epoch_train_per_channel_l1_loss_true = sum_epoch_train_per_channel_l1_loss_true.detach().cpu() / len(train_loader.dataset)
            writer.add_scalar('MSE Validation Loss True', avg_eval_mse_loss, epoch)
            writer.add_scalar('MSE Training Loss Pseudo', avg_epoch_mse_train_loss_pseudo, epoch)
            writer.add_scalar('MSE Training Loss True', avg_epoch_mse_train_loss_true, epoch)
            output_str = "M {:d} E {:d} ".format(model_idx, epoch)
            output_str += "TrainMSETrue {:.6f} ".format(avg_epoch_mse_train_loss_true)
            output_str += "EvalMSE {:.6f} ".format(avg_eval_mse_loss)
            for dim in range(state_dim):
                writer.add_scalar('train l1 loss channel' + str(dim), avg_epoch_train_per_channel_l1_loss_true[dim], epoch)
                writer.add_scalar('eval l1 loss channel' + str(dim), avg_eval_per_channel_l1_loss[dim], epoch)
                output_str += 'TL1_'+str(dim)+" {:0.4f} ".format(avg_epoch_train_per_channel_l1_loss_true[dim])
                output_str += 'VL1_'+str(dim)+" {:0.4f} ".format(avg_eval_per_channel_l1_loss[dim])
                if log_to_wandb:
                    wandb.log({'train l1 loss channel ' + str(dim): avg_epoch_train_per_channel_l1_loss_true[dim], 
                               'eval l1 loss channel ' + str(dim): avg_eval_per_channel_l1_loss[dim]}, commit=False)
            tqdm.write(output_str)
            torch.save({'model':copy.deepcopy(model).to('cpu').state_dict(), 'training_dataset_stats':dataset_states_cpu}, os.path.join(checkpoints_dir, 'model_current.pth'))
            if log_to_wandb:
                wandb.log({'MSE Validation Loss':avg_eval_mse_loss, 'MSE Training Loss Pseudo': avg_epoch_mse_train_loss_pseudo, 'MSE Training Loss True': avg_epoch_mse_train_loss_true,
                           })
        
        if eval_loader is not None:
            avg_eval_mse_loss, avg_eval_per_channel_l1_loss = model_eval_analysis(model = model, data_loader = eval_loader, dataset_stats = dataset_stats, mode = 'eval', 
                                            gpu_device = gpu_device, eval_loss_quantile = eval_loss_quantile, epoch=epoch, log_to_wandb=log_to_wandb)
            print('--------Eval L1 loss per channel-------')
            print(avg_eval_per_channel_l1_loss)

        torch.save({'model':copy.deepcopy(model).to('cpu').state_dict(), 'training_dataset_stats':dataset_states_cpu},
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        return model


def model_val_analysis(model, data_loader, dataset_stats, mode, gpu_device):
    model.eval()
    with torch.no_grad():
        sum_epoch_gt = None # used in percent error calculation only
        sum_epoch_abs_error = None
        sum_epoch_mse_loss = 0. # computed using unnormalized model outputs
        loss_fn = torch.nn.MSELoss(reduction='sum')
        for model_input, gt in data_loader: # inputs are already normalized according to training distribution
            model_input = model_input.to(gpu_device)
            gt = gt.to(gpu_device)
            if mode == 'train': # evaluate model performance on training set (usually done after training). gt for eval dataset need no unnormalization
                gt = gt * dataset_stats['labels_std'] + dataset_stats['labels_mean']
            model_output = model(model_input)[0] # in eval mode [0] argument is the unnormalized x dot
            sum_epoch_mse_loss += loss_fn(model_output, gt)
            if sum_epoch_abs_error is None:
                sum_epoch_abs_error = torch.sum(torch.abs(gt - model_output), dim = 0, keepdim = False)
                sum_epoch_gt = torch.sum(torch.abs(gt), dim = 0)
            else:
                sum_epoch_abs_error += torch.sum(torch.abs(gt - model_output), dim = 0, keepdim = False)
                sum_epoch_gt += torch.sum(torch.abs(gt), dim = 0)
        avg_epoch_abs_error = sum_epoch_abs_error / len(data_loader.dataset) # avg l1 loss of each channel per sample 
        avg_epoch_gt = sum_epoch_gt / len(data_loader.dataset) # used in percent error calculation only
        avg_epoch_abs_percent_error = (avg_epoch_abs_error / avg_epoch_gt) * 100 # avg percent l1 loss of each channel per sample
        avg_epoch_mse_loss = sum_epoch_mse_loss / (gt.shape[1] * len(data_loader.dataset))
    return avg_epoch_abs_error, avg_epoch_abs_percent_error, avg_epoch_mse_loss


# def model_eval_analysis(model, data_loader, dataset_stats, mode, gpu_device, eval_loss_quantile, epoch, log_to_wandb):
#     model.eval()
#     state_dim = 0
#     with torch.no_grad():
#         loss_fn = torch.nn.MSELoss(reduction='sum')
#         eval_set_mse_loss = 0.
#         eval_set_l1_loss_aggregate = None
#         eval_set_l1_percent_loss_aggregate = None
#         # dist_between_pusher_and_object_aggregate = None
#         gt_aggregate = None
#         for model_input, gt in data_loader: # inputs are already normalized according to training distribution
#             model_input = model_input.to(gpu_device)
#             gt = gt.to(gpu_device)
#             state_dim = gt.shape[1]
#             model_output = model(model_input)[0] # in eval mode [0] argument is the unnormalized x_dot
#             eval_set_mse_loss += loss_fn(model_output, gt)
#             eval_set_batch_l1_loss = torch.abs(model_output - gt)
#             dist_between_pusher_and_object = ((model_input[:,[0]]-model_input[:,[2]]) ** 2 + (model_input[:,[1]] - model_input[:,[3]]) ** 2) ** 0.5
#             eval_set_batch_l1_percent_loss = torch.abs(eval_set_batch_l1_loss / gt) * 100
#             if eval_set_l1_loss_aggregate is None:
#                 eval_set_l1_loss_aggregate = eval_set_batch_l1_loss
#                 eval_set_l1_percent_loss_aggregate = eval_set_batch_l1_percent_loss
#                 # dist_between_pusher_and_object_aggregate = dist_between_pusher_and_object
#                 gt_aggregate = gt
#             else:
#                 eval_set_l1_loss_aggregate = torch.vstack((eval_set_l1_loss_aggregate, eval_set_batch_l1_loss))
#                 eval_set_l1_percent_loss_aggregate = torch.vstack((eval_set_l1_percent_loss_aggregate, eval_set_batch_l1_percent_loss))
#                 # dist_between_pusher_and_object_aggregate = torch.vstack((dist_between_pusher_and_object_aggregate, dist_between_pusher_and_object))
#                 gt_aggregate = torch.vstack((gt_aggregate, gt))
#         eval_set_mse_loss = eval_set_mse_loss / (len(data_loader.dataset) * gt.shape[1])
#         eval_set_qx_l1_loss_per_channel = torch.quantile(eval_set_l1_loss_aggregate, eval_loss_quantile, dim=0)
#         eval_set_median_l1_loss_per_channel = torch.median(eval_set_l1_loss_aggregate, dim=0)[0]
#         # eval_set_qx_percent_l1_loss_per_channel = torch.quantile(eval_set_l1_percent_loss_aggregate, eval_loss_quantile, dim=0)
#         if log_to_wandb:
#             for dim in range(state_dim):
#                 wandb.log({str(int(eval_loss_quantile*100)) + ' quantile l1 validation loss channel ' + str(dim): eval_set_qx_l1_loss_per_channel[dim], 
#                                 'median l1 validation loss channel ' + str(dim):eval_set_median_l1_loss_per_channel[dim]}, commit=False)
#                 # fig = px.scatter(x=dist_between_pusher_and_object_aggregate.squeeze().detach().cpu().numpy(), y=eval_set_l1_loss_aggregate[:,dim].detach().cpu().numpy(), color = torch.abs(gt_aggregate[:,dim]).detach().cpu().numpy(), labels={'x':'pusher object distance','y':'l1 loss per channel', 'color':'gt l1 value'})
#                 # wandb.log({"pusher object dist vs. l1 loss channel " + str(dim): fig}, commit = False)
#     return eval_set_mse_loss, eval_set_qx_l1_loss_per_channel, eval_set_median_l1_loss_per_channel


def model_eval_analysis(model, data_loader, dataset_stats, mode, gpu_device, eval_loss_quantile, epoch, log_to_wandb):
    model.eval()
    state_dim = 0
    with torch.no_grad():
        loss_fn = torch.nn.MSELoss(reduction='sum')
        sum_eval_mse_loss = 0. # sum of error squared for all sampled across all channel
        eval_raw_l1_loss_aggregate = None
        model_input_x_y_aggregate = None
        for model_input, gt in data_loader: # inputs are already normalized according to training distribution
            model_input = model_input.to(gpu_device)
            gt = gt.to(gpu_device)
            state_dim = gt.shape[1]
            model_output = model(model_input)[0] # in eval mode [0] argument is the unnormalized x_dot
            batch_eval_raw_l1_loss = torch.abs(model_output - gt)
            if eval_raw_l1_loss_aggregate is None:
                eval_raw_l1_loss_aggregate = batch_eval_raw_l1_loss
                model_input_x_y_aggregate = model_input[:, :2]
            else:
                eval_raw_l1_loss_aggregate = torch.vstack((eval_raw_l1_loss_aggregate, batch_eval_raw_l1_loss))
                model_input_x_y_aggregate = torch.vstack((model_input_x_y_aggregate, model_input[:, :2]))
            sum_eval_mse_loss += loss_fn(model_output, gt)


        avg_eval_per_channel_l1_loss = torch.mean(eval_raw_l1_loss_aggregate, dim = 0)
        avg_eval_mse_loss = sum_eval_mse_loss / (len(data_loader.dataset) * gt.shape[1])
        
        # model_input_x_y_aggregate = model_input_x_y_aggregate.detach().cpu()
        # channel_omega_raw_l1_loss = eval_raw_l1_loss_aggregate[:,3].detach().cpu()

        # sc = plt.scatter(model_input_x_y_aggregate[:,0], model_input_x_y_aggregate[:,1], c = channel_omega_raw_l1_loss, cmap="PuBu")
        # plt.colorbar(sc)
        # plt.savefig('figs/epoch'+str(epoch)+'.png')
        # plt.close()




        # if log_to_wandb:
        #     for dim in range(state_dim):
        #         wandb.log({str(int(eval_loss_quantile*100)) + ' quantile l1 validation loss channel ' + str(dim): eval_set_qx_l1_loss_per_channel[dim], 
        #                         'median l1 validation loss channel ' + str(dim):eval_set_median_l1_loss_per_channel[dim]}, commit=False)
                # fig = px.scatter(x=dist_between_pusher_and_object_aggregate.squeeze().detach().cpu().numpy(), y=eval_set_l1_loss_aggregate[:,dim].detach().cpu().numpy(), color = torch.abs(gt_aggregate[:,dim]).detach().cpu().numpy(), labels={'x':'pusher object distance','y':'l1 loss per channel', 'color':'gt l1 value'})
                # wandb.log({"pusher object dist vs. l1 loss channel " + str(dim): fig}, commit = False)
    return avg_eval_mse_loss, avg_eval_per_channel_l1_loss
 
 