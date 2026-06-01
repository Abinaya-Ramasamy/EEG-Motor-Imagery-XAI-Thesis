import os
from argparse import ArgumentParser
from torch import load as torch_load, save
from pandas import DataFrame, ExcelWriter
from numpy import load, unique, array, swapaxes, expand_dims, argmax
from pandas import set_option
from time import time
from os import path as os_path
from warnings import filterwarnings
filterwarnings('ignore')

from sys import path
path.append('/projects/sciences/computing/ramab620/eegxai-spec-rnn/utils')
from xai_eval import GetRobustness
from eeg_models import SelectModel
from eeg_train import DataGenerator, DataLoader, EvalModel, GetDataTest
from eeg_dbf import LblEncoding

from copy import deepcopy
import torch
from torch import nn

def RandomiseWeightsAny(model, only_types=(nn.LSTM, nn.GRU, nn.RNN, nn.Linear, nn.Conv1d, nn.Conv2d)):
    """
    Randomise weights for robustness testing.
    Resets parameters of supported layers (LSTM, Linear, etc.)
    """
    print("Randomising weights for model...")
    rand_model = deepcopy(model)

    for m in rand_model.modules():
        if isinstance(m, only_types) and hasattr(m, "reset_parameters"):
            m.reset_parameters()

    return rand_model

# File arguments
#parser = ArgumentParser()
#parser.add_argument('--domain', type=str, required=False, choices=['temporal', 'spectral', 'spatial'], default="temporal")
#parser.add_argument('--model', type=str, required=False, choices=['TwoDCNN', 'EEGNet'], default="TwoDCNN")
#parser.add_argument('--xai_methods', nargs='+', type=str, required=False, 
 #                   default= ['Saliency', 'Deconv', 'Guid-BP' ,'Guid-GCam', 'DeepLift', 'LRP',
  #                  'IxG', 'GradCam', 'GradCam++', 'ScoreCam','FullGrad', 'LayerCAM', 'IG'])
#parser.add_argument('--eval_method', type=str, required=False, choices=['PC', 'SSIM'], default="PC") 
#parser.add_argument('--rand_name', type=str, required=False,choices=['weight', 'label'], default="weight") 
#parser.add_argument('--n_instance', type=int, required=False, default=None)

#args = parser.parse_args()
domain = 'spectral'
model_name = 'LSTM'
xai_methods_list = ['Saliency', 'DeepLift', 'IxG', 'IG', 'LRP']                                                                                                                                                             
n_instance = None 
#model_name = args.model
#xai_methods_list = args.xai_methods
eval_method = 'PC'
rand_name = 'weight'
snr =  '-3.5'
set_option('display.float_format', '{:.6f}'.format)
#n_instance = args.n_instance

print('\n--------------------------------------------------------------------------------------------------------')
print(f'\nEvaluation robustness on {domain} eeg data, with {snr} noise using {model_name}') 
print(f'Xai_methods : {xai_methods_list}')
print(f'Evaluation matrics: {eval_method}')
print(f'Random name: {rand_name}')


# Provide link for data and result 
data_link = f'/projects/sciences/computing/ramab620/eegxai-spec-rnn/dataset/{snr}_{domain}'
model_link = f'/projects/sciences/computing/ramab620/eegxai-spec-rnn/dataset/{model_name}/'
result_link = f'/projects/sciences/computing/ramab620/eegxai-spec-rnn/result/{model_name}/'
os.makedirs(result_link, exist_ok=True)

# Load eeg data
try :
    print(f'Opening eeg data files...')
    eeg_data = load(f'{data_link}_data.npy', mmap_mode='r')
    print(f'Opening ground-truth data files...')
    gt_data = load(f'{data_link}_gt.npy', mmap_mode='r')
    print(f'Opening label data files...')
    labels = load(f'{data_link}_labels.npy', mmap_mode='r')
    rand_labels = load(f'{data_link}_random_labels.npy', mmap_mode='r')
    channel_names = load(f'{data_link}_channel_names.npy')
except ValueError as e:
    print(f"Error in loading file: {e}")

n_class = len(unique(labels))
n_chan = array(eeg_data).shape[1]
n_sample = array(eeg_data).shape[2]
#eeg_data = expand_dims(eeg_data, axis=1)

labels_encoded = LblEncoding (labels, num_classes = n_class)
random_labels_encoded = LblEncoding (rand_labels, num_classes = n_class)
if n_instance == None:
    n_instance = len(eeg_data)

print(f'Dataset size: {eeg_data.shape}')
print(f'Sample size: {n_instance}')

# Implement XAI methods
n_fold = 5
generator_size = len(eeg_data)//n_fold
batch_size = len(eeg_data)//100

all_time = []
scores_fold = DataFrame(columns=xai_methods_list)
all_fold = DataFrame(columns=xai_methods_list)

# Implement XAI methods
for fold_id in range (n_fold): 
    print('=========================================================================')
    print(f"Fold-{fold_id}")

    # Load data every fold
    X_test, gt_test, y_test = GetDataTest(eeg_data, gt_data, labels_encoded, fold_id, generator_size)
    _, _, y_rand = GetDataTest(eeg_data, gt_data, random_labels_encoded, fold_id, generator_size)

    # LSTM expects (N, T, C) not (N, C, T)
    #X_test = swapaxes(X_test, 1, 2)

    X_test_gen = DataGenerator(X_test, y_test)
    ori_loader = DataLoader(X_test_gen, batch_size=batch_size)
    xb, yb = next(iter(ori_loader))
    print("Batch X shape:", xb.shape)

    labels_test = argmax(y_test, axis= 1)
    
    #Eval model 
    ori_model_path = f'{model_link}{snr}_{domain}_best_fold{fold_id}_{model_name}.pth'
    print("Loading model:", ori_model_path)
    ori_model = SelectModel(model_name, n_class=n_class, n_chan=n_chan, n_samples =n_sample)
    ori_model.load_state_dict(torch_load(ori_model_path))
    ori_model.eval()
    _, ori_preds = EvalModel(ori_model, ori_loader)
    
     #load randomised model
    rand_model = SelectModel(model_name, n_class=n_class, n_chan=n_chan, n_samples =n_sample)
    rand_model_path = f'{model_link}{snr}_{domain}_random_{rand_name}_fold{fold_id}_{model_name}.pth'
    if rand_name =="label":
        #load and eval randomised labels model
        rand_labels = load(f'{data_link}_random_labels.npy', mmap_mode='r')
        rand_data = DataGenerator(X_test, y_rand)
        rand_loader = DataLoader(rand_data, batch_size=batch_size, shuffle=False)
        random_labels_test = argmax(y_rand, axis= 1)

        rand_model.load_state_dict(torch_load(rand_model_path))
        rand_model.eval()
        _, rand_preds = EvalModel (rand_model, rand_loader)

    elif rand_name =="weight":  
        if os_path.exists(rand_model_path):
            print('Load randomised weight model...')
            rand_model.load_state_dict(torch_load(rand_model_path))
        else:
            print('Randomised weight model...')
            rand_model = RandomiseWeightsAny(ori_model)
            # -------- ADD THIS BLOCK HERE --------
            import torch
            with torch.no_grad():
                ow = next(ori_model.parameters()).flatten()[:5].cpu()
                rw = next(rand_model.parameters()).flatten()[:5].cpu()
            print("ori weight sample:", ow)
            print("rnd weight sample:", rw)
        # -------------------------------------
            save(rand_model.state_dict(), rand_model_path) 

        rand_model.eval()
        _, rand_preds = EvalModel (rand_model, ori_loader)
        random_labels_test = argmax(y_test, axis= 1)   #using original labels

      
    #Apply XAI methods 
    start_time = time()
    scores_fold, _, _= GetRobustness (ori_model, rand_model, X_test[:n_instance], labels_test[:n_instance], random_labels_test[:n_instance], 
                                      ori_preds[:n_instance], rand_preds[:n_instance],xai_methods_list = xai_methods_list, eval_method=eval_method, index=None)
    end_time = time()
    execution_time = end_time - start_time
    
    #average all samples in this fold
    average_scores = scores_fold.abs().mean()
    std_scores = scores_fold.abs().std()
    scores_fold.loc['avg'] = average_scores
    scores_fold.loc['std'] = std_scores
    all_fold.loc[fold_id] = average_scores
    all_time.append(int(execution_time))

    #save fold result
    result_fold_path = f'{result_link}{snr}_{domain}_{model_name}_{eval_method}_{rand_name}_fold{fold_id}.xlsx'
    with ExcelWriter(result_fold_path, engine='openpyxl') as writer: 
        scores_fold.to_excel(writer, sheet_name=f'fold{fold_id}', index=True)

    print(f'Evaluate model: {ori_model_path}')
    print(f'Number of sample in this fold: {len(X_test)}')
    print(f'\nResult {eval_method}, on fold {fold_id+1}:\n {scores_fold.tail(10)}\n')

#averaged all fold
all_fold['time'] = all_time
average_all_scores = all_fold.abs().mean()
std_all_scores = all_fold.abs().std()
all_fold.loc['avg'] = average_all_scores
all_fold.loc['std'] = std_all_scores
print(f'\nResult Robustness Test ({rand_name} random-{eval_method}) using {model_name} on {domain} domain :\n {all_fold}\n')

data_info = DataFrame(columns=['Variable', 'value'])
data_info.loc [0] = ['Domain', domain]
data_info.loc[1] = ['SNR', snr]
data_info.loc[2] = ['Model', model_name]
data_info.loc[3] = ['Eval method', eval_method]
data_info.loc[4] = ['Random', rand_name]

#Save all result
result_path = f'{result_link}{snr}_{domain}_{model_name}_{eval_method}_{rand_name}.xlsx'
with ExcelWriter(result_path, engine='openpyxl') as writer: 
    data_info.to_excel(writer, sheet_name='info', index=False)
    all_fold.to_excel(writer, sheet_name='all fold', index=True)