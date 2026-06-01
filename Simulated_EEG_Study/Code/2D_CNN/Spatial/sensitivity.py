from argparse import ArgumentParser
from torch import load as torch_load
from pandas import DataFrame, ExcelWriter
from numpy import load, unique, array, swapaxes, expand_dims, argmax
from time import time
from pandas import set_option
from warnings import filterwarnings
filterwarnings('ignore')

from sys import path
path.append('/projects/sciences/computing/ramab620/eegxai/utils')
from eeg_train import DataGenerator, DataLoader, EvalModel, GetDataTest
from eeg_dbf import LblEncoding
from xai_eval import GetSenitivity
from eeg_models import SelectModel

# File arguments
#parser = ArgumentParser()
#parser.add_argument('--domain', type=str, required=False,choices=['temporal', 'spectral', 'spatial'], default="temporal")
#parser.add_argument('--model', type=str, required=False, choices=['TwoDCNN', 'EEGNet'], default="TwoDCNN")
#parser.add_argument('--xai_methods', nargs='+', type=str, required=False, 
                    #default= ['Saliency', 'Deconv', 'Guid-BP' ,'Guid-GCam', 'DeepLift', 'LRP',
                    #'IxG', 'GradCam', 'GradCam++', 'ScoreCam','FullGrad', 'LayerCAM', 'IG']) 
#parser.add_argument('--n_instance', type=int, required=False, default=None)


#args = parser.parse_args()
domain = 'spatial'
model_name = 'TwoDCNN'
xai_methods_list = ['Saliency', 'Deconv', 'Guid-BP' ,'Guid-GCam', 'DeepLift', 'LRP',
                    'IxG', 'GradCam', 'GradCam++', 'ScoreCam','FullGrad', 'LayerCAM', 'IG']
n_instance = None
if domain == 'spatial':
    eval_method = 'CS'
else:
    eval_method = 'RMA'
snr = '-3.5'
set_option('display.float_format', '{:.6f}'.format)


print('\n--------------------------------------------------------------------------------------------------------')
print(f'\nEvaluation sensitivity on {domain} eeg data, with {snr} noise using {model_name}') 
print(f'Xai_methods : {xai_methods_list}')
print(f'Evaluation matrics: {eval_method}')
print(f'Model: {model_name}')

# Provide link for data and result 
data_link = f'/projects/sciences/computing/ramab620/eegxai/dataset/{snr}_{domain}'
model_link = f'/projects/sciences/computing/ramab620/eegxai/dataset/{model_name}/'
result_link = f'/projects/sciences/computing/ramab620/eegxai/result/{model_name}/'

# Load eeg data
try :
    print(f'Opening eeg data files...')
    eeg_data = load(f'{data_link}_data.npy', mmap_mode='r')
    print(f'Opening ground-truth data files...')
    gt_data = load(f'{data_link}_gt.npy', mmap_mode='r')
    print(f'Opening label data files...')
    labels = load(f'{data_link}_labels.npy', mmap_mode='r')
    channel_names = load(f'{data_link}_channel_names.npy')
except ValueError as e:
    print(f"Error in loading file: {e}")

n_class = len(unique(labels))
n_chan = array(eeg_data).shape[1]
n_sample = array(eeg_data).shape[2]
eeg_data = expand_dims(eeg_data, axis=1)

labels_encoded = LblEncoding (labels, n_class)
if n_instance == None:
    n_instance = len(eeg_data)

print(f'Dataset size: {eeg_data.shape}')
print(f'Sample size: {n_instance}')

#parameters
n_fold = 5
generator_size = len(eeg_data)//n_fold
batch_size = len(eeg_data)//100

all_time = []
scores_fold = DataFrame(columns=xai_methods_list)
all_fold = DataFrame(columns=xai_methods_list)

th_gt = 0.05
th_xai_list = [ 0.25, 0.75] #[th captum, th grad cam]
   
# Implement XAI methodss
for fold_id in range(n_fold): 
    print('=========================================================================')
    print(f"Fold-{fold_id+1}")

    # Load data every fold
    X_test, gt_test, y_test = GetDataTest(eeg_data, gt_data, labels_encoded, fold_id, generator_size)
    X_test_gen = DataGenerator(X_test, y_test)
    X_test_loader = DataLoader(X_test_gen, batch_size=batch_size)

    #Eval model 
    fold_model_path = f'{model_link}{snr}_{domain}_best_fold{fold_id}_{model_name}.pth'
    fold_model = SelectModel(model_name, n_class=n_class, n_chan=n_chan, n_samples =n_sample)
    fold_model.load_state_dict(torch_load(fold_model_path))
    _, fold_preds = EvalModel (fold_model, X_test_loader)
    label_test = argmax(y_test, axis= 1)

    #Apply XAI methods 
    start_time = time()
    scores_fold, _= GetSenitivity (fold_model, X_test[:n_instance], label_test[:n_instance], fold_preds[:n_instance], gt_test[:n_instance], th_gt, th_xai_list,  
                    xai_methods_list= xai_methods_list, eval_method=eval_method, index= None)
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
    result_fold_path = f'{result_link}{snr}_{domain}_{model_name}_{eval_method}_fold{fold_id}.xlsx'
    with ExcelWriter(result_fold_path, engine='openpyxl') as writer: 
        scores_fold.to_excel(writer, sheet_name=f'fold{fold_id}', index=True)
    print(f'Best model in this fold saved at: {result_fold_path}')

    print(f'Evaluate model: {fold_model_path}')
    print(f'Number of sample in this fold: {len(X_test)}')
    print(f'\nResult {eval_method}, on fold {fold_id+1}:\n {scores_fold.tail(10)}\n')

#averaged all fold
all_fold['time'] = all_time
average_all_scores = all_fold.abs().mean()
std_all_scores = all_fold.abs().std()
all_fold.loc['avg'] = average_all_scores
all_fold.loc['std'] = std_all_scores
print(f'\nResult Sensitivity Test ({eval_method}) using {model_name} on {domain} domain :\n {all_fold}\n')

#Save all result
result_path = f'{result_link}{snr}_{domain}_{model_name}_{eval_method}.xlsx'
with ExcelWriter(result_path, engine='openpyxl') as writer: 
    all_fold.to_excel(writer, sheet_name='all fold', index=True)
print(f'Best model overall saved at: {result_path}')