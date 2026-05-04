# ======================== LIBRARIES ========================
# external libraries
import os
import argparse
import random
import warnings
import numpy as np
from sklearn.utils import check_random_state
import datetime
import torch
import yaml
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
# custom libraries
from src.models.training_code import train_model
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent

# ======================== HANDLING PARAMETERS ========================
parser = argparse.ArgumentParser()
parser.add_argument('--use_defaults', type = bool, help='whether to use defaults parameters for training and testing', default=True)
parser.add_argument('--randomseed', type=int, default=2, help='randomseed in split')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--batch_size_val', type=int, default=1)
parser.add_argument('--num_epoch', type=int, default=50)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--num_worker', type=int, default=2)
parser.add_argument('--predthreshold', type=float, default=0.5)
parser.add_argument('--setting', type=str, default='pointnet')
parser.add_argument('--datapath', type=str, default=f"{BASE_DIR}/outputs", help='path to the dataset')
opt = parser.parse_args()
# if use_defaults is True, use the default parameters in config/pointnet.yaml, otherwise use the parameters from command line
default_config = None
if opt.use_defaults:
    with open(str(BASE_DIR/"config"/f"{opt.setting}.yaml"), "r") as f:
        default_config = yaml.safe_load(f)
randomseed = opt.randomseed if default_config is None else default_config['randomseed']
batch_size = opt.batch_size if default_config is None else None if opt.setting == "xgboost" else default_config['batch_size']
batch_size_val = opt.batch_size_val if default_config is None else None if opt.setting == "xgboost" else default_config['batch_size_val']
num_epoch = opt.num_epoch if default_config is None else None if opt.setting == "xgboost" else default_config['num_epoch']
learning_rate = opt.learning_rate if default_config is None else None if opt.setting == "xgboost" else default_config['learning_rate']
num_worker = opt.num_worker if default_config is None else default_config['num_worker']
predthreshold = opt.predthreshold if default_config is None else default_config['predthreshold']
setting = opt.setting if default_config is None else default_config['setting']
raw_datapath=opt.datapath if default_config is None else default_config.get("datapath", f"{BASE_DIR}/outputs")
raw_output_path=getattr(opt, "output_path", f"{BASE_DIR}/outputs") if default_config is None else default_config.get("output_path", f"{BASE_DIR}/outputs/")
datapath=os.path.normpath(os.path.join(BASE_DIR, raw_datapath))
output_path=os.path.normpath(os.path.join(BASE_DIR, raw_output_path))
config = dict(
    randomseed=randomseed,
    batch_size=batch_size,
    batch_size_val=batch_size_val,
    num_epoch=num_epoch,
    learning_rate=learning_rate,
    num_worker=num_worker,
    predthreshold=predthreshold,
    setting=setting,
    datapath=datapath
)
print("actual config: ", config)
# handle chekcpoint direcoty
checkpoint = f"{BASE_DIR}/outputs/checkpoint/seed{randomseed}/{setting}"
if not os.path.exists(checkpoint):
    os.makedirs(checkpoint)

# ======================== UTILITIES========================
# handling cuda determinism
torch.manual_seed(randomseed)
torch.cuda.manual_seed(randomseed)
warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# handling numpy determinism
np.random.seed(randomseed)
# handling random determinism
random.seed(randomseed)
# handling scikit-learn determinism
sklearn_random_state = check_random_state(randomseed)
# handling device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================== MAIN ========================
if __name__ == '__main__':
    now = datetime.datetime.now()
    time_name = now.strftime("%Y-%m-%d-%H:%M")
    time_now = now.strftime("%Y-%m-%d.%H.%M.%S")
    print('#########################################################################################')
    # TensorBoard & csv
    checkpoint_names = []
    boards = []
    trainFs = []
    output_path = os.path.join(checkpoint)
    for k in range(6):
        os.makedirs(os.path.join(output_path, f'exp_fold{k}'), exist_ok=True)
        checkpoint_names.append(os.path.join(output_path, f'exp_fold{k}', time_now))
        boards.append(SummaryWriter(log_dir=checkpoint_names[k]))
        trainFs.append(open(os.path.join(output_path, f'exp_fold{k}', '{}.csv'.format(time_name)), 'w'))
    metrics,  metric_test = train_model(setting = setting, boards = boards, trainFs = trainFs, 
                                        batch_size = batch_size, batch_size_val = batch_size_val, 
                                        num_worker = num_worker, learning_rate = learning_rate, 
                                        device = device, datapath = datapath, output_path = output_path,
                                        num_epoch=num_epoch, predthreshold = predthreshold)   # train
    max_auc = max([metric['auc_bestauc'] for metric in metrics])
    best_fold = [metric['auc_bestauc'] for metric in metrics].index(max_auc)
    # printing results
    acc = metrics[0]['acc'] + metrics[1]['acc'] + metrics[2]['acc'] + metrics[3]['acc'] + metrics[4]['acc']
    auc = metrics[0]['auc'] + metrics[1]['auc'] + metrics[2]['auc'] + metrics[3]['auc'] + metrics[4]['auc']
    acc_bestauc = metrics[0]['acc_bestauc'] + metrics[1]['acc_bestauc'] + metrics[2]['acc_bestauc'] + metrics[3]['acc_bestauc'] + metrics[4]['acc_bestauc']
    auc_bestauc = metrics[0]['auc_bestauc'] + metrics[1]['auc_bestauc'] + metrics[2]['auc_bestauc'] + metrics[3]['auc_bestauc'] + metrics[4]['auc_bestauc']
    recall_bestauc = metrics[0]['recall_bestauc'] + metrics[1]['recall_bestauc'] + metrics[2]['recall_bestauc'] + metrics[3]['recall_bestauc'] + metrics[4]['recall_bestauc']
    precision_bestauc = metrics[0]['precision_bestauc'] + metrics[1]['precision_bestauc'] + metrics[2]['precision_bestauc'] + metrics[3]['precision_bestauc'] + metrics[4]['precision_bestauc']
    f1_score_bestauc = metrics[0]['f1_score_bestauc'] + metrics[1]['f1_score_bestauc'] + metrics[2]['f1_score_bestauc'] + metrics[3]['f1_score_bestauc'] + metrics[4]['f1_score_bestauc']
    print(
        'best acc for fold0: {}, fold1: {}, fold2: {}, fold3: {}, fold4: {}.'.format(
            metrics[0]['acc'], metrics[1]['acc'], metrics[2]['acc'], metrics[3]['acc'], metrics[4]['acc']
        ))
    print('best acc under 5-fold :', acc / 5)

    print(
        'best auc for fold0: {}, fold1: {}, fold2: {}, fold3: {}, fold4: {}.'.format(
            metrics[0]['auc'], metrics[1]['auc'], metrics[2]['auc'], metrics[3]['auc'], metrics[4]['auc']
        ))
    print('best auc under 5-fold :', auc / 5)

    print(
        'acc_bestauc for fold0: {}, fold1: {}, fold2: {}, fold3: {}, fold4: {}.'.format(
            metrics[0]['acc_bestauc'], metrics[1]['acc_bestauc'], metrics[2]['acc_bestauc'], metrics[3]['acc_bestauc'], metrics[4]['acc_bestauc']
        ))
    print('acc_bestauc under 5-fold :', acc_bestauc / 5)

    print(
        'auc_bestauc for fold0: {}, fold1: {}, fold2: {}, fold3: {}, fold4: {}.'.format(
            metrics[0]['auc_bestauc'], metrics[1]['auc_bestauc'], metrics[2]['auc_bestauc'], metrics[3]['auc_bestauc'], metrics[4]['auc_bestauc']
        ))
    print('auc_bestauc under 5-fold :', auc_bestauc / 5)

    print(
        'precision_bestauc for fold0: {}, fold1: {}, fold2: {}, fold3: {}, fold4: {}.'.format(
            metrics[0]['precision_bestauc'], metrics[1]['precision_bestauc'], metrics[2]['precision_bestauc'], metrics[3]['precision_bestauc'], metrics[4]['precision_bestauc']
        ))
    print('precision_bestauc under 5-fold :', precision_bestauc / 5)

    print(
        'f1_score_bestauc for fold0: {}, fold1: {}, fold2: {}, fold3: {}, fold4: {}.'.format(
            metrics[0]['f1_score_bestauc'], metrics[1]['f1_score_bestauc'], metrics[2]['f1_score_bestauc'], metrics[3]['f1_score_bestauc'], metrics[4]['f1_score_bestauc']
        ))
    print('f1_score_bestauc under 5-fold :', f1_score_bestauc / 5)
    print(f"Best model is from fold {best_fold} with AUC: {max_auc}")
    print('acc_test      : {}'.format(metric_test['acc_test']))
    print('auc_test      : {}'.format(metric_test['auc_test']))
    print('recall_test   : {}'.format(metric_test['recall_test']))
    print('precision_test: {}'.format(metric_test['precision_test']))
    print('f1_score_test : {}'.format(metric_test['f1_score_test']))
    print('')
    # ========= output csv =========
    for k in range(6):
        if k < 5:
            trainFs[k].write('fold:,fold{}\n'.format(k))
            trainFs[k].write('batch_size,{}\n'.format(batch_size))
            trainFs[k].write('batch_size_val,{}\n'.format(batch_size_val))
            trainFs[k].write('num_epoch,{}\n'.format(num_epoch))
            trainFs[k].write('learning_rate,{}\n'.format(learning_rate))
            trainFs[k].write('num_worker,{}\n'.format(num_worker))
            trainFs[k].write('*****************************************************\n')
            trainFs[k].write('acc,{}\n'.format(metrics[k]['acc']))
            trainFs[k].write('auc,{}\n'.format(metrics[k]['auc']))
            trainFs[k].write('acc_bestauc,{}\n'.format(metrics[k]['acc_bestauc']))
            trainFs[k].write('auc_bestauc,{}\n'.format(metrics[k]['auc_bestauc']))
            trainFs[k].write('recall_bestauc,{}\n'.format(metrics[k]['recall_bestauc']))
            trainFs[k].write('precision_bestauc,{}\n'.format(metrics[k]['precision_bestauc']))
            trainFs[k].write('f1_score_bestauc,{}\n'.format(metrics[k]['f1_score_bestauc']))
            trainFs[k].write('####################################################\n')
            trainFs[k].close()
        else:
            trainFs[k].write('fold:average 5 fold\n')
            trainFs[k].write('batchsize,{}\n'.format(batch_size))
            trainFs[k].write('batchsize_val,{}\n'.format(batch_size_val))
            trainFs[k].write('num_epoch,{}\n'.format(num_epoch))
            trainFs[k].write('learning_rate,{}\n'.format(learning_rate))
            trainFs[k].write('predthreshold,{}\n'.format(predthreshold))
            trainFs[k].write('acc,{}\n'.format(acc / 5))
            trainFs[k].write('auc,{}\n'.format(auc / 5))
            trainFs[k].write('acc var,{}\n'.format(np.var([metrics[0]['acc'], metrics[1]['acc'], metrics[2]['acc'], metrics[3]['acc'], metrics[4]['acc']])))
            trainFs[k].write('auc var,{}\n'.format(np.var([metrics[0]['auc'], metrics[1]['auc'], metrics[2]['auc'], metrics[3]['auc'], metrics[4]['auc']])))
            trainFs[k].write('acc_bestauc,{}\n'.format(acc_bestauc / 5))
            trainFs[k].write('auc_bestauc,{}\n'.format(auc_bestauc / 5))
            trainFs[k].write('acc_bestauc var,{}\n'.format(np.var(
                [metrics[0]['acc_bestauc'], metrics[1]['acc_bestauc'], metrics[2]['acc_bestauc'], metrics[3]['acc_bestauc'], metrics[4]['acc_bestauc']])))
            trainFs[k].write('auc_bestauc var,{}\n'.format(np.var(
                [metrics[0]['auc_bestauc'], metrics[1]['auc_bestauc'], metrics[2]['auc_bestauc'], metrics[3]['auc_bestauc'], metrics[4]['auc_bestauc']])))
            trainFs[k].write(f"Best model is from fold {best_fold} with AUC: {max_auc}\n")
            trainFs[k].write('acc_test,{}\n'.format(metric_test['acc_test']))
            trainFs[k].write('auc_test,{}\n'.format(metric_test['auc_test']))
            trainFs[k].write('recall_test,{}\n'.format(metric_test['recall_test']))
            trainFs[k].write('precision_test,{}\n'.format(metric_test['precision_test']))
            trainFs[k].write('f1_score_test,{}\n'.format(metric_test['f1_score_test']))
            if setting == "pinn":
                trainFs[k].write('test_time,{}\n'.format(metric_test['test_time']))
                trainFs[k].write('wss_error,{}\n'.format(metric_test['wss_error']))
                trainFs[k].write('pressure_error,{}\n'.format(metric_test['pressure_error']))
                trainFs[k].write('velocity_error_x,{}\n'.format(metric_test['velocity_error_x']))
                trainFs[k].write('velocity_error_y,{}\n'.format(metric_test['velocity_error_y']))
                trainFs[k].write('velocity_error_z,{}\n'.format(metric_test['velocity_error_z']))
            trainFs[k].write('####################################################\n')
            trainFs[k].close()
    # ========= ROC plotting =========
    # validation
    fig_val, ax_val = plt.subplots()
    ax_val.plot([0, 1], [0, 1], color='r', linestyle='--')
    ax_val.plot(metrics[0]['fpr_bestauc'], metrics[0]['tpr_bestauc'], label='Fold1 auc={}'.format(format(metrics[0]['auc_bestauc'], '.4f')))
    ax_val.plot(metrics[1]['fpr_bestauc'], metrics[1]['tpr_bestauc'], label='Fold2 auc={}'.format(format(metrics[1]['auc_bestauc'], '.4f')))
    ax_val.plot(metrics[2]['fpr_bestauc'], metrics[2]['tpr_bestauc'], label='Fold3 auc={}'.format(format(metrics[2]['auc_bestauc'], '.4f')))
    ax_val.plot(metrics[3]['fpr_bestauc'], metrics[3]['tpr_bestauc'], label='Fold4 auc={}'.format(format(metrics[3]['auc_bestauc'], '.4f')))
    ax_val.plot(metrics[4]['fpr_bestauc'], metrics[4]['tpr_bestauc'], label='Fold5 auc={}'.format(format(metrics[4]['auc_bestauc'], '.4f')))
    ax_val.set_xlabel('False Positive Rate')
    ax_val.set_ylabel('True Positive Rate')
    ax_val.set_title('ROC Curve on Validation Sets')
    ax_val.legend(loc='lower right')
    # testing
    fig_test, ax_test = plt.subplots()
    ax_test.plot([0, 1], [0, 1], color='r', linestyle='--')
    ax_test.plot(metric_test['fpr_test'], metric_test['tpr_test'], label='auc={}'.format(format(metric_test['auc_test'], '.4f')))
    ax_test.set_xlabel('False Positive Rate')
    ax_test.set_ylabel('True Positive Rate')
    ax_test.set_title('ROC Curve on Test Set')
    ax_test.legend(loc='lower right')
    fig_test.savefig(os.path.join(checkpoint, 'roc_test.png'), dpi=300, transparent=True)
    fig_val.savefig(os.path.join(checkpoint, 'roc_valid.png'), dpi=300, transparent=True)
    plt.show()
    print('#########################################################################################')
