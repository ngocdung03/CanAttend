import os
from re import L
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import warnings
import torch
from scipy.stats import zscore
# torch.multiprocessing.set_start_method('spawn') #0829 context has already been set
import lightning as pl

# from survival.lib.models import deepsurv
warnings.filterwarnings("ignore")
HOME = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(HOME+'/lib')
from preprocessing import *
from dataloader import *
from models import *
from pylightning0902 import pl_lstm, pl_mtl_lstm


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    return 

def parse_args():
    desc = 'Evaluate survival analysis models'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--n_trials', type=int, default=1, help='number of trials')
    parser.add_argument('--model', type=str, default='cox', help='surival model name, default cox, {cox, deepsurv, deephit, lstm, mtl_lstm}')
    parser.add_argument('--dataset', type=str, help='data file')
    parser.add_argument('--feature', type=str, help='feature file')
    parser.add_argument('--cancer', type=str, help='cancer file')
    parser.add_argument('--version', type=int, default=2, help='version of list of features')
    parser.add_argument('--imputation', type=str, default='mean', help='imputation method: mean, median, regression')
    parser.add_argument('--by_sex', type=str2bool, default=True, help='appply group by with sex')
    parser.add_argument('--test_rate', type=float, default=0.2, help='the rate of testset')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate') ##
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer') ##
    parser.add_argument('--dropout', type=float, default=0.1, help='drop out reate') ##
    parser.add_argument('--tasks', nargs='+', default=None, help='different cancers for multitask model')
    parser.add_argument('--new', type=str2bool, default=False, help='new dataset or old dataset')
    parser.add_argument('--drop_normal', type=str2bool, default=False, help='drop normal observations')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    parser.add_argument('--pfi_calculation', type=str2bool, default=False, help='calculate permutation feature importance only')
    parser.add_argument('--save_mod', type=str2bool, default=False, help='save model')
    return check_args(parser.parse_args())

def check_args(args):
    try:
        assert args.n_trials > 0 and args.n_trials <= 200
    except AssertionError: 
        print('n_trials should be greater than 0')
        sys.exit(1)
    models = ['cox', 'deepsurv', 'deephit', 'lstm', 'mtl_lstm', 'pl_lstm', 'pl_mtl_lstm']
    try:
        assert args.model in models
    except AssertionError: 
        print('ERROR', 'Not support model(%s), only support' % (args.model) , str(models))
        sys.exit(1)
    try:
        assert args.dataset is not None
        if not os.path.exists(args.dataset):
            print('Wrong dataset file(%s)' % (args.dataset))
            sys.exit(1)
    except AssertionError:
        print('No dataset file, --dataset should be required.')
        sys.exit(1)
    try:
        assert args.feature is not None
        if not os.path.exists(args.feature):
            print('Wrong feature file(%s)' % (args.feature))
            sys.exit(1)
    except AssertionError:
        print('No feature file, --feature should be required.')
        sys.exit(1)
    try:
        assert args.cancer is not None
        if not os.path.exists(args.cancer):
            print('Wrong cancer file(%s)' %(args.cancer))
            sys.exit(1)
    except AssertionError:
        print('No cancer file, --cancer should be required.')
        sys.exit(1)
    try:
        assert type(args.version) == int
    except AssertionError:
        raise TypeError('version should be integer')
    try:
        method = ['mean', 'median', 'regression']
        assert args.imputation in method
    except AssertionError:
        print('Wrong imputation method(%s), only support' % (args.imputation), str(method))
        sys.exit(1)
    try:
        assert type(args.by_sex) == bool
    except AssertionError:
        raise TypeError('by_sex should be bool')
    try:
        assert args.test_rate <= 1 and args.test_rate >= 0
    except AssertionError:
        raise TypeError('test_rate should be between 0 and 1')
    try:
        assert args.batch_size >= 32
    except AssertionError:
        raise TypeError('batch_size should be greater than 32')
    try:
        assert args.n_epochs > 0
    except AssertionError:
        raise TypeError('n_epochs should be greater than 0')
    try:
        cancers, _ = get_cancers(args.cancer)
        if args.tasks:
            for task in args.tasks:
                assert task in cancers 
    except AssertionError:
        raise TypeError('Wrong cancer(%s), only support' % (task, str(cancers)))
    try:
        assert type(args.new) == bool
    except AssertionError:
        raise TypeError('new should be bool')
    try:
        assert type(args.drop_normal) == bool
    except AssertionError:
        raise TypeError('drop_normal should be bool')
    try:
        assert type(args.pfi_calculation) == bool
    except AssertionError:
        raise TypeError('pfi_calculation) should be bool')
    return args


def dat_process(fulldata, df, feature, tasks, threshold_sd=6):
    # wrangling BMI_B, SBP, DBP
    rev_ind = df[df['SBP_B'] < df['DBP_B']].index # There maybe unsuitable values after imputation
    SBP_neg = df.loc[rev_ind, 'SBP_B']
    DBP_neg = df.loc[rev_ind, 'DBP_B']
    df.loc[rev_ind, 'SBP_B']= DBP_neg.values
    df.loc[rev_ind, 'DBP_B']= SBP_neg.values
    
    df['BMI_B'] = df['WT_B']/(df['HT_B']/100)**2  
    
    # second feature
    df = make_2nd_variable('ccr', make_2nd_variable('pp', df=df))
    df = make_2nd_variable('egfr',df=df)

    # filter for outlier > threshold*SD
    zscr_transform = df.apply(zscore, nan_policy='omit')
    outlier_dat = zscr_transform[(zscr_transform[feature] > threshold_sd).sum(axis=1) > 0]
    df = df.drop(outlier_dat.index)[feature]

    # event and time
    # if args.tasks:
    df = pd.concat([fulldata['YID'], df, fulldata[tasks], fulldata['time'].fillna(-1)], join='inner', axis=1) ## YID

    # drop left censor
    df = df[df['time'] > 0] 
    
    return df
    
def load_dataset2(args, validation=None, target= None, new_data=False, drop_normal=False):   
    if args is None or target is None: 
        return

    # lodd dataset
    data = pd.read_csv(args.dataset)
    data = data.sample(frac=1) #try shufflilng to see lstm results
    
    # load feature and event 
    feature = get_features(args.feature)
    feature = feature + ['SMOK2', 'SMOKA_MOD', 'SMOKC_MOD', 'alco1', 'ALCO_AMOUNT', 'F1', 'F12', 'F61'] if args.version==3 else feature
    
    # feature = ['AGE_B', 'SEX1', 'WT_B', 'HT_B', 'WC', 'BMI_B', 'SBP_B', 'DBP_B', 'CHO_B', 'HDL_B', 'TG_B', 'LDL_B', 'FVC_B', 'FEV1_B', 'ALB_B', 'GLOBULIN_B', 'AGR', 'BIL_B', 'DBIL_B', 'ALP_B', 'AST', 'ALT', 'GGTP', 'GLU', 'AMYLASE_B', 'CREAT_B', 'BUN_B', 'SG', 'PH']
    if feature[0] == 'YID':
        feature.remove('YID')
    
    cancer, cancer_time = get_cancers(args.cancer)

    # making outcome of some cancers
    data['CRC'] = data['COLON'] + data['RECTM'] if new_data else data['CRC']
    data['LYMPH'] = data.LYMPH_C81 + data.LYMPH_C82 + data.LYMPH_C83 + data.LYMPH_C84 + data.LYMPH_C85 + data.LYMPH_C86
    data['GALL_BILE'] = data.GALLB + data.BILE
    data['LUNG_LAR'] = data.LUNG + data.LARYN
    data['UTE_CER'] = data.UTERI_C54 + data.UTERI_C55 + data.CERVI
    
    # change values > 1 (due to duplication) into 1 ##0202
    new_col = ['CRC', 'LYMPH', 'GALL_BILE', 'LUNG_LAR', 'UTE_CER']
    for col in new_col:
        data[col][data[col]>1] = 1

    # define 'time' column
    data['time'] = data.iloc[:,100:168].mode(axis=1) if new_data else data.iloc[:, -19:].mode(axis=1)
    
    # drop na for age and sex
    data = data[~data['AGE_B'].isna()][~data['SEX1'].isna()]
    
    # demo: drop normal obs. #TEST
    if drop_normal:
        data['event'] = data[args.tasks].sum(axis=1)
        data = data[data['event'] != 0] 

    # split testset
    data_x = data[feature]
    test = data_x.sample(frac=args.test_rate) #random_state=1
    train = data_x.drop(test.index)
    if validation:
        val = train.sample(frac=validation)
        train = train.drop(val.index)
    
    # imputation
    if args.imputation == 'regression':
        df_train, refer = imputation(train, method=args.imputation, bysex=args.by_sex)
        df_test = imputation(test, method=args.imputation, refer=refer, bysex=args.by_sex)
        df_val = imputation(val, method=args.imputation, refer=refer, bysex=args.by_sex) if validation else None

    else:
        df_train = imputation(train, method=args.imputation, bysex=args.by_sex)
        df_test = imputation(test, method=args.imputation, refer=train, bysex=args.by_sex)
        df_val = imputation(val, method=args.imputation, refer=train, bysex=args.by_sex) if validation else None
        
    if args.version==2:
        feature += ['PP', 'CCR']
        
    if args.version==3:
        dropped = ['FVC_B', 'FEV1_B', 'ALB_B', 'GLOBULIN_B', 'AGR', 'BIL_B', 'DBIL_B', 'ALP_B', 'AMYLASE_B', 'BUN_B', 'SG', 'PH']
        for i in dropped:
            while i in feature:
                feature.remove(i)
                
        feature += ['PP', 'CCR', 'eGFR']
    
    # making 2nd variables, drop outliers and negative time
    df_train = dat_process(data, df_train, feature, args.tasks)
    df_test = dat_process(data, df_test, feature, args.tasks)
    
    if validation:
        df_val = dat_process(data, df_val, feature, args.tasks)
        return df_train, df_test, df_val, feature
    return df_train, df_test, feature

def run_cph(train, test, feature, event=None, duration=None):
    if event is None or duration is None:
        return
    train_c, test_c = cox_regression((train, test), feature=feature, event=event, duration=duration)
    print(str(args), '\ttrain:', train_c, 'test:', test_c)

def run_lstm(args, train, train_f, test, event_ind=None, valid=None, pl2=False):
    n=[x['X'].shape for x in test][0][-1]
    if pl2:
        train_result, test_result = pl_lstm(train, train_f, test, valid, n, str(args.tasks[0]), n_epochs=args.n_epochs)
        print(str(args), '\ttrain:', train_result, 'test:', test_result)
    else:
        model = lstm(n, dropout=args.dropout)
        train_c, test_c = evaluation_f(model, train, train_f, test, learning_rate=args.learning_rate, n_epochs=args.n_epochs, E=event_ind, save_mod=args.save_mod) #evaluate_lstm
        print(str(args), '\ttrain:', train_c, 'test:', test_c)
    
def run_deepsurv(train, test, args, feature = None, event=None, duration=None, device=device):
    if feature is None or event is None or duration is None:
        return
    train_c, test_c = deepsurv((train, test), feature = feature, event=event, duration=duration, device=device)
    print(str(args), '\ttrain:', train_c, 'test:', test_c) 
    
def run_mtl_deephit(train, train_pred, test, args, device):
    n = [x.shape for x,_,_ in test][0][-1]
    num_tasks = len(args.tasks)
    hparams1 = {'p_dropout': 0.05343304105619939, 'learning_rate': 0.003661511694319197} #n_epochs 7
    hparams2 = {'p_dropout': 0.5834374419730315, 'learning_rate': 0.007842602453415332} #n_epochs 9
    if args.tasks[0] == 'THROI':
        hparams = hparams1
    elif args.tasks[0] == 'PROST':
        hparams = hparams2
    model = mtl_deephit(n, num_tasks, p_dropout=hparams['p_dropout']) #1025
    results = evaluation_mtl_deephit(model, train, train_pred, test, num_tasks, n_epochs=args.n_epochs, loss_func=loss_func, device=device, learning_rate=hparams['learning_rate'])
    for i in range(num_tasks):
        print(str(args),'\nEvent ',i+1, '\ttrain:', results[i][0], 'test:', results[i][1])
        
def run_mtl_lstm(args, train, train_f, test, valid=None, device=None, pl2=False): #0826
    n=[x['X'].shape for x in test][0][-1]
    num_tasks = len(args.tasks)
    if pl2:
        train_result, test_result = pl_mtl_lstm(train, train_f, test, valid, n, num_tasks=num_tasks, n_epochs=args.n_epochs) 
        print(str(args), '\ttrain:', train_result, 'test:', test_result)
    else:
        model_name = args.tasks[0] + str(len(args.tasks))
        model = mtl_lstm(n, num_tasks=num_tasks)
        results = evaluation_mtl_lstm(model, train, train_f, test, num_tasks, model_name=model_name, n_epochs = args.n_epochs, save_mod=args.save_mod, device=device) #1118
        for i in range(num_tasks):
            print(str(args), '\nEvent', i+1, '\ttrain:', results[i][0], 'test:', results[i][1])
            
if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)" ##
    args = parse_args()
    if args is None:
        sys.exit(1)
    # import pickle
    # train3, test3, val3 = load_dataset2(args, validation = 0.2, target='thyroid', new_data=args.new)
    # with open('train3.pkl', 'wb') as f:
    #     pickle.dump(train3, f)
    # with open('test3.pkl', 'wb') as f:
    #     pickle.dump(test3, f)
    # with open('val3.pkl', 'wb') as f:
    #     pickle.dump(val3, f)
    # print(train[args.tasks+['event']])
    # print(test[args.tasks+['event']])

####
    device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # feature = get_features(args.feature)
    # feature = feature + ['PP', 'CCR'] if args.version==2 else feature
    cancer, cancer_time = get_cancers(args.cancer)
    event_map = {x:cancer_time[i] for i, x in enumerate(cancer)}
    target =  event_map[args.tasks[0]]
    for i in range(args.n_trials):
        # a = time.time()
        # train, test = load_dataset(args, target=target) #, target=target
        train2, test2, feature = load_dataset2(args, target=target, new_data=args.new, drop_normal = args.drop_normal) 
        train3, test3, val3, feature = load_dataset2(args, validation = 0.2, target=target, new_data=args.new, drop_normal = args.drop_normal)
        
        print('start %dth trial' % (i+1))
        if args.model == 'cox':
            # print(feature)
            run_cph(train2, test2, feature, event=args.tasks[0], duration='time') ##
        elif args.model in ['lstm', 'pl_lstm']:
            worker = 0
            train_loader, train_loader_f = get_loader2(train2, covariates=feature, outcome=args.tasks, batch_size=args.batch_size, device = device, num_workers=worker) ## args.tasks[0] 
            test_loader = get_loader2(test2, covariates=feature, outcome=args.tasks, device = device, num_workers=worker) ## args.tasks[0]
            
            train_loader2, train_loader_f2 = get_loader2(train3, covariates=feature, outcome=args.tasks[0], batch_size=args.batch_size, device = device, num_workers=worker) ## drop 'YID'
            valid_loader2, _ = get_loader2(val3, covariates=feature, outcome=args.tasks[0], batch_size=args.batch_size, device = device, num_workers=worker) 
            test_loader2 = get_loader2(test3, covariates=feature, outcome=args.tasks[0], device = device, num_workers=worker) ## 
            
            if args.model == 'lstm':
                for i in range(len(args.tasks)):
                    run_lstm(args, train_loader, train_loader_f, test_loader, event_ind=i, pl2=False)
                # print("Remodify run_lstm!")
            elif args.model == 'pl_lstm':
                run_lstm(args, train_loader2, train_loader_f2, test_loader2, valid_loader2, pl2=True)
            
        elif args.model == 'deepsurv':
            run_deepsurv(train2, test2, args, feature=feature, event=args.tasks[0], duration='time', device=device) ## 
        elif args.model == 'deephit':
            outcome = args.tasks
            train_loader, train_loader_f = get_loader_deephit(train2, feature, outcome, 'time', batch_size = args.batch_size)
            test_loader = get_loader_deephit(test2, feature, outcome, 'time')
            run_mtl_deephit(train_loader, train_loader_f, test_loader, args, device=device)
        elif args.model in ['mtl_lstm', 'pl_mtl_lstm']:
            worker = args.num_workers
            train_loader, train_loader_f = get_loader2(train2, covariates = feature, outcome= args.tasks, batch_size = args.batch_size, num_workers=worker)
            test_loader = get_loader2(test2, covariates = feature, outcome = args.tasks, num_workers=worker)
            
            train_loader2, train_loader_f2 = get_loader2(train3, covariates=feature, outcome=args.tasks, batch_size=args.batch_size, device = device, num_workers=worker) ## drop 'YID'
            valid_loader2, _ = get_loader2(val3, covariates=feature, outcome=args.tasks, batch_size=args.batch_size, device = device, num_workers=worker) 
            test_loader2 = get_loader2(test3, covariates=feature, outcome=args.tasks, device = device, num_workers=worker)
            
            if args.model == 'mtl_lstm':
                run_mtl_lstm(args, train_loader2, train_loader_f2, test_loader2, device=device, pl2=False)
            elif args.model == 'pl_mtl_lstm':
                run_mtl_lstm(args, train_loader2, train_loader_f2, test_loader2, valid_loader2, pl2=True) #device=device

               
            
