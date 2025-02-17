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
# torch.multiprocessing.set_start_method('spawn') #0829 context has already been set
import pytorch_lightning as pl

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
    parser.add_argument('--second_feature', type=str2bool, default=True, help='apply second feature')
    parser.add_argument('--imputation', type=str, default='mean', help='imputation method: mean, median, regression')
    parser.add_argument('--by_sex', type=str2bool, default=True, help='appply group by with sex')
    parser.add_argument('--test_rate', type=float, default=0.2, help='the rate of testset')
    parser.add_argument('--target_cancer', type=str, default='STOMA', help='target cancer to analyze')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--tasks', nargs='+', default=None, help='different cancers for multitask model')
    parser.add_argument('--new', type=str2bool, default=False, help='new dataset or old dataset')
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
        assert type(args.second_feature) == bool
    except AssertionError:
        raise TypeError('second_feature should be bool')
    try:
        method = ['mean', 'median', 'regression']
        assert args.imputation in method
    except AssertionError:
        print('Wrong imputation method(%s), only support' % (args.imputation), str(method))
        sys.exit(1)
    try:
        assert type(args.by_sex) == bool
    except AssertionError:
        raise TypeError('second_feature should be bool')
    try:
        assert args.test_rate <= 1 and args.test_rate >= 0
    except AssertionError:
        raise TypeError('test_rate should be between 0 and 1')
    try:
        cancers, _ = get_cancers(args.cancer)
        assert args.target_cancer in cancers
    except AssertionError:
        raise TypeError('Wrong cancer(%s), only support' % (args.target_cancer), str(cancers))
    try:
        assert args.batch_size >= 32
    except AssertionError:
        raise TypeError('batch_size should be greater than 32')
    try:
        assert args.n_epochs > 0
    except AssertionError:
        raise TypeError('n_epochs should be greater than 0')
    try:
        if args.tasks:
            for task in args.tasks:
                assert task in cancers 
    except AssertionError:
        raise TypeError('Wrong cancer(%s), only support' % (args.target_cancer), str(cancers))
    try:
        assert type(args.new) == bool
    except AssertionError:
        raise TypeError('new should be bool')
    return args

def load_dataset2(args, validation=None, target= None, new_data=False):   
    if args is None or target is None: 
        return

    # lodd dataset
    data = pd.read_csv(args.dataset)

    # load feature and event 
    # feature = get_features(args.feature)
    feature = ['AGE_B', 'SEX1', 'WT_B', 'HT_B', 'WC', 'BMI_B', 'SBP_B', 'DBP_B', 'CHO_B', 'HDL_B', 'TG_B', 'LDL_B', 'FVC_B', 'FEV1_B', 'ALB_B', 'GLOBULIN_B', 'AGR', 'BIL_B', 'DBIL_B', 'ALP_B', 'AST', 'ALT', 'GGTP', 'GLU', 'AMYLASE_B', 'CREAT_B', 'BUN_B', 'SG', 'PH']
    # ['YID']
    cancer, cancer_time = get_cancers(args.cancer)
    
    # data['time'] = data[cancer_time].mode(axis=1)  ##
    data['time'] = data.iloc[:,100:168].mode(axis=1) if new_data else data.iloc[:, -19:].mode(axis=1)

    data['CRC'] = data['COLON'] + data['RECTM'] if new_data else data['CRC']
    
    # drop na for age and sex
    data = data[~data['AGE_B'].isna()][~data['SEX1'].isna()]

    # split testset
    data_x = data[feature]
    test = data_x.sample(frac=args.test_rate) #random_state=1
    train = data_x.drop(test.index)
    if validation:
        val = train.sample(frac=validation)
        train = train.drop(val.index)
    
    # imputation
    if args.imputation == 'regression':
        df_train = imputation2(train, method=args.imputation, feature=feature) #, bysex=args.by_sex, added feature
        df_test = imputation2(test, method=args.imputation, feature=feature) #, refer=refer, bysex=args.by_sex, added feature
        df_val = imputation2(val, method=args.imputation, feature=feature) if validation else None
    else:
        df_train = imputation(train, method=args.imputation, bysex=args.by_sex)
        df_test = imputation(test, method=args.imputation, refer=train, bysex=args.by_sex)
        df_val = imputation(val, method=args.imputation, refer=train, bysex=args.by_sex) if validation else None
    
    # second feature
    df_train = make_2nd_variable('ccr', make_2nd_variable('pp', df=df_train)) #introduce NA?
    df_test = make_2nd_variable('ccr', make_2nd_variable('pp', df=df_test))
    df_val = make_2nd_variable('ccr', make_2nd_variable('pp', df=df_val)) if validation else None

    # scaling
    feature = feature + ['PP', 'CCR']
    scaler = StandardScaler()
    # df_train = pd.DataFrame(scaler.fit_transform(df_train[feature]), columns=feature)
    # df_test = pd.DataFrame(scaler.transform(df_test[feature]), columns=feature)

    # event and time
    if args.tasks:
        df_train = pd.concat([data[['YID']], df_train, data[args.tasks], data[['time']].fillna(-1)], join='inner', axis=1) ## YID
        df_test = pd.concat([data['YID'], df_test, data[args.tasks], data[['time']].fillna(-1)], join='inner', axis=1) ##
        df_val = pd.concat([data['YID'], df_val, data[args.tasks], data[['time']].fillna(-1)], join='inner', axis=1) if validation else None
    else:
        df_train = pd.concat([df_train, 
                              data[[args.target_cancer]], 
                              data[['time']].fillna(-1)], join='inner', axis=1) ##
        df_test = pd.concat([df_test, 
                             data[[args.target_cancer]], 
                             data[['time']].fillna(-1)], join='inner', axis=1)
        df_val = pd.concat([df_val, 
                             data[[args.target_cancer]], 
                             data[['time']].fillna(-1)], join='inner', axis=1) if validation else None

    # drop left censor
    df_train = df_train[df_train['time'] > 0] # Empty
    df_test = df_test[df_test['time'] > 0]
    df_val = df_val[df_val['time'] > 0] if validation else None
    
    if validation:
        return (df_train, df_test, df_val)
    return (df_train, df_test)

def run_cph(train, test, args, event=None, duration=None):
    if event is None or duration is None:
        return
    train_c, test_c = cox_regression((train, test), event=event, duration=duration)
    print(str(args), '\ttrain:', train_c, 'test:', test_c)

def run_lstm(args, train, train_f,  test, valid=None, pl2=False):
    n=[x['X'].shape for x in test][0][-1]
    if pl2:
        # model = LightninglstmClassifier(n)
        # trainer = pl.Trainer(max_epochs=1) #
        # trainer.fit(model, train, valid) #test
        # train_result = trainer.test(model, dataloaders=train_f, verbose=False)
        # test_result = trainer.test(model, dataloaders=test, verbose=False)
        train_result, test_result = pl_lstm(train, train_f, test, valid, n)
        print(str(args), '\ttrain:', train_result, 'test:', test_result)
    else:
        model = lstm(n)
        train_c, test_c = evaluation_f(model, train, train_f, test, n_epochs=args.n_epochs, E=1) #evaluate_lstm
        print(str(args), '\ttrain:', train_c, 'test:', test_c)
    
def run_deepsurv(train, test, args, feature = None, event=None, duration=None, device=device):
    if feature is None or event is None or duration is None:
        return
    train_c, test_c = deepsurv((train, test), feature = feature, event=event, duration=duration, device=device)
    print(str(args), '\ttrain:', train_c, 'test:', test_c) #train
    # print(train[[duration]],'\t', type(train[[duration]]))
    # print(train[[event]],'\t', type(train[[event]]))
    
def run_mtl_deephit(train, train_pred, test, args, device):
    n = [x.shape for x,_,_ in test][0][-1]
    num_tasks = len(args.tasks)
    model = mtl_deephit(n, num_tasks)
    results = evaluation_mtl_deephit(model, train, train_pred, test, num_tasks, n_epochs = args.n_epochs, loss_func=loss_func, device=device)
    for i in range(num_tasks):
        print(str(args),'\nEvent ',i+1, '\ttrain:', results[i][0], 'test:', results[i][1])
        
def run_mtl_lstm(args, train, train_f, test, valid=None, device=None, pl2=False): #0826
    n=[x['X'].shape for x in test][0][-1]
    num_tasks = len(args.tasks)
    if pl2:
        # model = Lightningmtl_lstm(n)
        # trainer = pl.Trainer(max_epochs=1) 
        # trainer.fit(model, train, test) #?device
        # train_result = trainer.test(model, dataloaders=train_f, verbose=False)
        # test_result = trainer.test(model, dataloaders=test, verbose=False) #device
        train_result, test_result = pl_mtl_lstm(train, train_f, test, valid, n, n_epochs=args.n_epochs) 
        print(str(args), '\ttrain:', train_result, 'test:', test_result)
    else:
        model = mtl_lstm(n, num_tasks=num_tasks)
        results = evaluation_mtl_lstm(model, train, train_f, test, num_tasks, n_epochs = args.n_epochs, device=device) #0826
        for i in range(num_tasks):
            print(str(args), '\nEvent', i+1, '\ttrain:', results[i][0], 'test:', results[i][1])
            
if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)" ##
    args = parse_args()
    if args is None:
        sys.exit(1)
    # import pickle
    # # train, test = load_dataset2(args, target='thyroid', new_data=args.new)
    # train3, test3, val3 = load_dataset2(args, validation = 0.2, target='thyroid', new_data=args.new)
    # with open('train3.pkl', 'wb') as f:
    #     pickle.dump(train3, f)
    # with open('test3.pkl', 'wb') as f:
    #     pickle.dump(test3, f)
    # with open('val3.pkl', 'wb') as f:
    #     pickle.dump(val3, f)
    # print(args.tasks)
    # print(type(args.tasks))
####
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature = get_features(args.feature)
    # feature = feature[1:] + ['PP', 'CCR']
    cancer, cancer_time = get_cancers(args.cancer)
    event_map = {x:cancer_time[i] for i, x in enumerate(cancer)}
    target =  event_map[args.target_cancer]
    for i in range(args.n_trials):
        a = time.time()
        # train, test = load_dataset(args, target=target) #, target=target
        train2, test2 = load_dataset2(args, target=target, new_data=args.new) 
        train3, test3, val3 = load_dataset2(args, validation = 0.2, target=target, new_data=args.new)
        
        print('start %dth trial' % (i+1))
        if args.model == 'cox':
            run_cph(train2, test2, args, event=args.target_cancer, duration='time') ##
        elif args.model in ['lstm', 'pl_lstm']:
            worker = 0
            train_loader, train_loader_f = get_loader2(train2, covariates=feature[1:], outcome=args.target_cancer, batch_size=args.batch_size, device = device, num_workers=worker) ## drop 'YID' 
            test_loader = get_loader2(test2, covariates=feature[1:], outcome=args.target_cancer, device = device, num_workers=worker) ## 
            
            train_loader2, train_loader_f2 = get_loader2(train3, covariates=feature[1:], outcome=args.target_cancer, batch_size=args.batch_size, device = device, num_workers=worker) ## drop 'YID'
            valid_loader2, _ = get_loader2(val3, covariates=feature[1:], outcome=args.target_cancer, batch_size=args.batch_size, device = device, num_workers=worker) 
            test_loader2 = get_loader2(test3, covariates=feature[1:], outcome=args.target_cancer, device = device, num_workers=worker) ## 
            
            # is_lightning = True if args.model == 'pl_lstm' else False
            # run_lstm((train_loader, train_loader_f, test_loader), args, pl2=is_lightning)
            
            if args.model == 'lstm':
                run_lstm(args, train_loader, train_loader_f, test_loader, pl2=False)
            elif args.model == 'pl_lstm':
                run_lstm(args, train_loader2, train_loader_f2, test_loader2, valid_loader2, pl2=True)
            
        elif args.model == 'deepsurv':
            run_deepsurv(train2, test2, args, feature=feature[1:], event=args.target_cancer, duration='time', device=device) ## 
        elif args.model == 'deephit':
            outcome = args.tasks
            train_loader, train_loader_f = get_loader_deephit(train2, feature[1:], outcome, 'time', batch_size = args.batch_size)
            test_loader = get_loader_deephit(test2, feature[1:], outcome, 'time')
            run_mtl_deephit(train_loader, train_loader_f, test_loader, args, device=device)
        elif args.model in ['mtl_lstm', 'pl_mtl_lstm']:
            worker = 0
            train_loader, train_loader_f = get_loader2(train2, covariates = feature[1:], outcome= args.tasks, batch_size = args.batch_size, num_workers=worker)
            test_loader = get_loader2(test2, covariates = feature[1:], outcome = args.tasks, num_workers=worker)
            
            train_loader2, train_loader_f2 = get_loader2(train3, covariates=feature[1:], outcome=args.tasks, batch_size=args.batch_size, device = device, num_workers=worker) ## drop 'YID'
            valid_loader2, _ = get_loader2(val3, covariates=feature[1:], outcome=args.tasks, batch_size=args.batch_size, device = device, num_workers=worker) 
            test_loader2 = get_loader2(test3, covariates=feature[1:], outcome=args.tasks, device = device, num_workers=worker)
            
            # is_lightning = True if args.model == 'pl_lstm' else False
            # run_mtl_lstm((train_loader, train_loader_f, test_loader), args, device, is_lightning)
            
            if args.model == 'mtl_lstm':
                run_mtl_lstm(args, train_loader2, train_loader_f2, test_loader2, device=device, pl2=False)
            elif args.model == 'pl_mtl_lstm':
                run_mtl_lstm(args, train_loader2, train_loader_f2, test_loader2, valid_loader2, pl2=True) #device=device

               
            
