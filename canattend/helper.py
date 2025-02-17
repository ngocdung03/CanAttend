import numpy as np
import pandas as pd
import os, sys
import copy 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from survtrace.utils import LabelTransform

sys.path.append('/home/nguyen/survival_serv/')
os.chdir('..')
from surv0913.lib.preprocessing import get_cancers, get_features, imputation
from surv0913.run_surv import dat_process
print(os.getcwd())
os.chdir('SurvTRACE-main')

def load_dataset0(args, new_data=False, drop_normal=False):  # Modified from run_surv.py 
    # lodd dataset
    data = pd.read_csv(args.dataset)
    data = data.sample(frac=1) #try shufflilng to see lstm results
    
    # load feature and event 
    feature = get_features(args.feature)
    feature = feature + ['SMOK2', 'SMOKA_MOD', 'SMOKC_MOD', 'alco1', 'ALCO_AMOUNT', 'F1', 'F12', 'F61'] if args.version==3 else feature
    
    if feature[0] == 'YID':
        feature.remove('YID')
    
    # cancer, cancer_time = get_cancers(args.cancer)

    # making outcome of some cancers
    data['CRC'] = data['COLON'] + data['RECTM'] if new_data else data['CRC']
    data['LYMPH'] = data.LYMPH_C81 + data.LYMPH_C82 + data.LYMPH_C83 + data.LYMPH_C84 + data.LYMPH_C85 + data.LYMPH_C86
    data['GALL_BILE'] = data.GALLB + data.BILE
    data['LUNG_LAR'] = data.LUNG + data.LARYN
    data['UTE_CER'] = data.UTERI_C54 + data.UTERI_C55 + data.CERVI
    
    # change values > 1 (due to duplication) into 1 ##0202
    new_col = ['CRC', 'LYMPH', 'GALL_BILE', 'LUNG_LAR', 'UTE_CER']
    for col in new_col:
        data.loc[data[col]>1, col] = 1
        
    # define 'time' column
    data['time'] = data.iloc[:,100:168].mode(axis=1) if new_data else data.iloc[:, -19:].mode(axis=1)
    
    # drop na for age and sex
    data = data[~data['AGE_B'].isna()][~data['SEX1'].isna()]
    
    # demo: drop normal obs. #TEST
    if drop_normal:
        data['event'] = data[args.tasks].sum(axis=1)
        data = data[data['event'] != 0] 
    
    return data
    # # split testset
    # data_x = data[feature]
    # test = data_x.sample(frac=args.test_rate) #random_state=1
    # train = data_x.drop(test.index)
    # if validation:
    #     val = train.sample(frac=validation)
    #     train = train.drop(val.index)

    # imputation
def dsets_process(args, data, fulldata, feature, validation=True):
    train, test, val = data
    if args.imputation == 'regression':
        df_train, refer = imputation(train, method=args.imputation, bysex=args.by_sex)
        df_test = imputation(test, method=args.imputation, refer=refer, bysex=args.by_sex)
        df_val = imputation(val, method=args.imputation, refer=refer, bysex=args.by_sex) if validation else None

    else:
        df_train = imputation(train, method=args.imputation, bysex=args.by_sex)
        df_test = imputation(test, method=args.imputation, refer=train, bysex=args.by_sex)
        df_val = imputation(val, method=args.imputation, refer=train, bysex=args.by_sex) if validation else None

    if args.version==2:
        feature2 = feature + ['PP', 'CCR']
        
    if args.version==3:
        dropped = ['FVC_B', 'FEV1_B', 'ALB_B', 'GLOBULIN_B', 'AGR', 'BIL_B', 'DBIL_B', 'ALP_B', 'AMYLASE_B', 'BUN_B', 'SG', 'PH']
        for i in dropped:
            while i in feature:
                feature.remove(i)
                
        feature2 = feature + ['PP', 'CCR', 'eGFR']
    
    # making 2nd variables, drop outliers and negative time
    df_train = dat_process(fulldata, df_train, feature2, args.tasks)
    df_test = dat_process(fulldata, df_test, feature2, args.tasks)
    
    if validation:
        df_val = dat_process(fulldata, df_val, feature2, args.tasks)
        return df_train, df_test, df_val, feature2
    return df_train, df_test, feature2

def dsets_process2(dats, dfs, cols_std, cols_cat):
    dts_concat = []
    # print('len(dats[0])', len(dats[0]))
    # print('dfs[0] index', len(dfs[0].index), dfs[0].index)
    for i in range(len(dats)):
        # print('i', i)
        dats[i] = pd.DataFrame(dats[i], columns=cols_std, 
                               index=dfs[i].index) ## impose index from fulldata
        dat_concat = pd.concat([dfs[i][cols_cat], dats[i]], axis=1)
        dts_concat.append(dat_concat)
    return dts_concat

def load_data(config, args):
    '''load data, return updated configuration.
    '''
    data = config['data']
    horizons = config['horizons']
    assert data in ["metabric", "nwtco", "support", "gbsg", "flchain", "seer", "ysdat"], "Data Not Found!" ##
    get_target = lambda df: (df['duration'].values, df['event'].values)
    feature0 = get_features(args.feature)[1:] 

    if data == "ysdat":
        df0 = load_dataset0(args, new_data=True) 
        # print('config', config)
        df = copy.deepcopy(df0)
        df = df.rename(columns={"time": "duration"})
        df['event'] = df[config['task']]
        # print('df[config["task"]]', df[config['task']].head())
        # print('df.event', df['event'])                  
        
        times = np.quantile(df["duration"][df["event"]==1.0], horizons).tolist() 
        # print('times', times)
        
        df_feat = df.drop(["duration","event"],axis=1)
        
        ## change split dat sets into here
        # get the largest duration time 
        max_duration_idx = df["duration"].idxmax()
        
        # split sets 
        df_test = df_feat.drop(max_duration_idx).sample(frac=0.3) #? why drop max_duration ##?should drop index instead of label
        df_train = df_feat.drop(df_test.index)
        df_val = df_train.drop(max_duration_idx).sample(frac=0.1) ##/should drop index instead of label 
        df_train = df_train.drop(df_val.index)
        
        ## Missing imputation and create new variables
        df_train, df_test, df_val, feature = dsets_process(args, (df_train, df_test, df_val), df0, feature0, validation=True)
        # print('dsets_process df_train', df_train.head())
        # print('df train and val cols 1', df_train.columns, df_val.columns)
    
        cols_categorical = ['SEX1']
        cols_standardize = [i for i in feature if i not in cols_categorical]
        df_train_std, df_test_std, df_val_std = df_train[cols_standardize], df_test[cols_standardize], df_val[cols_standardize]
        
        # print('df_train_std', df_train_std.head())
        
        # Missings were imputed in dsets_process() so array output of scaler retains order
        scaler = StandardScaler()
        df_train_std_disc = scaler.fit_transform(df_train_std)
        df_test_std_disc, df_val_std_disc = scaler.transform(df_test_std), scaler.transform(df_val_std)

        # print('cols_std', cols_standardize)
        
        # Concat numerical and categorical cols
        # Impose index into df_std_disc, as long as no change in order
        dts_concat = dsets_process2([df_train_std_disc, df_test_std_disc, df_val_std_disc],
                                    [df_train, df_test, df_val], 
                                    cols_standardize, cols_categorical)
        
        df_train_prc, df_test_prc, df_val_prc = dts_concat[0], dts_concat[1], dts_concat[2]
        # print('df train and val cols 2', df_train_prc.columns, df_val_prc.columns)
        
        # Label encode for categorical cols
        vocab_size = 0
        for _,feat in enumerate(cols_categorical):
            encoder = LabelEncoder()
            df_train_prc[feat] = encoder.fit_transform(df_train_prc[feat]).astype(float) + vocab_size
            df_test_prc[feat] = encoder.transform(df_test_prc[feat]).astype(float) + vocab_size ##
            df_val_prc[feat] = encoder.transform(df_val_prc[feat]).astype(float) + vocab_size ##
            vocab_size = df_feat[feat].max() + 1
        
        df_train_ftr, df_test_ftr, df_val_ftr = df_train_prc, df_test_prc, df_val_prc
        # print('df train and val cols 3', df_train_ftr.columns, df_val_ftr.columns)

        # assign cuts - Later
        labtrans = LabelTransform(cuts=np.array([0]+times+[df["duration"].max()]))
        labtrans.fit(*get_target(df.loc[df_train_ftr.index]))
        y = labtrans.transform(*get_target(df)) # y = (discrete duration, event indicator)
        
        ## Redefine df_y dataframe because idx not retained
        df_y = pd.DataFrame({"duration": y[0], "event": y[1], "proportion": y[2]}, index=df.index) #? is this order right
        # print('df_y', df_y.head())
        
        df_y_train = df_y.loc[df_train_ftr.index]
        df_y_val = df_y.loc[df_val_ftr.index]
        df_y_test = df.loc[df_test_ftr.index, ['duration', 'event']] ## should be original df
        # print('df_y_test', df_y_test)
        
        # out of if else - does it change STConfig
        config['labtrans'] = labtrans
        config['num_numerical_feature'] = int(len(cols_standardize))
        config['num_categorical_feature'] = int(len(cols_categorical))
        config['num_feature'] = int(len(df_train_ftr.columns))
        config['vocab_size'] = int(vocab_size)
        config['duration_index'] = labtrans.cuts
        config['out_feature'] = int(labtrans.out_features)
        # print('labstrans.cut', labtrans.cuts)
        # print('config duration', config['duration_index'])
        # print('df_y_test min', df_y_test['duration'].min())
        # print('df_y_test max', df_y_test['duration'].max())
        
        return df, df_train_ftr, df_y_train, df_test_ftr, df_y_test, df_val_ftr, df_y_val