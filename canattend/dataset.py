JUPYTER=False
import os, sys
import copy
import numpy as np
import pandas as pd
from pycox.datasets import metabric, support
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .utils import LabelTransform
from sklearn.utils import resample

### kcpsii ###
sys.path.append('/Users/moadata/Downloads/survival_serv/')

if JUPYTER:
    os.chdir('..')
    from surv0913.lib.preprocessing import get_features, imputation
    from surv0913.run_surv import dat_process
    os.chdir('canattend')
    
else:
    os.chdir('../surv0913')
    HOME = os.getcwd() #os.path.split(os.path.abspath(__file__))[0]
    sys.path.append(HOME)
    sys.path.append(HOME+'/lib')
    from preprocessing import get_features, imputation
    from run_surv import dat_process
    os.chdir('../canattend')

def load_data(config, sample=None, control_stimu=False, n_controls=300): ##

    data = config['data']
    horizons = config['horizons']
    assert data in ["seer2", "kcpsii"], "Data Not Found!"
    get_target = lambda df: (df['duration'].values, df['event'].values)

    if data == "seer2":
        PATH_DATA = "./data/seer_processed0909_2.csv" ##
        df = pd.read_csv(PATH_DATA)
        # Shuffling
        df = df.sample(frac=1) 
        
        print(f"Year of diagnosis: {df['Year of diagnosis'].min()}-{df['Year of diagnosis'].max()}")
        
        if sample:
            df = df.sample(frac=sample)
            print('Use sampled data with len:', len(df))
        # else:
            # print('Number of obs: ', len(df))
        
        ##
        df['duration'] = df['duration']/12 # Convert duration from months to years
        
        df['Cervix'] = df['Cervix Uteri']  #+ df['Corpus Uteri'] + df['Uterus, NOS'] #'Cervix-uterus'
        df['Lymphoma'] = df['NHL - Nodal'] + df['NHL - Extranodal'] + df['Hodgkin - Nodal'] + df['Hodgkin - Extranodal']
        df['Colorectal'] = df['Rectum'] + df['Sigmoid Colon'] + df['Cecum'] + df['Ascending Colon'] + df['Rectosigmoid Junction'] + df['Transverse Colon'] + df['Descending Colon'] + df['Hepatic Flexure'] + df['Splenic Flexure'] + df['Appendix'] + df['Large Intestine, NOS']
        df['Lung-bronchus'] = df['Lung and Bronchus']
        df['Kidney-rpelvis'] = df['Kidney and Renal Pelvis']
        
        event_list = config['task'] if type(config['task']) == list else [config['task']]
        for event in event_list:
            df.loc[df[event] > 1, event] = 1
        
        # Cancer cases listing
    
        cols_categorical = ["Sex", "Year of diagnosis", "Race recode (W, B, AI, API)", "Histologic Type ICD-O-3",
            "Laterality", "Sequence number", "ER Status Recode Breast Cancer (1990+)",
            "PR Status Recode Breast Cancer (1990+)", "Summary stage 2000 (1998-2017)",
            "RX Summ--Surg Prim Site (1998+)", "Reason no cancer-directed surgery", "First malignant primary indicator",
            "Diagnostic Confirmation", "Median household income inflation adj to 2022"] ## 2019
        cols_standardize = ["Regional nodes examined (1988+)", "CS tumor size (2004-2015)", "Total number of benign/borderline tumors for patient",
            "Total number of in situ/malignant tumors for patient",]
        ##
        
        cols_to_drop = ["duration"] + event_list
        df_feat = df.drop(cols_to_drop,axis=1)
        
        df_feat_standardize = df_feat[cols_standardize]        
        df_feat_standardize_disc = StandardScaler().fit_transform(df_feat_standardize)
 
        df_feat_standardize_disc = pd.DataFrame(df_feat_standardize_disc, columns=cols_standardize, index=df_feat_standardize.index) ## Error if not specify index
        df_feat = pd.concat([df_feat[cols_categorical], df_feat_standardize_disc], axis=1)

        vocab_size = 0
        for i,feat in enumerate(cols_categorical):
            df_feat[feat] = LabelEncoder().fit_transform(df_feat[feat]).astype(float) + vocab_size
            vocab_size = df_feat[feat].max() + 1
        
        ## Stimulate control data 
        if control_stimu: ##
            Features = df_feat.columns
            control_covs = resample(df[Features], replace=True, n_samples=n_controls, random_state=42)
            censoring_times = np.random.uniform(7, 16, size=n_controls) 
            control_data = control_covs.copy()
            df_feat = pd.concat([df_feat, control_data], ignore_index=True)
            
            control_data['duration'] = censoring_times
            control_data[event_list] = 0
            df = pd.concat([df, control_data], ignore_index=True)
        
        # Train test split
        # get the largest duration time
        max_duration_idx = df["duration"].idxmax() ## argmax()
        df_test = df_feat.drop(max_duration_idx).sample(frac=0.3)
        df_train = df_feat.drop(df_test.index)
        df_val = df_train.drop(max_duration_idx).sample(frac=0.1)
        df_train = df_train.drop(df_val.index)
        
        # Starting Lab Trans
        ## If the given horizons in this form [0.25, .5, .75], discretize time by percentiles. 
        if config['horizons'][0] <1: ##
            times = np.quantile(df["duration"][df[event_list].sum(axis=1)> 0.0], horizons).tolist()
            input_cuts = np.array([0]+times+[df["duration"].max()])

        else:
            input_cuts = np.array([0]+config['horizons']+[df["duration"].max()])
            
        # assign cuts
        labtrans = LabelTransform(cuts=input_cuts)
        get_target = lambda df,event: (df['duration'].values, df[event].values)
        # this datasets have two competing events!
        df_y_train = pd.DataFrame({"duration":df["duration"][df_train.index]})
        df_y_test = pd.DataFrame({"duration":df["duration"][df_test.index]})
        df_y_val = pd.DataFrame({"duration":df["duration"][df_val.index]})
        
        for i,event in enumerate(event_list):
            labtrans.fit(*get_target(df.loc[df_train.index], event))
            y = labtrans.transform(*get_target(df, event)) # y = (discrete duration, event indicator) # helper.py 
            event_name = "event_{}".format(i)
            df[event_name] = y[1]
            df_y_train[event_name] = df[event_name][df_train.index]
            df_y_val[event_name] = df[event_name][df_val.index]
            df_y_test[event_name] = df[event_name][df_test.index]
        # discretized duration
        df["duration_disc"] = y[0]
        # proportion is the same for all events
        df["proportion"] = y[2]
        df_y_train["proportion"] = df["proportion"][df_train.index]
        df_y_val["proportion"] = df["proportion"][df_val.index]
        df_y_test["proportion"] = df["proportion"][df_test.index]
        
        if config['discrete_time']: ##
            df_y_train["duration"] = df["duration_disc"][df_train.index]
            df_y_val["duration"] = df["duration_disc"][df_val.index]
            
        if config['discrete_time_test']: ##
            df_y_test["duration"] = df["duration_disc"][df_test.index]
        
        # set number of events
        config['num_event'] = len(event_list)

    elif data == 'kcpsii':
        data = config['data']
        horizons = config['horizons']
        assert data in [ "kcpsii"], "Data Not kcpsii!" ##
        get_target = lambda df, event: (df['duration'].values, df[event].values) ##

        df0 = load_dataset0(args, new_data=True)
        df0 = df0.sample(frac=sample) if sample else df0
        
        df = copy.deepcopy(df0)
        df = df.rename(columns={"time": "duration"})
        # df['event'] = df[config['task']]
        event_list = df[config['task']] if type(config['task']) == list else df[[config['task']]] ##
                                                                                
        # times = np.quantile(df["duration"][df[config['task']].sum(axis=1) > 0.0], horizons).tolist() ## df['event'] == 1.0 
                        
        drop_col = ['duration'] + config['task'] if type(config['task']) == list else ['duration'] + [config['task']]
        df_feat = df.drop(drop_col,axis=1) ##
        
        ## change split dat sets into here
        # get the largest duration time 
        max_duration_idx = df["duration"].idxmax()
        
        # split sets 
        df_test = df_feat.drop(max_duration_idx).sample(frac=0.3) #? why drop max_duration ##?should drop index instead of label
        df_train = df_feat.drop(df_test.index)
        df_val = df_train.drop(max_duration_idx).sample(frac=0.1) ##/should drop index instead of label 
        df_train = df_train.drop(df_val.index)
        
        ## Missing imputation and create new variables
        feature0 = get_features(args.feature)[1:] 
        df_train, df_test, df_val, feature = dsets_process(args, (df_train, df_test, df_val), df0, feature0, validation=True)
        
        cols_categorical = ['SEX1']
        cols_standardize = [i for i in feature if i not in cols_categorical]
        df_train_std, df_test_std, df_val_std = df_train[cols_standardize], df_test[cols_standardize], df_val[cols_standardize]
        
        # Missings were imputed in dsets_process() so array output of scaler retains order
        scaler = StandardScaler()
        df_train_std_disc = scaler.fit_transform(df_train_std)
        df_test_std_disc, df_val_std_disc = scaler.transform(df_test_std), scaler.transform(df_val_std)
        
        # Concat numerical and categorical cols
        # Impose index into df_std_disc, as long as no change in order
        dts_concat = dsets_process2([df_train_std_disc, df_test_std_disc, df_val_std_disc],
                                    [df_train, df_test, df_val], 
                                    cols_standardize, cols_categorical)
        
        df_train_prc, df_test_prc, df_val_prc = dts_concat[0], dts_concat[1], dts_concat[2]
        
        # Label encode for categorical cols
        vocab_size = 0
        for _,feat in enumerate(cols_categorical):
            encoder = LabelEncoder()
            df_train_prc[feat] = encoder.fit_transform(df_train_prc[feat]).astype(float) + vocab_size
            df_test_prc[feat] = encoder.transform(df_test_prc[feat]).astype(float) + vocab_size ##
            df_val_prc[feat] = encoder.transform(df_val_prc[feat]).astype(float) + vocab_size ##
            vocab_size = df_feat[feat].max() + 1
        
        df_train_ftr, df_test_ftr, df_val_ftr = df_train_prc, df_test_prc, df_val_prc
        
        ## If the given horizons in this form [0.25, .5, .75], discretize time by percentiles. 
        if config['horizons'][0] < 1: ##
            times = np.quantile(df["duration"][df[config['task']].sum(axis=1) > 0.0], horizons).tolist()
            input_cuts = np.array([0]+times+[df["duration"].max()])
            
        else:
            input_cuts = np.array([0]+config['horizons']+[df["duration"].max()])

        # assign cuts - Later
        labtrans = LabelTransform(cuts=input_cuts) 

        ## Multi transform 0802
        df_y_train = pd.DataFrame({"duration":df["duration"][df_train.index]})
        df_y_test = pd.DataFrame({"duration":df["duration"][df_test.index]})
        df_y_val = pd.DataFrame({"duration":df["duration"][df_val.index]})
        
        for i,event in enumerate(event_list):
            # print('dataset - event', event)
            labtrans.fit(*get_target(df.loc[df_train_ftr.index], event))
            y = labtrans.transform(*get_target(df, event)) # y = (discrete duration, event indicator) ERROR
            
            event_name = "event_{}".format(i)
            df[event_name] = y[1]
            df_y_train[event_name] = df[event_name][df_train.index]
            df_y_val[event_name] = df[event_name][df_val.index]
            df_y_test[event_name] = df[event_name][df_test.index]
        
        # discretized duration
        df["duration_disc"] = y[0]
        
        # proportion is the same for all events
        df["proportion"] = y[2]
        
        df_y_train["proportion"] = df["proportion"][df_train.index]
        df_y_val["proportion"] = df["proportion"][df_val.index]
        df_y_test["proportion"] = df["proportion"][df_test.index]
        
        if config['discrete_time']: ##
            df_y_train["duration"] = df["duration_disc"][df_train.index]
            df_y_val["duration"] = df["duration_disc"][df_val.index]
                
        if config['discrete_time_test']: ##
            df_y_test["duration"] = df["duration_disc"][df_test.index]
            
    config['labtrans'] = labtrans
    config['num_numerical_feature'] = int(len(cols_standardize))
    config['num_categorical_feature'] = int(len(cols_categorical))
    config['num_feature'] = int(len(df_train.columns))
    config['vocab_size'] = int(vocab_size)
    config['duration_index'] = labtrans.cuts
    # print('ST dataset labtrans.out_features: ', labtrans.out_features)
    config['out_feature'] = int(labtrans.out_features)
    config['num_event'] = len(config['task']) if type(config['task']) == list else 1
    
    return df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val

def load_dataset0(args, new_data=False, drop_normal=False):  # Modified from run_surv.py 
    # lodd dataset
    data = pd.read_csv(args.dataset)
    data = data.sample(frac=1) #try shufflilng to see lstm results
    
    # load feature and event 
    feature = get_features(args.feature)
    feature = feature + ['SMOK2', 'SMOKA_MOD', 'SMOKC_MOD', 'alco1', 'ALCO_AMOUNT', 'F1', 'F12', 'F61'] if args.version==3 else feature
    
    if feature[0] == 'YID':
        feature.remove('YID')

    # making outcome of some cancers
    data['CRC'] = data['COLON'] + data['RECTM'] if new_data else data['CRC']
    data['LYMPH'] = data.LYMPH_C81 + data.LYMPH_C82 + data.LYMPH_C83 + data.LYMPH_C84 + data.LYMPH_C85 + data.LYMPH_C86
    data['GALL_BILE'] = data.GALLB + data.BILE
    data['LUNG_LAR'] = data.LUNG + data.LARYN
    data['UTE_CER'] = data.CERVI  #data.UTERI_C54 + data.UTERI_C55 +
    
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
    for i in range(len(dats)):
        dats[i] = pd.DataFrame(dats[i], columns=cols_std, 
                               index=dfs[i].index) ## impose index from fulldata
        dat_concat = pd.concat([dfs[i][cols_cat], dats[i]], axis=1)
        dts_concat.append(dat_concat)
    return dts_concat