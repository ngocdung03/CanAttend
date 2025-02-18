from collections import defaultdict

import numpy as np
from sksurv.metrics import brier_score, concordance_index_ipcw


class Evaluator:
    def __init__(self, df, train_index):
        self.df_train_all = df.loc[train_index]
 
    def eval_single(self, model, test_set, val_batch_size=None, time_unit='quantile', allow_all_censored=False):
        
        df_train_all = self.df_train_all
        get_target = lambda df: (df['duration'].values, df['event_0'].values) #event
        durations_train, events_train = get_target(df_train_all)
        et_train = np.array([(events_train[i], durations_train[i]) for i in range(len(events_train))],
                        dtype = [('e', bool), ('t', float)])
        times = model.config['duration_index'][1:-1]
        horizons = model.config['horizons']

        df_test, df_y_test = test_set
        surv = model.predict_surv(df_test, batch_size=val_batch_size)
        risk = 1 - surv

        durations_test, events_test = get_target(df_y_test)
        et_test = np.array([(events_test[i], durations_test[i]) for i in range(len(events_test))],
                    dtype = [('e', bool), ('t', float)])

        metric_dict = defaultdict(list)
        brs = brier_score(et_train, et_test, surv.to("cpu").numpy()[:,1:-1], times)[1]

        cis = []
        # print('evel_util allow all censored', allow_all_censored)
        for i, _ in enumerate(times):
            cis.append(
                concordance_index_ipcw(et_train, et_test, estimate=risk[:, i+1].to("cpu").numpy(), tau=times[i], allow_all_censored=allow_all_censored)[0]
                )
            metric_dict[f'{horizons[i]}_ipcw'] = cis[i]
            metric_dict[f'{horizons[i]}_brier'] = brs[i]

        for horizon in enumerate(horizons):
            print(f"For {horizon[1]} {time_unit},")
            print("TD Concordance Index - IPCW:", cis[horizon[0]])
            print("Brier Score:", brs[horizon[0]])
        
        return metric_dict

    def eval_multi(self, model, test_set, val_batch_size=None, allow_all_censored=False, time_unit='quantile'): ## val_batch_size=10000
        times = model.config['duration_index'][1:-1]
        horizons = model.config['horizons']
        df_train_all = self.df_train_all
        get_target = lambda df, risk: (df['duration'].values, df['event_{}'.format(risk)].values)  # noqa: E731
        df_test, df_y_test = test_set

        metric_dict = defaultdict(list)
        for risk_idx in range(model.config.num_event):
            # print('risk_idx', risk_idx) ##
            durations_train, events_train = get_target(df_train_all, risk_idx)
            durations_test, events_test = get_target(df_y_test, risk_idx)
            
            surv = model.predict_surv(df_test, batch_size=val_batch_size, event=risk_idx)
            
            risk = 1 - surv
            
            et_train = np.array([(events_train[i], durations_train[i]) for i in range(len(events_train))],
                            dtype = [('e', bool), ('t', float)])
            et_test = np.array([(events_test[i], durations_test[i]) for i in range(len(events_test))],
                        dtype = [('e', bool), ('t', float)])
            
            
            brs = brier_score(et_train, et_test, surv.to("cpu").numpy()[:,1:-1], times)[1]
            cis = []
            for i, _ in enumerate(times):
                
                ### 
                mask = durations_test < times[i]
                event_1st_time = events_test[mask]
                if not np.any(event_1st_time):
                    print(f'All censored ({len(event_1st_time)} obs) at risk {risk_idx} and {i}th time')

                cis.append(concordance_index_ipcw(et_train, et_test, risk[:, i+1].to("cpu").numpy(), times[i], allow_all_censored=allow_all_censored)[0])  ##
                metric_dict[f'{horizons[i]}_ipcw_{risk_idx}'] = cis[i]
                metric_dict[f'{horizons[i]}_brier_{risk_idx}'] = brs[i]

            for horizon in enumerate(horizons):
                print("Event: {} For {} {},".format(risk_idx, horizon[1], time_unit))
                print("TD Concordance Index - IPCW:", cis[horizon[0]])
                print("Brier Score:", brs[horizon[0]])
        
        return metric_dict

    def eval(self, model, test_set, confidence=None, val_batch_size=None, allow_all_censored=False, time_unit='quantile'):
        print("***"*10) 
        print("start evaluation")
        print("***"*10)

        if confidence is None:
            if model.config['num_event'] > 1:
                return self.eval_multi(model, test_set, val_batch_size, allow_all_censored=allow_all_censored, time_unit=time_unit)
            else:
                return self.eval_single(model, test_set, val_batch_size, allow_all_censored=allow_all_censored, time_unit=time_unit)

        else:
            # do bootstrapping
            stats_dict = defaultdict(list)
            for i in range(10):
                df_test = test_set[0].sample(test_set[0].shape[0], replace=True)
                df_y_test = test_set[1].loc[df_test.index]
                
                if model.config['num_event'] > 1:
                    res_dict = self.eval_multi(model, (df_test, df_y_test), val_batch_size, allow_all_censored=allow_all_censored, time_unit=time_unit)
                else:
                    res_dict = self.eval_single(model, (df_test, df_y_test), val_batch_size, allow_all_censored=allow_all_censored, time_unit=time_unit) #

                for k in res_dict.keys():
                    stats_dict[k].append(res_dict[k])

            metric_dict = {}
            # compute confidence interveal 95%
            alpha = confidence
            p1 = ((1-alpha)/2) * 100
            p2 = (alpha+((1.0-alpha)/2.0)) * 100
            for k in stats_dict.keys():
                stats = stats_dict[k]
                lower = max(0, np.percentile(stats, p1))
                upper = min(1.0, np.percentile(stats, p2))
                # print(f'{alpha} confidence interval {lower} and {upper}')
                print(f'{alpha} confidence {k} average:', (upper+lower)/2)
                print(f'{alpha} confidence {k} interval:', (upper-lower)/2)
                metric_dict[k] = [(upper+lower)/2, (upper-lower)/2]

            return metric_dict
