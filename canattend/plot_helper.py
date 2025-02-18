import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import sklearn.metrics as metrics
import torchtuples as tt

## Functions related to RSF and DeepHit
# DeepHit definition
class CauseSpecificNet(torch.nn.Module):

	def __init__(self, in_features, num_nodes_shared, num_nodes_indiv, num_risks,
				 out_features, batch_norm=True, dropout=None):
		super().__init__()
		self.shared_net = tt.practical.MLPVanilla(
			in_features, num_nodes_shared[:-1], num_nodes_shared[-1],
			batch_norm, dropout,
		)
		self.risk_nets = torch.nn.ModuleList()
		for _ in range(num_risks):
			net = tt.practical.MLPVanilla(
				num_nodes_shared[-1], num_nodes_indiv, out_features,
				batch_norm, dropout,
			)
			self.risk_nets.append(net)

	def forward(self, input):
		residual = input
		out = self.shared_net(input)
		out = torch.cat((out, residual), dim=1) ##
		out = [net(out) for net in self.risk_nets]
		out = torch.stack(out, dim=1)
		return out

# Function for RSF
def calculate_hazard(survival_probs, time_points):
	survival_probs = np.array(survival_probs)
	survival_probs += 1e-50 # This prevents Nan value in calculating hazard rates
	
	time_points = np.array(time_points) # Should be in [0, 5, 10, 15] format

	# Ensure the inputs are sorted by time
	if not np.all(np.diff(time_points) > 0):
		raise ValueError("Time points must be strictly increasing.")

	# Compute time differences
	dt = np.diff(time_points)

	# Compute the hazard rates
	# hazard_rates = -np.diff(np.log(survival_probs)) / dt #?
	hazard_rates = (survival_probs[:, :-1] - survival_probs[:, 1:]) / survival_probs[:, :-1]
	# hazard_rates = -np.log((survival_probs[:, 1:]/survival_probs[:, :-1]))/dt
	return hazard_rates

# Function for DeepHit
def de_encode(data):
	event_df = data.iloc[:, 1:-1]

	# Assign 0 to rows with all zeros
	decoded = pd.Series(0, index=event_df.index)
	all_zero_rows = event_df.sum(axis=1) == 0
 
	# For other rows, assign index of max value (shifted by 1)
	decoded[~all_zero_rows] = event_df[~all_zero_rows].idxmax(axis=1).apply(lambda x: int(x.split("_")[-1]) + 1)

	data['label'] = decoded
	
	return data['duration'].values.astype('int64'), data['label'].values.astype('int64')

## Functions for plotting attention matrix
# To plot single attention heatmap
def attention_heatmap(attn_mtrx, layer_no, head_no, ftr, ax=None, fig=None, xticksize=8, yticksize=7):
	# try summing attn matrix along batch axis
	all_encode_tensors = torch.stack(attn_mtrx).cpu() #output[1]
	sum_tensors = torch.sum(all_encode_tensors, dim=1)

	# Create a figure and axis
	if not ax:
		fig, ax = plt.subplots()

	# Plot the heatmap
	cax = ax.imshow(sum_tensors[layer_no][head_no]/all_encode_tensors.shape[1], cmap='viridis', aspect='auto')

	# Add a colorbar
	fig.colorbar(cax, ax=ax)
	ax.set_xticks(range(0, 31))
	ax.set_xticklabels([i for i in range(31)], rotation=90, fontsize=xticksize)
	ax.set_yticks(range(0, 31))
	ax.set_yticklabels([i for i in range(31)], fontsize=yticksize)
	if ftr:
		ax.tick_params(axis='x', rotation=90)
		ax.set_xticklabels(ftr)
		ax.set_yticklabels(list(ftr))  #[::-1]

	# Customize axis labels
	ax.set_title(f'Layer {layer_no+1} - Head {head_no+1}')
	
## Functions for plotting Kaplan Meier classfication curves
# To calculate the C-index for a given threshold
def calculate_c_index_for_threshold(df_tuple, event_idx, cut_idx, risk_scores=None, c_ind=True, model=None, verbose=False):
	print(f'## Event: {event_idx} ##')
	df_x, df_y = df_tuple
	
	# Calculate risk score
	if risk_scores is None:
		preds = model.predict(df_x, batch_size=None, event=event_idx)[0] ## no need [0] before
		risk_scores = preds[:, cut_idx+1].detach().numpy()

	times_test, events_test = df_y['duration'], df_y[f'event_{event_idx}'] #['event']
	
	# Define a range of thresholds to test
	thresholds = np.linspace(np.min(risk_scores), np.max(risk_scores), 100)
	best_threshold = None
	best_criterion = -np.inf
 
	# Iterate over thresholds to find the best one
	for threshold in thresholds:
		if verbose:
			print('Threshold: ', threshold)
		high_risk = risk_scores > threshold
		low_risk = risk_scores <= threshold
		
		# Create a combined array of times and events for high and low risk groups
		combined_times = np.concatenate([times_test[high_risk], times_test[low_risk]])
		combined_events = np.concatenate([events_test[high_risk], events_test[low_risk]])

		# Create combined risk scores with high risk group as -1 and low risk as 1
		combined_risk = np.concatenate([-np.ones(sum(high_risk)), np.ones(sum(low_risk))])
	
		# current_c_index = calculate_c_index_for_threshold(event_idx=0, cut_idx=1, threshold=threshold)
		if c_ind: 
			current_c_index = concordance_index(combined_times, combined_risk, combined_events)
			if current_c_index > best_criterion:
				best_criterion = current_c_index
				best_threshold = threshold

		else:
			fpr, tpr, threshold0 = metrics.roc_curve(events_test, high_risk) #y_test, y_probas
			roc_auc = metrics.auc(fpr, tpr)
			if roc_auc > best_criterion:
				best_criterion = roc_auc
				best_threshold = threshold

		if verbose:
			print(f"Best Threshold: {best_threshold}, Best Value: {best_criterion}")
  
	return best_threshold

# To plot KM curves
def plot_km_curv(task, thres, preds, df_y_test_tpl, ax=None):
	df_y_test_duration, df_y_test_event = df_y_test_tpl
	# THRES = best_threshold
	clf = preds > thres #[:, 1]

	kmf0 = KaplanMeierFitter()
	kmf1 = KaplanMeierFitter()

	kmf0.fit(df_y_test_duration[~clf], df_y_test_event[~clf], label='Low risk') #duration, event
	kmf1.fit(df_y_test_duration[clf], df_y_test_event[clf], label='High risk')

	pred_risk = pd.merge(kmf0.survival_function_, 
			kmf1.survival_function_, 
			left_index=True, right_index=True) 
			
	# Log-rank test
	results = logrank_test(df_y_test_duration[clf], 
					   df_y_test_duration[~clf], 
					   event_observed_A=df_y_test_event[clf], 
					   event_observed_B=df_y_test_event[~clf])
	
	if ax:
		ax.plot(pred_risk)
		ax.set_ylabel(task)
		ax.legend(['Predicted as low risk', 'Predicted as high risk'], loc='lower left')
		# plt.text(0, -1, f'Log-rank test:p={0.4}', fontsize=8)
		if results.p_value > 0.0001:
			ax.text(0.43, 0.05, 'Log-rank test:p= %0.4f' % results.p_value, transform=ax.transAxes)
			
		else:
			ax.text(0.43, 0.05, 'Log-rank test:p<0.0001', transform=ax.transAxes)
		
	else:
		pred_risk.plot()

# To plot individual vs reference curves
def km_curves(idx, event_idx, model, df_test_tuple, threshold_list, cut_expand, ax=None):
	CUT_IDX = 1
	df_test, df_y_test = df_test_tuple
 
	# Reference curves
	threshold0 = threshold_list[event_idx] 
	preds = model.predict(df_test, batch_size=None, event=event_idx)[0]
	risk_scores = preds[:, CUT_IDX+1].detach().numpy()

	kmf0 = KaplanMeierFitter()
	kmf1 = KaplanMeierFitter()

	clf = risk_scores > threshold0 
	kmf0.fit(df_y_test['duration'][~clf], df_y_test[f'event_{event_idx}'][~clf], label='Low risk') #duration, event
	kmf1.fit(df_y_test['duration'][clf], df_y_test[f'event_{event_idx}'][clf], label='High risk')

	# Plotting 
	# pred_risk = kmf.survival_function_  #merge with kmf0, kmf1 later
	pred_surv = model.predict_surv(pd.DataFrame(df_test.loc[idx]).T, event=event_idx)
	
	pred_risk0 = pd.DataFrame(pred_surv.numpy(), columns=cut_expand, index=[idx]).T

	pred_risk = pd.merge(
						#  pred_risk,
						kmf0.survival_function_,
						kmf1.survival_function_,
						how = 'inner',
						left_index=True,
						right_index=True
						#  kmf1.survival_function_
						)
	pred_risk = pd.merge(pred_risk,
						pred_risk0,
						how='outer',
						left_index=True,
						right_index=True
						)

	pred_risk[idx] = pred_risk[idx].interpolate()

	if ax:
		ax.plot(pred_risk.iloc[:, :-1], alpha=0.4) # reference lines
		ax.plot(pred_risk.iloc[:, -1]) # idx line
		ax.set_title(f'{idx}') 
	else:
		pred_risk.iloc[:, :-1].plot(alpha=0.4)
		pred_risk.iloc[:, -1].plot()
  
## Functions for plotting ROC
# To plot ROC for each model
def plot_roc(event_idx, model, df_test, df_y_test, cut_idx=1, cuts=[5, 10, 15], model_type='st', ax=None, event_name=None, preds=None): #, threshold
	if model_type == 'st':
		preds = model.predict(df_test, batch_size=None, event=event_idx)[0]
		clf = preds[:, cut_idx] #> threshold
		clf = clf.detach().numpy()
		clf_np = clf.astype(int)
  
	if model_type == 'st1':
		preds = model.predict(df_test, batch_size=None) #[0]
		clf = preds[:, cut_idx+1] #> threshold
		clf = clf.detach().numpy()
		clf_np = clf.astype(int)
  
	elif model_type == 'rsf':
		surv_func = model.predict_survival_function(df_test, return_array=True)
		time_points = [0] + cuts
		preds = calculate_hazard(surv_func, time_points)
		clf = preds[:, cut_idx] #> threshold
		clf_np = clf.astype(int)
	
	elif model_type == 'deephit':
		if preds is None:
			preds = model.predict(df_test.values.astype('float32'))
		clf = preds[:, event_idx, cut_idx+1] > threshold
		clf_np = clf.astype(int)

	y_test = df_y_test[f'event_{event_idx}'] # ground truth labels
	y_probas = clf_np  # predicted probabilities generated by sklearn classifier
	fpr, tpr, threshold = metrics.roc_curve(y_test, y_probas)
	roc_auc = metrics.auc(fpr, tpr)

	event_name = event_name if event_name else event_idx

	if ax:
		ax.set_title(f'Receiver Operating Characteristic - {event_name}')
		ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
		ax.plot([0, 1], [0, 1],'r--')
		ax.set_xlim([0, 1])
		ax.set_ylim([0, 1])
		ax.set_ylabel('True Positive Rate')
		ax.set_xlabel('False Positive Rate')
		ax.legend(loc = 'lower right')
		# plt.show()
	else:
		plt.title(f'Receiver Operating Characteristic - {event_name}')
		plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
		plt.legend(loc = 'lower right')
		plt.plot([0, 1], [0, 1],'r--')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.show()
 
# To plot combined ROC
def plot_roc_combo(event_idx, 
                #    threshold_dict, 
                   model_list, model_name_list, df_test, df_y_test, cut_idx=1, cuts=[5, 10, 15], ax=None, event_name=None, preds=None):
	y_test = df_y_test[f'event_{event_idx}'] 
	
	# print('Len model_list: ', len(model_list))
	for j, mod_name in enumerate(model_name_list):
		color_list = ['m', 'c', 'y', 'r', 'b'] 
  
		if mod_name == 'DeepHit':
			# threshold = threshold_dict[mod_name][event_idx]
			preds = model_list[j].predict(df_test.values.astype('float32'))
			clf = preds[:, event_idx, cut_idx+1] ## cut+1
		 
		elif mod_name == 'RSF':
			# threshold = threshold_dict[mod_name][event_idx]
			surv_func = model_list[j][f'model_{event_idx}'].predict_survival_function(df_test, return_array=True)
			time_points = [0] + cuts
			preds = calculate_hazard(surv_func, time_points)
			clf = preds[:, cut_idx+1] #> threshold
   
		elif mod_name == 'ST':
			# threshold = threshold_dict[mod_name][event_idx_mod]
			preds = model_list[j].predict(df_test, batch_size=None) #[0]
			clf = preds[:, cut_idx+1]  ## cut+1
			clf = clf.detach().numpy()
   
		elif mod_name == 'CanAttend group 1':
			event_idx_mod =  event_idx if event_idx in [0, 1] else event_idx - 3 
			if event_idx in [0, 1]:
				color_list = ['m', 'c', 'y', 'r']  ##
    
			# threshold = threshold_dict[mod_name][event_idx_mod]
			preds = model_list[j].predict(df_test, batch_size=None, event=event_idx_mod)[0]
			clf = preds[:, cut_idx+1]  ## cut+1
			clf = clf.detach().numpy()
  		
		elif mod_name == 'CanAttend group 2':
			event_idx_mod = event_idx - 2
			if event_idx in [2, 3, 4]:
				color_list = ['m', 'c', 'y', 'b'] ## 'c', 'm'? 

			preds = model_list[j].predict(df_test, batch_size=None, event=event_idx_mod)[0]
			clf = preds[:, cut_idx+1]  ## cut+1
			clf = clf.detach().numpy()
   
		y_probas = clf # clf.astype(int) 
		fpr, tpr, threshold = metrics.roc_curve(y_test, y_probas)
		roc_auc = metrics.auc(fpr, tpr)

		if ax:
			ax.plot(fpr, tpr, color_list[j], label = f'{mod_name} AUC = {roc_auc:.2f}', ls='-', marker=None)
		
		else:
			plt.plot(fpr, tpr, color_list[j], label = f'{mod_name} AUC = {roc_auc:.2f}')
  
	if ax:
		ax.set_title(f'Receiver Operating Characteristic - {event_name}')
		ax.plot([0, 1], [0, 1],'k:', label='Reference line')
		ax.set_xlim([0, 1])
		ax.set_ylim([0, 1])
		ax.set_ylabel('True Positive Rate')
		ax.set_xlabel('False Positive Rate')
		ax.legend(loc = 'lower right')
	
	else:
		plt.title(f'Receiver Operating Characteristic - {event_name}')
		plt.legend(loc = 'lower right')
		plt.plot([0, 1], [0, 1],'k:', label='Reference line')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.show()
		
## Other functions 
def get_hp(filename):
		with open(filename, 'r') as f:
			return [x.strip('\n') for x in f.readlines() if '#' not in x]
