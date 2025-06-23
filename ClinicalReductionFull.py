import pandas as pd
from scipy.stats import kendalltau, linregress
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc, accuracy_score
import scipy.stats as stats
from preprocessing_new import preprocessing_data
from scipy.interpolate import interp1d

def get_prediction_distribution(test_list, fea_name, SubjectList_tv, values):
    results = {}
    for value in values:
        results[value] = []

    for i in range(0, 2000):
        # print(i)
        train_val_list = SubjectList_tv.sample(n=len(SubjectList_tv), replace=True)

        train_x = train_val_list[fea_name]
        train_y = train_val_list[target]

        lm = linear_model.LogisticRegression(max_iter=100000, class_weight='balanced',
                                             random_state=np.random.randint(0, 10000))
        lm.fit(train_x, train_y)
        fpr, tpr, thresholds = roc_curve(train_y, lm.predict_proba(train_x)[:, -1])
        roc_auc_train = auc(fpr, tpr)

        opt_thres = find_opt_thres(lm, train_x, train_y)


        if (lm.coef_[0][2] > 0) & (lm.coef_[0][3] > 0) & (roc_auc_train>0.65):  # & ((pred_prob-opt_thres)>0):

            for value in values:
                test_list_change = test_list.copy()
                test_list_change.loc[0, fea_name[-1]] = value
                pred_prob = lm.predict_proba(test_list_change[fea_name])[:, -1]
                y_pred_opt = pred_prob - opt_thres + 0.5
                results[value].append(y_pred_opt[0])

    ## calculate the mean of results for each value
    results = pd.DataFrame(results)
    final_mean = np.nanmean(results, axis=0)
    f = interp1d(final_mean, values)
    try:
        opt_v = f(0.5)
        test_list_change = test_list.copy()
        test_list_change.loc[0, fea_name[-1]] = opt_v
        opt_list = get_original_data(test_list_change.copy(), fea_name[2:4])[fea_name]
        opt_value = opt_list[fea_name[-1]].values[0]
    except:
        opt_value = 0
        print(final_mean)

    return opt_value

def get_original_data(data, orig_list):
    ct = np.load(r'ct.npy', allow_pickle=True).item()
    ### ct includes one hot encoder and minmax scaler.  Here to use minmax scaler to inverse transform data to original scale
    fitted_minmax_scaler = ct.named_transformers_['NumTrans']
    orig_data = fitted_minmax_scaler.inverse_transform(data[ct.transformers_[1][2]])

    for name in orig_list:
        index_orig = ct.transformers_[1][2].index(name)
        index = data.columns.get_loc(name)
        ## replace the data with original data
        data.loc[:, name] = orig_data[:, index_orig]

    return data

def find_opt_thres(lm, train_x, train_y):
    pred_prob = lm.predict_proba(train_x)[:, -1]
    fpr, tpr, thresholds = roc_curve(train_y, pred_prob)
    # max_index = (tpr - fpr).argmax()
    max_index = np.argmax(np.sqrt(tpr * (1 - fpr)))
    opt_thres = thresholds[max_index]
    return opt_thres

def get_threshold_toxicity(shap_values, orig_data, fea_name, point):
    shap_y = shap_values.values[:, -1]
    orig_x = orig_data[fea_name[-1]].values
    ## find the linear relationship between shap_y and orig_x
    slope, intercept, r_value, p_value, std_err = linregress(orig_x, shap_y)
    ## find the threshold of orig_x
    thres = (point - intercept) / slope
    return thres

def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

data = pd.read_csv(r'ClinicalDose_f.csv')
if 'Unnamed: 0' in data.keys():
    total_data = data.drop('Unnamed: 0', axis=1)


fea_name = ['regcohortno_Arm B (Standard + Irinotecan)',
            'demecog_Fully Active',
            'age_years',
            'small_bowel_v10',]
target = 'Inc_2'

SubjectList_whole = preprocessing_data(total_data, '')
auc_list = []
acc_list_ext = []

## load counts_new.npy and data_dict_new.npy
data_dict = np.load('data_dict_new2.npy', allow_pickle=True).item()
key1 = [key for key in data_dict.keys() if 'count1' in key]
sub1 = data_dict[key1[0]]
key2 = [key for key in data_dict.keys() if 'count2' in key]
sub2 = data_dict[key2[0]]
## combine sub1 and sub2 as group1
group1 = sub1.copy()
group1.extend(sub2)

orig_v =[]
reduce_v = []

for i, sub in enumerate(group1):
    print(i)
    test_list = SubjectList_whole[SubjectList_whole['subject_label'] == sub]
    test_orig = get_original_data(test_list.copy(), fea_name[2:4])
    orig_v.append(test_orig[fea_name[-1]].values[0])
    ## remove test_list from SubjectList
    SubjectList_tv = SubjectList_whole.drop(test_list.index, axis=0).reset_index(drop=True)
    test_list = test_list.reset_index(drop=True)

    ## space from 1 to 0, with step from bigger to smaller

    start = test_list[fea_name[-1]].values[0]*0.95
    stop = 0
    num_steps = 5

    step_size = (start - stop) / num_steps

    values =[]
    for j in range(num_steps + 1):
        value = start - j * step_size
        values.append(value)

    # values = np.arange(test_list[fea_name[-1]].values[0]*0.95, 0.02,  -0.05)
    # values = np.append(values, 0)

    opt_v = get_prediction_distribution(test_list, fea_name, SubjectList_tv, values)
    reduce_v.append(opt_v)


## save the orig_v and reduce_v
orig_v = pd.DataFrame(orig_v)
reduce_v = pd.DataFrame(reduce_v)
orig_v.to_csv('orig_v_group1.csv')
reduce_v.to_csv('reduce_v_group1.csv')

print(np.mean(orig_v))
print(np.mean(reduce_v))


# if original_prediction < 0.5:
#     if arm_info == 0:
#         test_list_change_arm = test_list.copy()
#         test_list_change_arm[fea_name[0]] = 1
#         prediction_arm_change = get_prediction_distribution(test_list_change_arm, fea_name, SubjectList_tv)
#         if prediction_arm_change >= 0.5:
#             test_list_change_arm_V = test_list_change_arm.copy()
#             test_list_change_arm_V.loc[0, fea_name[-1]] = 0
#             reduced_V0_prediction_arm_change = get_prediction_distribution(test_list_change_arm_V, fea_name, SubjectList_tv)
#             if reduced_V0_prediction_arm_change < 0.5:
#                 group2.append(sub)




