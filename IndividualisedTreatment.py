import pandas as pd
from scipy.stats import kendalltau, linregress
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc, accuracy_score
import scipy.stats as stats
from preprocessing_new import preprocessing_data

def get_prediction_distribution(test_x, fea_name, SubjectList_tv):
    results = []
    for i in range(0, 5000):
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
        pred_prob = lm.predict_proba(test_x)[:, -1]

        y_pred_opt = pred_prob - opt_thres + 0.5
        results.append(y_pred_opt[0])

    final_mean = np.nanmean(results, axis=0)
    return final_mean

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

tpr_list = []
fig, ax = plt.subplots()
count1 = 0 ## >0.5, arm a, reduce V can <0.5
count2 = 0 ## >0.5, arm b, reudce V can <0.5
count3 = 0 ## >0.5, arm b, reduce V cannot <0.5, change arm can <0.5
count4 = 0 ## >0.5, arm b, reduce V cannot <0.5, change arm cannot <0.5


count5 = 0 ## <0.5, good arm B
count6 = 0 ## <0.5, arm A + Irinotecan still <0.5
count7 = 0 ## <0.5, +Irinotecan, reduce V can <0.5
count8 = 0 ## <0.5, +Irinotecan, reduce V cannot <0.5

## patient list for each cohort
patient_list_count1 = []
patient_list_count2 = []
patient_list_count3 = []
patient_list_count4 = []
patient_list_count5 = []
patient_list_count6 = []
patient_list_count7 = []
patient_list_count8 = []

for i, sub in enumerate(SubjectList_whole['subject_label']):
    print(i)

    test_list = SubjectList_whole[SubjectList_whole['subject_label'] == sub]
    ## remove test_list from SubjectList
    SubjectList_tv = SubjectList_whole.drop(test_list.index, axis=0).reset_index(drop=True)
    test_list = test_list.reset_index(drop=True)
    test_list = test_list[fea_name]
    test_x = test_list[fea_name]
    original_prediction = get_prediction_distribution(test_x, fea_name, SubjectList_tv)

    arm_info = test_list[fea_name[0]].values[0]

    if original_prediction >= 0.5:
        test_list_change = test_list.copy()
        test_list_change.loc[0, fea_name[-1]] = 0
        reduced_V_prediction = get_prediction_distribution(test_list_change, fea_name, SubjectList_tv)
        if reduced_V_prediction < 0.5:
            if arm_info == 0:
                count1 += 1
                patient_list_count1.append(sub)
            else:
                count2 += 1
                patient_list_count2.append(sub)

        else:
            if arm_info == 1:
                test_list_change_arm = test_list.copy()
                test_list_change_arm[fea_name[0]] = 0
                prediction_arm_change = get_prediction_distribution(test_list_change_arm, fea_name, SubjectList_tv)
                if prediction_arm_change < 0.5:
                    count3 += 1
                    patient_list_count3.append(sub)
                else:
                    count4 += 1
                    patient_list_count4.append(sub)

    if original_prediction < 0.5:
        if arm_info == 1:
            count5 += 1
            patient_list_count5.append(sub)

        if arm_info == 0:
            test_list_change_arm = test_list.copy()
            test_list_change_arm[fea_name[0]] = 1
            prediction_arm_change = get_prediction_distribution(test_list_change_arm, fea_name, SubjectList_tv)
            if prediction_arm_change < 0.5:
                count6 += 1
                patient_list_count6.append(sub)
            else:
                test_list_change_arm_V = test_list_change_arm.copy()
                test_list_change_arm_V.loc[0, fea_name[-1]] = 0
                reduced_V0_prediction_arm_change = get_prediction_distribution(test_list_change_arm_V, fea_name, SubjectList_tv)
                if reduced_V0_prediction_arm_change < 0.5:
                    count7 += 1
                    patient_list_count7.append(sub)
                else:
                    count8 += 1
                    patient_list_count8.append(sub)



counts = [count1, count2, count3, count4, count5, count6, count7, count8]
## save the counts
np.save('counts_new2.npy', counts)

data_dict = {'count1_'+str(count1): patient_list_count1,
             'count2_'+str(count2): patient_list_count2,
             'count3_'+str(count3): patient_list_count3,
             'count4_'+str(count4): patient_list_count4,
             'count5_'+str(count5): patient_list_count5,
             'count6_'+str(count6): patient_list_count6,
             'count7_'+str(count7): patient_list_count7,
             'count8_'+str(count8): patient_list_count8}


np.save('data_dict_new2.npy', data_dict)

print('count1:', count1)
print('count2:', count2)
print('count3:', count3)
print('count4:', count4)
print('count5:', count5)
print('count6:', count6)
print('count7:', count7)
print('count8:', count8)
