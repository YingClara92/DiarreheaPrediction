import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc
from numpy import interp
from SplitNew import split_data
from preprocessing_new import preprocessing_data


def find_opt_thres(lm, train_x, train_y):
    pred_prob = lm.predict_proba(train_x)[:, -1]
    fpr, tpr, thresholds = roc_curve(train_y, pred_prob)
    # max_index = (tpr - fpr).argmax()
    max_index = np.argmax(np.sqrt(tpr * (1 - fpr)))
    opt_thres = thresholds[max_index]
    return opt_thres


data = pd.read_csv(r'ClinicalDose_f.csv')
# data = pd.read_csv(r'../check_whole_rm_target_origv_d.csv')
if 'Unnamed: 0' in data.keys():
    data = data.drop('Unnamed: 0', axis=1)

target = 'Inc_2'
fea_name = [
    'regcohortno_Arm B (Standard + Irinotecan)',
    'demecog_Fully Active',
    'age_years',
    'small_bowel_v10',
]
## auc
auc_list_ext = []
auc_list_train = []
## precision, recall, specificity, sensitivity

precision_list_ext = []
recall_list_ext = []
specificity_list_ext = []

precision_list_train = []
recall_list_train = []
specificity_list_train = []
## tpr
tpr_list = []
idx_distance = 0
file_name = 'whole/distance_record_whole_lg_weight_10.csv'
SubjectList_whole = preprocessing_data(data, file_name)
distance_record = pd.read_csv(file_name)
if 'Unnamed: 0' in distance_record.keys():
    distance_record = distance_record.drop('Unnamed: 0', axis=1)

record_dict ={}
record_train = {}
for i in range(1000):  ## 9
    SubjectList, SubjectList_test, flag = split_data(data, fea_name)
    if flag == 1:
        ## add SubjectList and SubjectList_test to SubjectWhole
        external_x = SubjectList_test[fea_name]
        external_y = SubjectList_test[target]

        for j in range(0, 2000):

            train_list = SubjectList.sample(n=len(SubjectList), replace=True)
            train_list = train_list.reset_index(drop=True)

            train_x = train_list[fea_name]
            train_y = train_list[target]


            lm = linear_model.LogisticRegression(max_iter=10000, class_weight='balanced',
                                                 random_state=np.random.randint(0, 10000))
            lm.fit(train_x, train_y)

            fpr, tpr, thresholds = roc_curve(train_y, lm.predict_proba(train_x)[:, -1])
            roc_auc_train = auc(fpr, tpr)

            auc_list_train.append(roc_auc_train)

            pred_prob_test_external = lm.predict_proba(external_x)[:, -1]  # decision_function(X[test])
            fpr_e, tpr_e, thresholds_e = roc_curve(external_y, pred_prob_test_external)

            x = np.arange(0, 1.0, 0.05)
            tpr_list.append(interp(x, fpr, tpr))

            roc_auc_ext = auc(fpr_e, tpr_e)
            auc_list_ext.append(roc_auc_ext)

            opt_thres = find_opt_thres(lm, train_x, train_y)
            label_pred_ext = pred_prob_test_external>opt_thres

            ## calculate precision, recall, specificity, sensitivity for external test set
            TP_ext = np.sum((external_y == 1) & (label_pred_ext == 1))
            TN_ext = np.sum((external_y == 0) & (label_pred_ext == 0))
            FP_ext = np.sum((external_y == 0) & (label_pred_ext == 1))
            FN_ext = np.sum((external_y == 1) & (label_pred_ext == 0))
            precision_ext = TP_ext / (TP_ext + FP_ext)
            recall_ext = TP_ext / (TP_ext + FN_ext)
            specificity_ext = TN_ext / (TN_ext + FP_ext)
            sensitivity_ext = TP_ext / (TP_ext + FN_ext)
            precision_list_ext.append(precision_ext)
            recall_list_ext.append(recall_ext)
            specificity_list_ext.append(specificity_ext)

            ##calculate precision, recall, specificity, sensitivity for train set
            label_pred_train = lm.predict(train_x)
            TP_train = np.sum((train_y == 1) & (label_pred_train == 1))
            TN_train = np.sum((train_y == 0) & (label_pred_train == 0))
            FP_train = np.sum((train_y == 0) & (label_pred_train == 1))
            FN_train = np.sum((train_y == 1) & (label_pred_train == 0))
            precision_train = TP_train / (TP_train + FP_train)
            recall_train = TP_train / (TP_train + FN_train)
            specificity_train = TN_train / (TN_train + FP_train)
            sensitivity_train = TP_train / (TP_train + FN_train)
            precision_list_train.append(precision_train)
            recall_list_train.append(recall_train)
            specificity_list_train.append(specificity_train)

            ## record the index of the train_list in the SubjectList_whole based on subject_label
            idx = SubjectList_whole[SubjectList_whole['subject_label'].isin(train_list['subject_label'])].index
            ## get the probablity of whole data
            pred_prob_whole = lm.predict_proba(SubjectList_whole[fea_name])[:, -1]
            label_pred_whole = pred_prob_whole > opt_thres
            ## change the type of label_pred_whole to int16
            label_pred_whole = label_pred_whole.astype(np.float16)
            ## set idx of the label_pred_whole to nan
            # label_pred_whole[idx] = np.nan
            record_dict['run_' + str(idx_distance)] = label_pred_whole
            ## create the same all zeros metric as label_pred_whole
            flag_train = np.zeros(len(SubjectList_whole))
            flag_train[idx] = 1
            record_train['run_' + str(idx_distance)] = flag_train
            idx_distance += 1


## save distance_record to csv file
record_data = pd.DataFrame(record_dict)
distance_record = pd.concat([distance_record, record_data], axis=1)
distance_record.to_csv(file_name)

record_train = pd.DataFrame(record_train)
record_train = pd.concat([distance_record, record_train], axis=1)
record_train.to_csv('whole/distance_record_train_flag_lg_weight_10.csv')

## save auc_list_train, auc_list_val, auc_list_ext to npy file
data_dict={'train': auc_list_train, 'ext': auc_list_ext}
np.save('whole/auc_list_10.npy', data_dict)
## save precision_list_val, recall_list_val, specificity_list_val, sensitivity_list_val, precision_list_ext, recall_list_ext, specificity_list_ext, sensitivity_list_ext to npy file
data_dict={ 'precision_ext': precision_list_ext,
            'recall_ext': recall_list_ext,
            'specificity_ext': specificity_list_ext,
            'precision_train': precision_list_train,
            'recall_train': recall_list_train,
            'specificity_train': specificity_list_train
            }
np.save('whole/precision_recall_list_10.npy', data_dict)
