import pandas as pd
from scipy.stats import kendalltau, linregress
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_curve, auc, accuracy_score
from numpy import interp
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
# import shap
from imblearn.over_sampling import RandomOverSampler
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier, RUSBoostClassifier, \
    EasyEnsembleClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.inspection import DecisionBoundaryDisplay
import cv2
from sklearn.model_selection import train_test_split
# from Split import split_data
from Split2 import split_data
import shap
from scipy.stats import gaussian_kde

def get_original_data(data, orig_list):
    # orig_list = ['age_years', 'intestine_target_v45']
    ct = np.load('ct.npy', allow_pickle=True).item()
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


data = pd.read_csv(r'ClinicalDose_f.csv')
if 'Unnamed: 0' in data.keys():
    data = data.drop('Unnamed: 0', axis=1)


target = 'Inc_2'
fea_name = [
    'regcohortno_Arm B (Standard + Irinotecan)',
    'demecog_Fully Active',
    'age_years',
    'small_bowel_v10',
]
iteration = 0
n = len(fea_name)
masv = {}
for fea in fea_name:
    masv[fea+'_orig'] = []
    masv[fea+'_shap'] = []
fig, ax = plt.subplots(2, 2, figsize=(20, 16))
base_value_list = []
for i in range(2000):  ## 9
    SubjectList, SubjectList_test, whole, flag = split_data(data, fea_name)
    if flag == 1:
        SubjectList_whole = whole#pd.concat([SubjectList, SubjectList_test])
        external_x = SubjectList_test[fea_name]
        external_y = SubjectList_test[target]

        c0 = SubjectList[SubjectList[target] == 0]
        c1 = SubjectList[SubjectList[target] == 1]
        for j in range(0, 100):
            # val_list = pd.concat([c0.sample(5), c1.sample(5)])
            # train_list = SubjectList.drop(val_list.index, axis=0)
            train_list, val_list = train_test_split(SubjectList, test_size=0.1, random_state=np.random.randint(0, 10000))
            val_list = val_list.reset_index(drop=True)
            train_list = train_list.reset_index(drop=True)

            train_x = train_list[fea_name]
            train_y = train_list[target]

            val_x = val_list[fea_name]
            val_y = val_list[target]

            if len(val_y.value_counts())>1:

                lm = linear_model.LogisticRegression(max_iter=100000, class_weight='balanced',
                                                     random_state=np.random.randint(0, 10000))
                lm.fit(train_x, train_y)

                fpr, tpr, thresholds = roc_curve(train_y, lm.predict_proba(train_x)[:, -1])
                roc_auc_train = auc(fpr, tpr)

                prediction = lm.predict(val_x)
                pred_prob_val = lm.predict_proba(val_x)[:, -1]
                fpr_val, tpr_val, thresholds_val = roc_curve(val_y, pred_prob_val)

                roc_auc_val = auc(fpr_val, tpr_val)

                pred_prob_test_external = lm.predict_proba(external_x)[:, -1]  # decision_function(X[test])
                fpr_e, tpr_e, thresholds_e = roc_curve(external_y, pred_prob_test_external)
                roc_auc_ext = auc(fpr_e, tpr_e)
                # if roc_auc_val > 0.6:
                if roc_auc_val > 0.65 and roc_auc_ext > 0.65:
                    iteration += 1
                    # print('optmium threshold', thresholds[np.argmax(tpr - fpr)])

                    explainer = shap.Explainer(lm, train_x, feature_names=fea_name)
                    shap_values = explainer(SubjectList_whole[fea_name])
                    # print(shap_values.base_values[0])
                    base_value_list.append(shap_values.base_values[0])
                    orig_data = get_original_data(SubjectList_whole.copy(), fea_name[2:])[fea_name]
                    shap_values.data = np.array(orig_data)

                    for fea in fea_name:
                        name = fea
                        col_ind = train_x.columns.get_loc(name)
                        orig_list = orig_data.values[:, col_ind]
                        shap_list = shap_values.values[:, col_ind]

                        masv[fea+'_orig'].append(orig_list)
                        masv[fea+'_shap'].append(shap_list)

                    if iteration > 5000:
                        # for i, fea in enumerate(fea_name):
                        #     shap_values.values[:, i] = np.array(masv[fea + '_shap']).mean(axis=0)
                        #     shap_values.data[:, i] = np.array(masv[fea + '_orig']).mean(axis=0)
                        #
                        #
                        # shap_values.feature_names = ['Arm (1: Arm B, 0: Arm A)', 'Fully Active(1: Yes, 0: No)', 'Age',
                        #                              'Small Bowel V10']
                        # plt.rcParams["figure.autolayout"] = True
                        # ## set y label font size
                        # plt.yticks(fontsize=15)
                        # plt.xticks(fontsize=15)
                        # ## set label font bold
                        # # plt.xlabel('SHAP value', fontsize=15, fontweight='bold')
                        #
                        # shap.plots.beeswarm(shap_values, max_display=15, order=shap.Explanation.abs.mean(0),
                        #                     plot_size=(12, 5))
                        # plt.show()
                        # exit(0)
                        for i, fea in enumerate(fea_name):
                            fea_shap = shap_values[:, i]
                            fea_shap.values = np.array(masv[fea + '_shap']).mean(axis=0)
                            fea_shap.data = np.array(masv[fea + '_orig']).mean(axis=0)
                            # ## sort the fea_shap.data based on sorted index
                            sorted_index = np.argsort(fea_shap.data)
                            # fea_shap.data = fea_shap.data[sorted_index]
                            # fea_shap.values = fea_shap.values[sorted_index]
                            ## calculate the 5% and 95% confidence interval
                            ci_5 = np.percentile(np.array(masv[fea + '_shap']), 5, axis=0)
                            ci_95 = np.percentile(np.array(masv[fea + '_shap']), 95, axis=0)

                            ## sort the ci_5 and ci_95 based on sorted index
                            ci_95 = ci_95[sorted_index]
                            ci_5 = ci_5[sorted_index]
                            sorted_data = fea_shap.data[sorted_index]


                            if i<2:
                                shap.plots.scatter(fea_shap, ax=ax[0, i], show=False, alpha=0)
                                ## show botplot of shap value for fea with transparency of 0.5, mean shown as a red dot
                                seaborn.boxplot(x=np.array(masv[fea + '_orig']).flatten(),
                                                y=np.array(masv[fea + '_shap']).flatten(),
                                                ax=ax[0, i], color='blue',
                                                boxprops=dict(alpha=.3), width=0.5,
                                                showmeans=True,
                                                meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "red"},)
                                ## set x label
                                if i == 0:
                                    ax[0, i].set_xlabel('Arm (1: Arm B, 0: Arm A)', fontsize=20)
                                    ax[0, i].set_ylabel('SHAP value', fontsize=20)
                                else:
                                    ax[0, i].set_xlabel('Fully Active(1: Yes, 0: No)', fontsize=20)
                                    ax[0, i].set_ylabel('')
                                ## set x tick label size
                                ax[0, i].tick_params(axis='x', labelsize=20)
                                ## set y tick label size
                                ax[0, i].tick_params(axis='y', labelsize=20)
                                ax[0, i].grid(axis='y')
                                ## add a horizontal line at y = 0
                                ax[0, i].axhline(0, color='green', linewidth=4)

                            else:
                                shap.plots.scatter(fea_shap, ax=ax[1, i-2], show=False, alpha=1, color='red')
                                ax[1, i-2].fill_between(sorted_data, ci_5, ci_95, color='blue', alpha=0.3)

                                ax[1, i-2].yaxis.set_tick_params(labelsize=20)
                                ax[1, i-2].yaxis.label.set_size(20)
                                ax[1, i-2].xaxis.label.set_size(20)
                                ## increase the font size of xticks
                                ax[1, i-2].xaxis.set_tick_params(labelsize=20)
                                ax[1, i - 2].yaxis.set_tick_params(labelsize=20)
                                ax[1, i -2].grid(axis='y')
                                ## add a horizontal line at y = 0
                                ax[1, i -2].axhline(0, color='green', linewidth=4)
                                if i == 2:
                                    ax[1, i-2].set_ylabel('SHAP value', fontsize=20)
                                else:
                                    ax[1, i-2].set_ylabel('')

                        ax[1, 0].set_xlabel('Age(years)', fontsize=20)
                        ax[1, 1].set_xlabel('Small bowel V10', fontsize=20)
                        ## save figure as eps
                        fig.savefig("scatter.eps")
                        fig.savefig("Four.png")
                        plt.show()
                        print(base_value_list)
                        print('mean base', np.mean(base_value_list))
                        ## calculate the distribution of base value
                        prob, edges = np.histogram(base_value_list, bins=100, density=True)
                        ## get the value of highest prob
                        max_index = np.argmax(prob)
                        ## get the mean of the value of highest prob
                        max_value_hist = np.mean([edges[max_index], edges[max_index + 1]])
                        ## get the value of highest density
                        kde = gaussian_kde(base_value_list)
                        xx = np.linspace(-1, 1, 1000)
                        yy = kde(xx)
                        max_index = np.argmax(yy)
                        max_value = xx[max_index]
                        print('high prob base value', max_value)
                        print('high hist base value', max_value_hist)
                        exit(0)



