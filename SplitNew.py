import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np
import os
def split_data(data, fea_name):
    # data = pd.read_csv(r'check_whole_two_rm_target_10_d.csv')
    if 'Unnamed: 0' in data.keys():
        data = data.drop('Unnamed: 0', axis=1)
    id = data['subject_label']
    institute = data['institute']
    data = data.drop(['subject_label', 'institute'], axis=1)

    ## drop data whose demecog is nan
    # data = data.dropna(subset=['demecog'])

    cat_cols = ['regcohortno', 'regsex', 'demecog']

    target = 'Inc_2'
    num_cols = list(set(data.keys()).difference(set(cat_cols)).difference(set([target])))
    ct = ColumnTransformer(
        [("CatTrans", OneHotEncoder(), cat_cols),
         ("NumTrans", MinMaxScaler(), num_cols), ],
        remainder='passthrough', verbose_feature_names_out=False)
    SubjectList = pd.DataFrame(ct.fit_transform(data), columns=ct.get_feature_names_out())

    clinical_cols = list(set(ct.get_feature_names_out()).difference(set([target])))

    np.save('ct.npy', ct)

    for col in clinical_cols:
        if 'nan' in col:
            clinical_cols.remove(col)

    clinical_cols.append('subject_label')
    clinical_cols.append('Inc_2')

    imp_mean = KNNImputer()

    imp_mean.fit(SubjectList)
    SubjectList = pd.DataFrame(imp_mean.transform(SubjectList), columns=SubjectList.columns)

    SubjectList['subject_label'] = id
    SubjectList['institute'] = institute
    data_csv = SubjectList

    ss = list(data_csv['institute'].value_counts().keys())
    ## create an empty dataframe
    data_test = pd.DataFrame()
    flag = 1
    for i in range(len(ss)):
        ## randomize ss
        np.random.shuffle(ss)
        ## get patient whose institute is ss[i]
        data_i = data_csv[data_csv['institute'] == ss[i]]
        ## add data_i to data_test
        data_test = pd.concat([data_test, data_i], axis=0)
        data_test = data_test.drop_duplicates()
        if len(data_test)>50:
            break
    data_train = data_csv.drop(data_test.index)

    data_test = data_test.reset_index(drop=True)
    data_train = data_train.reset_index(drop=True)

    # print(len(data_csv))
    # print(len(data_test)+len(data_train))

    bb = fea_name[0:1]
    bb.append(target)
    for fea in bb:
        st = data_train[fea].value_counts()
        p_st = st[1.0]/(st[0.0]+st[1.0])

        tt =data_test[fea].value_counts()
        p = tt[1.0]/(tt[0.0]+tt[1.0])

        if abs(p_st-p) > 0.1:
            flag = 0
            break

    if flag == 1:
        data_train = data_train[clinical_cols]
        data_test = data_test[clinical_cols]
        return data_train, data_test, flag
    else:
        return None, None,flag