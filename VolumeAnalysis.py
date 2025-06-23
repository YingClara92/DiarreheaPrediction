import numpy as np
import pandas as pd

## load data from orig_v_group1.csv and reduce_v_group1.csv
orig_v_group1 = pd.read_csv('orig_v_group1.csv')
reduce_v_group1 = pd.read_csv('reduce_v_group1.csv')

data_dict = np.load('data_dict_new2.npy', allow_pickle=True).item()
key1 = [key for key in data_dict.keys() if 'count1' in key][0]
key2 = [key for key in data_dict.keys() if 'count2' in key][0]
key3 = [key for key in data_dict.keys() if 'count3' in key][0]
key4 = [key for key in data_dict.keys() if 'count4' in key][0]
key5 = [key for key in data_dict.keys() if 'count5' in key][0]
key6 = [key for key in data_dict.keys() if 'count6' in key][0]
key7 = [key for key in data_dict.keys() if 'count7' in key][0]
key8 = [key for key in data_dict.keys() if 'count8' in key][0]

sub1 = data_dict[key1]
sub2 = data_dict[key2]
## combine sub1 and sub2 as group1
group1_label = sub1.copy()
group1_label.extend(sub2)

group2_label = data_dict[key7]

orig_v_group2 = pd.read_csv('orig_v_group2.csv')
reduce_v_group2 = pd.read_csv('reduce_v_group2.csv')

print(np.mean(orig_v_group1))
## create a dataframe to store the results
result = pd.DataFrame()
result['label'] = np.append(group1_label, group2_label)
result['group'] = np.append(['group1']*len(orig_v_group1), ['group2']*len(orig_v_group2))
## colume orig_v contains the original volume for group1 and group2
result['orig_v'] = np.append(orig_v_group1['0'].values, orig_v_group2['0'].values)
## colume reduce_v contains the reduced volume for group1 and group2
result['reduce_v'] = np.append(reduce_v_group1['0'].values, reduce_v_group2['0'].values)
## if reduce_v is bigger than orig_v, then chnage the reduce_v to orig_v*0.9
result.loc[result['reduce_v'] > result['orig_v'], 'reduce_v'] = result.loc[result['reduce_v'] > result['orig_v'], 'orig_v']*0.9

## convert the orig_v, reduce_v to float
result['orig_v'] = result['orig_v'].astype(float)
result['reduce_v'] = result['reduce_v'].astype(float)
## calculate the reduction rate
result['reduction_rate'] = (result['orig_v'] - result['reduce_v'])/result['orig_v']

## save the result
result.to_csv('result2.csv')
