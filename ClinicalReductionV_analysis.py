import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
## read the group data
group_data = pd.read_csv('result.csv')

# group_str = 'group2'
# orig_v = group_data[group_data['group']==group_str]['orig_v'].values
# reduce_v = group_data[group_data['group']==group_str]['reduce_v'].values
# ratio = group_data[group_data['group']==group_str]['reduction_rate'].values

orig_v = group_data['orig_v'].values
reduce_v = group_data['reduce_v'].values
ratio = group_data['reduction_rate'].values

print('feasible ratio:', ratio[ratio <= 0.5].shape[0])
print(len(ratio))

fig, ax = plt.subplots(figsize=(5, 6))
sns.histplot(ratio, kde=True, color='orange', ax=ax)

plt.xlabel('Ratio of the volume to be reduced to achieve <0.5 (%)')
plt.ylabel('Frequency')
plt.show()
## plot the histogram of orig_v and reduce_v
fig, ax = plt.subplots(figsize=(5, 6))
sns.histplot(orig_v, kde=True, color='blue', ax=ax)
sns.histplot(reduce_v, kde=True, color='red', ax=ax)
## add the legend
plt.legend(['Original', 'Reduced'])
plt.xlabel('V10 (cc)')
plt.ylabel('Frequency')

## remove the 0 value of orig_v
orig_v = orig_v[reduce_v != 0]
reduce_v = reduce_v[reduce_v != 0]
## add the mean value of orig_v and reduce_v
plt.axvline(x=orig_v.mean(), color='blue', linestyle='dashed', linewidth=1)
plt.axvline(x=reduce_v.mean(), color='red', linestyle='dashed', linewidth=1)
## add text for the mean value of orig_v and reduce_v, keep 2 decimal places
str_orig = "{:.2f}".format(orig_v.mean())
str_reduce = "{:.2f}".format(reduce_v.mean())

plt.text(orig_v.mean()+10, 17.5, str_orig, rotation=0, color='blue')
plt.text(reduce_v.mean()+10, 17.5, str_reduce, rotation=0, color='red')
plt.show()

## remove 1.0 value of ratio
ratio = ratio[ratio != 1.0]

print('ratio max:', ratio.max())
print('ratio min:', ratio.min())

