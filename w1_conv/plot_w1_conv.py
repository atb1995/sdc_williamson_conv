import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

# df =pd.read_csv('../data/w1_base2.csv')

# df2=df.groupby(['k','dt']).sum()
# df2=df2.reset_index()
# df3=pd.DataFrame(df2)
# df4 = df3[' errornorm_D']/df3[' norm_D']
 
# df5 = pd.concat([df4, df3], axis=1)

# cols = ['b','g','k']


fig,ax=plt.subplots(1)

# # k8 = df5[df5['k'] == 8]
# # k6 = df5[df5['k'] == 6]
# k3 = df5[df5['k'] == 3]
# k2 = df5[df5['k'] == 2]
# k1 = df5[df5['k'] == 1]

# i = 0
# for k_dataframe in [k3, k2, k1]:
#     # Plot points
#     x = np.log(k_dataframe['dt'].values)
#     y = np.log(k_dataframe[0].values)
#     ax.scatter(x, y, label='SDC%s'%(k_dataframe['k'].values[0]), color=cols[i])
#     # Make best fit line
#     slope, intercept = np.polyfit(x, y, 1)
#     print(slope)
#     print(intercept)
#     ax.plot(x, slope*x+intercept, color = cols[i], label='%s'%(slope))
#     i = i + 1

df =pd.read_csv('w1_sdc_conv_rk.csv')

df2=df.groupby(['k','dt']).sum()
df2=df2.reset_index()
df3=pd.DataFrame(df2)
df4 = df3['errornorm']/df3['norm']
 
df5 = pd.concat([df4, df3], axis=1)
# s5 = df5[df5['k'] == 5]
#s4 = df5[df5['k'] == 4]
# s3 = df5[df5['k'] == 3]
s2 = df5[df5['k'] == 2]
s1 = df5[df5['k'] == 1]
s0 = df5[df5['k'] == 0]

# df =pd.read_csv('w1_exp_sdc_r5_d1_k3.csv')

# df2=df.groupby(['k','dt']).sum()
# df2=df2.reset_index()
# df3=pd.DataFrame(df2)
# df4 = df3['errornorm']/df3['norm']
 
# df5 = pd.concat([df4, df3], axis=1)
#s3 = df5[df5['k'] == 3]

i = 0
cols = ['r','c','m','g','b']
names = ['FE_SDC0','FE_SDC1', 'FE_SDC2','FE_SDC3','FE_SDC4','FE_SDC5']
#names = ['TrapeziumRule','FE_SDC3', 'BE_SDC3','SDC3','SDC4','SDC5']
for s_dataframe in [s0,s2]:
    # Plot points
    x = np.log(s_dataframe['dt'].values)
    y = np.log(s_dataframe[0].values)
    ax.scatter(x, y, label=names[i], color=cols[i])
    # Make best fit line
    slope, intercept = np.polyfit(x, y, 1)
    print(slope)
    print(intercept)
    ax.plot(x, slope*x+intercept, color = cols[i], label='%s'%(slope))
    i = i + 1
plt.title("Williamson 1 SDC FE Convergence")
plt.xlabel(r'$log(\Delta t$)')
plt.ylabel(r'$log(L_2(D_{sol}-D_{true})/L_2(D_{true}))$')
plt.legend()
plt.show()