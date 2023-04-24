# -*- coding: utf-8 -*-
import pandas as pd
df = pd.read_csv("diri05.csv")

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False

fig = plt.figure(figsize=(18,6))

x=np.arange(200).tolist()

plt.subplot(1,3,1)

plt.plot(x,df['FedAvg'].tolist()[0:200],label='FedAvg',linewidth=2, marker='D', markevery=20)
plt.plot(x,df['FedProx'].tolist()[0:200],label='FedProx',linewidth=2, marker='o', markevery=20)
plt.plot(x,df['PruneFL'].tolist()[0:200],label='PruneFL',linewidth=2, marker='^', markevery=20)
plt.plot(x,df['FedDST'].tolist()[0:200],label='FedDST',linewidth=2, color='m', marker='s', markevery=20)
plt.plot(x,df['mine'].tolist()[0:200],label=u'本文算法',color='red',linewidth=3, linestyle='--')

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel(xlabel=u'通信轮数'+'$t$', fontsize=20)
plt.ylabel(ylabel=u'测试准确率', fontsize=20)
plt.xlim((0,200))
plt.ylim((0.45,0.65))
plt.grid(linestyle = '--')

plt.subplot(1,3,2)

plt.plot(df['flops00'].tolist()[0:200],df['FedAvg'].tolist()[0:200],linewidth=2, marker='D', markevery=5)
plt.plot(df['flops00'].tolist()[0:200],df['FedProx'].tolist()[0:200],linewidth=2, marker='o', markevery=5)
plt.plot(df['flops09'].tolist()[0:200],df['PruneFL'].tolist()[0:200],linewidth=2, marker='^', markevery=20)
plt.plot(df['flops09'].tolist()[0:200],df['FedDST'].tolist()[0:200],linewidth=2, color='m', marker='s', markevery=20)
plt.plot(df['flops09'].tolist()[0:200],df['mine'].tolist()[0:200],color='red',linewidth=3, linestyle='--')

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel(xlabel=u'平均FLOPs（'+r'$\times 10^{12}$'+u'）', fontsize=20)
plt.xlim((0,60))
plt.ylim((0.45,0.65))
plt.grid(linestyle = '--')
ax = plt.gca()
ax.axes.yaxis.set_ticklabels([])

plt.subplot(1,3,3)

plt.plot(df['upload00'].tolist()[0:200],df['FedAvg'].tolist()[0:200],linewidth=2, marker='D', markevery=5)
plt.plot(df['upload00'].tolist()[0:200],df['FedProx'].tolist()[0:200],linewidth=2, marker='o', markevery=5)
plt.plot(df['upload09'].tolist()[0:200],df['PruneFL'].tolist()[0:200],linewidth=2, marker='^', markevery=20)
plt.plot(df['upload09'].tolist()[0:200],df['FedDST'].tolist()[0:200],linewidth=2, color='m', marker='s', markevery=20)
plt.plot(df['upload09'].tolist()[0:200],df['mine'].tolist()[0:200],color='red',linewidth=3, linestyle='--')

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel(xlabel=u'平均上传开销 [MiB]', fontsize=20)
plt.xlim((0,60))
plt.ylim((0.45,0.65))
plt.grid(linestyle = '--')
ax = plt.gca()
ax.axes.yaxis.set_ticklabels([])

plt.subplots_adjust(wspace=0.1)
fig.legend(prop = {'size':20}, fontsize=20, ncol=6, frameon=False, loc='center', bbox_to_anchor=(0.5, -0.05))

plt.savefig('fig_final_1.pdf', bbox_inches='tight')