# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False

x=[0.9, 0.7, 0.5, 0.3, 0.1, 0.0]
y1=[0.592653979,0.600723398,0.60610301,0.607896215,0.598930194,0.60162]
y2=[0.570270338,0.584776215,0.589562769,0.594029154,0.587349323,0.58563]
y3=[0.56029, 0.57665, 0.58267, 0.59004, 0.58345, 0.553965]
y4=[0.529961411,0.54933043,0.56401494,0.560267332,0.54933043,0.52434]

plt.figure(figsize=(15, 7.5))
plt.plot(x,y1,'D-',label=r'$\alpha=0.80$')
plt.plot(x,y2,'o-',label=r'$\alpha=0.85$')
plt.plot(x,y3,'^-',label=r'$\alpha=0.90$')
plt.plot(x,y4,'s-',label=r'$\alpha=0.95$')


# plt.axvline(0.5, color='red', linestyle='-' ,linewidth=2)
# plt.axvline(0.8, color='red', linestyle='-' ,linewidth=2)

plt.xticks(x, ('0.9', '0.7', '0.5', '0.3', '0.1', '0.0'), fontsize=20)
plt.yticks(fontsize=20)
plt.ylim((0.5,0.62))

plt.xlabel(xlabel=u'初始调整系数'+r'$\lambda^0$', fontsize=20)
plt.ylabel(ylabel=u'测试准确率', fontsize=20)

plt.legend(prop = {'size':20}, fontsize=20)
plt.grid(linestyle = '--')
plt.savefig('fig_dynamic_ratio.pdf', bbox_inches='tight')