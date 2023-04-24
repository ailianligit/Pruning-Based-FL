# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False
 
name = ['0 (0.90)','1 (0.90)','2 (0.45)','3 (0.45)','4 (0.45)','5 (0.20)','6 (0.20)','7 (0.20)','8 (0.00)','9 (0.00)']

num1 = np.array([0.01984999604220948,0.026367244826096172, 0, 0, 0, 0, 0, 0, 0, 0])
num2 = np.array([0, 0, 0.08448145831335233,0.07790836600103106,0.07515437507503055, 0, 0, 0, 0, 0])
num3 = np.array([0, 0, 0, 0, 0, 0.12541226310125872,0.12571845338839088,0.1299121478660409, 0, 0])
num4 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0.16599411573282855,0.16920157965376137])

# num0 = np.array([2605,602,7,1092,37,87,26,4,0,0,0,109,114,18,204,32,10,0,0,1])
# num1 = np.array([42,2188,202,2,1,0,793,1496,0,0,0,233,7,3,1,0,8,0,23,1])
# num2 = np.array([0,2232,1263,128,8,0,141,0,1,0,1,73,207,5,1,0,363,0,1,0])
# num3 = np.array([441,4,447,0,1025,0,763,0,352,1420,0,0,1,28,0,305,1,0,0,1])
# num4 = np.array([0,0,0,1854,18,0,377,0,1,32,1556,4,912,241,1,3,0,0,1,0])
# num5 = np.array([669,122,4,0,799,684,0,201,14,5,0,91,0,648,0,207,7,5,1,0])
# num6 = np.array([531,0,0,2,6,0,6,36,2,82,365,0,1,0,2811,220,289,648,0,1])
# num7 = np.array([26,18,0,96,5,2330,1,129,1689,211,0,0,337,0,12,146,0,0,0,0])
# num8 = np.array([1,126,2365,0,44,0,601,718,88,0,1,0,5,514,3,0,500,15,0,1])
# num9 = np.array([336,1,0,813,878,20,40,11,25,286,186,1593,0,47,0,0,0,25,0,0])

plt.figure(figsize=(15, 7.5))
 
plt.bar(name, num1, ec='k', color='w', hatch='-')
plt.bar(name, num2, ec='k', color='w', hatch='\\')
plt.bar(name, num3, ec='k', color='w', hatch='x')
plt.bar(name, num4, ec='k', color='w', hatch='|')
# plt.bar(name, num3, ec='k', color='w', hatch='.')
# plt.bar(name, num4, ec='k', color='w', hatch='|')
# plt.bar(name, num5, ec='k', color='w', hatch='x')
# plt.bar(name_list, num1, label=u'标签1', bottom=num0, hatch='*')
# plt.bar(name_list, num2, label=u'标签2', bottom=num1+num0, hatch='\\')
# plt.bar(name_list, num3, label=u'标签3', bottom=num2+num1+num0, hatch='.')
# plt.bar(name_list, num4, label=u'标签4', bottom=num3+num2+num1+num0, hatch='|')
# plt.bar(name_list, num5, label=u'标签5', bottom=num4+num3+num2+num1+num0, hatch='O')
# plt.bar(name_list, num6, label=u'标签6', bottom=num5+num4+num3+num2+num1+num0, hatch='-')
# plt.bar(name_list, num7, label=u'标签7', bottom=num6+num5+num4+num3+num2+num1+num0, hatch='o')
# plt.bar(name_list, num8, label=u'标签8', bottom=num7+num6+num5+num4+num3+num2+num1+num0, hatch='+')
# plt.bar(name_list, num9, label=u'标签9', bottom=num8+num7+num6+num5+num4+num3+num2+num1+num0, hatch='x')

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.ylim((0.09,0.11))

plt.xlabel(xlabel=u'客户端'+r'$n$'+u'（噪声比例）', fontsize=20)
plt.ylabel(ylabel=u'客户端'+r'$n$'+u'的聚合权重'+r'$r_{n}$', fontsize=20)

# plt.legend(prop = {'size':20}, fontsize=20)
plt.savefig('fig_noisy_weight.pdf', bbox_inches='tight')