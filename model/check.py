# -*- coding: utf-8 -*-
"""
卡方检验
"""
#卡方检验是一种用途很广的基于卡方分布的假设检验方法，
#其根本思想就是在于比较理论频数和实际频数的吻合程度或拟合优度问题,主要用于留存率，渗透率等漏斗指标
#预言抛一个色子，各面向上的概率都相同。为了验证本身理论的正确性，该科学家抛了600次硬币
from scipy import stats

obs = [112, 102, 96, 105, 95, 100] #观察值
exp = [100, 100, 100, 100, 100, 100] #期望值
print(stats.chisquare(obs, f_exp = exp))

#-------------------------------------------------------------------------------
"""方差分析"""
#方差分析的主要功能就是验证两组样本，或者两组以上的样本均值是否有显著性差异，即均值是否一样。
#这里有两个大点需要注意：①方差分析的原假设是：样本不存在显著性差异（即，均值完全相等）；②两样本数据无交互作用
#（即，样本数据独立）这一点在双因素方差分析中判断两因素是否独立时用
#做方差分析的时候数据需要满足正态分布；方差齐性等。
#正常拿到数据后需要对数据是否符合正态分布和组间方差是否一致做检验。
"""
探究施肥是否会对促进植株生成（植株生长以树高作为指标来衡量）。
试验为： - 对照组：清水 - 实验组： 某肥料四个浓度梯度，分别是A,B,C,D，
施肥一段时间之后测量树高（要控制其他变量保持一致，比如施肥之前的树高要基本保持一致，生长势基本保持一致等等）
"""
import numpy as np
df = {'ctl':list(np.random.normal(10,5,100)),
      'treat1':list(np.random.normal(15,5,100)),\
      'treat2':list(np.random.normal(20,5,100)),\
      'treat3':list(np.random.normal(30,5,100)),\
      'treat4':list(np.random.normal(31,5,100))}
#组合成数据框
import pandas as pd
df = pd.DataFrame(df)
df_melt = df.melt()
df_melt.columns = ['Treat','Value']
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
model = ols('Value~C(Treat)',data=df_melt).fit()
anova_table = anova_lm(model, typ = 2)
print(anova_table)

#多重检验:比较常用的检验方法是邓肯多重检验（Tukey HSD test）
from statsmodels.stats.multicomp import MultiComparison
mc = MultiComparison(df_melt['Value'],df_melt['Treat'])
tukey_result = mc.tukeyhsd(alpha = 0.5)
print(tukey_result)






