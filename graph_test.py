import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# -*- coding: utf-8 -*-
# データの読み込み
df = pd.read_csv("/home/user/anaconda3/envs/machine_lear/machine_lear/AGC_累積.csv", encoding="UTF-8", index_col=0)
# グラフ作成
plt.rcParams['font.family'] = "IPAexGothic" # 日本語化の設定
#df.plot(figsize=(10, 6), xlim=[2000, 2014], ylim=[0, 100])

ax = df.plot(x='date',y = 'accumulated_opening_price',kind = 'line', label='accumulated_opening_price')
df.plot(x='date',y = 'Cumulative_High_Value',kind = 'line', label='Cumulative_High_Value',ax=ax)
plt.legend()
'''
,''cumulative_low'',\
            'Cumulative_Terminal_Value','cumulative_trading_volume','Cumulative_corrected_final_value',\
            'Accumulation_of_credit','Accumulated_credit_for_disability','Accumulated_Credit_Multiplier',\
            'Accumulation_of_reverse sun-rays','Accumulated_days','Cumulative_loan_residue',\
            'Accumulated_Financing_Disability'
#fig = plt.figure()
'''
plt.show()

'''
#add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
ax1 = fig.add_subplot(5, 3, 1)
ax2 = fig.add_subplot(5, 3, 2)
ax3 = fig.add_subplot(5, 3, 3)
ax4 = fig.add_subplot(5, 3, 4)
ax5 = fig.add_subplot(5, 3, 5)
ax6 = fig.add_subplot(5, 3, 6)
ax7 = fig.add_subplot(5, 3, 7)
ax8 = fig.add_subplot(5, 3, 8)
ax9 = fig.add_subplot(5, 3, 9)
ax10 = fig.add_subplot(5, 3, 10)
ax11 = fig.add_subplot(5, 3, 11)
ax12 = fig.add_subplot(5, 3, 12)
ax13 = fig.add_subplot(5, 3, 13)
#13

t = np.linspace(-10, 10, 1000)
y1 = np.sin(t)
y2 = np.cos(t) 
y3 = np.abs(np.sin(t))
y4 = np.sin(t)**2

c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13 =   \
                '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22'\
                '#17becf','#1f77b4','#ff7f0e','#2ca02c'     # 各プロットの色
l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13  = 'accumulated_opening_price','Cumulative_High_Value','cumulative_low',\
            'Cumulative_Terminal_Value','cumulative_trading_volume','Cumulative_corrected_final_value',\
            'Accumulation_of_credit','Accumulated_credit_for_disability','Accumulated_Credit_Multiplier',\
            'Accumulation_of_reverse sun-rays','Accumulated_days','Cumulative_loan_residue',\
            'Accumulated_Financing_Disability'

ax1.plot(t, y1, color=c1, label=l1)
ax2.plot(t, y2, color=c2, label=l2)
ax3.plot(t, y3, color=c3, label=l3)
ax4.plot(t, y4, color=c4, label=l4)
ax5.plot(t, y5, color=c4, label=l4)
ax6.plot(t, y6, color=c4, label=l4)
ax7.plot(t, y4, color=c4, label=l4)
ax8.plot(t, y4, color=c4, label=l4)
ax9.plot(t, y4, color=c4, label=l4)
ax10.plot(t, y4, color=c4, label=l4)
ax11.plot(t, y4, color=c4, label=l4)

ax1.legend(loc = 'upper right') #凡例
ax2.legend(loc = 'upper right') #凡例
ax3.legend(loc = 'upper right') #凡例
ax4.legend(loc = 'upper right') #凡例
fig.tight_layout()              #レイアウトの設定
plt.show()


df.plot(x='date',y='accumulated_opening_price',color=c1)
df.plot(x='date',y='Cumulative_High_Value',color=c2)
df.plot(x='date',y='cumulative_low',color=c3)
df.plot(x='date',y='Cumulative_Terminal_Value',color=c4)
df.plot(x='date',y='cumulative_trading_volume',color=c5)
df.plot(x='date',y='Cumulative_corrected_final_value',color=c6)
df.plot(x='date',y='Accumulation_of_credit',color=c7)
df.plot(x='date',y='Accumulated_credit_for_disability',color=c8)
df.plot(x='date',y='Accumulated_Credit_Multiplier',color=c9)
df.plot(x='date',y='Accumulation_of_reverse sun-rays',color=c10)
df.plot(x='date',y='Accumulated_days',color=c11)
df.plot(x='date',y='Cumulative_loan_residue',color=c12)
df.plot(x='date',y='Accumulated_Financing_Disability',color=c13)


OrderedDict([('tab:blue', ),
    #=>              ('tab:orange', ),
    #=>              ('tab:green', ''),
    #=>              ('tab:red', ''),
    #=>              ('tab:purple', ''),
    #=>              ('tab:brown', ''),
    #=>              ('tab:pink', ''),
    #=>              ('tab:gray', ''),
    #=>              ('tab:olive', ''),
    #=>              ('tab:cyan', '')])
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10) # xは0～10
y = 2 * x + 25 # y = 2x + 5

# グラフ
plt.plot(x, y)

# 表示
plt.show()
'''
