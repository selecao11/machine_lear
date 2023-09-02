import pandas as pd
import matplotlib.pyplot as plt
# データの読み込み
df = pd.read_csv("/home/user/anaconda3/envs/machine_lear/machine_lear/AGC_累積.csv", encoding="UTF-8", index_col=0)
# グラフ作成
plt.rcParams['font.family'] = "IPAexGothic" # 日本語化の設定

ax = df.plot(x='date',y = 'accumulated_opening_price',kind = 'line', label='accumulated_opening_price')
df.plot(x='date',y = 'Cumulative_High_Value',kind = 'line', label='Cumulative_High_Value',ax=ax)
df.plot(x='date',y = 'cumulative_low',kind = 'line', label='cumulative_low',ax=ax)
df.plot(x='date',y = 'Cumulative_Terminal_Value',kind = 'line', label='Cumulative_Terminal_Value',ax=ax)
#df.plot(x='date',y = 'cumulative_trading_volume',kind = 'line', label='cumulative_trading_volume',ax=ax)
df.plot(x='date',y = 'Cumulative_corrected_final_value',kind = 'line', label='Cumulative_corrected_final_value',ax=ax)
#df.plot(x='date',y = 'Accumulation_of_credit',kind = 'line', label='Accumulation_of_credit',ax=ax)
#df.plot(x='date',y = 'Accumulated_credit_for_disability',kind = 'line', label='Accumulated_credit_for_disability',ax=ax)
#df.plot(x='date',y = 'Accumulated_Credit_Multiplier',kind = 'line', label='Accumulated_Credit_Multiplier',ax=ax)
#df.plot(x='date',y = 'Accumulation_of_reverse sun-rays',kind = 'line', label='Accumulation_of_reverse sun-rays',ax=ax)
#df.plot(x='date',y = 'Accumulated_days',kind = 'line', label='Accumulated_days',ax=ax)
df.plot(x='date',y = 'Cumulative_loan_residue',kind = 'line', label='Cumulative_loan_residue',ax=ax)
df.plot(x='date',y = 'Accumulated_Financing_Disability',kind = 'line', label='Accumulated_Financing_Disability',ax=ax)
plt.legend()
plt.show()

