
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
%matplotlib inline
import datetime as dt
import warnings
warnings.filterwarnings('ignore')
import time



database = 'dbname.db' # the relative path to your database file
path = "results/"

conn = sqlite3.connect(database)
df_meals = pd.read_sql_query("""SELECT * FROM meal_receipt WHERE id_canteen=1;""", conn, parse_dates=['meal_date'])
conn.close()

df_meals['year']=df_meals.meal_date.dt.year

conn = sqlite3.connect(database)
df_dish = pd.read_sql_query("""SELECT * FROM dish_bought ;""", conn)
conn.close()

descr=pd.read_csv('piatto.csv', sep=';')

df_dishes_bought=pd.merge(df_dish, descr, on='id_dish', how='left')

df=pd.merge(df_meals, df_dishes_bought, on=['id_receipt'], how='left')

df['hour']=df.meal_date.dt.hour
df['month']=df.meal_date.dt.month

df['season']=np.array(['']*df.shape[0])
conditions = [
    (df.month.isin([4,5,6])),
    (df.month.isin([7,8])),
    (df.month.isin([9,10,11]))
]
choices = ['spring', 'summer', 'autumn']

df['season'] = np.select(conditions, choices, default='winter')
choices.append('winter')

df['meal'] = np.where(df['hour']>=16, 'dinner', 'lunch')

df.meal.value_counts()


# ##### controlla se per tutte le id_receipt c'è lo stesso numero di composition e dish_description

prova=pd.pivot_table(df,index='id_receipt', values=['dish_composition','dish_description'], aggfunc='count').reset_index()

prova=prova[prova['dish_composition']==prova['dish_description']]
prova=prova[['id_receipt']]

pasti=pd.merge(prova,df, on='id_receipt', how='left')

# elimino gli studenti con meno di 10 pasti
totali=pd.pivot_table(pasti, index='id_stu', values='id_receipt', aggfunc='count').fillna(0).reset_index()
totali=totali[totali.id_receipt>=10]
totali=totali[['id_stu']]

pasti=pd.merge(totali,pasti, on='id_stu', how='left')

composition1=pd.pivot_table(pasti, index=['id_stu','month','meal'], columns='dish_composition', values='id_receipt', aggfunc='count').fillna(0).reset_index()
tot1=pd.pivot_table(pasti, index=['id_stu','month','meal'], values='id_receipt', aggfunc=lambda x: len(x.unique())).fillna(0).reset_index()
composition1=pd.merge(composition1, tot1, on=['id_stu','month','meal'], how='left')
composition1.rename(columns={'id_receipt': 'n_meals'}, inplace=True)

composition2=pd.pivot_table(pasti, index=['id_stu','year','month','meal'], columns='dish_composition', values='id_receipt', aggfunc='count').fillna(0).reset_index()
tot2=pd.pivot_table(pasti, index=['id_stu','year','month','meal'], values='id_receipt', aggfunc=lambda x: len(x.unique())).fillna(0).reset_index()
composition2=pd.merge(composition2, tot2, on=['id_stu','year','month','meal'], how='left')
composition2.rename(columns={'id_receipt': 'n_meals'}, inplace=True)

composition3=pd.pivot_table(pasti, index=['id_stu','year','season','meal'], columns='dish_composition', values='id_receipt', aggfunc='count').fillna(0).reset_index()
tot3=pd.pivot_table(pasti, index=['id_stu','year','season','meal'], values='id_receipt', aggfunc=lambda x: len(x.unique())).fillna(0).reset_index()
composition3=pd.merge(composition3, tot3, on=['id_stu','year','season','meal'], how='left')
composition3.rename(columns={'id_receipt': 'n_meals'}, inplace=True)

composition1['c8']=((composition1[8]/composition1['n_meals']).round(6))*100
composition1['c10']=((composition1[10]/composition1['n_meals']).round(6))*100
composition1['c11']=((composition1[11]/composition1['n_meals']).round(6))*100
composition1['c12']=((composition1[12]/composition1['n_meals']).round(6))*100
composition1['c13']=((composition1[13]/composition1['n_meals']).round(6))*100
composition1['c20']=((composition1[20]/composition1['n_meals']).round(6))*100
composition1['c21']=((composition1[21]/composition1['n_meals']).round(6))*100
composition1['c22']=((composition1[22]/composition1['n_meals']).round(6))*100
composition1['c23']=((composition1[23]/composition1['n_meals']).round(6))*100
composition1['c31']=((composition1[31]/composition1['n_meals']).round(6))*100
composition1['c32']=((composition1[32]/composition1['n_meals']).round(6))*100
composition1['c33']=((composition1[33]/composition1['n_meals']).round(6))*100
composition1['c34']=((composition1[34]/composition1['n_meals']).round(6))*100
composition1['c51']=((composition1[51]/composition1['n_meals']).round(6))*100
composition1['c52']=((composition1[52]/composition1['n_meals']).round(6))*100
composition1['c53']=((composition1[53]/composition1['n_meals']).round(6))*100
composition1['c60']=((composition1[60]/composition1['n_meals']).round(6))*100
composition1['c62']=((composition1[62]/composition1['n_meals']).round(6))*100
composition1['c71']=((composition1[71]/composition1['n_meals']).round(6))*100
composition1['c81']=((composition1[81]/composition1['n_meals']).round(6))*100
composition1['c82']=((composition1[82]/composition1['n_meals']).round(6))*100
composition1['c83']=((composition1[83]/composition1['n_meals']).round(6))*100
composition1['c91']=((composition1[91]/composition1['n_meals']).round(6))*100
composition1['c92']=((composition1[92]/composition1['n_meals']).round(6))*100
composition1['c93']=((composition1[93]/composition1['n_meals']).round(6))*100
composition1['c101']=((composition1[101]/composition1['n_meals']).round(6))*100
composition1['c102']=((composition1[102]/composition1['n_meals']).round(6))*100
composition1['c212']=((composition1[212]/composition1['n_meals']).round(6))*100
composition1['c213']=((composition1[213]/composition1['n_meals']).round(6))*100
composition1['c415']=((composition1[415]/composition1['n_meals']).round(6))*100
composition1['c416']=((composition1[416]/composition1['n_meals']).round(6))*100

composition1=composition1[[u'id_stu',   u'month',    u'meal',u'n_meals',
            u'c8',     u'c10',     u'c11',     u'c12',     u'c13',     u'c20',
           u'c21',     u'c22',     u'c23',     u'c31',     u'c32',     u'c33',
           u'c34',     u'c51',     u'c52',     u'c53',     u'c60',     u'c62',
           u'c71',     u'c81',     u'c82',     u'c83',     u'c91',     u'c92',
           u'c93',    u'c101',    u'c102',    u'c212',    u'c213',    u'c415',
          u'c416']]

composition2['c8']=((composition2[8]/composition2['n_meals']).round(6))*100
composition2['c10']=((composition2[10]/composition2['n_meals']).round(6))*100
composition2['c11']=((composition2[11]/composition2['n_meals']).round(6))*100
composition2['c12']=((composition2[12]/composition2['n_meals']).round(6))*100
composition2['c13']=((composition2[13]/composition2['n_meals']).round(6))*100
composition2['c20']=((composition2[20]/composition2['n_meals']).round(6))*100
composition2['c21']=((composition2[21]/composition2['n_meals']).round(6))*100
composition2['c22']=((composition2[22]/composition2['n_meals']).round(6))*100
composition2['c23']=((composition2[23]/composition2['n_meals']).round(6))*100
composition2['c31']=((composition2[31]/composition2['n_meals']).round(6))*100
composition2['c32']=((composition2[32]/composition2['n_meals']).round(6))*100
composition2['c33']=((composition2[33]/composition2['n_meals']).round(6))*100
composition2['c34']=((composition2[34]/composition2['n_meals']).round(6))*100
composition2['c51']=((composition2[51]/composition2['n_meals']).round(6))*100
composition2['c52']=((composition2[52]/composition2['n_meals']).round(6))*100
composition2['c53']=((composition2[53]/composition2['n_meals']).round(6))*100
composition2['c60']=((composition2[60]/composition2['n_meals']).round(6))*100
composition2['c62']=((composition2[62]/composition2['n_meals']).round(6))*100
composition2['c71']=((composition2[71]/composition2['n_meals']).round(6))*100
composition2['c81']=((composition2[81]/composition2['n_meals']).round(6))*100
composition2['c82']=((composition2[82]/composition2['n_meals']).round(6))*100
composition2['c83']=((composition2[83]/composition2['n_meals']).round(6))*100
composition2['c91']=((composition2[91]/composition2['n_meals']).round(6))*100
composition2['c92']=((composition2[92]/composition2['n_meals']).round(6))*100
composition2['c93']=((composition2[93]/composition2['n_meals']).round(6))*100
composition2['c101']=((composition2[101]/composition2['n_meals']).round(6))*100
composition2['c102']=((composition2[102]/composition2['n_meals']).round(6))*100
composition2['c212']=((composition2[212]/composition2['n_meals']).round(6))*100
composition2['c213']=((composition2[213]/composition2['n_meals']).round(6))*100
composition2['c415']=((composition2[415]/composition2['n_meals']).round(6))*100
composition2['c416']=((composition2[416]/composition2['n_meals']).round(6))*100

composition2=composition2[['id_stu','year','month','meal', u'n_meals',
            u'c8',     u'c10',     u'c11',     u'c12',     u'c13',     u'c20',
           u'c21',     u'c22',     u'c23',     u'c31',     u'c32',     u'c33',
           u'c34',     u'c51',     u'c52',     u'c53',     u'c60',     u'c62',
           u'c71',     u'c81',     u'c82',     u'c83',     u'c91',     u'c92',
           u'c93',    u'c101',    u'c102',    u'c212',    u'c213',    u'c415',
          u'c416']]

composition3['c8']=((composition3[8]/composition3['n_meals']).round(6))*100
composition3['c10']=((composition3[10]/composition3['n_meals']).round(6))*100
composition3['c11']=((composition3[11]/composition3['n_meals']).round(6))*100
composition3['c12']=((composition3[12]/composition3['n_meals']).round(6))*100
composition3['c13']=((composition3[13]/composition3['n_meals']).round(6))*100
composition3['c20']=((composition3[20]/composition3['n_meals']).round(6))*100
composition3['c21']=((composition3[21]/composition3['n_meals']).round(6))*100
composition3['c22']=((composition3[22]/composition3['n_meals']).round(6))*100
composition3['c23']=((composition3[23]/composition3['n_meals']).round(6))*100
composition3['c31']=((composition3[31]/composition3['n_meals']).round(6))*100
composition3['c32']=((composition3[32]/composition3['n_meals']).round(6))*100
composition3['c33']=((composition3[33]/composition3['n_meals']).round(6))*100
composition3['c34']=((composition3[34]/composition3['n_meals']).round(6))*100
composition3['c51']=((composition3[51]/composition3['n_meals']).round(6))*100
composition3['c52']=((composition3[52]/composition3['n_meals']).round(6))*100
composition3['c53']=((composition3[53]/composition3['n_meals']).round(6))*100
composition3['c60']=((composition3[60]/composition3['n_meals']).round(6))*100
composition3['c62']=((composition3[62]/composition3['n_meals']).round(6))*100
composition3['c71']=((composition3[71]/composition3['n_meals']).round(6))*100
composition3['c81']=((composition3[81]/composition3['n_meals']).round(6))*100
composition3['c82']=((composition3[82]/composition3['n_meals']).round(6))*100
composition3['c83']=((composition3[83]/composition3['n_meals']).round(6))*100
composition3['c91']=((composition3[91]/composition3['n_meals']).round(6))*100
composition3['c92']=((composition3[92]/composition3['n_meals']).round(6))*100
composition3['c93']=((composition3[93]/composition3['n_meals']).round(6))*100
composition3['c101']=((composition3[101]/composition3['n_meals']).round(6))*100
composition3['c102']=((composition3[102]/composition3['n_meals']).round(6))*100
composition3['c212']=((composition3[212]/composition3['n_meals']).round(6))*100
composition3['c213']=((composition3[213]/composition3['n_meals']).round(6))*100
composition3['c415']=((composition3[415]/composition3['n_meals']).round(6))*100
composition3['c416']=((composition3[416]/composition3['n_meals']).round(6))*100

composition3=composition3[['id_stu','year','season','meal', u'n_meals',
            u'c8',     u'c10',     u'c11',     u'c12',     u'c13',     u'c20',
           u'c21',     u'c22',     u'c23',     u'c31',     u'c32',     u'c33',
           u'c34',     u'c51',     u'c52',     u'c53',     u'c60',     u'c62',
           u'c71',     u'c81',     u'c82',     u'c83',     u'c91',     u'c92',
           u'c93',    u'c101',    u'c102',    u'c212',    u'c213',    u'c415',
          u'c416']]

composition1.to_csv(path+'pap_composition1.csv', sep=';', index=False)
composition2.to_csv(path+'pap_composition2.csv', sep=';', index=False)
composition3.to_csv(path+'pap_composition3.csv', sep=';', index=False)

