
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt

import matplotlib.cm as cm

import seaborn as sn

import datetime as dt
import warnings

import time
import scipy.stats
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs

from __future__ import print_function

path = "results/"


comp3=pd.read_csv(path+'pap_composition3.csv', sep=';')

comp3b=comp3.copy()
comp3b=comp3b.set_index('id_stu')
tab=comp3b.drop(['year','season','meal','n_meals'],1)


kmeans = KMeans(n_clusters=50, random_state=10).fit(tab)
labels = kmeans.labels_
centroid=kmeans.cluster_centers_
tab['clusters'] = labels
tab.head()


Ztab= pd.DataFrame(centroid, columns=['c8', 'c10', 'c11', 'c12', 'c13', 'c20', 'c21', 'c22', 'c23', 'c31',
       'c32', 'c33', 'c34', 'c51', 'c52', 'c53', 'c60', 'c62', 'c71', 'c81',
       'c82', 'c83', 'c91', 'c92', 'c93', 'c101', 'c102', 'c212', 'c213',
       'c415', 'c416'])
               

lista=[7,13,30, 33, 26, 41, 6, 47, 19]
selection_clusters=Ztab.loc[Ztab.index.isin(lista)]


selection_clusters_t=selection_clusters.transpose()


selection_clusters_t=selection_clusters_t.reset_index()
selection_clusters_t.rename(columns={'index': 'cat'}, inplace=True)

category=pd.read_csv('food_categories.csv', sep=';')

selection_clusters_t=pd.merge(selection_clusters_t, category, on='cat', how='left')
selection_clusters_t.set_index('cat', inplace=True)

selection_clusters_t=selection_clusters_t.sort_values(by=6, ascending=False)
sel=selection_clusters_t.head(10)

fig, ax = plt.subplots()
fig.set_size_inches(12, 10)
ax = sns.barplot(x=6, y='foodcat', data=sel, palette="Wistia_r")
ax.set_xlabel('Proportion of food categories for cluster 4', size=26)
ax.set_ylabel('', size=22)
# plt.ylabel('')
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
# props={'facecolor':'white', 'alpha':0.4, 'pad':10}
plt.text(75, 2,"N: 4864 \n%: 10.6", size=24, bbox=props)
plt.tick_params(axis='both', which='major', labelsize=24)
fig.savefig(path+'foodcat_cluster_4.png', bbox_inches='tight')
plt.show()


selection_clusters_t=selection_clusters_t.sort_values(by=7, ascending=False)
sel=selection_clusters_t.head(10)

fig, ax = plt.subplots()
fig.set_size_inches(12, 10)
ax = sns.barplot(x=7, y='foodcat', data=sel, palette="Reds_r")
ax.set_xlabel('Proportion of food categories for cluster 0, attribute 7', size=26)
ax.set_ylabel('', size=18)
# plt.ylabel('')
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
# props={'facecolor':'white', 'alpha':0.4, 'pad':10}
plt.text(75, 2,"N: 22438 \n%: 48.8", size=24, bbox=props)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.xlim(0,100)
fig.savefig(path+'foodcat_cluster_0_a7.png', bbox_inches='tight')
plt.show()


selection_clusters_t=selection_clusters_t.sort_values(by=13, ascending=False)
sel=selection_clusters_t.head(10)

fig, ax = plt.subplots()
fig.set_size_inches(12, 10)
ax = sns.barplot(x=13, y='foodcat', data=sel, palette="Reds_r")
ax.set_xlabel('Proportion of food categories for cluster 0, attribute 13', size=26)
ax.set_ylabel('', size=18)
# plt.ylabel('')
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
# props={'facecolor':'white', 'alpha':0.4, 'pad':10}
plt.text(75, 2,"N: 22438 \n%: 48.8", size=24, bbox=props)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.xlim(0,100)
fig.savefig(path+'foodcat_cluster_0_a13.png', bbox_inches='tight')
plt.show()


selection_clusters_t=selection_clusters_t.sort_values(by=30, ascending=False)
sel=selection_clusters_t.head(10)

fig, ax = plt.subplots()
fig.set_size_inches(12, 10)
ax = sns.barplot(x=30, y='foodcat', data=sel, palette="Reds_r")
ax.set_xlabel('Proportion of food categories for cluster 0, attribute 30', size=26)
ax.set_ylabel('', size=18)
# plt.ylabel('')
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
# props={'facecolor':'white', 'alpha':0.4, 'pad':10}
plt.text(75, 2,"N: 22438 \n%: 48.8", size=24, bbox=props)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.xlim(0,100)
fig.savefig(path+'foodcat_cluster_0_a30.png', bbox_inches='tight')
plt.show()


selection_clusters_t=selection_clusters_t.sort_values(by=19, ascending=False)
sel=selection_clusters_t.head(10)
fig, ax = plt.subplots()
fig.set_size_inches(12, 10)
ax = sns.barplot(x=19, y='foodcat', data=sel, palette="PuRd_r")
ax.set_xlabel('Proportion of food categories for cluster 6', size=26)
ax.set_ylabel('', size=18)
# plt.ylabel('')
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
# props={'facecolor':'white', 'alpha':0.4, 'pad':10}
plt.text(75, 2.25,"N: 3829 \n%: 8.3", size=24, bbox=props)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.xlim(0,100)
fig.savefig(path+'foodcat_cluster_6.png', bbox_inches='tight')
plt.show()


selection_clusters_t=selection_clusters_t.sort_values(by=26, ascending=False)
sel=selection_clusters_t.head(10)
fig, ax = plt.subplots()
fig.set_size_inches(12, 10)
ax = sns.barplot(x=26, y='foodcat', data=sel, palette="Greens_d")
ax.set_xlabel('Proportion of food categories for cluster 2', size=26)
ax.set_ylabel('', size=18)
# plt.ylabel('')
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
# props={'facecolor':'white', 'alpha':0.4, 'pad':10}
plt.text(75, 2,"N: 4688 \n%: 10.2", size=24, bbox=props)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.xlim(0,100)
fig.savefig(path+'foodcat_cluster_2.png', bbox_inches='tight')
plt.show()


selection_clusters_t=selection_clusters_t.sort_values(by=33, ascending=False)
sel=selection_clusters_t.head(10)
fig, ax = plt.subplots()
fig.set_size_inches(12, 10)
ax = sns.barplot(x=33, y='foodcat', data=sel, palette="Blues_r")
ax.set_xlabel('Proportion of food categories for cluster 1', size=26)
ax.set_ylabel('', size=18)
# plt.ylabel('')
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
# props={'facecolor':'white', 'alpha':0.4, 'pad':10}
plt.text(75, 2, "N: 5493 \n%: 12.0", size=24, bbox=props)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.xlim(0,100)

fig.savefig(path+'foodcat_cluster_1.png', bbox_inches='tight')
plt.show()



selection_clusters_t=selection_clusters_t.sort_values(by=41, ascending=False)
sel=selection_clusters_t.head(10)
fig, ax = plt.subplots()
fig.set_size_inches(12, 10)
ax = sns.barplot(x=41, y='foodcat', data=sel, palette="Purples_r")
ax.set_xlabel('Proportion of food categories for cluster 3', size=26)
ax.set_ylabel('', size=18)
# plt.ylabel('')
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
# props={'facecolor':'white', 'alpha':0.4, 'pad':10}
plt.text(75, 2,"N: 2979 \n%: 6.5", size=24, bbox=props)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.xlim(0,100)

fig.savefig(path+'foodcat_cluster_3.png', bbox_inches='tight')
plt.show()


selection_clusters_t=selection_clusters_t.sort_values(by=47, ascending=False)
sel=selection_clusters_t.head(10)
fig, ax = plt.subplots()
fig.set_size_inches(12, 10)
ax = sns.barplot(x=47, y='foodcat', data=sel, palette="copper")
ax.set_xlabel('Proportion of food categories for cluster 5', size=26)
ax.set_ylabel('', size=18)
# plt.ylabel('')
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
# props={'facecolor':'white', 'alpha':0.4, 'pad':10}
plt.text(75, 2.25,"N: 1661 \n%: 3.6", size=24, bbox=props)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.xlim(0,100)
fig.savefig(path+'foodcat_cluster_5.png', bbox_inches='tight')
plt.show()



result=pd.merge(comp3b[['year','season','meal','n_meals']], tab,left_index=True, right_index=True)

result_sel=result[result.clusters.isin(lista)]

stagioni=pd.pivot_table(result_sel, index='clusters', values='year', columns='season', aggfunc='count').reset_index()

anni=pd.pivot_table(result_sel, index='clusters', values='n_meals', columns='year', aggfunc='count').reset_index()


annistagioni=pd.pivot_table(result_sel, index=['year', 'season'], columns='clusters', values='n_meals', aggfunc='count').reset_index()

annipasti=pd.pivot_table(result_sel, index=['year', 'meal'], columns='clusters', values='n_meals', aggfunc='count').reset_index()


pasti=pd.pivot_table(result_sel, index='clusters', values='year', columns='meal', aggfunc='count').reset_index()
pasti.to_csv(path+'distr_pasti.csv', sep=';')


numpasti=pd.pivot_table(result_sel, index='clusters', values='n_meals', aggfunc='mean').reset_index()
numpasti.to_csv(path+'distr_numero_pasti.csv', sep=';')

passo2=pd.pivot_table(result, index='id_stu', columns='clusters', values='n_meals', aggfunc='count').fillna(0).reset_index()
tot=pd.pivot_table(result, index='id_stu',values='n_meals', aggfunc='count').fillna(0).reset_index().rename(columns={'n_meals': 'n'})
passo2=pd.merge(passo2, tot, on='id_stu', how='left')

passo3=passo2.copy()
passo3['cl0']=(passo2[0]/passo2['n'].round(4))*100
passo3['cl1']=(passo2[1]/passo2['n'].round(4))*100
passo3['cl2']=(passo2[2]/passo2['n'].round(4))*100
passo3['cl3']=(passo2[3]/passo2['n'].round(4))*100
passo3['cl4']=(passo2[4]/passo2['n'].round(4))*100
passo3['cl5']=(passo2[5]/passo2['n'].round(4))*100
passo3['cl6']=(passo2[6]/passo2['n'].round(4))*100
passo3['cl7']=(passo2[7]/passo2['n'].round(4))*100
passo3['cl8']=(passo2[8]/passo2['n'].round(4))*100
passo3['cl9']=(passo2[9]/passo2['n'].round(4))*100
passo3['cl10']=(passo2[10]/passo2['n'].round(4))*100
passo3['cl11']=(passo2[11]/passo2['n'].round(4))*100
passo3['cl12']=(passo2[12]/passo2['n'].round(4))*100
passo3['cl13']=(passo2[13]/passo2['n'].round(4))*100
passo3['cl14']=(passo2[14]/passo2['n'].round(4))*100
passo3['cl15']=(passo2[15]/passo2['n'].round(4))*100
passo3['cl16']=(passo2[16]/passo2['n'].round(4))*100
passo3['cl17']=(passo2[17]/passo2['n'].round(4))*100
passo3['cl18']=(passo2[18]/passo2['n'].round(4))*100
passo3['cl19']=(passo2[19]/passo2['n'].round(4))*100
passo3['cl20']=(passo2[20]/passo2['n'].round(4))*100
passo3['cl21']=(passo2[21]/passo2['n'].round(4))*100
passo3['cl22']=(passo2[22]/passo2['n'].round(4))*100
passo3['cl23']=(passo2[23]/passo2['n'].round(4))*100
passo3['cl24']=(passo2[24]/passo2['n'].round(4))*100
passo3['cl25']=(passo2[25]/passo2['n'].round(4))*100
passo3['cl26']=(passo2[26]/passo2['n'].round(4))*100
passo3['cl27']=(passo2[27]/passo2['n'].round(4))*100
passo3['cl28']=(passo2[28]/passo2['n'].round(4))*100
passo3['cl29']=(passo2[29]/passo2['n'].round(4))*100
passo3['cl30']=(passo2[30]/passo2['n'].round(4))*100
passo3['cl31']=(passo2[31]/passo2['n'].round(4))*100
passo3['cl32']=(passo2[32]/passo2['n'].round(4))*100
passo3['cl33']=(passo2[33]/passo2['n'].round(4))*100
passo3['cl34']=(passo2[34]/passo2['n'].round(4))*100
passo3['cl35']=(passo2[35]/passo2['n'].round(4))*100
passo3['cl36']=(passo2[36]/passo2['n'].round(4))*100
passo3['cl37']=(passo2[37]/passo2['n'].round(4))*100
passo3['cl38']=(passo2[38]/passo2['n'].round(4))*100
passo3['cl39']=(passo2[39]/passo2['n'].round(4))*100
passo3['cl40']=(passo2[40]/passo2['n'].round(4))*100
passo3['cl41']=(passo2[41]/passo2['n'].round(4))*100
passo3['cl42']=(passo2[42]/passo2['n'].round(4))*100
passo3['cl43']=(passo2[43]/passo2['n'].round(4))*100
passo3['cl44']=(passo2[44]/passo2['n'].round(4))*100
passo3['cl45']=(passo2[45]/passo2['n'].round(4))*100
passo3['cl46']=(passo2[46]/passo2['n'].round(4))*100
passo3['cl47']=(passo2[47]/passo2['n'].round(4))*100
passo3['cl48']=(passo2[48]/passo2['n'].round(4))*100
passo3['cl49']=(passo2[49]/passo2['n'].round(4))*100

passo4=passo3.drop(passo3.columns[1:52],1)

passo4.set_index('id_stu', inplace=True)


inertia = []
# i=3

for i in range(2,10):
    km = KMeans(n_clusters=i, random_state=10)
    labels = km.fit_predict(passo4)
    inertia.append(km.inertia_)
    print('k:',i)

print(inertia)

pos = list(range(2,10))
pos1 = list(range(0,8))
fig=plt.figure(figsize=(8, 6))
# plt.title("Inertia",fontsize=16)
plt.plot(inertia,"-",marker='o', color='green',linewidth=3.0)
plt.xticks(pos1,pos,fontsize=10)
# plt.xticks(fontsize=10)
plt.ylabel("INERTIA", fontsize=11)
plt.xlabel("K", fontsize=11)
plt.savefig(path+'inertia_fase2_comp2.pdf', format='pdf')
# plt.yscale('log')
plt.tight_layout()
plt.show() 



kmeans = KMeans(n_clusters=7, random_state=10).fit(passo4)
labels = kmeans.labels_
centro=kmeans.cluster_centers_.astype(float)
passo4['clusters'] = labels

passo4.clusters.value_counts().sort_index()


Z = pd.DataFrame(centro, 
                 columns=['c0','c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11',
       'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'c19', 'c20', 'c21',
       'c22', 'c23', 'c24', 'c25', 'c26', 'c27', 'c28', 'c29', 'c30', 'c31',
       'c32', 'c33', 'c34', 'c35', 'c36', 'c37', 'c38', 'c39', 'c40', 'c41',
       'c42', 'c43', 'c44', 'c45', 'c46', 'c47', 'c48', 'c49'])


Z.to_csv('centroidi_comp3_k7.csv', sep=';')

