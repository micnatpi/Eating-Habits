
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import seaborn as sn
import datetime as dt
import warnings
warnings.filterwarnings('ignore')
import time
import scipy.stats
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs

path = "results/"

comp1=pd.read_csv(path+'pap_composition1.csv', sep=';')

tab=comp1.drop(['id_stu','month','meal','n_meals'],1)

inertia = []

kappa=[25,50,75,100,125,150]

for i in kappa:
    km = KMeans(n_clusters=i, random_state=10)
    labels = km.fit_predict(tab)
    inertia.append(km.inertia_)
    print('k:',i)

print(inertia)


pos = kappa
pos1 = list(range(0,len(kappa)))
fig=plt.figure(figsize=(10, 5))
# plt.title("Inertia",fontsize=16)
plt.plot(inertia,"-",marker='o', color='green',linewidth=3.0)
plt.xticks(pos1,pos,fontsize=10)
plt.xticks(fontsize=10)
plt.ylabel("INERTIA", fontsize=11)
plt.xlabel("K", fontsize=11)
plt.savefig(path+'inertiaComp1.pdf', format='pdf')
plt.tight_layout()
plt.show() 


range_n_clusters=[25,50,75,100,125,150]
valori=[]
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(tab) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(tab)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values =             sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()


# cambia il K
kmeans = KMeans(n_clusters=4, random_state=10).fit(tab)
labels = kmeans.labels_
centro=kmeans.cluster_centers_.astype(float)
tab['clusters'] = labels


comp2=pd.read_csv(path+'pap_composition2.csv', sep=';')

tab2=comp2.drop(['id_stu','month','meal','n_meals'],1)

inertia2 = []

kappa=[25,50,75,100,125,150]

for i in kappa:
    km = KMeans(n_clusters=i, random_state=10)
    labels = km.fit_predict(tab2)
    inertia2.append(km.inertia_)
    print('k:',i)

print(inertia2)

pos = kappa
pos1 = list(range(0,len(kappa)))
fig=plt.figure(figsize=(10, 5))
# plt.title("Inertia",fontsize=16)
plt.plot(inertia2,"-",marker='o', color='green',linewidth=3.0)
plt.xticks(pos1,pos,fontsize=10)
plt.xticks(fontsize=10)
plt.ylabel("INERTIA 2", fontsize=11)
plt.xlabel("K", fontsize=11)
plt.savefig(path+'inertiaC2.pdf', format='pdf')
# plt.yscale('log')
plt.tight_layout()
plt.show() 


comp3=pd.read_csv(path+'pap_composition3.csv', sep=';')

tab3=comp3.drop(['id_stu','year','season','meal','n_meals'],1)

plt.matshow(tab3.corr())
plt.savefig(path+'correlationC3.pdf', format='pdf')
plt.show()

inertia3 = []

kappa=[25,50,75,100,125]

for i in kappa:
    km = KMeans(n_clusters=i, random_state=10)
    labels = km.fit_predict(tab3)
    inertia3.append(km.inertia_)
    print('k:',i)

print(inertia3)

inertia3b = []

kappa2=[150,175]

for i in kappa2:
    km = KMeans(n_clusters=i, random_state=10)
    labels = km.fit_predict(tab3)
    inertia3.append(km.inertia_)
    print('k:',i)

print(inertia3)

pos = kappa+kappa2
pos1 = list(range(0,len(kappa+kappa2)))
fig=plt.figure(figsize=(10, 5))
# plt.title("Inertia",fontsize=16)
plt.plot(inertia3,"-",marker='o', color='green',linewidth=3.0)
plt.xticks(pos1,pos,fontsize=10)
plt.xticks(fontsize=10)
plt.ylabel("INERTIA 3", fontsize=11)
plt.xlabel("K", fontsize=11)
plt.savefig(path+'inertiaC3.pdf', format='pdf')
# plt.yscale('log')
plt.tight_layout()
plt.show() 
