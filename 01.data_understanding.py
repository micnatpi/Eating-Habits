
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.ticker as ticker
from stats_utils import *
from datetime import datetime
import datetime as dt
from scipy.stats import spearmanr, pearsonr, chisquare, ttest_ind, f_oneway


database = 'dbname.db'


# # 1) Investigating the number of meals per individual and the role of grants


conn = sqlite3.connect(database)
df = pd.read_sql_query("""SELECT * FROM meal_receipt ;""", conn, parse_dates=['meal_date']) #
conn.close()


df['anno']=df.meal_date.dt.year

# MEALS PER YEAR
pd.pivot_table(df, index='anno', aggfunc=lambda x: len(x.unique()), values='id_receipt').plot(kind='bar', color='darkgreen')

plt.savefig('/home/michela/Download/mealsperyear.pdf', format='pdf')
plt.show()



# STUDENTS PER YEAR
pd.pivot_table(df, index='anno', aggfunc=lambda x: len(x.unique()), values='id_stu').plot(kind='bar', color='darkgreen')

plt.savefig('/home/michela/Download/studentsperyear.pdf', format='pdf')
plt.show()


conn = sqlite3.connect(database)
df_grants = pd.read_sql_query("""SELECT * FROM meal_receipt WHERE id_rate in (2, 3, 4)""", conn)#, parse_dates=['meal_date'])
conn.close()


# ## Some statistics of the dataset


print '\x1b[2;30;47m' + 'Studenti totali:' + '\x1b[0m'
print "\tNumero di pasti:\t%s" %df.shape[0]
print "\tNumero di studenti:\t%s" %df.id_stu.unique().shape[0]
print "\tTipologie di pasto:\t%s" %df.id_meal_type.unique().shape[0]
print "\tNumero di tariffe:\t%s" %df.id_rate.unique().shape[0]
print "\tNumero di mense:\t%s" %df.id_canteen.unique().shape[0]

first_meal_date = df.meal_date.min()
last_meal_date=df.meal_date.max()

print "\n\tPeriodo di osservazione:\t%s" %str(last_meal_date - first_meal_date).split(',')[0]
print "\tPrima data:\t\t%s" %first_meal_date
print "\tUltima data:\t\t%s" %last_meal_date

##############################################

print '\n\x1b[2;30;47m' + 'Studenti con borsa di studio:' + '\x1b[0m'
print "\tNumero di pasti\t%s" %df_grants.shape[0]
print "\tNumero di studenti:\t%s" %df_grants.id_stu.unique().shape[0]
print "\tTipologie di pasto:\t%s" %df_grants.id_meal_type.unique().shape[0]
print "\tNumero di tariffe:\t%s" %df_grants.id_rate.unique().shape[0]
print "\tNumero di mense:\t%s" %df_grants.id_canteen.unique().shape[0]

# length of the period of observation (in days)
first_meal_date = datetime.strptime(df_grants.ix[0].meal_date, "%d/%m/%Y %H:%M:%S")
last_meal_date = datetime.strptime(df_grants.ix[df_grants.shape[0] - 1].meal_date, "%d/%m/%Y %H:%M:%S")
print "\n\tPeriodo di osservazione:\t%s" %str(last_meal_date - first_meal_date).split(',')[0]
print "\tPrima data:\t\t%s" %first_meal_date
print "\tUltima data:\t\t%s" %last_meal_date


# ## Number of meals per student


stu_groups = df.groupby('id_stu')
meals_per_stu = stu_groups.count().id_receipt.values

stu_grants_groups = df_grants.groupby('id_stu')
meals_per_stu_grants = stu_grants_groups.count().id_receipt.values


x, y = zip(*lbpdf(1.2, list(meals_per_stu)))
x_gr, y_gr = zip(*lbpdf(1.2, list(meals_per_stu_grants)))


fig = plt.figure(figsize=(6, 6))
plt.plot(x, y, 'o', linewidth=0, color='darkgreen',label='all students')
plt.plot(x_gr, y_gr, 's', linewidth=0, label='grant students', color='orange')
plt.loglog()
plt.xlabel('Meals per student', fontsize=11)
plt.ylabel('Probability', fontsize=11)
plt.legend(numpoints=1, fontsize=11)
plt.savefig('/home/michela/Download/distr_stu_grant.pdf', format='pdf')
plt.show()



df['time']=df.meal_date.dt.time


df['meal']='lunch'
df['hour']=df['meal_date'].dt.hour 

df.meal[df['hour']>15]='dinner'
pd.pivot_table(df, index='meal', values='id_receipt', aggfunc='count')

 
# ## The impact of filtering on the data size

def filter_students(stu_groups, min_meals=10):
    """
    Filter the groupby dataframe by eliminating all the individuals
    with a number of meals less than "min_meals"
    
    Parameters
    ----------
    stu_groups: DataFrameGroupBy
    
    min_meals: int (default 10)
    
    Returns
    -------
    a filtered DataFrameGroupBy
    """
    filtered_stu_groups = stu_groups.filter(lambda x: len(x) > min_meals).groupby('id_stu')
    return filtered_stu_groups

stu_groups_10 = filter_students(stu_groups, min_meals=10)
stu_groups_100 = filter_students(stu_groups, min_meals=100)
stu_groups_1000 = filter_students(stu_groups, min_meals=1000)

stu_groups_10_gr = filter_students(stu_grants_groups, min_meals=10)
stu_groups_100_gr = filter_students(stu_grants_groups, min_meals=100)
stu_groups_1000_gr = filter_students(stu_grants_groups, min_meals=1000)


n_stu_groups_10 = len(stu_groups_10)
n_stu_groups_100 = len(stu_groups_100)
n_stu_groups_1000 = len(stu_groups_1000)

n_stu_groups_10_gr = len(stu_groups_10_gr)
n_stu_groups_100_gr = len(stu_groups_100_gr)
n_stu_groups_1000_gr = len(stu_groups_1000_gr)


print '\x1b[2;30;47m' + 'All individuals:' + '\x1b[0m'
print "Students with at least 10 meals:\t%s" %n_stu_groups_10
print "Students with at least 100 meals:\t%s" %n_stu_groups_100
print "Students with at least 1000 meals:\t%s" %n_stu_groups_1000

print '\n\x1b[2;30;47m' + 'Individuals with grant:' + '\x1b[0m'
print "Grants with at least 10 meals:\t\t%s" %n_stu_groups_10_gr
print "Grants with at least 100 meals:\t\t%s" %n_stu_groups_100_gr
print "Grants with at least 1000 meals:\t%s" %n_stu_groups_1000_gr


# # 2) THE DISHES

conn = sqlite3.connect(database)
df_dishes_bought = pd.merge(pd.merge(pd.read_sql_query("""SELECT * FROM dish_bought""", conn),
                            df[['id_receipt', 'meal_date']], 
                            on='id_receipt'),
                           pd.read_sql_query("""SELECT * FROM dish""", conn)[['id_dish', 'dish_description']],
                           on='id_dish')
conn.close()



dish_popularity = df_dishes_bought.groupby('id_dish').count().id_receipt.values


x, y = zip(*lbpdf(1.5, list(dish_popularity)))

fig = plt.figure(figsize=(8, 6))
plt.plot(x, y, 'o', linewidth=0, color='darkgreen')
plt.loglog()
plt.xlabel('Dish popularity', fontsize=16)
plt.ylabel('P(Dish popularity)', fontsize=16)
plt.show()



# dish per meal
dishes_per_meal = df_dishes_bought.groupby('id_receipt').count().id_dish.values
print min(dishes_per_meal), max(dishes_per_meal)

count_dish=df_dishes_bought.groupby('id_receipt').count().reset_index()
print(len(count_dish))


counting=count_dish[['id_receipt', 'id_dish']]


uno=pd.pivot_table(counting, index='id_dish', values='id_receipt', aggfunc='count').reset_index()


uno=uno[uno.id_dish<6]

fig = plt.figure(figsize=(6,5))
plt.bar(uno.id_dish, uno.id_receipt, color='darkgreen')
plt.xlabel('Number of dishes', fontsize=11)
plt.ylabel('Meals', fontsize=11)
plt.show()



# ## At what extent is dish popularity related to dish life?


last_date = df.meal_date.max()

def inter_time_length(group):
    """
    For every dish compute the difference (in days) between the first
    time that dish was eaten and the last time it has been eaten
    """
    time_diff = group.meal_date.max() - group.meal_date.min()
    return int(str(time_diff).split(' ')[0].split(' ')[0])

time_of_dish = df_dishes_bought.groupby('id_dish').apply(inter_time_length)

def dish_life_since_first(group):
    time_diff = last_date - group.meal_date.min()
    return int(str(time_diff).split(' ')[0].split(' ')[0])

time_of_dish_since = df_dishes_bought.groupby('id_dish').apply(dish_life_since_first)


fig = plt.figure(figsize=(6, 6))
plt.hist(time_of_dish, bins=100, rwidth=0.85, linewidth=0, alpha=0.75, color='darkgreen')
plt.xlabel('inter time [days]', fontsize=20)
plt.ylabel('P(inter time)', fontsize=20)
plt.show()



time=pd.DataFrame(time_of_dish_since).reset_index()

time.rename(columns={0: 'days'}, inplace=True)
zeri=time[time['days']<100]


fig = plt.figure(figsize=(8, 6))
plt.hist(time_of_dish_since, bins=40, rwidth=0.85, linewidth=0, alpha=0.75, color='darkgreen')
plt.xlabel('dish life [days]', fontsize=16)
plt.ylabel('P(dish life)', fontsize=16)
plt.xlim(0,2600)
plt.show()


# Distribution of time between the first time a dish was consumed and the last time it has been consumed. We observe that there are two peaks: 
# one peak close to zero indicates that many dishes have been introduced recently. 
# The other peak is toward the maximum, indicating that many dishes have been introduced long time ago.

# The distribution of the life of a dish, i.e., time passed since the first time it was introduced. 
# The majority of dishes have a long life, i.e., they have been introduced 6 years ago.

# ## The correlation between dish popularity and dish life


r_pearson = round(pearsonr(time_of_dish, dish_popularity)[0], 2)
r_spearman = round(spearmanr(time_of_dish, dish_popularity)[0], 2)

fig = plt.figure(figsize=(10, 10))
plt.scatter(time_of_dish, dish_popularity, linewidth=0, color='darkgreen')

plt.semilogy()
plt.xlabel('Inter time [days]', fontsize=18)
plt.ylabel('Dish popularity', fontsize=18)

plt.annotate('$r_{pearson}=%s$\n$r_{spearman}=%s$' %(r_pearson, r_spearman), 
             xy=(0.15, 0.80), xycoords='axes fraction',
            fontsize=18)

plt.show()


# We observe a strong non-linear positive correlation between dish popularity and inter time. 
# This means that the dishes introduced long time ago tend to be the most popular 
# (why? because the students eat what they see or because the dishes have been proposed due to their success).
:


r_pearson = round(pearsonr(time_of_dish_since, dish_popularity)[0], 2)
r_spearman = round(spearmanr(time_of_dish_since, dish_popularity)[0], 2)

fig = plt.figure(figsize=(6, 6))
plt.scatter(time_of_dish_since, dish_popularity, linewidth=0)

plt.semilogy()
plt.xlabel('dish life [days]', fontsize=20)
plt.ylabel('dish popularity', fontsize=20)

plt.annotate('$r_{pearson}=%s$\n$r_{spearman}=%s$' %(r_pearson, r_spearman), 
             xy=(0.15, 0.80), xycoords='axes fraction',
            fontsize=18)

plt.show()


# ## Top 100 most popular dishes

top_dishes = df_dishes_bought.groupby('id_dish').count().sort_values(by='id_receipt', 
                                                        ascending=False).head(100).reset_index()



conn = sqlite3.connect(database)
df_dishes = pd.read_sql_query("""SELECT * FROM dish""", conn)
conn.close()

top = pd.merge(top_dishes, df_dishes, on='id_dish')[['id_dish', 'dish_description_x', 'dish_description_y']]
top.columns = ['id_dish', 'dish_popularity', 'dish_description']


# # tendency of grant people to take everything


## WHAT IS the typical meal of a student, i.e., primo-secondo-contorno, or just primo-contorno?
conn = sqlite3.connect(database)
db = pd.read_sql_query("""SELECT * FROM dish_bought""", conn) 
conn.close()

gby = db.groupby(by='id_receipt').count().reset_index()


print(min(gby['id_dish']), max(gby['id_dish']))

conn = sqlite3.connect(database)
pasti = pd.read_sql_query("""SELECT id_receipt,id_rate, id_meal_type FROM meal_receipt WHERE id_canteen=1""", conn) 
conn.close()


info=pd.merge(pasti, gby, on='id_receipt', how='left')
print(len(gby),len(pasti), len(info))
info.head()


info['id_rate'] = info['id_rate'].replace([2,3,4], 'grant')
info['id_rate'] = info['id_rate'].replace([1,5,30,31,32], 'no grant')
info['id_rate'] = info['id_rate'].replace([6,7,8,9,26,28,33,34,36,38,40], 'other')


conn = sqlite3.connect(database)
mealtype = pd.read_sql_query("""SELECT * FROM meal_type""", conn) 
conn.close()
mealtype.head(6)


freq=info.pivot_table(index='id_rate', columns='id_meal_type', values='id_dish', aggfunc='count',margins=True).reset_index()
print('Frequenza di pasti per tariffa e tipologia di pasto \n' )
print(freq.to_string(na_rep=''))
freq2=freq.to_string(na_rep='')




chisquare(freq[[1,2,3,4]])


distrperc=freq.copy()
distrperc[-1]=(distrperc[-1]/distrperc['All']*100).fillna(0).round(2)
distrperc[1]=(distrperc[1]/distrperc['All']*100).fillna(0).round(2)
distrperc[2]=(distrperc[2]/distrperc['All']*100).fillna(0).round(2)
distrperc[3]=(distrperc[3]/distrperc['All']*100).fillna(0).round(2)
distrperc[4]=(distrperc[4]/distrperc['All']*100).fillna(0).round(2)

distrperc['All']=(distrperc['All']/distrperc['All']*100).fillna(0).round(1)
print('Distribuzione percentuale dei pasti consumati per tariffa e tipologia \n')
distrperc


mean=info.pivot_table(index='id_rate', columns='id_meal_type', values='id_dish', aggfunc='mean',margins=True).reset_index().round(2)
print('Numero medio di piatti consumati per tariffa e tipologia di pasto \n')
print(mean.to_string(na_rep=''))


info2=info[info['id_meal_type']!=-1]
print(len(info), len(info2))



medie=info2.pivot_table(index='id_rate', values='id_dish', aggfunc='count')
medie


len(info2[info2['id_rate']=='no grant']), len(info2[info2['id_rate']=='grant'])



info3=info2[['id_receipt','id_rate','id_dish']]
info3.head()


info3=info3.dropna() 

ttest_ind(info3[info3['id_rate'].isin(['no grant'])]['id_dish'],info3[info3['id_rate']=='grant']['id_dish'])

f_val, p_val = f_oneway(list(info3[info3['id_rate']=='no grant']['id_dish']), list(info3[info3['id_rate']=='grant']['id_dish']),list(info3[info3['id_rate']=='other']['id_dish']) )
'F value:',f_val,'p value:', p_val

len(info2[info2['id_rate'].isin([1,30,31])]['id_dish'].isnull()),len(info2[info2['id_rate'].isin([1,30,31])]['id_dish'])


# ## Chi sono gli studenti??

conn = sqlite3.connect(database)
ids = pd.read_sql_query("""SELECT id_stu, id_pers, id_course, enroll_year FROM relation2 ;""", conn) #
conn.close()

conn = sqlite3.connect(database)
person = pd.read_sql_query("""SELECT * FROM person ;""", conn) #
conn.close()

conn = sqlite3.connect(database)
geo = pd.read_sql_query("""SELECT * FROM geo ;""", conn) #
conn.close()


stu=df[['id_stu']]
print len(stu.id_stu)
stu=stu.drop_duplicates()
print len(stu.id_stu)


dfstu=pd.merge(stu, ids, on='id_stu', how='left')

pd.pivot_table(dfstu, index='enroll_year', values='id_stu', aggfunc='count')

dfpers=dfstu.drop_duplicates('id_pers', keep='last')

person['id_pers']=person['id_pers'].astype(float)

dfpers2=pd.merge(dfpers, person, on='id_pers', how='left')
print len(dfpers2), len(dfpers), len(person)

pd.pivot_table(dfpers2, index='sex', values='id_pers', aggfunc='count')

geo['id_birth_town']=geo['id_birth_town'].astype(float)


dfpers3=pd.merge(dfpers2, geo, on='id_birth_town', how='left')
print len(dfpers3), len(dfpers2), len(geo)


pd.pivot_table(dfpers3, index='IT', values='id_stu', aggfunc='count')

pd.pivot_table(dfpers3[dfpers3.IT==1], index='region', values='id_stu', aggfunc='count')


pd.pivot_table(dfpers3[dfpers3['region'].isin(['Toscana', 'Tuscany', 'Tuscany '])], index='province_code', values='id_stu', aggfunc='count')


# ## Cosa studiano

conn = sqlite3.connect(database)
course = pd.read_sql_query("""SELECT * FROM course ;""", conn) 
conn.close()

conn = sqlite3.connect(database)
types = pd.read_sql_query("""SELECT * FROM course_type ;""", conn) 
conn.close()

course['id_course']=course['id_course'].astype(float)
dfcourse=pd.merge(dfstu, course, on='id_course', how='left')

types['id_type']=types['id_type'].astype(float)
dfcourse2=pd.merge(dfcourse, types, on='id_type', how='left')

pd.pivot_table(dfcourse2, index='type_description', values='id_stu', aggfunc='count').reset_index()
