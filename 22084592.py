import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from scipy.stats import t
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

ds_2 = pd.read_csv("german_credit_data.csv")


numerical = ['Credit amount','Age','Duration']
categorical = ['Gender','Job','Housing','Saving accounts','Checking account','Purpose']
unused = ['Unnamed: 0']
ds_2 = ds_2.drop(columns = unused)

stats = ds_2[numerical].describe().rename_axis('Stats').reset_index()
describe_table = stats.round(2)
fig, ax = plt.subplots(figsize=(5, 3))
ax.axis('off')
table_data = [describe_table.columns] + describe_table.values.tolist()
cell_colors = [['#DFF2FF']*len(describe_table.columns)]*(len(describe_table)+1)
table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center', cellColours=cell_colors)
plt.title('Statistical Parameters Of Numerical Features')
plt.savefig('table.png',bbox_inches='tight')


def update_diag_func(data, label, color):
    for val in data.quantile([.25, .5, .75]):
        plt.axvline(data.quantile(.25), ls=':', color='red')
        plt.axvline(data.quantile(.5), ls=':', color='green')
        plt.axvline(data.quantile(.75), ls=':', color='red')

sns.set(rc={'figure.figsize':(10,30)})
sns.set_context("paper", rc={"axes.labelsize":18})
sns_pairplot =  sns.pairplot(ds_2[numerical],height=2, aspect=2,diag_kws={'color':'darkslateblue'},corner=True)
sns_pairplot.map_diag(update_diag_func)
sns_pairplot.fig.suptitle('Pairplots of Feture to show Skewness and Correlation',y=1.05,x=0.5,fontsize=20)
sns_pairplot.savefig("22084592.png", bbox_inches="tight",dpi=300)

corr = ds_2.corr(method = 'pearson')
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots(figsize = (15,12))
fig.set_size_inches(15,15)
sns.heatmap(corr, mask = mask, vmax = 0.9, square = True, annot = True)

ds_2_cluster = pd.DataFrame()
ds_2_cluster['Credit amount'] = ds_2['Credit amount']
ds_2_cluster['Age'] = ds_2['Age']
ds_2_cluster['Duration'] = ds_2['Duration']


fig = plt.figure(figsize = (15,10))
axes = 220
for num in numerical:
    axes += 1
    fig.add_subplot(axes)
    sns.boxplot(data = ds_2, x = num)

fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,8))
sns.distplot(ds_2["Age"], ax=ax1)
sns.distplot(ds_2["Credit amount"], ax=ax2)
sns.distplot(ds_2["Duration"], ax=ax3)
plt.tight_layout()
plt.legend()


ds_2_cluster_log = np.log(ds_2_cluster[['Age', 'Credit amount','Duration']])

fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,8))
sns.distplot(ds_2_cluster_log["Age"], ax=ax1)
sns.distplot(ds_2_cluster_log["Credit amount"], ax=ax2)
sns.distplot(ds_2_cluster_log["Duration"], ax=ax3)
plt.tight_layout()


ds_2['Saving accounts'] = ds_2['Saving accounts'].fillna(ds_2['Saving accounts'].mode())
ds_2['Checking account'] = ds_2['Checking account'].fillna(ds_2['Checking account'].mode())


explode = (0.05,0,0.05,0.05,0,0.1,0.,0.1)
fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(ds_2.Purpose.value_counts().values, labels=['']*len(ds_2.Purpose.value_counts().index), 
                                  autopct='%1.1f%%', startangle=90,textprops={'fontsize': 14},pctdistance=1.2,
                                 explode=explode)

centre_circle = plt.Circle((0, 0), 0.3, color='white', edgecolor='black', linewidth=0.8)
ax.add_patch(centre_circle)
ax.axis('equal')
ax.legend(wedges, ds_2.Purpose.value_counts().index, title="Categories", loc="center left", 
          bbox_to_anchor=(0.7, 0, 0.5, 1), fontsize=12)
plt.title('Share Of Credit Amount For Different Purposes',fontsize=18,y=1.1);

fig, ax = plt.subplots(figsize=(14,6))
box = sns.boxplot(x='Purpose',y='Credit amount' , hue='Gender', data=ds_2)
box.set_xticklabels(box.get_xticklabels(), rotation=15,fontsize=14)
fig.subplots_adjust(bottom=0.2)
plt.title('Determine the Credit Amount For Each Gender By Different Purposes',fontsize=22);
plt.xlabel('Purpose',fontsize=16)
plt.ylabel('Credit Amount',fontsize=16)
plt.xticks(fontsize=18)
plt.tight_layout()

sns.set(rc={'figure.figsize':(15,5)})
sns.lineplot(data=ds_2, x='Duration', y='Credit amount', hue='Gender', lw=2, palette='deep');
plt.xlabel('Duration (Months)',fontsize=18);
plt.ylabel('Credit Amount',fontsize=18);
plt.title('Trend of Credit Amount with Duration For Each Gender',fontsize=22);

ds_2_cluster_log['Job'] = ds_2.Job

scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(ds_2_cluster_log)

Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(cluster_scaled)
    Sum_of_squared_distances.append(km.inertia_)
plt.figure(figsize=(5,5))
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k clusters',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("22084592.png", bbox_inches="tight",dpi=300)


fig = plt.figure(num=None, figsize=(15, 10), dpi=50, facecolor='white', edgecolor='k')
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ds_2_cluster['Age'], ds_2_cluster['Credit amount'], ds_2_cluster['Duration'], c=kmeans_labels, cmap='rainbow')
ax.set_xlabel('Age')
ax.set_ylabel('Credit Amount')
ax.set_zlabel('Duration')
ax.set_facecolor('white')
ax.set_xlabel('Age', fontsize=20)
ax.set_ylabel('Credit Amount', fontsize=20)
ax.set_zlabel('Duration', fontsize=20)


plt.title('3D Scatter Plot with K-Means Clustering',fontsize=28)
plt.tight_layout()
plt.savefig("22084592.png", bbox_inches="tight",dpi=300)


DS2_clustered_kmeans = ds_2_cluster.assign(Cluster=kmeans_labels)
grouped_kmeans = DS2_clustered_kmeans.groupby(['Cluster']).mean().round(1)


X = DS_2['Duration'].values
y = DS_2['Credit amount'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def err_ranges(model, x, params, mse, alpha=0.05):
    n = len(y_test)
    dof = n - len(params)
    t_value = t.ppf(1 - alpha / 2, dof)
    confidence_interval = t_value * np.sqrt(mse / n)  # assuming normal distribution
    
    upper_bound = model(x, *params) + confidence_interval
    lower_bound = model(x, *params) - confidence_interval
    
    return lower_bound, upper_bound

def exponential_growth(x, a, b, c):
    return a * np.exp(b * x) + c

def linear_function(x, m, b):
    return m * x + b

def logistic_function(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))

exp_params, _ = curve_fit(exponential_growth, X_train, y_train)
y_exp = exponential_growth(X_test, *exp_params)
exp_mse = mean_squared_error(y_test,y_exp)

lin_params, _ = curve_fit(linear_function, X_train,y_train)
y_lin = linear_function(X_test, *lin_params)
lin_mse = mean_squared_error(y_test,y_lin)

log_params, _ = curve_fit(logistic_function, X_train, y_train)
y_log = logistic_function(X_test, *log_params)
log_mse = mean_squared_error(y_test,y_log)


plt.figure(figsize=(5,5))
lower_bound, upper_bound = err_ranges(linear_function, X_test, lin_params, lin_mse)
plt.scatter(X_test, y_test, label='Actual Data')
plt.plot(X_test, y_lin, label=f'Linear Function (MSE: {lin_mse:.2f})', color='blue')
plt.fill_between(X_test.flatten(), lower_bound, upper_bound, color='blue', alpha=0.3, label='95% Confidence Interval')
plt.legend()
plt.xlabel('Independent Variable',fontsize=10)
plt.ylabel('Dependent Variable',fontsize=10)
plt.title('Linear Function Fitting and Prediction\nwith Confidence Interval',fontsize=12)
plt.savefig("22084592.png", bbox_inches="tight",dpi=300)

plt.figure(figsize=(5,5))
lower_bound, upper_bound = err_ranges(logistic_function, X_test, log_params, log_mse)
plt.scatter(X_test, y_test, label='Actual Data')
plt.plot(X_test, y_log, label=f'Logistic Function (MSE: {log_mse:.2f})', color='blue')
plt.fill_between(X_test.flatten(), lower_bound, upper_bound, color='blue', alpha=0.3, label='95% Confidence Interval')
plt.legend()
plt.xlabel('Independent Variable',fontsize=10)
plt.ylabel('Dependent Variable',fontsize=10)
plt.title('Logistic Function Fitting and Prediction\nwith Confidence Interval',fontsize=12)
plt.savefig("22084592.png", bbox_inches="tight",dpi=300)

plt.figure(figsize=(5,5))
lower_bound, upper_bound = err_ranges(exponential_growth, X_test, exp_params, exp_mse)
plt.scatter(X_test, y_test, label='Actual Data')
plt.plot(X_test, y_exp, label=f'Exponential Growth (MSE: {exp_mse:.2f})', color='blue')
plt.fill_between(X_test.flatten(), lower_bound, upper_bound, color='blue', alpha=0.3, label='95% Confidence Interval')
plt.legend()
plt.xlabel('Independent Variable',fontsize=10)
plt.ylabel('Dependent Variable',fontsize=10)
plt.title('Exponential Growth Function Fitting and Prediction\nwith Confidence Interval',fontsize=12)
plt.savefig("22084592.png", bbox_inches="tight",dpi=300)


plt.figure()
plt.savefig("22084592.png", bbox_inches="tight",dpi=300)

