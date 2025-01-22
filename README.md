
Using machine learning for disease prediction involves teaching computers to study lots of medical information to guess if someone might get sick. For example, with heart disease prediction using machine learning, computers can look at factors like age, blood pressure, and cholesterol levels to guess who might have heart problems in the future. This helps doctors catch issues early and keep people healthy.

Importing Necessary Libraries
Data Loading 
Plotting Librariesimport pandas as pd import numpy as np import matplotlib.pyplot as plt import seaborn as sns import cufflinks as cf %matplotlib inline

Metrics for Classification techniquefrom sklearn.metrics import classification_report,confusion_matrix,accuracy_score

Scalerfrom sklearn.preprocessing import StandardScaler from sklearn.model_selection import RandomizedSearchCV, train_test_split

Model buildingfrom xgboost import XGBClassifier from catboost import CatBoostClassifier from sklearn.ensemble import RandomForestClassifier from sklearn.neighbors import KNeighborsClassifier from sklearn.svm import SVC

Data Loading
Here we will be using the pandas read_csv function to read the dataset. Specify the location of the dataset and import them.

Importing Datadata = pd.read_csv(“heart.csv”) data.head(6) # Mention no of rows to be displayed from the top in the argument

Output:

Exploratory Data Analysis
Now, let’s see the size of the datasetdata.shape

Output:(303, 14)

Inference: We have a dataset with 303 rows which indicates a smaller set of data.

As above we saw the size of our dataset now let’s see the type of each feature that our dataset holds.

Python Code:

Inference: The inference we can derive from the above output is:

Out of 14 features, we have 13 int types and only one with the float data types.
Woah! Fortunately, this dataset doesn’t hold any missing values.
As we are getting some information from each feature so let’s see how statistically the dataset is spread.data.describe()

Output:

Exploratory Data Analysis
It is always better to check the correlation between the features so that we can analyze that which feature is negatively correlated and which is positively correlated so, Let’s check the correlation between various features.plt.figure(figsize=(20,12)) sns.set_context(‘notebook’,font_scale = 1.3) sns.heatmap(data.corr(),annot=True,linewidth =2) plt.tight_layout()

Output:

output , heart disease prediction using Machine learning
By far we have checked the correlation between the features but it is also a good practice to check the correlation of the target variable.

So, let’s do this!sns.set_context(‘notebook’,font_scale = 2.3) data.drop(‘target’, axis=1).corrwith(data.target).plot(kind=’bar’, grid=True, figsize=(20, 10), title=”Correlation with the target feature”) plt.tight_layout()

Output:

Correlation with the Target Feature , 
Inference: Insights from the above graph are:

Four feature( “cp”, “restecg”, “thalach”, “slope” ) are positively correlated with the target feature.
Other features are negatively correlated with the target feature.
So, we have done enough collective analysis now let’s go for the analysis of the individual features which comprises both univariate and bivariate analysis.

Age(“age”) Analysis
Here we will be checking the 10 ages and their counts.plt.figure(figsize=(25,12)) sns.set_context(‘notebook’,font_scale = 1.5) sns.barplot(x=data.age.value_counts()[:10].index,y=data.age.value_counts()[:10].values) plt.tight_layout()

Output:

Age Analysis| Heart Disease Prediction 
Inference:  Here we can see that the 58 age column has the highest frequency.

Let’s check the range of age in the dataset.minAge=min(data.age) maxAge=max(data.age) meanAge=data.age.mean() print(‘Min Age :’,minAge) print(‘Max Age :’,maxAge) print(‘Mean Age :’,meanAge)

Output:

Output | Heart Disease Prediction 
Min Age : 29 Max Age : 77 Mean Age : 54.366336633663366

We should divide the Age feature into three parts – “Young”, “Middle” and “Elder”Young = data[(data.age>=29)&(data.age<40)] Middle = data[(data.age>=40)&(data.age<55)] Elder = data[(data.age>55)] plt.figure(figsize=(23,10)) sns.set_context(‘notebook’,font_scale = 1.5) sns.barplot(x=[‘young ages’,’middle ages’,’elderly ages’],y=[len(Young),len(Middle),len(Elder)]) plt.tight_layout()

Output:

Heart Disease Prediction 
Inference: Here we can see that elder people are the most affected by heart disease and young ones are the least affected.

To prove the above inference we will plot the pie chart.colors = [‘blue’,’green’,’yellow’] explode = [0,0,0.1] plt.figure(figsize=(10,10)) sns.set_context(‘notebook’,font_scale = 1.2) plt.pie([len(Young),len(Middle),len(Elder)],labels=[‘young ages’,’middle ages’,’elderly ages’],explode=explode,colors=colors, autopct=’%1.1f%%’) plt.tight_layout()

Output:

Sex(“sex”) Feature Analysis
Sex feature analysis | Heart Disease Prediction 
plt.figure(figsize=(18,9)) sns.set_context(‘notebook’,font_scale = 1.5) sns.countplot(data[‘sex’]) plt.tight_layout()

Output:

Inference: Here it is clearly visible that, Ratio of Male to Female is approx 2:1.

Now let’s plot the relation between sex and slope.plt.figure(figsize=(18,9)) sns.set_context(‘notebook’,font_scale = 1.5) sns.countplot(data[‘sex’],hue=data[“slope”]) plt.tight_layout()

Output:

Output of Sex Analysis,
Inference: Here it is clearly visible that the slope value is higher in the case of males(1).

Chest Pain Type(“cp”) Analysis
plt.figure(figsize=(18,9)) sns.set_context(‘notebook’,font_scale = 1.5) sns.countplot(data[‘cp’]) plt.tight_layout()

Output:

Chest Pain
Inference: As seen, there are 4 types of chest pain

status at least
condition slightly distressed
condition medium problem
condition too bad
Analyzing cp vs target column

Heart Disease Prediction 
Inference: From the above graph we can make some inferences,

People having the least chest pain are not likely to have heart disease.
People having severe chest pain are likely to have heart disease.
Elderly people are more likely to have chest pain.

Thal Analysis
plt.figure(figsize=(18,9)) sns.set_context(‘notebook’,font_scale = 1.5) sns.countplot(data[‘thal’]) plt.tight_layout()

Output:

Thal Analysis
Target
plt.figure(figsize=(18,9)) sns.set_context(‘notebook’,font_scale = 1.5) sns.countplot(data[‘target’]) plt.tight_layout()

Output:

Target | Heart Disease Prediction 
Inference: The ratio between 1 and 0 is much less than 1.5 which indicates that the target feature is not imbalanced. So for a balanced dataset, we can use accuracy_score as evaluation metrics for our model.

Feature Engineering
Now we will see the complete description of the continuous data as well as the categorical datacategorical_val = [] continous_val = [] for column in data.columns: print(“——————–“) print(f”{column} : {data[column].unique()}”) if len(data[column].unique()) <= 10: categorical_val.append(column) else: continous_val.append(column)

Output:

Feature Engineering Output | Heart Disease Prediction 
Now here first we will be removing the target column from our set of features then we will categorize all the categorical variables using the get dummies method which will create a separate column for each category suppose X variable contains 2 types of unique values then it will create 2 different columns for the X variable.categorical_val.remove(‘target’) dfs = pd.get_dummies(data, columns = categorical_val) dfs.head(6)

Output:

Output | Heart Disease Prediction 
Now we will be using the standard scaler method to scale down the data so that it won’t raise the outliers also dataset which is scaled to general units leads to having better accuracy.sc = StandardScaler() col_to_scale = [‘age’, ‘trestbps’, ‘chol’, ‘thalach’, ‘oldpeak’] dfs[col_to_scale] = sc.fit_transform(dfs[col_to_scale]) dfs.head(6)

Output:

Output | Heart Disease Prediction 
Modeling
Splitting our DatasetX = dfs.drop(‘target’, axis=1) y = dfs.target X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

The KNN Machine Learning Algorithm
knn = KNeighborsClassifier(n_neighbors = 10) knn.fit(X_train,y_train) y_pred1 = knn.predict(X_test) print(accuracy_score(y_test,y_pred1))

Output:0.8571428571428571

Conclusion on Heart Disease Prediction
1. We did data visualization and data analysis of the target variable, age features, and whatnot along with its univariate analysis and bivariate analysis.

2. We also did a complete feature engineering part in this article which summons all the valid steps needed for further steps i.e.
model building.

3. From the above model accuracy, KNN is giving us the accuracy which is 89%.

Conclusion
Heart disease prediction using machine learning utilizes algorithms to analyze medical data like age, blood pressure, and cholesterol levels, aiding in early detection and prevention. Machine learning greatly enhances disease prediction by analyzing large datasets, identifying patterns, and making accurate forecasts, ultimately improving healthcare outcomes and saving lives.![image 17](https://github.com/user-attachments/assets/3de7f3e2-5696-48d9-90bf-b34023b6d632)
![image 14](https://github.com/user-attachments/assets/286aef63-e25d-4cee-9be1-d17fb5746437)
![image 13](https://github.com/user-attachments/assets/e71ba993-90de-49d5-83ea-135b067d4c74)
![image 12](https://github.com/user-attachments/assets/363e71fe-0ff8-4612-a274-54294638836d)
![image 11](https://github.com/user-attachments/assets/fb1558e8-7b50-4358-b055-8fd39918056f)
![image 10](https://github.com/user-attachments/assets/965c2d23-e708-4137-b800-efe86035346a)
![image 9](https://github.com/user-attachments/assets/57e6ece6-68f2-4b34-b2dd-2770e2459967)
![image 8](https://github.com/user-attachments/assets/9ca1950c-16b9-45ee-b3e0-cc9c0c22421e)
![image 7](https://github.com/user-attachments/assets/3303cae0-b222-48ed-bd52-b55a705241db)
![image 6](https://github.com/user-attachments/assets/fb05ab5e-b60a-41c6-a1fc-63d89065e19d)
![image 5](https://github.com/user-attachments/assets/c067762c-4dc7-42d0-9f97-04ea350c7671)
![image 4](https://github.com/user-attachments/assets/0fad4a9d-1888-4bf9-9021-79b1c8689a3d)
![image 3](https://github.com/user-attachments/assets/3d8f2c2e-ef49-46f1-9bfe-d86e86d0033c)
![image 2](https://github.com/user-attachments/assets/88d47e03-9692-4995-be8e-e63e5429f218)
![Image 1](https://github.com/user-attachments/assets/6d15a43e-d2fd-4448-a077-dd1c0d838923)
