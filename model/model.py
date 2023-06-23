# Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

#ignore warning messages
import warnings
warnings.filterwarnings('ignore')
# ML Libraries
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
# Setting figure size
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 8
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size
# Get the data
df = pd.read_csv("C:\\Users\\FreeComp\\OneDrive\\Desktop\\New folder\\diabetes.csv")
df.head()
# df.shapedf.info()
# df.describe()
# replacting 0 with nan
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = \
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
df.head(3)
# Distribution of all variables
f, ax= plt.subplots(figsize=(15, 10))

ax.set(xlim=(-.05, 200))
plt.ylabel('Variables')
plt.title("Overview Data Set")
ax = sns.boxplot(data = df, orient = 'v', palette = 'Blues')
# Checking for null values
df.isnull().sum()
# function to find the mean
def median_target(var):
    temp = df[df[var].notnull()]
    temp = round(temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].mean().reset_index(), 1)
    return temp
median_target("Glucose")
df.loc[(df['Outcome'] == 0 ) & (df['Glucose'].isnull()), 'Glucose'] = 110.6
df.loc[(df['Outcome'] == 1 ) & (df['Glucose'].isnull()), 'Glucose'] = 142.3
median_target("BloodPressure")
df.loc[(df['Outcome'] == 0 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 70.9
df.loc[(df['Outcome'] == 1 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 75.3
median_target("SkinThickness")
df.loc[(df['Outcome'] == 0 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 27.2
df.loc[(df['Outcome'] == 1 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 33.0
median_target("Insulin")
df.loc[(df['Outcome'] == 0 ) & (df['Insulin'].isnull()), 'Insulin'] = 130.3
df.loc[(df['Outcome'] == 1 ) & (df['Insulin'].isnull()), 'Insulin'] = 206.8
median_target("BMI")
df.loc[(df['Outcome'] == 0 ) & (df['BMI'].isnull()), 'BMI'] = 30.9
df.loc[(df['Outcome'] == 1 ) & (df['BMI'].isnull()), 'BMI'] = 35.4


g = sns.histplot(data=df, x='Outcome', hue='Outcome', multiple='stack', palette='Set3')
g.set_title('Count of Outcome Variable')
g.set_xlabel('Outcome')
g.set_ylabel('Count')

plt.show()
# Correlation plot
f, ax = plt.subplots(figsize=(11, 9))

mask = np.triu(np.ones_like(df.corr(), dtype=bool))

sns.heatmap(df.corr(), mask=mask, vmax=.3, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, cmap='BuGn')
import seaborn as sns
import matplotlib.pyplot as plt

for column in df.columns:
    sns.histplot(df[column], kde=True, bins=20, color='blue', edgecolor='black')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.title('Distribution of {}'.format(column))
    plt.show()
# Histogram for each column
for col in df.columns:
    plt.hist(df[col], bins=20, alpha=0.5, color='blue', edgecolor='black')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.title('Histogram of {}'.format(col))
    plt.show()
    # Glucose vs BP
    plt.rcParams["figure.figsize"] = (10, 8)
    custom_palette = ['red', 'green']
    sns.scatterplot(x='Glucose', y='BloodPressure', hue='Outcome', data=df, s=60, alpha=0.8, color='green',
                    palette=custom_palette)
    plt.title('Glucose vs Blood Pressure')
    # Insulin vs Blood Pressure
    plt.rcParams["figure.figsize"] = (10, 8)
    custom_palette = ['red', 'green', 'blue', 'orange']
    sns.scatterplot(x='Insulin', y='BloodPressure', hue='Outcome', data=df, s=60, alpha=0.8, color='orange',
                    palette=custom_palette)
    plt.xticks([0, 166, 200, 400, 600])
    plt.title('Insulin vs Blood Pressure')
    # Glucose vs Age
    plt.rcParams["figure.figsize"] = (10, 8)
    custom_palette = ['red', 'green', 'blue', 'orange', 'teal']
    sns.scatterplot(x='Glucose', y='Age', hue='Outcome', data=df, s=60, alpha=0.8, color='teal', palette=custom_palette)
    plt.title('Glucose vs Age')
    # BMI vs Age
    plt.rcParams["figure.figsize"] = (10, 8)
    custom_palette = ['red', 'green', 'blue', 'orange']
    sns.scatterplot(x='BMI', y='Age', hue='Outcome', data=df, s=60, alpha=0.8, color='orange', palette=custom_palette)
    plt.title('BMI vs Age')
    # Skin Thickness vs DPF
    plt.rcParams["figure.figsize"] = (10, 8)
    custom_palette = ['red', 'green', 'blue', 'purple']
    sns.scatterplot(x='SkinThickness', y='DiabetesPedigreeFunction', hue='Outcome', data=df, s=60, alpha=0.8,
                    color='purple', palette=custom_palette)
    plt.title('Skin Thickness vs DPF')
    # splitting columns
    X = df.drop(columns='Outcome')
    y = df['Outcome']
    # scaling
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X),
                     columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                              'DiabetesPedigreeFunction', 'Age'])
    # Split the dataset into 70% Training set and 30% Testing set
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


    def classification_model(y_test, prediction, model, x_train, y_train):
        # Accuracy score
        accuracy = accuracy_score(y_test, prediction)
        print("Accuracy Score: {:.3%}".format(accuracy))

        # F1 score
        f1 = f1_score(y_test, prediction)
        print("F1 Score: {:.3%}".format(f1))

        # Cross validation with 5 folds
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        kf.get_n_splits(x_train)

        accuracy_model = []

        for train_index, test_index in kf.split(x_train):
            # Split train-test
            X_train, X_test = x_train.iloc[train_index], x_train.iloc[test_index]
            Y_train, Y_test = y_train.iloc[train_index], y_train.iloc[test_index]
            # Train the model
            model.fit(X_train, Y_train)
            # Append to accuracy_model the accuracy of the model
            accuracy_model.append(accuracy_score(Y_test, model.predict(X_test)))

        print("Cross-Validation Score: {:.3%}".format(np.mean(accuracy_model)))


    # Function for histogram plot
    def histogram_plot(df):
        fig, axes = plt.subplots(nrows=len(df.columns), figsize=(10, 8 * len(df.columns)))

        for i, column in enumerate(df.columns):
            sns.histplot(data=df, x=column, kde=True, color='purple', ax=axes[i])
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('Density')
            axes[i].set_title('Distribution of {}'.format(column))

        plt.tight_layout()
        plt.show()


gb = GradientBoostingClassifier()
gb.fit(x_train, y_train)
prediction4 = gb.predict(x_test)

print(classification_report(y_test, prediction4))
sns.heatmap(confusion_matrix(y_test, prediction4), annot=True, cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

plt.show()


feature_importances = gb.feature_importances_

plt.barh(x_train.columns, feature_importances, color='orange')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance of Variables')

plt.show()
x_tr = x_train.loc[:,['Insulin','Glucose','BMI','Age','SkinThickness']]
x_te = x_test.loc[:,['Insulin','Glucose','BMI','Age','SkinThickness']]


gb2 = GradientBoostingClassifier()
gb2.fit(x_tr, y_train)
prediction5 = gb2.predict(x_te)

print(classification_report(y_test, prediction5))
sns.heatmap(confusion_matrix(y_test, prediction5), annot=True, cmap='Greens')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

plt.show()
