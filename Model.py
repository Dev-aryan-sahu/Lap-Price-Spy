# %%
import numpy as np # type: ignore 
import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore # type: ignore

# %%
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.preprocessing import OneHotEncoder # type: ignore
from sklearn.metrics import r2_score,mean_absolute_error # type: ignore
from sklearn.linear_model import LinearRegression,Ridge # type: ignore
from sklearn.tree import DecisionTreeRegressor # type: ignore
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor # type: ignore
from xgboost import XGBRegressor # type: ignore

# %%
df = pd.read_csv('laptop_data.csv')
df.head()

# %%
df.shape

# %%
df.info()

# %%
df.duplicated().sum()

# %%
df.isnull().sum()

# %%
df.drop(columns=['Unnamed: 0'],inplace=True)
df.head()

# %%
df['Ram'] = df['Ram'].str.replace('GB','')
df['Weight'] = df['Weight'].str.replace('kg','')
df.head()

# %%
df['Ram'] = df['Ram'].astype('int32')
df['Weight'] = df['Weight'].astype('float32')
df.info()

# %%
sns.distplot(df['Price'])

# %%
df['Company'].value_counts().plot(kind='bar')

# %%
sns.barplot(x=df['Company'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

# %%
df['TypeName'].value_counts().plot(kind='bar')

# %%
sns.barplot(x=df['TypeName'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

# %%
sns.distplot(df['Inches'])

# %%
sns.barplot(x=df['Inches'],y=df['Price'])

# %%
df['ScreenResolution'].value_counts()

# %%
df['Touchscreen'] = df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
df.sample(5)

# %%
df['Touchscreen'].value_counts().plot(kind='bar')

# %%
sns.barplot(x=df['Touchscreen'],y=df['Price'])

# %%
df['Ips'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)
df.head()

# %%
sns.barplot(x=df['Ips'],y=df['Price'])

# %%
# Use regex to extract the resolution from the 'ScreenResolution' column
# This extracts numbers before and after the 'x' in the resolution string
df['X_res'] = df['ScreenResolution'].str.extract(r'(\d+)x')[0]
df['Y_res'] = df['ScreenResolution'].str.extract(r'x(\d+)')[0]

# Convert the extracted resolutions to numeric values
df['X_res'] = pd.to_numeric(df['X_res'])
df['Y_res'] = pd.to_numeric(df['Y_res'])

# Show a random sample of 5 rows to verify the result
df.sample(5)


# %%
# Ensure X_res is treated as a string before replacing commas
df['X_res'] = df['X_res'].astype(str)

# Replace commas, extract numeric values, and handle decimal points in 'X_res'
df['X_res'] = df['X_res'].str.replace(',', '')  # Remove commas if any
df['X_res'] = df['X_res'].str.findall(r'(\d+\.?\d*)').apply(lambda x: x[0] if len(x) > 0 else None)  # Extract first match if available

# Convert 'X_res' to numeric after cleaning
df['X_res'] = pd.to_numeric(df['X_res'], errors='coerce')

# Display the first few rows to verify the results
df.head()


# %%
df['X_res'] = df['X_res'].astype('int')
df['Y_res'] = df['Y_res'].astype('int')
df.info()

# %%
# Find non-numeric values in the 'Price' column
df[~df['Price'].apply(lambda x: isinstance(x, (int, float)))]
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df.dropna(subset=['Price'], inplace=True)
# Select only numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=['number'])

# Calculate correlation
correlation = numeric_df.corr()['Price']
print(correlation)


# %%
# Calculate PPI (pixels per inch)
df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5 / df['Inches']).astype(float)

# Select only numeric columns for correlation calculation
numeric_df = df.select_dtypes(include='number')

# Calculate correlations with Price
correlation_with_price = numeric_df.corr()['Price']

# Display the DataFrame and the correlation results
print("DataFrame:")
print(df)
print("\nCorrelation with Price:")
print(correlation_with_price)

# %%
df.drop(columns=['ScreenResolution'],inplace=True)
df.head()

# %%
df.drop(columns=['Inches','X_res','Y_res'],inplace=True)
df.head()

# %%
df['Cpu'].value_counts()

# %%
df['Cpu Name'] = df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))
df.head()

# %%
def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'
df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)
df.head()

# %%
df['Cpu brand'].value_counts().plot(kind='bar')

# %%
sns.barplot(x=df['Cpu brand'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

# %%
df.drop(columns=['Cpu','Cpu Name'],inplace=True)
df.head()

# %%
df['Ram'].value_counts().plot(kind='bar')

# %%
sns.barplot(x=df['Ram'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

# %%
df['Memory'].value_counts()

# %%
# Ensure 'Memory' column is clean before splitting
df['Memory'] = df['Memory'].astype(str).replace(r'\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '', regex=False)
df["Memory"] = df["Memory"].str.replace('TB', '000', regex=False)

# Split the 'Memory' column into two parts, if "+" exists
new = df["Memory"].str.split("+", n=1, expand=True)

# Ensure that the split creates two columns, 'first' and 'second'
df["first"] = new[0].str.strip()  # Strip any leading/trailing spaces
df["second"] = new[1]  # Second part, could be NaN if no "+" exists

# Handle cases where 'second' might be NaN
df["second"].fillna("0", inplace=True)

# Flag storage types in the 'first' and 'second' parts
df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

# Remove all non-numeric characters to keep only memory size values
df['first'] = df['first'].str.extract(r'(\d+)')  # Extract only the numeric part from 'first'
df['second'] = df['second'].str.extract(r'(\d+)')  # Extract only the numeric part from 'second'

# Replace NaN values with 0 in case of empty strings
df["first"].fillna(0, inplace=True)
df["second"].fillna(0, inplace=True)

# Convert 'first' and 'second' columns to integers
df["first"] = df["first"].astype(int)
df["second"] = df["second"].astype(int)

# Calculate total memory for each type of storage
df["HDD"] = (df["first"] * df["Layer1HDD"] + df["second"] * df["Layer2HDD"])
df["SSD"] = (df["first"] * df["Layer1SSD"] + df["second"] * df["Layer2SSD"])
df["Hybrid"] = (df["first"] * df["Layer1Hybrid"] + df["second"] * df["Layer2Hybrid"])
df["Flash_Storage"] = (df["first"] * df["Layer1Flash_Storage"] + df["second"] * df["Layer2Flash_Storage"])

# Drop the intermediate columns
df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
                 'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
                 'Layer2Flash_Storage'], inplace=True)



# %%
df.sample(5)

# %%
df.drop(columns=['Memory'],inplace=True)
df.head()

# %%
# Find non-numeric values in the 'Price' column
df[~df['Price'].apply(lambda x: isinstance(x, (int, float)))]
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df.dropna(subset=['Price'], inplace=True)
# Select only numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=['number'])

# Calculate correlation
correlation = numeric_df.corr()['Price']
print(correlation)

# %%
df.drop(columns=['Hybrid','Flash_Storage'],inplace=True)
df.head()

# %%
df['Gpu'].value_counts()

# %%
df['Gpu brand'] = df['Gpu'].apply(lambda x:x.split()[0])
df.head()

# %%
df['Gpu brand'].value_counts()

# %%
df = df[df['Gpu brand'] != 'ARM']
df['Gpu brand'].value_counts()

# %%
sns.barplot(x=df['Gpu brand'],y=df['Price'],estimator=np.median)
plt.xticks(rotation='vertical')
plt.show()

# %%
df.drop(columns=['Gpu'],inplace=True)

# %%
df.head()

# %%
df['OpSys'].value_counts()

# %%
sns.barplot(x=df['OpSys'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

# %%
def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'
df['os'] = df['OpSys'].apply(cat_os)

# %%
df.head()

# %%
df.drop(columns=['OpSys'],inplace=True)

# %%
sns.barplot(x=df['os'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

# %%
sns.displot(df['Weight'])

# %%
sns.scatterplot(x=df['Weight'],y=df['Price'])

# %%
# Find non-numeric values in the 'Price' column
df[~df['Price'].apply(lambda x: isinstance(x, (int, float)))]
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df.dropna(subset=['Price'], inplace=True)
# Select only numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=['number'])

# Calculate correlation
correlation = numeric_df.corr()['Price']
print(correlation)

# %%
# Select only numeric columns for the correlation matrix
numeric_df = df.select_dtypes(include=['number'])

# Set the figure size for better visibility
plt.figure(figsize=(10, 8))

# Create the heatmap using the correlation matrix of numeric columns
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Add title for clarity
plt.title('Correlation Heatmap', fontsize=16)

# Show the plot
plt.show()

# %%
sns.displot(np.log(df['Price']))

# %%
X = df.drop(columns=['Price'])
y = np.log(df['Price'])
X

# %%
y

# %%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=2)

# %%
X_train

# %%
X = df.drop(columns=['Price'])  # Replace 'Price' with your target variable
y = df['Price']  # Your target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=50)

# Create ColumnTransformer for preprocessing
step1 = ColumnTransformer(
    transformers=[
        ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])  # Use sparse_output instead of sparse
    ],
    remainder='passthrough'
)

# Create Linear Regression model
step2 = LinearRegression()

# Create the pipeline
pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Fit the pipeline on the training data
pipe.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipe.predict(X_test)

# Evaluate the model
print('R² score:', r2_score(y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))

# %%
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.7, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
plt.xlabel('True Values (Price)')
plt.ylabel('Predicted Values (Price)')
plt.title('Linear Regression')

plt.tight_layout()
plt.show()

# %%
X = df.drop(columns=['Price'])  # Replace 'Price' with your target variable
y = df['Price']  # Your target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=50)

# Create ColumnTransformer for preprocessing
step1 = ColumnTransformer(
    transformers=[
        ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])  # Use sparse_output instead of sparse
    ],
    remainder='passthrough'
)

# Create Ridge regression model
step2 = Ridge(alpha=5)

# Create the pipeline
pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Fit the pipeline on the training data
pipe.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipe.predict(X_test)

# Evaluate the model
print('R² score:', r2_score(y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))

# %%
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred, alpha=0.7, color='g')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
plt.xlabel('True Values (Price)')
plt.ylabel('Predicted Values (Price)')
plt.title('Ridge Regression')
plt.tight_layout()
plt.show()

# %%
X = df.drop(columns=['Price'])  # Replace 'Price' with your target variable
y = df['Price']  # Your target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=50)

# Create ColumnTransformer for preprocessing
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

# Define the RandomForest model
step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

# Create the pipeline
pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Fit the pipeline on the training data
pipe.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipe.predict(X_test)

# Evaluate the model
print('R2 score:', r2_score(y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))

# %%
plt.figure(figsize=(8, 5 ))
plt.scatter(y_test, y_pred, alpha=0.7, color='b')  # Scatter plot of true vs predicted values
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)  # Line for perfect predictions
plt.xlabel('True Values (Price)')
plt.ylabel('Predicted Values (Price)')
plt.title('True vs Predicted Values')
plt.show()

# %%
X = df.drop(columns=['Price'])  # Replace 'Price' with your target variable
y = df['Price']  # Your target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=50)

# Create ColumnTransformer for preprocessing
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])  # Adjust indices based on your DataFrame
], remainder='passthrough')

# Create Gradient Boosting Regressor model
step2 = GradientBoostingRegressor(n_estimators=500)

# Create the pipeline
pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Fit the pipeline on the training data
pipe.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipe.predict(X_test)

# Evaluate the model
print('R² score:', r2_score(y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))

# %%
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
plt.xlabel('True Values (Price)')
plt.ylabel('Predicted Values (Price)')
plt.title('Gradient Boosting Regressor')
plt.grid()
plt.tight_layout()
plt.show()

# %%
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

# Define the pipeline with bootstrap=True
step2 = ExtraTreesRegressor(
    n_estimators=100,
    random_state=3,
    bootstrap=True,         # Enable bootstrapping
    max_samples=0.5,        # Now max_samples can be used
    max_features=0.75,
    max_depth=15
)

# Create a pipeline
pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Fit the pipeline on the training data
pipe.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipe.predict(X_test)

# Evaluate the model
print('R² score:', r2_score(y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))


# %%
# Plot graph of true vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='b', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')

# Labels and title
plt.xlabel('True Values (Price)')
plt.ylabel('Predicted Values (Price)')
plt.title('True vs Predicted Values: ExtraTreesRegressor')
plt.legend()
plt.grid(True)
plt.show()

# %%
# Define feature and target variables
X = df.drop(columns=['Price'])  # Replace 'Price' with your target variable
y = df['Price']  # Your target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create ColumnTransformer for preprocessing
step1 = ColumnTransformer(
    transformers=[
        ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])  # Corrected the parameter name
    ],
    remainder='passthrough'
)

# Create XGBRegressor model
step2 = XGBRegressor(n_estimators=45, max_depth=5, learning_rate=0.5)

# Create the pipeline
pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Fit the pipeline on the training data
pipe.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipe.predict(X_test)

# Evaluate the model
print('R² score:', r2_score(y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))

# %%
# Plotting the graph
plt.figure(figsize=(10, 6))

# Scatter plot of True vs Predicted values
plt.scatter(y_test, y_pred, alpha=0.7, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)  # Line for perfect predictions

# Add labels and title
plt.xlabel('True Values (Price)')
plt.ylabel('Predicted Values (Price)')
plt.title('True vs Predicted Values for XGBRegressor')

plt.show()

# %%
import pickle

pickle.dump(df,open('df.pkl','wb'))
pickle.dump(pipe,open('pipe.pkl','wb'))

# %%
df

# %%
X_train


