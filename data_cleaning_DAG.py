#Step 4
#Functions for Data Prep Pipelines
#%%
#Load Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

#Load Dataset
def load_data(path, columns_to_keep=None):
    """Load dataset from CSV and keep only specified columns."""
    df = pd.read_csv(path)
    if columns_to_keep:
        df = df[columns_to_keep]
    return df

#Drop Unneeded Columns
def drop_columns(df, columns_to_drop):
    """Drop specified columns from the dataframe."""
    df = df.drop(columns=columns_to_drop, errors='ignore')
    return df

#Correct Variable Types
def convert_types(df, categorical_columns=None):
    """Convert specified columns to categorical type."""
    if categorical_columns:
       for col in categorical_columns:
           df[col] = df[col].astype('category')
           return df

#Create Target Variable
def create_target_variable(df, target_col, target_type='binary', threshold=None, mapping=None):
    """Create target variable based on specified type. If binary, use threshold; if categorical, use mapping."""
    if target_type == 'binary' and threshold is not None:
        df['target'] = (df[target_col] >= threshold).astype(int)
    elif target_type == 'categorical' and mapping is not None:
        df['target'] = df[target_col].map(mapping).astype(int)
    return df

#Calculate Target Variable Prevalence
def calculate_prevalence(df, target_col):
    """Calculate and return the prevalence of the target variable."""
    prevalence = df[target_col].value_counts(normalize=True)
    return prevalence

#One-Hot Encode and Normalize
def preprocess_features(df, categorical_cols, numeric_cols):
    """One-hot encode categorical variables and normalize numeric variables."""
    X = df[categorical_cols + numeric_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numeric_cols)
        ]
    )
    X_processed = preprocessor.fit_transform(X)
    return X_processed, preprocessor

#Split Data into Partitions
def split_data(X, y, train_size=0.6, tune_size=0.2, test_size=0.2, random_state=42):
    """Split data into training, tuning, and testing sets using a 20-20-60 split."""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_size), random_state=random_state)
    relative_tune_size = tune_size / (tune_size + test_size)
    X_tune, X_test, y_tune, y_test = train_test_split(X_temp, y_temp, test_size=(1 - relative_tune_size), random_state=random_state)
    return X_train, X_tune, X_test, y_train, y_tune, y_test

#%%
#Test Functions with College Completion Data Set
#Load dataset
college_cols = [
    'level','control','hbcu','flagship',
    'student_count','ft_pct','fte_value','aid_value','grad_100_value'
]
df_college = load_data("cc_institution_details.csv", columns_to_keep=college_cols)

# Correct variable type/class
categorical_cols = ['level','control','hbcu','flagship']
df_college = convert_types(df_college, categorical_cols)

#Create target variable: graduation rate (high = 1, low = 0)
median_grad = df_college['grad_100_value'].median()
df_college = create_target_variable(df_college, 'grad_100_value', target_type='binary', threshold=median_grad)

#Check target prevalence
print("College Completion - Target Prevalence:")
print(calculate_prevalence(df_college, 'target'))

#One-hot encode categorical variables and normalize numeric variables
numeric_cols = ['student_count','ft_pct','fte_value','aid_value']
X_college, preprocessor_college = preprocess_features(df_college, categorical_cols, numeric_cols)
y_college = df_college['target']

#Split data into necessary data partitions
X_train_col, X_tune_col, X_test_col, y_train_col, y_tune_col, y_test_col = split_data(X_college, y_college)
print("College Completion Data Shapes:")
print("Train:", X_train_col.shape, "Tune:", X_tune_col.shape, "Test:", X_test_col.shape)

# %%
#Test Functions with Campus Recruitment Data Set
#Load dataset
placement_cols = ['workex', 'degree_p', 'status']
df_campus = load_data("Placement_Data_Full_Class.csv", columns_to_keep=placement_cols)

#Correct variable type/class
categorical_cols_campus = ['status', 'workex']
df_campus = convert_types(df_campus, categorical_cols_campus)

#Create target variable: placement status (placed = 1, not placed = 0)
mapping = {'Placed': 1, 'Not Placed': 0}
df_campus = create_target_variable(df_campus, 'status', target_type='categorical', mapping=mapping)

#Check target prevalence
print("Campus Recruitment - Target Prevalence:")
print(calculate_prevalence(df_campus, 'target'))

#One-hot encode categorical variables and normalize numeric variables
numeric_cols_campus = ['degree_p']
X_campus, preprocessor_campus = preprocess_features(df_campus, categorical_cols_campus, numeric_cols_campus)
y_campus = df_campus['target']

#Split data into necessary data partitions
X_train_camp, X_tune_camp, X_test_camp, y_train_camp, y_tune_camp, y_test_camp = split_data(X_campus, y_campus)
print("Campus Recruitment Data Shapes:")
print("Train:", X_train_camp.shape, "Tune:", X_tune_camp.shape, "Test:", X_test_camp.shape)