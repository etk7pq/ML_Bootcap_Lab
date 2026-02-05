#%%
#Step 1
# Question for the College Completion Data Set
question1 = "Does a higher level of financial aid indicate higher a higher 4-year graduation rate?"
print("Question for the College Completion Data Set:", question1)

# Question for the Campus Recruitment Data Set
question2 = "Is work experience or degree percentage more indicative of placement"
print("Question for the Campus Recruitment Data Set:", question2)

#%%
#Step 2

#College Recruitement Data Set
# ----------------------------------
# What is a independent Business Metric for your problem?
IBM1 = "average financial aid awarded per student"
print("Independent Business Metric for College Completion Data Set:", IBM1)

#Data Preparation
#Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
#Load dataset
df_college = pd.read_csv("cc_institution_details.csv")

#Drop unneeded columns
df_college = df_college[[
    'level', 'control', 'hbcu', 'flagship',  #categorical variables
    'student_count', 'ft_pct', 'fte_value',  #numeric variables
    'aid_value', #independent business metric                            
    'grad_100_value' #target variable                         
]]

#Correct variable type/class
categorical_cols = ['level', 'control', 'hbcu', 'flagship']
for col in categorical_cols:
    df_college[col] = df_college[col].astype('category')

#Create target variable: graduation rate (high = 1, low = 0)
med = df_college['grad_100_value'].median()
df_college['high_grad_100'] = (df_college['grad_100_value'] > med).astype(int)

#Calculate target variable prevalence
print("High 4-year graduation (grad_100) prevalence:")
print(df_college['high_grad_100'].value_counts(normalize=True))

#One-hot encode categorical variables and normalize numeric variables
numeric_cols = [col for col in df_college.columns if col not in categorical_cols + ['grad_100_value', 'high_grad_100']]
X = df_college[categorical_cols + numeric_cols]
y = df_college['high_grad_100']  
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ]
)

#Split data into necessary data partitions
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_tune, X_test, y_tune, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Train size:", X_train.shape)
print("Tune size:", X_tune.shape)
print("Test size:", X_test.shape)

#%%
#Campus Recruitment Data Set
# ----------------------------------
# What is a independent Business Metric for your problem?
IBM2 = "work experience; also degree percentage"
print("Independent Business Metric for Campus Recruitment Data Set:", IBM2)

#Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

#Load Dataset
df_placement = pd.read_csv("Placement_Data_Full_Class.csv") 

#Drop unneeded columns
df_placement = df_placement[[
    'degree_p',   #degree percentage   
    'workex',   #work experience (categorical)     
    'status'    #placement status      
]]

#Correct variable type/class
df_placement['workex'] = df_placement['workex'].astype('category')
df_placement['status'] = df_placement['status'].astype('category')

#Create target variable: placement status (placed = 1, not placed = 0)
df_placement['placed'] = (df_placement['status'] == 'Placed').astype(int)

#Calculate target variable prevalence
print("Placement prevalence:")
print(df_placement['placed'].value_counts(normalize=True))

#One-hot encode categorical variables and normalize numeric variables
categorical_cols = ['workex']
numeric_cols = ['degree_p']
X = df_placement[categorical_cols + numeric_cols]
y = df_placement['placed']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ]
)

#Split data into necessary data partitions
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_tune, X_test, y_tune, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print("Train size:", X_train.shape)
print("Tune size:", X_tune.shape)
print("Test size:", X_test.shape)


#%%
#Step 3
#College Completion Data Set
#What do your instincts tell you about the data. Can it address your problem, what areas/items are you worried about?
print("""
      College Completion Data Set:
      It seems as thought the data can address the problem because it includes both the independent business metric (average financial aid awarded per student) and the target variable (graduation rate). 
      Additionally, the target variable is fairly balanced, so classification models can detect patterns without being biased towards one class.
      However, this dataset has a lot of features (most of which I dropped) that could also potentially influence the graduation rate, such as student demographics, SAT performance, and the quality of the institution.
      I am worried about the potential for confounding variables that could influence the relationship between financial aid and graduation rates. For example, students who receive more financial aid may also have other advantages (such as higher SAT scores or better access to resources) that contribute to their success, rather than the financial aid itself being the causal factor.
      In summary, I am worried about being able to identify the difference between correlation and causation in this dataset.
      """
)
#Campus Recruitment Data Set
#What do your instincts tell you about the data. Can it address your problem, what areas/items are you worried about?
print("""
      Campus Recruitment Data Set:
      Once again, I think that this data can address my problem because it includes both the independent business metric(s) (work experience and degree percentage) and the target variable (placement status).
      This target variable is more imbalanced than the previous dataset, which could make it more difficult for classification models to detect patterns without being biased towards the majority class (placed). However, the imbalance is not extreme, so I think it can still be addressed.
      I am worried about the the small sample size of this dataset, as well as identifying the difference between correlation and causation.
      """)