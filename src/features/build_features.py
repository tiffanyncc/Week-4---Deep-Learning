import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    # Separating target variable and other variables
    Y = df['Attrition']
    X = df.drop(columns=['Attrition'])
    
    # Scaling the data
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return df, X_scaled, Y

def prepare_data(X_scaled, Y):
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=1, stratify=Y)
    return x_train, x_test, y_train, y_test