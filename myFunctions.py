import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pyreadstat
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier



def clean_data(csv_file):
    # Load the data into a dataframe
    data = pd.read_csv(csv_file)
    df = pd.DataFrame(data)

    # List of numeric columns
    numeric_columns = ['Rk', 'G', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%',
                       'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'W/L%']
    
    # Make respective columns numeric
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Remove the rank column
    df = df.drop(columns=['Rk'])

    #drop the games column
    df = df.drop(columns=['G'])
    
    return df

def clean_season_halfs(csv_file_1):
    # Load the data into a dataframe
    data = pd.read_csv(csv_file_1)
    df = pd.DataFrame(data)

    # List of numeric columns
    numeric_columns = ['Rk', 'G', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%',
                       'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'W/L%']
    
    # Make respective columns numeric
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Remove the rank column
    df = df.drop(columns=['Rk'])

    #drop the games column
    df = df.drop(columns=['G'])
    
    return df


def find_most_important_metric(df):
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Calculate correlation coefficients with W/L%
    correlation_with_WL = numeric_df.corr()['W/L%']

    # Drop W/L% as it's a correlation with itself
    correlation_with_WL = correlation_with_WL.drop('W/L%')

    # Find the metric with highest absolute correlation
    most_important_metric = correlation_with_WL.abs().idxmax()

    # Print the top 5 metrics
    print(correlation_with_WL.abs().sort_values(ascending=False).head(5))

    print(f"The most important metric for W/L% is: {most_important_metric}")

    # Return the correlation values for plotting
    return correlation_with_WL

def bar_plot_correlation(correlation_with_WL, title):
    plt.figure(figsize=(20, 10))
    sns.barplot(x=correlation_with_WL.index, y=correlation_with_WL.values)
    plt.title(title)
    #label the y-axis
    plt.ylabel('Correlation coefficient with respect to W/L%')
    plt.xlabel('Team Metrics')
    plt.xticks(rotation=45)
    plt.show()

# ML Analysis
def train_and_predict(df, df2):
    correlation_with_WL = find_most_important_metric(df)

    # Convert W/L% to binary category for both df and df2
    df['W/L%'] = df['W/L%'].apply(lambda x: 'Winning' if x > 52.5 else 'Losing')
    df2['W/L%'] = df2['W/L%'].apply(lambda x: 'Winning' if x > 52.5 else 'Losing')

    # Define features and target for df (training set), determined from correlation coefficient on first half of data
    selected_features = ['3P%', 'PTS', 'FG%', 'DRB', 'FT'] 
    X_train = df[selected_features]
    y_train = df['W/L%']

    # Define features and target for df2 (testing set)
    X_test = df2[selected_features]
    y_test = df2['W/L%']

    # Scale features
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)



    # Create a dummy classifier ()
    dummy = DummyClassifier(strategy='most_frequent') #predicting the most common outcome in the set, either winning or losing

    # Train the dummy classifier
    dummy.fit(X_train, y_train)

    # Make predictions on the testing set
    y_dummy_pred = dummy.predict(X_test)

    # Calculate accuracy
    dummy_accuracy = accuracy_score(y_test, y_dummy_pred)
    print("Dummy Accuracy:", dummy_accuracy)

