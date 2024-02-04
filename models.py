from matplotlib import category
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score, r2_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
    
# Učitajte .csv datoteku
#df = pd.read_csv('data/Top_10000_Movies_IMDb.csv')

def get_unique_genres(dataframe, genre_column='Genre'):
    # Extract genres from the specified column
    genres_list = dataframe[genre_column].str.split(', ').explode().unique()
    
    return genres_list


def create_genre_columns(dataframe, unique_genres, genre_column='Genre'):
    # Create binary columns for each unique genre
    for genre in unique_genres:
        dataframe[f'Genre_{genre}'] = dataframe[genre_column].apply(lambda x: 1 if genre in x else 0)
    
    return dataframe


def preprocess_data(df):
    
    #print(df.columns)
    
    unique_genres = get_unique_genres(df)
    # Print the unique genres
    #print(unique_genres)

    print("Kreiranje kolona za svaki Genre . . .")
    df = create_genre_columns(df, unique_genres)
    print('Kreiranje nove Genre Kolone.')
    
    print("Obrada Runtime kolone . . .")
    df['Runtime'] = df['Runtime'].apply(lambda x: float(x.split(' ')[0]) if 'min' in x else 0)
    print('Kolona Runtime obradjena.')

    print('Kreiranje novih kolona za svakog Director i Star . . .')
    for i in range(5):  # Assuming a maximum of 5 directors or stars
        df[f'Director_{i+1}'] = df['Directors'].apply(lambda x: eval(x)[i] if len(eval(x)) > i else None)
        df[f'Star_{i+1}'] = df['Stars'].apply(lambda x: eval(x)[i] if len(eval(x)) > i else None)
        
        # Use label encoding for each new column
        director_label_encoder = LabelEncoder()
        star_label_encoder = LabelEncoder()

        df[f'Director_{i+1}'] = director_label_encoder.fit_transform(df[f'Director_{i+1}'].astype(str))
        df[f'Star_{i+1}'] = star_label_encoder.fit_transform(df[f'Star_{i+1}'].astype(str))
        
    print('Napravljenje nove Director i Star kolone.')
    df = df.drop(['Directors', 'Stars'], axis=1)
        
    null_values = df.isnull().sum()
    #print("Null vrednosti po kolonama:")
    #print(null_values)
    df = df.dropna()
    print('Obriasni redovi sa null vrednostima.')
        
        
    return df
    

def get_movie_list(df):
    return df['Movie Name'].tolist()



def random_forest_regression_model(df, title = ''):
    
    index_to_drop = df[df['movie_title'] == title].index
    #print(f'IZBACIO GF? {df.shape[0]}')
    # Dropovanje reda na osnovu indeksa
    df = df.drop(index_to_drop)

    #print(f'IZBACIO GF? {df.shape[0]}')
    # Definisanje feature matrice X i ciljnog vektora y
    #X = df.drop(['ID', 'Movie Name', 'Rating', 'Plot', 'Link', 'Genre'], axis=1)  # Uklanjamo nepotrebne kolone
    X = df.drop(['color','director_name','actor_2_name','genres','actor_1_name','movie_title','actor_3_name',
                 'plot_keywords','movie_imdb_link',
                 'language','country','imdb_score','content_rating'], axis=1)
    
    y = df['imdb_score']

    #print(df.columns)

    X.to_csv("korisceni_atributi_predikcija.csv",index=False)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Initialize the RandomForestRegressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model on the training data
    rf_model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    p = X_test.shape[1]  # Broj atributa
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
     
    rmse = np.sqrt(mse)  # Calculate RMSE

    print('\n\n')
    print('-------------------Random Forest Regression model------------------')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
    print(f'R-squared adj: {r2_adj}')
    print(f'Root Mean Squared Error: {rmse}')
    print('\n\n')
    print("Feature importances:")
    atributi_nazivi = X_train.columns
    for naziv, importance in zip(atributi_nazivi, rf_model.feature_importances_):
        print(f"Feature '{naziv}': {importance}")
    indices = np.argsort(rf_model.feature_importances_)
    plt.figure(figsize=(12,9))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), rf_model.feature_importances_[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [X_train.columns[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    
    return rf_model

from sklearn.linear_model import LinearRegression

def linear_regression_model(df, title=''):
    
    index_to_drop = df[df['movie_title'] == title].index
    # Dropovanje reda na osnovu indeksa
    df = df.drop(index_to_drop)
    
    X = df.drop(['color','director_name','actor_2_name','genres','actor_1_name','movie_title','actor_3_name',
                 'plot_keywords','movie_imdb_link',
                 'language','country','imdb_score','content_rating'], axis=1)
    
    y = df['imdb_score']

    # Definisanje feature matrice X i ciljnog vektora y
   # X = df.drop(['ID', 'Movie Name', 'Rating', 'Plot', 'Link', 'Genre'], axis=1)  # Uklanjamo nepotrebne kolone
    #y = df['Rating']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Linear Regression model
    lr_model = LinearRegression()

    # Fit the model on the training data
    lr_model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = lr_model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)  # Calculate RMSE

    print('\n\n')
    print(f'-------------------Linear Regression Model Metrics-----------------------')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
    print(f'Root Mean Squared Error: {rmse}')
    print('\n\n')
    
    return lr_model

from sklearn.ensemble import GradientBoostingRegressor

def gradient_boosting_regression_model(df, title = ''):
    index_to_drop = df[df['movie_title'] == title].index
    # Dropovanje reda na osnovu indeksa
    df = df.drop(index_to_drop)
    # Defining feature matrix X and target vector y
    X = df.drop(['color','director_name','actor_2_name','genres','actor_1_name','movie_title','actor_3_name',
                 'plot_keywords','movie_imdb_link',
                 'language','country','imdb_score','content_rating'], axis=1)
    
    y = df['imdb_score']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Gradient Boosting Regressor model
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

    # Fit the model on the training data
    gb_model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = gb_model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    p = X_test.shape[1]  # Broj atributa
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    rmse = np.sqrt(mse)  # Calculate RMSE

    print('\n\n')
    print(f'----------------Gradient Boosting Regressor Model Metrics-----------------')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
    print(f'R-squared adj: {r2_adj}')
    print(f'Root Mean Squared Error: {rmse}')
    print('\n\n')
    
    return gb_model

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

def lasso_regression_model(df, title = ''):
    
    index_to_drop = df[df['movie_title'] == title].index
    # Dropovanje reda na osnovu indeksa
    df = df.drop(index_to_drop)
    # Definisanje feature matrice X i ciljnog vektora y
    #X = df.drop(['ID', 'Movie Name', 'Rating', 'Plot', 'Link', 'Genre'], axis=1)  # Uklanjamo nepotrebne kolone
    #y = df['Rating']
    X = df.drop(['color','director_name','actor_2_name','genres','actor_1_name','movie_title','actor_3_name',
                 'plot_keywords','movie_imdb_link',
                 'language','country','imdb_score','content_rating'], axis=1)
    
    y = df['imdb_score']

    #print(df.columns)

    #X.to_csv("PROBA.csv", index=False)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Lasso Regression model
    lasso_model = Lasso(alpha=0.1)  # Možete eksperimentisati sa vrednostima alpha

    # Fit the model on the training data
    lasso_model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = lasso_model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    p = X_test.shape[1]  # Broj atributa
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
     
    rmse = np.sqrt(mse)  # Calculate RMSE

    print('\n\n')
    print('-------------------Lasso Regression model------------------')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
    print(f'R-squared adj: {r2_adj}')
    print(f'Root Mean Squared Error: {rmse}')
    print('\n\n')
    
    return lasso_model

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

def knn_regression_model(df):
    # NIJE DOBAR ZA PREDIKCIJU OCENE
    # Definisanje feature matrice X i ciljnog vektora y
    X = df.drop(['color','director_name','actor_2_name','genres','actor_1_name','movie_title','actor_3_name',
                 'plot_keywords','movie_imdb_link',
                 'language','country','imdb_score','content_rating'], axis=1)
    
    y = df['imdb_score']

    #print(df.columns)

    X.to_csv("PROBA.csv", index=False)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the KNN Regression model
    knn_model = KNeighborsRegressor(n_neighbors=5)  # Možete eksperimentisati sa brojem suseda (n_neighbors)

    # Fit the model on the training data
    knn_model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = knn_model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    p = X_test.shape[1]  # Broj atributa
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
     
    rmse = np.sqrt(mse)  # Calculate RMSE

    print('\n\n')
    print('-------------------K-Nearest Neighbors Regression model------------------')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
    print(f'R-squared adj: {r2_adj}')
    print(f'Root Mean Squared Error: {rmse}')
    print('\n\n')
    
    return knn_model




def predict_movie_rating(movie_name, model, df, features):
    # Find the row corresponding to the movie name
    movie_row = df[df['Movie Name'] == movie_name]
    
    #if movie_row.empty:
     #   return f"Movie '{movie_name}' not found in the dataset."

    # Extract features for prediction
    movie_features = movie_row[features]

    # Make prediction using the model
    rating_prediction = model.predict(movie_features)[0]
    
    # Get the actual rating
    actual_rating = movie_row['Rating'].values[0]

    return f"Predicted Rating for '{movie_name}': {rating_prediction:.2f}, Actual Rating: {actual_rating:.2f}", rating_prediction, actual_rating


def predict_movie_rating2(movie_name, model, df, features):
    # Find the row corresponding to the movie name
    movie_row = df[df['movie_title'] == movie_name]
    
#    print(f'MOVIE ROW: {movie_row}')
    #if movie_row.empty:
     #   return f"Movie '{movie_name}' not found in the dataset."

    # Extract features for prediction
    movie_features = movie_row[features]

    # Make prediction using the model
    rating_prediction = model.predict(movie_features)[0]
    
    # Get the actual rating
    actual_rating = movie_row['imdb_score'].values[0]

    return f"Predicted Rating for '{movie_name}': {rating_prediction:.2f}, Actual Rating: {actual_rating:.2f}", rating_prediction, actual_rating

