import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import models as PRED
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/Top_10000_Movies_IMDb.csv')

#df = df.dropna()
titles_list = df['Movie Name'].tolist()

# Preuzimanje reci koje se cesto ponavalju, ali nemaju puno znacenja
stop_words = set(stopwords.words('english'))

# Inicijalizujemo PorterStemmer za smanjenje reci na osnovni oblik
ps = PorterStemmer()

# Funkcija za izdvajanje ključnih reči iz teksta
def extract_keywords(plot):
    # Tokeniziramo tekst
    words = word_tokenize(plot)
    
    # Filtriramo stop reči
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    
    # Smanjimo reči na osnovni oblik
    stemmed_words = [ps.stem(word) for word in filtered_words]
    
    return ' '.join(stemmed_words)

# Primenimo funkciju na kolonu "Plot" i dodajemo novu kolonu "Keywords"
df['Keywords'] = df['Plot'].apply(extract_keywords)


def preprocess_data(df):
    
    # Predprocesiranje 'Runtime' kolone uklanjanjem min dela
    df['Runtime'] = df['Runtime'].apply(lambda x: float(x.split(' ')[0]) if 'min' in x else 0)

    # Konvertuj 'Votes' u float
    df['Votes'] = df['Votes'].astype(float)

    # Konvertujemo 'Gross' u float
    df['Gross'] = df['Gross'].replace(',', '', regex=True).astype(float)

    return df



def calculate_cosine_similarity(df, lista_uticaja=['y','n','n']):
    #print(lista_uticaja)
    columns_high_influence = ['Genre', 'Rating', 'Keywords', 'Votes']
    columns_low_influence = ['Gross', 'Runtime']
    
    # Kopiramo podatke sa kolonama visokog i niskog uticaja
    df_subset = df[columns_high_influence + columns_low_influence].copy()
    
    df_size = df_subset.shape
    #print(f"Veličina SUBSET DataFrame-a: {df_size[0]} redova x {df_size[1]} kolona")

    dodatno_df = pd.read_csv('glumci_reziseri_prep.csv')
    #dodatno_df = PRED.create_genre_columns(dodatno_df, PRED.get_unique_genres(dodatno_df))
    glumci_reziseri_df = dodatno_df.iloc[:, 12:]
    
    
    null_values = df_subset.isnull().sum()
    #print("Null vrednosti po kolonama:")
    #print(null_values)
    # Preprocesiraj podatke
    df_subset = preprocess_data(df_subset)

    # Normalizujemo numeričke vrednosti u kolonama s niskim uticajem
    scaler = MinMaxScaler()
    df_subset[columns_low_influence] = scaler.fit_transform(df_subset[columns_low_influence])

    # Izvršimo CountVectorizer na kolonama s visokim uticajem
    cv = CountVectorizer()
    #df_subset['Genre'] +' ' + df_subset['Keywords'] + ' ' + df_subset['Directors'].astype(str) + ' ' + df_subset['Stars'].astype(str)

    # Spojimo matrice i izračunamo kosinusnu sličnost
    # Glumci/Rez, Radnja, Zanr
    if(lista_uticaja == ['y', 'y', 'y']):
        print("Koristimo sve atribute")
        dtm_high_influence = cv.fit_transform(df_subset['Genre'] + df_subset['Keywords']).toarray()
        new_matrix = np.concatenate((dtm_high_influence, df_subset[columns_low_influence].values, glumci_reziseri_df.values), axis=1)
    elif(lista_uticaja == ['y', 'y', 'n']):
        print("Ne koristimo zanr")
        dtm_high_influence = cv.fit_transform(df_subset['Keywords']).toarray()
        new_matrix = np.concatenate((dtm_high_influence, df_subset[columns_low_influence].values, glumci_reziseri_df.values), axis=1)
    elif(lista_uticaja == ['y', 'n', 'y']):
        print("Ne koristimo radnju")
        dtm_high_influence = cv.fit_transform(df_subset['Genre']).toarray()
        new_matrix = np.concatenate((dtm_high_influence, df_subset[columns_low_influence].values, glumci_reziseri_df.values), axis=1)
    elif(lista_uticaja == ['y', 'n', 'n']):
        print("Ne koristimo zanr i radnju")
        new_matrix = np.concatenate((df_subset[columns_low_influence].values, glumci_reziseri_df.values), axis=1)
    elif(lista_uticaja == ['n', 'n', 'y']):
        print("Ne koristimo glumce i radnju")
        dtm_high_influence = cv.fit_transform(df_subset['Genre']).toarray()
        new_matrix = np.concatenate((dtm_high_influence, df_subset[columns_low_influence].values), axis=1)
    elif(lista_uticaja == ['n', 'y', 'y']):
        print("Ne koristimo glumce")
        dtm_high_influence = cv.fit_transform(df_subset['Genre'] + df_subset['Keywords']).toarray()
        new_matrix = np.concatenate((dtm_high_influence, df_subset[columns_low_influence].values), axis=1)
    elif(lista_uticaja == ['n', 'y', 'n']):
        print("Ne koristimo glumce i zanr")
        dtm_high_influence = cv.fit_transform(df_subset['Keywords']).toarray()
        new_matrix = np.concatenate((dtm_high_influence, df_subset[columns_low_influence].values), axis=1)
    else:
        print("Ne koristimo radnju i zanr i glumce")
        new_matrix = np.concatenate((df_subset[columns_low_influence].values, glumci_reziseri_df.values), axis=1)
        
    
    #np.savetxt('matrica_slicnosti.txt', new_matrix , fmt='%d')
    # Zamena NaN vrednosti sa nulama
    #new_matrix = glumci_reziseri_df.values
    #new_matrix = np.matrix(matrix_values)
    matrix_shape = new_matrix.shape
    
    #print(f"Matrix Size Slicnosti: {matrix_shape[0]} rows x {matrix_shape[1]} columns")
    
    #new_matrix = np.nan_to_num(new_matrix)

    similarities = cosine_similarity(new_matrix, dense_output=False)

    return similarities, new_matrix


# funkcije za mapiranje glumaca i rezisera, ranije resenje
import hashlib

def hash_name(name):
    return int(hashlib.sha256(name.encode('utf-8')).hexdigest(), 16) % 10**8


mapping = {}
counter = 0

def get_unique_label(name):
    global counter
    if name not in mapping:
        counter += 1
        mapping[name] = counter
        
    #print(f'Counter: {counter}')
    
    #print(mapping[name])

        
    return mapping[name]

    
def preprocess_data_for_rating(df):
    
    df = PRED.create_genre_columns(df, PRED.get_unique_genres(df))
    
    df = df.drop(['Genre','ID','Movie Name','Rating','Runtime','Genre','Metascore','Plot','Directors','Stars','Votes','Gross','Link'], axis=1)
    
    return df
    


from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process

def modelKNN():
    model_knn= NearestNeighbors(metric= 'euclidean', algorithm='brute',  leaf_size=20, p=1, n_neighbors=10)
    return model_knn



def recommender(movie_name, model, n_recommendations, matrica):
    #mat_movies=csr_matrix(movies.values)
    model.fit(matrica)
    idx=process.extractOne(movie_name, df['Movie Name'])[2]
    print('Movie Selected: ',df['Movie Name'][idx], 'Index: ',idx)
    print(matrica[idx])
    
    query_movie = matrica[idx].reshape(1, -1)
    print('Searching for recommendations.....')
    distances, indices=model.kneighbors(query_movie, n_neighbors=n_recommendations+1)
    indices = indices[0][1:]
    print('\n Movie list:')
    for i in indices:
        print(df['Movie Name'][i])
    
    
import numpy as np

def jaccard_similarity(row1, row2):
    intersection = np.sum(np.minimum(row1, row2))
    union = np.sum(np.maximum(row1, row2))
    return intersection / union if union != 0 else 0

def find_similar_rows(matrix, input_row_index, top_n=10):
    input_row = matrix[input_row_index]
    
    similarities = np.array([jaccard_similarity(input_row, row) for row in matrix])

    similar_indices = np.argsort(similarities)[::-1]

    similar_indices = similar_indices[similar_indices != input_row_index]

    top_similar_indices = similar_indices[:top_n]

    return top_similar_indices


    

def build_recommendations(title, similarities):
    try:
        title = title.lower()

        # Pronađi indeks filma u DataFrame-u na osnovu naslova
        idx = df[df['Movie Name'].apply(lambda x: x.lower()) == title].index[0]
        #print(f'INDEX FILMAAA: {idx} {df['Movie Name'][idx]}')
        
        #top_indices = np.argsort(vector)[-k:][::-1]
        top_indices = np.argsort(similarities[idx])[-10:][::-1]
        
        print(similarities[idx])
        #print(top_indices)
        selected_values = [similarities[idx][i] for i in top_indices]
        print(f'Kosinusne slicnosti: {selected_values}')
        # Sortiraj preporuke na osnovu kosinusne sličnosti
        #recommendations = df['Movie Name'].iloc[similarities[idx].argsort()[::-1]][0:15]
        
        recommendations = df.loc[top_indices, 'Movie Name']
        #print(f'Redovi df-a: {df.shape[0]}, Kolone df-a: {df.shape[1]}')
        #print(recommendations)
        # Sortiraj preporuke na temelju kosinusne sličnosti
        recommendations = df['Movie Name'].iloc[similarities[idx].argsort()[::-1]][0:500]

        # Stvori rečnik preporuka
        movie_recommendations = {rec: [df['ID'].iloc[rec], df['Movie Name'].iloc[rec]] for rec in
                                 recommendations.index}

        return pd.DataFrame(movie_recommendations).transpose().iloc[1:11]

    except:
        return None



def get_recommendations(title, similarities):
    recommendations = build_recommendations(title, similarities)
    if recommendations is None:
        return recommendations
    else:
        recommendations.rename(columns={0: 'tconst', 1: 'title'}, inplace=True)
        recommendations.reset_index(drop=True, inplace=True)
        recommendations['urls'] = [f'https://www.imdb.com/title/{title_id}/' for title_id in recommendations['tconst']]
        return recommendations.drop('tconst', axis=1)


def get_movie_data():
    return titles_list


def evaluate_model_2(title, similarities):
    # ground truth - neke od relevatninih gotovih preporuka
    ground_truth = df['Movie Name'][:10].tolist()

    # pridobijemo preporuke
    recommendations = build_recommendations(title, similarities)

    if recommendations is None:
        return None

    recommended_movies = recommendations['Movie Name'].tolist()

    true_positives = len(set(recommended_movies).intersection(ground_truth))
    precision = true_positives / len(recommended_movies) if len(recommended_movies) > 0 else 0
    recall = true_positives / len(ground_truth)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1_score:.2f}')

    return precision, recall, f1_score



def calculate_average_precision(actual, predicted):
    precision_at_k = 0
    #num_correct_predictions = 0

    relevant_at_k = sum(item in actual for item in predicted)
    
    # Calculate Precision at K
    precision_at_k = relevant_at_k / len(actual)
    

    if not actual:
        return 0
    
    print(f'Num of correct predictions: {relevant_at_k} / 10')

    return precision_at_k


def evaluate_model_map(titles, similarities, k=10):
    try:
        # ground truth - relevatnih gotvoih 10 preporucenih filmova
        #ground_truth = set(df['Movie Name'][:10].tolist())
        i = 0
        df3 = pd.read_csv('Stvarne_preporuke.csv')
        map_score_sum = 0
        
        for title in titles:
            
            i += 1
            indeks_reda = df3.index[df3['Naziv filma'] == title].tolist()[0]
            #print(f'Indeks reda {indeks_reda}')
            
            import json
            from ast import literal_eval
            
            # Zamena dvostrukih navodnika jednostrukim

            # Formatiranje stringa kao listu
            ground_truth = literal_eval(df3.at[indeks_reda, 'Preporučeni filmovi'])
            #ground_truth = [film.strip(" '") for film in string_vrednost.strip('[]').split(',')]
            
            # preporuke
            recommendations = build_recommendations(title, similarities)
            
            if recommendations is None:
                return None

            recommendations.to_csv("Preporuke.csv",index=False)
            #print(recommendations.columns)
            
            #recommended_movies = recommendations['1'][:k].tolist()
            recommended_movies = recommendations.iloc[:k, 1].tolist()
            
            print(f'Ground truth: {ground_truth}')
            print(f'Recommended movies: {recommended_movies}')
            
            #print(ground_truth)
            #print(recommended_movies)
            # izracunamo Mean Average Precision
            map_score = calculate_average_precision(ground_truth, recommended_movies)

            map_score_sum  += map_score
            # evaluation metric
            print(f'Mean Average Precision at {k} (K) for "{title}": {map_score:.2f}')
            print('\n\n')
            
            

        return map_score_sum/i

    except Exception as e:
        print(f'An error occurred: {e}')
        return None


import random



from sklearn.metrics import mean_squared_error
import numpy as np

df_inic = pd.read_csv('data/Top_10000_Movies_IMDb.csv')
#print(f'IZBACIO GF? {df_inic.shape[0]}')
            
df2 = PRED.preprocess_data(df_inic)
            
X = df2.drop(['ID', 'Movie Name', 'Rating', 'Plot', 'Link', 'Genre'], axis=1)



df_novi = pd.read_csv('data/movie_metadata.csv')

df_novi = df_novi.dropna()

X3 = df_novi.drop(['color','director_name','actor_2_name','genres','actor_1_name','movie_title','actor_3_name',
                 'plot_keywords','movie_imdb_link',
                 'language','country','content_rating'], axis=1)

df_novi['movie_title'] = df_novi['movie_title'].str.rstrip()
lista_filmova = df_novi['movie_title'].tolist()
#print(f'Broj filmova:{len(lista_filmova)}')
X2 = df_novi.drop(['color','director_name','actor_2_name','genres','actor_1_name','movie_title','actor_3_name',
                 'plot_keywords','movie_imdb_link',
                 'language','country','imdb_score','content_rating'], axis=1)

null_values = X2.isnull().sum()
#print("Null vrednosti po kolonama:")
#print(null_values)
#print(X.columns)
             
#rfr_model = PRED.random_forest_regression_model(df2)
rfr_model = PRED.random_forest_regression_model(df_novi)

linear_model = PRED.linear_regression_model(df_novi) 
  
lr_model = PRED.lasso_regression_model(df_novi)

gbr_model = PRED.gradient_boosting_regression_model(df_novi)



def main():
    
        
    total_mae = 0
    total_mae2 = 0
    total_mae3 = 0
    total_mae4 = 0
    count_below_03 = 0
    count_below_05 = 0
    count_below_075 = 0
    num_samples = 100
    lista_mae = []

    for _ in range(num_samples):
        # Odabir nasumičnog filma iz liste
        user_input = random.choice(lista_filmova)

        # Predviđanje ocene filma
        for_print, predicted_rating, actual_rating = PRED.predict_movie_rating2(user_input, rfr_model, df_novi, X2.columns)
        for_print2, predicted_rating2, actual_rating2 = PRED.predict_movie_rating2(user_input, gbr_model, df_novi, X2.columns)
        for_print3, predicted_rating3, actual_rating3 = PRED.predict_movie_rating2(user_input, lr_model, df_novi, X2.columns)
        for_print4, predicted_rating4, actual_rating4 = PRED.predict_movie_rating2(user_input, linear_model, df_novi, X2.columns)

        # Računanje MAE
        mae = abs(predicted_rating - actual_rating)
        lista_mae.append(mae)
        total_mae += mae
        
        mae2 = abs(predicted_rating2 - actual_rating2)
        total_mae2 += mae2
        
        mae3 = abs(predicted_rating3 - actual_rating3)
        total_mae3 += mae3

        mae4 = abs(predicted_rating4 - actual_rating4)
        total_mae4 += mae4
        
        # Provera uslova za statistiku
        if mae < 0.3:
            count_below_03 += 1
        if mae < 0.5:
            count_below_05 += 1
        if mae < 0.75:
            count_below_075 += 1
            
    avg_mae = total_mae / num_samples
    avg_mae2 = total_mae2 / num_samples
    avg_mae3 = total_mae2 / num_samples
    avg_mae4 = total_mae4 / num_samples


    models = ['Random Forest', 'Gradient Boosting', 'Lasso Regression', 'Linear Regression']

    avg_maes = [avg_mae, avg_mae2, avg_mae3, avg_mae4]

    import matplotlib.pyplot as plt
    
    plt.bar(models, avg_maes, color=['blue', 'green', 'red', 'purple'])
    plt.xlabel('Regression models - Mean Abosolute Error score')
    plt.ylabel('Average MAE value')
    plt.title('Average MAE for Different Regression Models')
    plt.show()


    # Računanje procentualne statistike
    percentage_below_03 = (count_below_03 / num_samples) * 100
    percentage_below_05 = (count_below_05 / num_samples) * 100
    percentage_below_075 = (count_below_075 / num_samples) * 100


    print('\n\n')
    print('-----------------Model Evaluation for Random Movies RFR Model-----------------')
    print(f'Average MAE: {avg_mae:.4f}')
    print(f'Percentage of movies with MAE below 0.3: {percentage_below_03:.2f}%')
    print(f'Percentage of movies with MAE below 0.5: {percentage_below_05:.2f}%')
    print(f'Percentage of movies with MAE below 0.75: {percentage_below_075:.2f}%')
    print('\n\n')


    # Sortiranje predicted vrednosti
    sorted_predictions = np.sort(lista_mae)

    fig, ax = plt.subplots(figsize=(12, 6))

    # plotovanje svih mae
    ax.scatter(range(1, num_samples + 1), sorted_predictions, color='darkblue', alpha=0.9, s=4, label='MAE values')

    # avg mae je crvena tacka
    ax.plot(num_samples / 2, avg_mae, marker='o', markersize=5, color='red', label='Average MAE')

    ax.text(num_samples / 2, 0.01, f'Avg MAE: {avg_mae:.4f}', color='red', ha='center', va='bottom')

        # dodatan info
    ax.text(num_samples / 2, 0.3, f'Below 0.3: {percentage_below_03:.2f}%', color='green', ha='center', va='bottom')
    ax.text(num_samples / 2, 0.5, f'Below 0.5: {percentage_below_05:.2f}%', color='orange', ha='center', va='bottom')
    ax.text(num_samples / 2, 0.75, f'Below 0.75: {percentage_below_075:.2f}%', color='red', ha='center', va='bottom')

        # labele i title
    ax.set_xlabel('Samples - 100 Random Movies (Sorted)')
    ax.set_ylabel('MAE')
    ax.set_title('Random Forest Regression Model MAE Values')
    ax.legend()

        
    plt.show()

    
    # Dobijamo  matricu sličnosti
    similarities, matrica = calculate_cosine_similarity(df)
    #np.savetxt('kosinusne_slicnosti.txt', similarities , fmt='%d')

    #print(similarities)

    np.savetxt('kosinusne_slicnosti.txt', similarities , fmt='%.2f')
    
    titles = ['The Godfather',
        'Avatar',
        'The Lord of the Rings: The Return of the King',
        'Mad Max',
        'The Conjuring',
        'Harry Potter and the Order of the Phoenix',
        'Indiana Jones and the Temple of Doom',
        'The Hunger Games: Catching Fire',
        'Spider-Man: Into the Spider-Verse',
        'The Dark Knight Rises',
        'Top Gun: Maverick',
        'Guardians of the Galaxy Vol. 3',
        'The Terminator',
        'Pirates of the Caribbean: The Curse of the Black Pearl',
        'Dead Poets Society',
        'The Exorcist',
        'How to Train Your Dragon',
        'Underground',
        'Donnie Darko',
        'Nausicaä of the Valley of the Wind']
     
     # TESTIRANJE KVALITETA PREPORUKE NA NEKOM TESTU SKUPU (SUBJEKTIVNO, ALI UVEK PROSECNO PREPORUCI BAR 2/10)       
    average_precision_at_k = evaluate_model_map(titles, similarities)
    
    df_sim = pd.read_csv('glumci_reziseri_prep.csv')
    df_sim = preprocess_data_for_rating(df_sim)
    your_matrix = df_sim.values
    
    
    #percentage_of_correct = average_precision_at_k * 100
            
    print(f'Average precision at K for training set (using Stars/Directors) equals: {average_precision_at_k}')
    print('\n\n')

    print("Welcome to the Movie Recommendation and Rating Prediction System!")
    print('\n\n')
    lista_odgovora = []
    lista_odgovora2 =[]

# Funkcija koja pita korisnika da li želi koristiti određeni atribut i dodaje odgovor u listu
    def pitaj_i_dodaj(atribut):
        odgovor = input(f"Želite li da koristite {atribut} kao faktor u preporuci filmova? (y/n): ").lower()
        while odgovor not in ['y', 'n']:
            print("Molim vas unesite validan odgovor (y/n).")
            odgovor = input(f"Želite li koristiti atribut {atribut}? (y/n): ").lower()
        
        lista_odgovora.append((atribut, odgovor))
        lista_odgovora2.append(odgovor)

    # Pitanje za svaki od atributa
    pitaj_i_dodaj("Glumce/Režisere")
    pitaj_i_dodaj("Radnju")
    pitaj_i_dodaj("Žanr")

    # Ispisivanje liste odgovora
    print("Lista odgovora:")
    for atribut, odgovor in lista_odgovora:
        print(f"{atribut}: {odgovor}")
        
    similarities2, matrica2 = calculate_cosine_similarity(df, lista_odgovora2)
        
    while True:
        try:
            user_input = input("Enter the title of a movie: ")
            idx=process.extractOne(user_input, df_novi['movie_title'])[2]
            print(idx)
            
            user_input = df_novi['movie_title'][idx]
            print(user_input)
            index = df[df['Movie Name'] == user_input].index[0]
           # print(index)
            top_similar_rows = find_similar_rows(your_matrix, index)
            if not user_input:
                print("Title cannot be empty. Please enter a valid movie title.")
                continue
            
             # Print column names of 'imdb' DataFrame for debugging
            #print("Columns in imdb DataFrame:")
            #print(imdb.columns)
            
            recommendations = get_recommendations(user_input, similarities2)
            import seaborn as sns
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10,7))
            sns.heatmap(X3.corr(), annot=True, mask = False, annot_kws={"size": 7})
            print(plt.show())
            
            rfr_model2 = PRED.random_forest_regression_model(df_novi, user_input)
            linear_model2 = PRED.linear_regression_model(df_novi) 
            lr_model2 = PRED.lasso_regression_model(df_novi)
            gbr_model2 = PRED.gradient_boosting_regression_model(df_novi)
            
            for_print, predicted_rating, actual_rating = PRED.predict_movie_rating2(user_input, rfr_model2, df_novi, X2.columns)
            
            for_print2, predicted_rating2, actual_rating2 = PRED.predict_movie_rating2(user_input, lr_model2, df_novi, X2.columns)
            
            for_print3, predicted_rating3, actual_rating3 = PRED.predict_movie_rating2(user_input, gbr_model2, df_novi, X2.columns)
            
            for_print4, predicted_rating4, actual_rating4 = PRED.predict_movie_rating2(user_input, linear_model2, df_novi, X2.columns)
            
            
            if recommendations is not None and predicted_rating is not None and actual_rating is not None:
                print("\nRecommended Movies based on Cosine Similarity Matrix Model:")
                print(recommendations)
                print('\n')
                print("\nRecommended Movies by KNN Model (based on Eucliedan distance):")
                recommender(user_input, modelKNN(),10, matrica2)
                print('\n')
                print('\n')
                print(f"Top 10 most similar rows to row {idx} are: {top_similar_rows}")
                print('Recommended Movies by Most Similar Vector Model:')
                for i in top_similar_rows:
                    print(df['Movie Name'][i])
                print('\n')
                print('Random Forest Regression Model:')
                print(for_print)
                print('\n')
                print('Lasso Regression Model:')
                print(for_print2)
                print('\n')
                print('Gradient Boosting Regression Model:')
                print(for_print3)
                print('\n')
                print('Linear Regression Model:')
                print(for_print4)
                print('\n')
                
            else:
                print("Movie not found in the dataset. Please try another title.")
                
        except Exception as e:
            print(f"An error occurred: {e}")
        
        user_choice = input("\nDo you want to search for another movie? (y/n): ").lower()
        if user_choice != 'y':
            print("Exiting the Movie and Rating Prediction Recommendation System. Thank you!")
            break

if __name__ == "__main__":
    main()
