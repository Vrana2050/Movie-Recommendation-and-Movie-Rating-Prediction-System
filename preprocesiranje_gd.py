import pandas as pd
import matplotlib.pyplot as plt

# NE POKRETATI, ISKORISCENO SAMO ZA PREPROCESIRANJE PODATAKA !!!
def allStarDirectors(df):
    all_directors = []
    all_stars = []

    for index, row in df.iterrows():
        directors = eval(row['Directors'])
        stars = eval(row['Stars'])
        all_directors.extend(directors)
        all_stars.extend(stars)

    # Count occurrences before making them unique
    # director_counts = {director: all_directors.count(director) for director in set(all_directors)}
    # star_counts = {star: all_stars.count(star) for star in set(all_stars)}

    unique_directors = list(set(all_directors))
    unique_stars = list(set(all_stars))

    # Count occurrences after making them unique
    unique_director_counts = {director: all_directors.count(director) for director in unique_directors}
    unique_star_counts = {star: all_stars.count(star) for star in unique_stars}
    
    top_directors = sorted(unique_director_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    director_names, director_counts = zip(*top_directors)

    plt.figure(figsize=(10, 5))
    plt.bar(director_names, director_counts, color='blue')
    plt.title('Top 5 Directors')
    plt.xlabel('Directors')
    plt.ylabel('Number of Appearances')
    #plt.show()

    # Plot the top 5 stars
    top_stars = sorted(unique_star_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    star_names, star_counts = zip(*top_stars)

    plt.figure(figsize=(10, 5))
    plt.bar(star_names, star_counts, color='orange')
    plt.title('Top 5 Stars')
    plt.xlabel('Stars')
    plt.ylabel('Number of Appearances')
    #plt.show()

    print("Number of directors before removing duplicates:", len(all_directors))
    print("Number of unique directors:", len(unique_directors))
    print("Number of directors that appeared more than 5 times:", len([count for count in unique_director_counts.values() if count > 5]))

    print("\nNumber of stars before removing duplicates:", len(all_stars))
    print("Number of unique stars:", len(unique_stars))
    print("Number of stars that appeared more than 5 times:", len([count for count in unique_star_counts.values() if count > 5]))

    # Filter directors and stars that appeared more than 5 times
    filtered_directors = [director for director in unique_directors if unique_director_counts[director] > 5]
    filtered_stars = [star for star in unique_stars if unique_star_counts[star] > 5]

    # Combine and remove duplicates using set
    combined_unique_values = list(set(filtered_directors + filtered_stars))

    print(len(combined_unique_values))
    return combined_unique_values

# Example usage:
# result_combined_unique_values = allStarDirectors(your_dataframe)

# Example usage:
# result_directors, result_stars = allStarDirectors(your_dataframe)

def addStarDirecorCols(df):
    unique_directors_stars = allStarDirectors(df)
    
    # Kreiranje DataFrame-a sa svim novim kolonama
    new_columns = pd.DataFrame(columns=unique_directors_stars)

    # Kopiranje originalnog DataFrame-a pre spajanja
    df = df.copy()

    # Spajanje svih novih kolona sa kopijom DataFrame-a
    df = pd.concat([df, new_columns], axis=1)

    i = 0
    # Popunjavanje vrednosti u novim kolonama
    for value in unique_directors_stars:
        print(i)
        df[value] = df.apply(lambda row: 1 if value in eval(row['Directors']) or value in eval(row['Stars']) else 0, axis=1)
        i += 1

    return df


df = pd.read_csv('data/Top_10000_Movies_IMDb.csv')
u1 = allStarDirectors(df)
df = addStarDirecorCols(df)

#df.to_csv("DODATNO.csv",index=False)
#print(df.head())
