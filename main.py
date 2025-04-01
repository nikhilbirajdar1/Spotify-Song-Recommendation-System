import threading
import webbrowser

import pandas as pd
import numpy as np
import json
import re
import sys
import itertools
import webbrowser

from tkinter import messagebox, ttk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import warnings
warnings.filterwarnings("ignore")



def main():
    client_id = '$YOUR_SPOTIFY_CLIENT_ID'
    client_secret= '$YOUR_SPOTIFY_CLIENT_SECRET'
    scope = 'user-library-read'


    ### Data Preprocesssing
    spotify_df = pd.read_csv('data/data.csv')
    data_w_genre = pd.read_csv('data/data_w_genres.csv')
    data_w_genre['genres_upd'] = data_w_genre['genres'].apply(lambda x: [re.sub(' ','_',i) for i in re.findall(r"'([^']*)'", x)])
    spotify_df['artists_upd_v1'] = spotify_df['artists'].apply(lambda x: re.findall(r"'([^']*)'", x))
    spotify_df[spotify_df['artists_upd_v1'].apply(lambda x: not x)].head(5)
    spotify_df['artists_upd_v2'] = spotify_df['artists'].apply(lambda x: re.findall('\"(.*?)\"',x))
    spotify_df['artists_upd'] = np.where(spotify_df['artists_upd_v1'].apply(lambda x: not x), spotify_df['artists_upd_v2'], spotify_df['artists_upd_v1'] )
    spotify_df['artists_song'] = spotify_df.apply(lambda row: row['artists_upd'][0]+row['name'],axis = 1)
    spotify_df['artists_song'] = spotify_df.apply(lambda row: row['artists_upd'][0]+row['name'],axis = 1)
    spotify_df.drop_duplicates('artists_song',inplace = True)
    artists_exploded = spotify_df[['artists_upd','id']].explode('artists_upd')
    artists_exploded_enriched = artists_exploded.merge(data_w_genre, how = 'left', left_on = 'artists_upd',right_on = 'artists')
    artists_exploded_enriched_nonnull = artists_exploded_enriched[~artists_exploded_enriched.genres_upd.isnull()]
    artists_genres_consolidated = artists_exploded_enriched_nonnull.groupby('id')['genres_upd'].apply(list).reset_index()
    artists_genres_consolidated['consolidates_genre_lists'] = artists_genres_consolidated['genres_upd'].apply(lambda x: list(set(list(itertools.chain.from_iterable(x)))))
    spotify_df = spotify_df.merge(artists_genres_consolidated[['id','consolidates_genre_lists']], on = 'id',how = 'left')

    ### Feature Engineering
    spotify_df['year'] = spotify_df['release_date'].apply(lambda x: x.split('-')[0])
    float_cols = spotify_df.dtypes[spotify_df.dtypes == 'float64'].index.values
    ohe_cols = 'popularity'
    spotify_df['popularity'].describe()
    spotify_df['popularity_red'] = spotify_df['popularity'].apply(lambda x: int(x/5))
    spotify_df['consolidates_genre_lists'] = spotify_df['consolidates_genre_lists'].apply(lambda d: d if isinstance(d, list) else [])


    # simple function to create OHE features
    # this gets passed later on
    def ohe_prep(df, column, new_name):
        """
        Create One Hot Encoded features of a specific column

        Parameters:
            df (pandas dataframe): Spotify Dataframe
            column (str): Column to be processed
            new_name (str): new column name to be used

        Returns:
            tf_df: One hot encoded features
        """
        tf_df = pd.get_dummies(df[column])
        feature_names = tf_df.columns
        tf_df.columns = [new_name + "|" + str(i) for i in feature_names]
        tf_df.reset_index(drop=True, inplace=True)
        return tf_df


    # function to build entire feature set
    def create_feature_set(df, float_cols):
        """
        Process spotify df to create a final set of features that will be used to generate recommendations

        Parameters:
            df (pandas dataframe): Spotify Dataframe
            float_cols (list(str)): List of float columns that will be scaled

        Returns:
            final: final set of features
        """
        # tfidf genre lists
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(df['consolidates_genre_lists'].apply(lambda x: " ".join(x)))
        genre_df = pd.DataFrame(tfidf_matrix.toarray())
        genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names_out()]
        genre_df.reset_index(drop=True, inplace=True)
        # explicity_ohe = ohe_prep(df, 'explicit','exp')
        year_ohe = ohe_prep(df, 'year', 'year') * 0.5
        popularity_ohe = ohe_prep(df, 'popularity_red', 'pop') * 0.15
        # scale float columns
        floats = df[float_cols].reset_index(drop=True)
        scaler = MinMaxScaler()
        floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns=floats.columns) * 0.2
        # concanenate all features
        final = pd.concat([genre_df, floats_scaled, popularity_ohe, year_ohe], axis=1)
        # add song id
        final['id'] = df['id'].values
        return final

    complete_feature_set = create_feature_set(spotify_df, float_cols=float_cols)


    ### Spotify API
    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    token = util.prompt_for_user_token(scope, client_id= client_id, client_secret=client_secret, redirect_uri='http://localhost:5173/callback')
    sp = spotipy.Spotify(auth=token)


    #gather playlist names.
    id_name = {}
    for i in sp.current_user_playlists()['items']:
        id_name[i['name']] = i['uri'].split(':')[2]


    def create_necessary_outputs(playlist_name, id_dic, df):
        """
        Pull songs from a specific playlist.

        Parameters:
            playlist_name (str): name of the playlist you'd like to pull from the spotify API
            id_dic (dic): dictionary that maps playlist_name to playlist_id
            df (pandas dataframe): spotify datafram

        Returns:
            playlist: all songs in the playlist THAT ARE AVAILABLE IN THE KAGGLE DATASET
        """
        # generate playlist dataframe
        playlist = pd.DataFrame()
        playlist_name = playlist_name
        for ix, i in enumerate(sp.playlist(id_dic[playlist_name])['tracks']['items']):
            # print(i['track']['artists'][0]['name'])
            playlist.loc[ix, 'artist'] = i['track']['artists'][0]['name']
            playlist.loc[ix, 'name'] = i['track']['name']
            playlist.loc[ix, 'id'] = i['track']['id']  # ['uri'].split(':')[2]
            playlist.loc[ix, 'url'] = i['track']['album']['images'][1]['url']
            playlist.loc[ix, 'date_added'] = i['added_at']

        playlist['date_added'] = pd.to_datetime(playlist['date_added'])
        playlist = playlist[playlist['id'].isin(df['id'].values)].sort_values('date_added', ascending=False)
        return playlist

    ##Getting the id from the playlist link
    playlist_link = link_text
    x = playlist_link.split("/")
    link_id = x[4]
    question_index = link_id.find("?")
    playlist_id = link_id[0: question_index]
    playlist_name= sp.playlist(playlist_id=playlist_id)["name"]
    id_name = {}
    id_name[playlist_name] = sp.playlist(playlist_id=playlist_id)['uri'].split(':')[2]
    str(playlist_name)

    user_playlist = create_necessary_outputs(f'{playlist_name}', id_name, spotify_df)

    ### Creating Playlist Vector

    def generate_playlist_feature(complete_feature_set, playlist_df, weight_factor):
        """
        Summarize a user's playlist into a single vector

        Parameters:
            complete_feature_set (pandas dataframe): Dataframe which includes all of the features for the spotify songs
            playlist_df (pandas dataframe): playlist dataframe
            weight_factor (float): float value that represents the recency bias. The larger the recency bias, the most priority recent songs get. Value should be close to 1.

        Returns:
            playlist_feature_set_weighted_final (pandas series): single feature that summarizes the playlist
            complete_feature_set_nonplaylist (pandas dataframe):
        """

        complete_feature_set_playlist = complete_feature_set[complete_feature_set['id'].isin(playlist_df['id'].values)]
        complete_feature_set_playlist = complete_feature_set_playlist.merge(playlist_df[['id', 'date_added']], on='id',how='inner')
        complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['id'].isin(playlist_df['id'].values)]

        playlist_feature_set = complete_feature_set_playlist.sort_values('date_added', ascending=False)

        most_recent_date = playlist_feature_set.iloc[0, -1]

        for ix, row in playlist_feature_set.iterrows():
            playlist_feature_set.loc[ix, 'months_from_recent'] = int(
                (most_recent_date.to_pydatetime() - row.iloc[-1].to_pydatetime()).days / 30)

        playlist_feature_set['weight'] = playlist_feature_set['months_from_recent'].apply(lambda x: weight_factor ** (-x))

        playlist_feature_set_weighted = playlist_feature_set.copy()
        playlist_feature_set_weighted.update(
            playlist_feature_set_weighted.iloc[:, :-4].mul(playlist_feature_set_weighted.weight, 0))
        playlist_feature_set_weighted_final = playlist_feature_set_weighted.iloc[:, :-4]
        return playlist_feature_set_weighted_final.sum(axis=0), complete_feature_set_nonplaylist
    try:
        complete_feature_set_playlist_vector, complete_feature_set_nonplaylist = generate_playlist_feature(complete_feature_set, user_playlist, 1.09)
    except IndexError:
        exitwindow = tk.Tk()
        exitwindow.title("Error!")
        exitwindow.geometry("250x250")
        exitwindow.iconbitmap("Images/Error.ico")
        x = (screen_width // 2) - (400 // 2)
        y = (screen_height // 2) - (500 // 2)
        exitwindow.geometry(f"250x250+{x}+{y}")
        exit_Label = Label(exitwindow, text="Sorry there was some error")
        exit_Label.place(anchor="center", relx=0.5, rely=0.5)

        def closeexitwindow():
            exitwindow.destroy()
            sys.exit()

        okbutton = Button(exitwindow, text="Ok",command=closeexitwindow, bg="White", fg="Black", width=4, height=2)
        okbutton.place(anchor="center", relx=0.5, rely=0.7)
        return

    def generate_playlist_recos(df, features, nonplaylist_features):
        """
        Pull songs from a specific playlist.

        Parameters:
            df (pandas dataframe): spotify dataframe
            features (pandas series): summarized playlist feature
            nonplaylist_features (pandas dataframe): feature set of songs that are not in the selected playlist

        Returns:
            non_playlist_df_top_40: Top 40 recommendations for that playlist
        """

        non_playlist_df = df[df['id'].isin(nonplaylist_features['id'].values)]
        non_playlist_df['sim'] = cosine_similarity(nonplaylist_features.drop('id', axis=1).values,features.values.reshape(1, -1))[:, 0]
        non_playlist_df_top_40 = non_playlist_df.sort_values('sim', ascending=False).head(40)
        non_playlist_df_top_40['url'] = non_playlist_df_top_40['id'].apply(
            lambda x: sp.track(x)['album']['images'][1]['url'])

        return non_playlist_df_top_40

    top40 = generate_playlist_recos(spotify_df, complete_feature_set_playlist_vector, complete_feature_set_nonplaylist)
    new_df = top40['name']
    new_df_uri = top40['id']
    new_list = new_df.values.tolist()
    new_list_uri = new_df_uri.values.tolist()
    global uri_list
    global song_list
    song_list = new_list
    uri_list = new_list_uri
    songwindow()



### GUI ###
def submit_text():
    global link_text

    entered_text = Link.get()
    link_text = entered_text
    try:
        backend_thread = threading.Thread(target=main())
        backend_thread.start()
    except:
        exitwindow = tk.Tk()
        exitwindow.title("Error!")
        exitwindow.geometry("350x350")
        exitwindow.iconbitmap("Images/Error.ico")
        x = (screen_width // 2) - (400 // 2)
        y = (screen_height // 2) - (500 // 2)
        exitwindow.geometry(f"350x350+{x}+{y}")
        exit_Label = Label(exitwindow, text="Error, Make sure you are connected with internet")
        exit_Label.place(anchor="center", relx=0.5, rely=0.5)

        def closeexitwindow():
            exitwindow.destroy()
            sys.exit()

        okbutton = Button(exitwindow, text="Ok",command=closeexitwindow, bg="White", fg="Black", width=4, height=2)
        okbutton.place(anchor="center", relx=0.5, rely=0.7)
        return



def songwindow():
    root1 = tk.Tk()
    root1.geometry("400x500")
    root1.title("Song List")
    x = (screen_width // 2) - (400 // 2)
    y = (screen_height // 2) - (500 // 2)
    root1.geometry(f"400x500+{x}+{y}")
    title_label = tk.Label(root1, text="Recommended Tracks Title")
    title_label.pack()
    listbox = tk.Listbox(root1,font=("Times New Roman", 10))
    listbox.pack()
    listbox.config(width=50, height=40,highlightthickness=0)

    def open_spotify(event):
        selected_index = listbox.curselection()
        if selected_index:
            selected_item = listbox.get(selected_index)
            song_id = uri_list[selected_index[0]]  # Assuming song_ids is a list of song IDs
            spotify_url = f"https://open.spotify.com/track/{song_id}"  # Construct the Spotify track URL
            webbrowser.open(spotify_url)
    for song_name in song_list:
        listbox.insert(tk.END, song_name)
    listbox.bind("<<ListboxSelect>>", open_spotify)

link_text = ''
root = tk.Tk()
root.title("Spotify Recommendation System")
root.iconbitmap("Images/Spotifyicon.ico")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = screen_width
window_height = screen_height
root.geometry(f"{window_width}x{window_height}")

background_image = Image.open("Images/Spotifybg.gif")
background_photo = ImageTk.PhotoImage(background_image)

background_label = tk.Label(root, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

Link = Entry(root, width=80)
Link.place(anchor='center', relx=0.5, rely=0.75,height=20)

submit_button = tk.Button(root, text="Search", command=submit_text, width=15, height=2, relief=tk.RAISED, bg="#007bff",
                          fg="white", font=("Times New Roman",12))
submit_button.place(anchor="center", relx=0.5, rely=0.85)

root.mainloop()
