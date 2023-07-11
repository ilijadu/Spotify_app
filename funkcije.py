#%%
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import lyricsgenius
from bs4 import BeautifulSoup
from selenium import webdriver
from string import punctuation
import nltk
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from random import seed
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import re
from sklearn.metrics import silhouette_score
from sklearn.decomposition import NMF
import requests


stopword = nltk.corpus.stopwords.words('english')
new_stop_words = ['ooh','yeah','hey','whoa','woah', 'ohh', 'was', 'mmm', 'oooh','yah','yeh','mmm', 'hmm','deh',
                  'doh','jah','wa', 'back','said','one', 'come','thing','get','would','like',
                  'know','go','let','cause','oh','could','got','uh','ah','lyric','outro']

stopword.extend(new_stop_words)
#%%
SPOTIPY_CLIENT_ID='c7f3eaa09bd044ffa1c2ca5cdc088aec'
SPOTIPY_CLIENT_SECRET='19f099110ebc43b9808017c6f6cee192'
SPOTIPY_REDIRECT_URI='http://localhost:8888/callback'

# Setting up the Spotify credentials and authentication scopes

scope = ['user-library-read','playlist-modify-public']

# Authenticate the user
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET, redirect_uri=SPOTIPY_REDIRECT_URI, scope=scope))

# Setting up the Genius API
genius = lyricsgenius.Genius('n-NvJGmw_MU_zTf4eORCJu-IUMhQwl9rCwfBWcdIUppRMjd7tp5YTJ2yh5GfJcSa')
#%%
def get_playlist(x:str):
    # Set up the playlist URI or URL
    playlist_uri = x

    # Get the tracks in the playlist
    results = sp.playlist_tracks(playlist_uri)
    tracks = results['items']

    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks
#%%
def get_playlist_name(x:str):
    playlist = sp.playlist(x)
    return (playlist['name'])
#%%
def get_playlist_creator(x:str):

    playlist = sp.playlist(x)
    return playlist['owner']['display_name']
#%%
# removing excess info from the track name
def remove_excess_info (x):
    error_words_with_feature=["(with","(feat."]
    error_words_with_feature_2=["[with","[feat."]
    error_words_with_hyphen =["remaster","bonus track","recorded at"," feat. ","- edit", "- video mix","remix"]

    for word in error_words_with_feature_2:
        if word in x.lower():
            x = x.rsplit(" [")[0]
            break
    for word in error_words_with_feature:
        if word in x.lower():
            x = x.rsplit(" (")[0]
            break
    print(x)
    for word in error_words_with_hyphen:
        if word in x.lower():
            x = x.rsplit(" -")[0]

    if " feat." in x:
        x = x.rsplit(" feat.")[0]

    return x


# Define a function to retrieve the lyrics for a track
def get_lyrics(track_name, artist_name):
    try:

        song = genius.search_song(track_name, artist_name)
        lyrics = song.lyrics
        return lyrics
    except:
        return 'No lyric found'
#%%
# Getting the info for the tracks for playlist
def get_playlist_info(tracks):
    track_info = []
    for track in tracks:
        track_name = track['track']['name']
        artist_name = track['track']['artists'][0]['name']
        # Retrieving the info about audio features of the track
        audio_features = sp.audio_features(track['track']['id'])[0]
        # Retrieving the lyrics via lyricsgenius library with the function get_lyrics
        info = {'Track Name': track_name, 'Artist Name': artist_name,"Lyrics":get_lyrics(track_name, artist_name)}
        info.update(audio_features)
        track_info.append(info)

    # Create a pandas dataframe with the track information
    df = pd.DataFrame(track_info)
    return df
#%%
def get_soup_selenium(url: str) -> BeautifulSoup:

    driver = webdriver.Chrome()
    driver.get(url)
    return BeautifulSoup(driver.page_source, 'html.parser')

def get_soup_selenium_request(url: str) -> BeautifulSoup:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36"
    }

    response = requests.get(url, allow_redirects=False,headers=headers)
    response_text = response.text
    return BeautifulSoup(response_text, 'html.parser')
#%%
def remove_rows(text):
    text=text.replace("\n"," ")
    return text

def replace_signs(x):
    signs=punctuation + " " + "’"
    x =[c for c in x.lower() if not c in signs]
    x="".join(x)
    return x.strip()


def remove_the(x):
    if "The" in x[0:3]:
        return x.replace("The","").strip()
    if "the" in x[0:3]:
        return x.replace("the","").strip()
    return x

#%%
def get_lyrics_selenium(track_name, artist_name):
    try:
        track_name=remove_excess_info(track_name)

        track_name_lc=[w.strip() for w in track_name.lower().split("(") if (not "feat" in w) ]

        if len(track_name_lc)>=1:
            track_name_lc="".join(track_name_lc)
        elif len(track_name_lc)==0:
            track_name_lc=track_name.lower().split(" feat.",maxsplit=1)[0]

        artist_name=remove_the(artist_name)

        url = "https://www.azlyrics.com/lyrics/"+replace_signs(artist_name)+"/"+replace_signs(track_name_lc)+".html"
        print(url)
        soup = get_soup_selenium(url)

        excess="<!-- Usage of azlyrics.com content by any third-party lyrics provider is prohibited by our licensing agreement. Sorry about that. -->"
        text =str(soup.find("div",{"class":"col-xs-12 col-lg-8 text-center"}).find("div",{"class":None,"style":None}))
        text= text.replace("<br/>","").replace(excess,"").replace("<div>","").replace("</div>","").replace("’","'").strip()


        return remove_rows(text.lower())
    except:
        return 'No lyric found (selenium)'

def get_lyrics_selenium_request(track_name, artist_name):
        try:
            track_name=remove_excess_info(track_name)

            track_name_lc=[w.strip() for w in track_name.lower().split("(") if (not "feat" in w) ]

            if len(track_name_lc)>=1:
                track_name_lc="".join(track_name_lc)
            elif len(track_name_lc)==0:
                track_name_lc=track_name.lower().split(" feat.",maxsplit=1)[0]

            artist_name=remove_the(artist_name)

            url = "https://www.azlyrics.com/lyrics/"+replace_signs(artist_name)+"/"+replace_signs(track_name_lc)+".html"
            print(url)
            soup = get_soup_selenium_request(url)

            excess="<!-- Usage of azlyrics.com content by any third-party lyrics provider is prohibited by our licensing agreement. Sorry about that. -->"
            text =str(soup.find("div",{"class":"col-xs-12 col-lg-8 text-center"}).find("div",{"class":None,"style":None}))
            text= text.replace("<br/>","").replace(excess,"").replace("<div>","").replace("</div>","").replace("’","'").strip()
            return remove_rows(text.lower())
        except:
            return 'No lyric found (selenium)'
#%%
def clean_lyrics(artist_name,track_name,lyric):

    track_name = remove_excess_info(track_name)

    #lyrics that is retrieved contains some unnecessary information that has to be removed
    if "lyrics[" in lyric:
        lyric = lyric.split("]",maxsplit=1)
    else:
        lyric = lyric.split(track_name+" lyrics",maxsplit=1)

    #if lyrics is retrieved, it would be split into two parts so that the second part represents the real lyrics;
    # if it's not, the retrieved text is either "No lyric found" or it is some unuseful text, and it wasn't split in the previous code.
    # Because of that web scraping is needed to retrieve the lyrics
    if len(lyric)==2:
        lyric = lyric[1]
    else:
        return get_lyrics_selenium(track_name,artist_name)

    lyric=lyric.replace("’","'").replace("embed","").replace("\'","'")

    result = re.sub('\[[^\]]*\W+[^\]]*\]', '', lyric)

    if result[len(result)-1].isnumeric():
        result = result.replace(result[len(result)-1],"")

    return remove_rows(result)

def clean_lyrics_request(artist_name,track_name,lyric):

    track_name = remove_excess_info(track_name)

    if "lyrics[" in lyric:
        lyric = lyric.split("]",maxsplit=1)
    else:
        lyric = lyric.split(track_name+" lyrics",maxsplit=1)

    if len(lyric)==2:
        lyric = lyric[1]
    else:
        return get_lyrics_selenium_request(track_name,artist_name)

    lyric=lyric.replace("’","'").replace("embed","").replace("\'","'")

    result = re.sub('\[[^\]]*\W+[^\]]*\]', '', lyric)

    if result[len(result)-1].isnumeric():
        result = result.replace(result[len(result)-1],"")
    return remove_rows(result)
#%%
def no_lyrics_songs(df):

    indexes = df.loc[df.clean_lyrics == "No lyric found (selenium)",:].index.tolist()
    df.drop(indexes,inplace=True)
    df.reset_index(inplace=True)
#%%
def noramlize_feature(x):
    if sum(x)==0:
        return x
    else:
        return (x - min(x))/(max(x) - min(x))
#%%
def normed_songs(df):
    k_means_columns2=["danceability","loudness","acousticness","speechiness","instrumentalness","valence","tempo"]
    songs_normed = df[k_means_columns2].apply(lambda x: noramlize_feature(x))
    return songs_normed

#%%
eval_metrics= pd.DataFrame(columns=["k", "tot.within.ss"])
def calculating_number_of_clusters(songs_normed):
    silhouettes=dict()
    for k in range(2,11):
        seed(10)
        kmeans= KMeans(n_clusters=k,n_init=1000,max_iter=20)
        kmeans.fit(songs_normed)
        labels=kmeans.labels_

        eval_metrics.loc[len(eval_metrics.index)]=[k,kmeans.inertia_]
        silhouette=silhouette_score(songs_normed,labels,metric="euclidean",sample_size=1000)

        silhouettes.update({k:silhouette})

    broj = sorted(silhouettes,key=silhouettes.get,reverse=True)[0:3]

    return broj



#%%
def elbow_method(songs_normed):


    # for k in range(2,16):
    #     seed(10)
    #
    #     kmeans= KMeans(n_clusters=k,n_init=1000,max_iter=20)
    #     kmeans.fit(songs_normed)
    #     eval_metrics.loc[len(eval_metrics.index)]=[k,kmeans.inertia_]

    return eval_metrics


#%%
def kmeans_clustering(songs_normed, number_of_clusters):
    kmeans= KMeans(n_clusters=number_of_clusters,n_init=1000,max_iter=20)
    print(f"Klasterovano je u {number_of_clusters} klastera")
    kmeans.fit(songs_normed)
    kmeans.labels_
    return kmeans.labels_

#%%
def number_of_columns(x):
    topics=[]
    for i in range(1,x+1):
        topics.append("topic_"+str(i))
    return topics

def vectorization(df,number_of_topics):
    vectorizer=TfidfVectorizer(stop_words=stopword, min_df=0.1)
    tfidf= vectorizer.fit_transform(df["clean_lyrics"])
    nmf=NMF(n_components=number_of_topics)
    topic_values= nmf.fit_transform(tfidf)
    df=df.join(pd.DataFrame(topic_values,columns=number_of_columns(number_of_topics)))
    return df
#%%
def topicing(df,number_of_topics):
    df=vectorization(df,number_of_topics)
    for i in range(1,number_of_topics+1):
        column_name="topic_"+str(i)
        df.loc[df[column_name]>=0.09,i]=True
        df.loc[df[column_name]<0.09,i]=False

    return df
#%%
songs_per_cluster_topics=defaultdict(list)

def check_topic(row,number_of_topics):

    for i in range (1,number_of_topics+1):
        cluster_str=str(row['cluster']+1) + "_" +str(i)

        if row[i]:
            songs_per_cluster_topics[cluster_str].append(row['id'])
#%%
def create_all_playlists(number_of_clusters,number_of_topics):

    k = 0
    for i in range(1,number_of_clusters+1):
        for j in range(1,number_of_topics+1):
            user_id = sp.me()['id']

            playlist_from_dict=str(i)+"_"+str(j)
            track_ids = songs_per_cluster_topics[playlist_from_dict]
            if len(track_ids)>2:

                k=k+1
                playlist_name = f' APP My Playlist number {k}'
                playlist_description = f'A playlist created using Python and Spotipy. It is based on cluster {i}'

                # add tracks to playlist

                playlist = sp.user_playlist_create(user_id, playlist_name, public=True, description=playlist_description)

                sp.user_playlist_add_tracks(user_id, playlist['id'], track_ids)
    return k


#%%
def button_scan(x:str):

    global df_songs
    df_songs=get_playlist_info(get_playlist(x))
    df_songs['clean_lyrics']=df_songs.apply(lambda x: clean_lyrics(x['Artist Name'].lower(),x['Track Name'].lower(),x['Lyrics'].lower()),axis=1)
    df_without_lyrics = df_songs.loc[df_songs.clean_lyrics == "No lyric found (selenium)",:]
    no_lyrics_songs(df_songs)
    global songs_normed

    songs_normed=normed_songs(df_songs)
    broj = calculating_number_of_clusters(songs_normed)
    eval_metrics=elbow_method(songs_normed)

    return df_songs,df_without_lyrics,broj,eval_metrics

def organize_button(nClust, nTopics):

    df_songs["cluster"]=kmeans_clustering(songs_normed,nClust)
    df=topicing(df_songs,nTopics)
    df.apply(check_topic,axis=1,args={nTopics})
    return create_all_playlists(nClust,nTopics)

#%%
