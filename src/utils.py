
import re
import json

orig_data_dir = '../OrigData/spotify_million_playlist_dataset/train_data'
orig_challenge_data_dir = '../OrigData/spotify_million_playlist_dataset_challenge'




test_fullpaths  = '../OrigData/spotify_million_playlist_dataset/test_data'
valid_fullpaths = '../OrigData/spotify_million_playlist_dataset/valid_data'


test_data_dir  = '../test_data'
valid_data_dir = '../valid_data'
train_data_dir = '../training_data'

train_json     = train_data_dir+'/train'
test_json      = test_data_dir+'/test'
valid_json     = valid_data_dir+'/valid'



MAX_TITLE_LEN = 20
chars = list('''abcdefghijklmnopqrstuvwxyz<>+-1234567890''')
char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
NUM_CHAR = len(chars)



def get_training_set_info():
    with open(train_json) as data_file:
        train = json.load(data_file)

    track_uri_to_id  = train['track_uri_to_id']
    artist_uri_to_id = train['artist_uri_to_id']
    all_track_uris   = set(train['all_track_uris'])
    return track_uri_to_id, artist_uri_to_id, all_track_uris
    


def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def title_to_indices(title, max_len):
    indices = []
    for char in title:
        if len(indices) >= max_len: break
        if char in char_to_idx:
            idx = char_to_idx[char]
            indices.append(idx)
            
    while len(indices) <= max_len: indices.append(-1)
        
    return indices