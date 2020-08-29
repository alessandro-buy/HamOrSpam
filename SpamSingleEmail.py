#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Print a string indicating directory')
parser.add_argument('-d', '--directory', metavar='', nargs = '+', help='a path for the function')
args = parser.parse_args()

import seaborn as sns
sns.set(style = "whitegrid",
        color_codes = True,
        font_scale = 1.5)

def spamOrHam(txt_file):

    """Some common utilities for classwork and homework in Berkeley's Data100."""
    def head(filename, lines=5):
        from itertools import islice
        with open(filename, "r") as f:
            return list(islice(f, lines))
    def fetch_and_cache(data_url, file, data_dir="data", force=False):
        import requests
        from pathlib import Path
        data_dir = Path(data_dir)
        data_dir.mkdir(exist_ok=True)
        file_path = data_dir/Path(file)
        if force and file_path.exists():
            file_path.unlink()
        if force or not file_path.exists():
            resp = requests.get(data_url, stream=True)
            file_size = int(resp.headers.get('content-length', 0))
            step = 40
            chunk_size = file_size//step
            with file_path.open('wb') as f:
                for chunk in resp.iter_content(chunk_size): # write file in chunks
                    f.write(chunk)
                    step -= 1
                    print('[' + '#'*(41 - step) + (step)*' ' + ']\r', end='')
            print(f"\nDownloaded {data_url.split('/')[-1]}!")
        else:
            import time
            time_downloaded = time.ctime(file_path.stat().st_ctime)
            print("Using version already downloaded:", time_downloaded)
        return file_path
    def line_count(file):
        with open(file, "r") as f:
            return sum(1 for line in f)
    def fetch_and_cache_gdrive(gdrive_id, file, data_dir="data", force=False):
        import requests
        from pathlib import Path

        data_dir = Path(data_dir)
        data_dir.mkdir(exist_ok=True)
        file_path = data_dir/Path(file)
        if force and file_path.exists():
            file_path.unlink()
        if force or not file_path.exists():
            download_file_from_google_drive(gdrive_id, file_path)
        else:
            import time
            time_downloaded = time.ctime(file_path.stat().st_ctime)
        return file_path
    def download_file_from_google_drive(id, destination):
        import requests
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params = { 'id' : id }, stream = True)
        token = get_confirm_token(response)
        if token:
            params = { 'id' : id, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)
        save_response_content(response, destination)
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None
    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            print("Downloading, this may take a few minutes.")
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
    fetch_and_cache_gdrive('1SCASpLZFKCp2zek-toR3xeKX3DZnBSyp', 'train.csv')
    fetch_and_cache_gdrive('1ZDFo9OTF96B5GP2Nzn8P8-AL7CTQXmC0', 'test.csv')


    original_training_data = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    original_training_data['email'] = original_training_data['email'].str.lower()
    test['email'] = test['email'].str.lower()

    from sklearn.model_selection import train_test_split
    train, val = train_test_split(original_training_data, test_size=0.1, random_state=42)

    def words_in_texts(words, texts):
        indicator_array = []
        for i in texts:
            current = []
            for j in words:
                if (j in i):
                    current.append(1)
                else:
                    current.append(0)
            indicator_array.append(current)
            current = []
        return indicator_array


    dir = str(txt_file[0])
    x = []
    file_in = open(dir, 'r')
    for y in file_in.read().split('\n'):
        x.append(y)
    email = ''.join(x)

    from sklearn.linear_model import LogisticRegression

    popular = np.array(pd.Series(''.join(train['email']).lower().split()).value_counts().head(100).index)

    def process_data_fm(data):
        X = pd.DataFrame(words_in_texts(popular, data['email']))
        X.columns = popular
        y = data['spam']
        return X, y


    final_model = LogisticRegression(penalty = 'l1', dual=False,max_iter=110, solver='liblinear')
    new_data = pd.read_csv('data/train.csv')

    train, test = train_test_split(new_data, test_size=0.2)
    x_train, y_train = process_data_fm(train)
    final_model.fit(x_train, y_train)

    def process(data):
        results = []
        for i in popular:
            if i in data:
                results.append(1)
            else:
                results.append(0)
        ret = []
        ret.append(results)
        ret = np.array(ret)
        X = pd.DataFrame(ret, columns = popular)
        return X
    data = [[email]]
    df = pd.DataFrame(data, columns = ['email'])
    classi = final_model.predict_proba(process(email))
    if classi[0][0] < .65:
        ans = 'Spam'
    else: ans = 'Ham'
    return ans


#spamOrHam('/Users/Alessandro/Desktop/sample_email 2.txt')

if __name__ == '__main__':
    print(spamOrHam(args.directory))
