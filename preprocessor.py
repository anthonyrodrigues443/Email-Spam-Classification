# Importing libraries
import time
initial_start = time.time()

import os
from email import policy
from email.parser import BytesParser
import pandas as pd
import numpy as np
import re
import pickle
from bs4 import BeautifulSoup

import nltk
nltk.data.path.append('/opt/render/nltk_data')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Lazy loading stopwords with fallback
def get_stopwords():
    try:
        return stopwords.words('english')
    except LookupError:
        nltk.download('stopwords', download_dir='/opt/render/nltk_data')
        return stopwords.words('english')

def get_lemmatizer():
    try:
        return WordNetLemmatizer()
    except LookupError:
        nltk.download('wordnet', download_dir='/opt/render/nltk_data')
        return WordNetLemmatizer()

stop_words = get_stopwords()
lem = WordNetLemmatizer()

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

end = round(time.time()-initial_start, 2)
print('✔️ Libraries imported \t\t', end)

start = time.time()
stop_words = stopwords.words('english')
lem = WordNetLemmatizer()

with open('feature_encoders/one_hot_encoder.pkl', 'rb') as pickle_file:
    ohe = pickle.load(pickle_file)

with open('scalers/rec_cnt_scaler.pkl', 'rb')as pickle_file:
    rec_cnt_scaler = pickle.load(pickle_file)
with open('scalers/sub_char_scaler.pkl', 'rb')as pickle_file:
    sub_char_scaler = pickle.load(pickle_file)
with open('scalers/content_char_scaler.pkl', 'rb')as pickle_file:
    content_char_scaler = pickle.load(pickle_file)

with open('vectorizors/tfidf_content.pkl', 'rb')as file:
    tfidf_content = pickle.load(file)
with open('vectorizors/tfidf_subject.pkl', 'rb')as file:
    tfidf_subject = pickle.load(file)

end = round(time.time()-start, 2)
print('✔️ Pickle files imported \t\t', end)


start = time.time()
model = tf.keras.models.load_model("dl_model/ann_model.h5")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
end = round(time.time()-start, 2)
print('✔️ DL model imported \t\t', end)

# Parse binary email file
def email_bytesparser(file, filepath=False):
    if filepath == False:
        parsed_email = BytesParser(policy=policy.default).parsebytes(file)
    else :
        with open(file, 'rb') as f:
            parsed_email = BytesParser(policy=policy.default).parse(f)
    return parsed_email

# html parser
def parse_text(text):
    try : 
        soup = BeautifulSoup(text, 'lxml')
        parsed_text = soup.text.strip()
    except Exception as ex :
        parsed_text = ""
    finally:
        parsed_text = re.sub(r'\s+', ' ', soup.text.strip())
    return parsed_text

# Extract required details
def extract_to_df(parsed_email):
    body = ""
    if parsed_email.is_multipart():
        for part in parsed_email.iter_parts():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                content_type = "plain"
                body = part.get_content()
                break
            elif content_type == "text/html":
                content_type = "html"
                body = part.get_content()
    else:
        content_type = parsed_email.get_content_type()
        body = parsed_email.get_content()

    recievers = parsed_email['To']
    subject = parsed_email['Subject']
    precedence = parsed_email['Precedence']
    parsed_email_text = parse_text(body)
    content = parsed_email_text
    list_unsub = parsed_email['List-Unsubscribe']
    replied_mail = parsed_email['Reply-To']
    list_sub = parsed_email['List-Subscribe']

    # Prepare the data as a list of one row
    data = pd.DataFrame([{
        'To': recievers,
        'Subject': subject,
        'Content-type': content_type,
        'Precendence': precedence,
        'Full_content': content,
        'List-unsubscribe': list_unsub,
        'List-subscribe': list_sub,
        'In-reply-to': replied_mail
    }])

    return data


# drop unwanted feature
def column_dropper(data, col):
    transformed_df = data.copy()
    transformed_df.drop(columns=col, inplace=True)
    return transformed_df


# Text cleaners
extra_space_remover = lambda x: re.sub(r'\s+', ' ', str(x)).strip() if pd.notna(x) else np.nan

def clean_text(text):
    if pd.isna(text):
        return np.nan
    else :
        # Replaces URLs in a string with a single space.
        url_pattern = re.compile(r'https?://\S+|www\.\S+|\b\S+\.(?:com|org|net|gov|edu|io|co|in|info)\b')
        no_links_text = url_pattern.sub(' ', text)

        # Removes all punctuations, numbers and symbols
        words = [word for word in no_links_text.split() if not ( '>' in word or '<' in word or '=' in word)]
        no_tags_text = ' '.join(words)
        cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', no_tags_text)
        if cleaned_text.strip() != '':
            return extra_space_remover(cleaned_text)
        else :
            return np.nan

# lemmatization and eliminating stopwords
def processing_text(text, stopwords=stop_words):
    words = text.split()
    words = [lem.lemmatize(word) for word in words if ((word not in stopwords) and (len(str(word)) > 1)) ]
    words = [word for word in words if len(word) > 1]
    return ' '.join(words)

def feature_engineering1(data):
    #Creating New bool feature 'Replied_mail'
    data['Replied_mail'] = np.where(pd.isna(data['In-reply-to']), 0, 1)
    data = column_dropper(data, 'In-reply-to')

    #Creating New numeric feature 'Recievers_count'
    data['Recievers_count'] = data['To'].apply(lambda x: len(str(x).split('@')) - 1)
    data = column_dropper(data, 'To')

    #Creating New bool feature 'Sub_Unsub_link'
    data['Sub_Unsub_link'] = np.where(
        (pd.notna(data['List-subscribe']) & pd.notna(data['List-unsubscribe'])), 1, 0)
    data = column_dropper(data, ['List-subscribe','List-unsubscribe'])
    
    # Cleaning Content-Type: feature
    def extract_cont_type(text):
        if pd.isna(text):
            return "none"
        elif 'html' in text :
            return 'html'
        elif 'plain' in text :
            return 'plain'
        else :
            return "none"
    
    data['Content-type'] = data['Content-type'].apply(extract_cont_type)
    
    # Cleaning text features
    data['Full_content'] = data['Full_content'].apply(clean_text)
    data['Subject'] = data['Subject'].apply(clean_text)


    # Creating new numeric features "sub_char" & "content_char"
    data['content_char'] = data['Full_content'].apply(len)
    data['sub_char'] = data['Subject'].apply(len)
    return data

# Encoding categorical features
def feature_encoding(data):
    encoded_content_type = ohe.transform(data[['Content-type']])
    encoded_df = pd.DataFrame(encoded_content_type, columns=ohe.get_feature_names_out(['Content-type']))

    data['Content_type_html'] = encoded_df['Content-type_html'].astype(int)
    data['Content_type_plain'] = encoded_df['Content-type_plain'].astype(int)
    data = column_dropper(data, 'Content-type')
    return data

# Scaling features with values > 1 or < 0
def feature_scaling(data):
    recievers_cnt = rec_cnt_scaler.transform(data[['Recievers_count']])
    data['Recievers_count'] = recievers_cnt

    sub_char = sub_char_scaler.transform(data[['sub_char']])
    data['sub_char'] = sub_char

    cont_char = content_char_scaler.transform(data[['content_char']])
    data['content_char'] = cont_char
    return data

# Applying lemmatization of text features
def apply_lemmatization(data):
    data['Subject'] = data['Subject'].apply(processing_text)
    data['Full_content'] = data['Full_content'].apply(processing_text)
    data.fillna("blank", inplace=True)
    return data

# Vectorization of text features
def vectorization(data):
    content_vectors = tfidf_content.transform(data['Full_content']).toarray()
    subject_vectors = tfidf_subject.transform(data['Subject']).toarray()
    return content_vectors, subject_vectors, data
    
# Concatenating the vectors of texts and numeric features
def data_concatenation(content_vectors, subject_vectors, data):
    X_1 = np.concatenate((content_vectors, subject_vectors), axis=1)
        
    X_2 = data[['content_char', 'sub_char', 'Replied_mail', 'Recievers_count',	'Sub_Unsub_link','Content_type_html','Content_type_plain']]
    X_2 = X_2.to_numpy()

    final_X = np.concatenate((X_1, X_2), axis=1)
    return final_X

def pipeline(file, filepath=False):
    start = time.time()
    parsed_mail = email_bytesparser(file, filepath)
    end = round(time.time()-start, 2)
    print('✔️ Email binary file parsing \t\t', end)
    start = time.time()
    data = extract_to_df(parsed_mail)
    end = round(time.time()-start, 2)
    print('✔️ Extracting required features in tabular form \t\t', end)
    start = time.time()
    data = feature_engineering1(data)
    end = round(time.time()-start, 2)
    print('✔️ Extracting numeric features \t\t', end)
    start = time.time()
    data = feature_encoding(data)
    end = round(time.time()-start, 2)
    print('✔️ Encoding categorical features \t\t', end)
    start = time.time()
    data = feature_scaling(data)
    end = round(time.time()-start, 2)
    print('✔️ Scaling features not in range (1,0) \t\t', end)
    start = time.time()
    data = apply_lemmatization(data)
    end = round(time.time()-start, 2)
    print('✔️ Lemmation of text \t\t', end)
    start = time.time()
    content_vectors, subject_vectors, data = vectorization(data)
    end = round(time.time()-start, 2)
    print('✔️ Vectorization of text \t\t', end)
    start = time.time()
    data = data_concatenation(content_vectors, subject_vectors, data)
    end = round(time.time()-start, 2)
    print('✔️ Concatenating vectors and numeric features \t\t', end)
    return data

def predictor(data):
    start = time.time()
    pred = model.predict(data)
    pred = round(pred[0][0], 2)
    end = round(time.time()-start, 2)
    print('✔️ Prediction \t\t', end)
    return pred