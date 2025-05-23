# Importing libraries
import time
initial_start = time.time()
import pandas as pd
import numpy as np
import re
import pickle
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
end = round(time.time()-initial_start, 2)
print('Libraries imported ✔️\t\t', end)

start = time.time()
# Loading saved pickle encoders,verctorizors,scalers and dl model
with open('preprocessing_dicts/preprocessing_dicts.pkl', 'rb') as pickle_file:
    reqd_features = pickle.load(pickle_file)

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
print('Pickle files imported ✔️\t\t', end)

start = time.time()
model = tf.keras.models.load_model("dl_model/ann_model.h5")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
end = round(time.time()-start, 2)
print('DL model imported ✔️\t\t', end)


#preprocessing functions
def return_dicts(email_text, reqd_features=reqd_features):
    data = dict()
    matches = re.findall(r'\b\w+(?:-\w+)*: ', email_text)
    matches = [match.strip() for match in matches]
    all_matches = ['From:']
    all_matches.extend(matches)
    split_text = re.split(r'\b\w+(?:-\w+)*: ', email_text)
    for i, j in zip(all_matches, split_text):
        for k in reqd_features:
            if k in i:
                data[i] = j
    return (
        data.get('To:', None),
        data.get('Subject:', None),
        data.get('Content-Type:', None),
        data.get('Precedence:', None),
        data.get('Content-Transfer-Encoding:', None),
        data.get('List-Unsubscribe:', None),
        data.get('List-Subscribe:', None),
        data.get('In-Reply-To:', None),
        data.get('wrote:', None),
    )


def column_dropper(data, col):
    transformed_df = data.copy()
    transformed_df.drop(columns=col, inplace=True)
    return transformed_df

def add_prec_content(row):
    try :
        return  row[5:].strip()
    except TypeError :
        return ""

def parse_text(data, col):
        for index, text in enumerate(data[col]):
                try : 
                        soup = BeautifulSoup(data[col][index], 'lxml')
                        parsed_text = soup.text.strip()
                except Exception as ex :
                        parsed_text = ""
                finally :
                        data.loc[index, col] = parsed_text
        return data

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

stop_words = stopwords.words('english')
lem = WordNetLemmatizer()

def preprocessing_text2(text, stopwords=stop_words):
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    words = [lem.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

email_text = '''
From gort44@excite.com Mon Jun 24 17:54:21 2002 Return-Path: gort44@excite.com Delivery-Date: Tue Jun 4 05:31:16 2002 Received: from mandark.labs.netnoteinc.com ([213.105.180.140]) by dogma.slashnull.org (8.11.6/8.11.6) with ESMTP id g544VFO20182 for <jm@jmason.org>; Tue, 4 Jun 2002 05:31:15 +0100 Received: from wi-poli.poli.cl ([200.54.149.34]) by mandark.labs.netnoteinc.com (8.11.2/8.11.2) with SMTP id g544VC729935; Tue, 4 Jun 2002 05:31:13 +0100 Received: from 216.77.61.89 (unverified [218.5.180.148]) by wi-poli.poli.cl (EMWAC SMTPRS 0.83) with SMTP id <B0000918901@wi-poli.poli.cl>; Tue, 04 Jun 2002 00:14:29 -0400 Message-Id: <B0000918901@wi-poli.poli.cl> To: <chrbader@telecom.at> From: ""irese"" <gort44@excite.com> Subject: Cash in on your home equity Date: Tue, 04 Jun 2002 00:18:34 -1600 MIME-Version: 1.0 Content-Type: text/plain; charset=""Windows-1252"" X-Keywords: Content-Transfer-Encoding: 7bit Mortgage Lenders & Brokers Are Ready to compete for your business. Whether a new home loan is what you seek or to refinance your current home loan at a lower interest rate, we can help! Mortgage rates haven't been this low in years take action now! Refinance your home with us and include all of those pesky credit card bills or use the extra cash for that pool you've always wanted... Where others say NO, we say YES!!! Even if you have been turned down elsewhere, we can help! Easy terms! Our mortgage referral service combines the highest quality loans with the most economical rates and the easiest qualifications! Take just 2 minutes to complete the following form. There is no obligation, all information is kept strictly confidential, and you must be at least 18 years of age. Service is available within the United States only. This service is fast and free. Free information request form: PLEASE VISIT http://builtit4unow.com/pos **************************************************************** Since you have received this message you have either responded to one of our offers in the past or your address has been registered with us. If you wish to ""OPT_OUT"" please visit: http://builtit4unow.com/pos ****************************************************************
'''
start = time.time()
data = pd.DataFrame(pd.Series(email_text).apply(return_dicts).tolist(), columns=reqd_features)
end = round(time.time()-start, 2)
print('Extracting required features \t\t', end)

start = time.time()
data['Replied_mail'] = np.where(pd.isna(data['In-Reply-To:']), 0, 1)
data = column_dropper(data, 'In-Reply-To:')

data['Recievers_count'] = data['To:'].apply(lambda x: len(str(x).split('@')) - 1)
data = column_dropper(data, 'To:')

data['Sub_Unsub_link'] = np.where(
    (pd.notna(data['List-Subscribe:']) & pd.notna(data['List-Unsubscribe:'])), 1, 0)

data = column_dropper(data, ['List-Subscribe:','List-Unsubscribe:'])


data['Prec_content'] = data['Precedence:'].apply(add_prec_content)
data = column_dropper(data, 'Precedence:')

data = parse_text(data, 'Content-Transfer-Encoding:')
data = parse_text(data, 'Prec_content')
data = parse_text(data, 'wrote:')

data['Wrote_content'] = data['wrote:'].apply(clean_text)
data = column_dropper(data, 'wrote:')

data['Content-Type:'] = data['Content-Type:'].str.lower()
data['Content-Type:'] = data['Content-Type:'].str.extract(r'(?:^.{5})?(.*?)(?:;|$)')[0].str.strip()

# Map content types
data['Content-Type:'] = np.select(
    [
        data['Content-Type:'].str.contains('html', na=False),
        data['Content-Type:'].str.contains('none', na=False)
    ],
    ['html', 'none'],
    default='plain'
)


encoded_content_type = ohe.transform(data[['Content-Type:']])
encoded_df = pd.DataFrame(encoded_content_type, columns=ohe.get_feature_names_out(['Content-Type:']))

data['Content_type_html'] = encoded_df['Content-Type:_html'].astype(int)
data['Content_type_plain'] = encoded_df['Content-Type:_plain'].astype(int)
data = column_dropper(data, 'Content-Type:')

data = parse_text(data, 'Subject:')
data['Subject'] = data['Subject:'].astype(str).str.strip().replace('', 'blank')
data = column_dropper(data, 'Subject:')

data['Full_content'] = data['Content-Transfer-Encoding:'].astype(str) +" "+ data['Prec_content'].astype(str) +" "+ data['Wrote_content'].astype(str)
data['Full_content'] = data['Full_content'].astype(str).str.strip().replace('', 'blank')
data = column_dropper(data, ['Prec_content','Wrote_content','Content-Transfer-Encoding:'])

data['content_char'] = data['Full_content'].apply(len)
data['sub_char'] = data['Subject'].apply(len)

recievers_cnt = rec_cnt_scaler.transform(data[['Recievers_count']])
data['Recievers_count'] = recievers_cnt

sub_char = sub_char_scaler.transform(data[['sub_char']])
data['sub_char'] = sub_char

cont_char = content_char_scaler.transform(data[['content_char']])
data['content_char'] = cont_char

preprocessed_df = data.copy()
preprocessed_df['Subject'] = preprocessed_df['Subject'].apply(preprocessing_text2)
preprocessed_df['Full_content'] = preprocessed_df['Full_content'].apply(preprocessing_text2)
preprocessed_df.fillna("blank", inplace=True)
end = round(time.time()-start, 2)
print('Processed data ✔️\t\t', end)

start = time.time()
content_vectors = tfidf_content.transform(preprocessed_df['Full_content']).toarray()
subject_vectors = tfidf_subject.transform(preprocessed_df['Subject']).toarray()

X_1 = np.concatenate((content_vectors, subject_vectors), axis=1)
X_2 = data[['content_char', 'sub_char', 'Replied_mail', 'Recievers_count',	'Sub_Unsub_link','Content_type_html','Content_type_plain']]
X_2 = X_2.to_numpy()

final_X = np.concatenate((X_1, X_2), axis=1)
print(final_X.shape)
end = round(time.time()-start, 2)
print('Vectorization done ✔️\t\t', end)

start = time.time()
pred = round(model.predict(final_X)[0][0])
end = round(time.time()-start, 2)
print('Prediction done  ✔️\t\t', end)
print(pred)
