# Importing libraries
import time
import logging
initial_start = time.time()

import email
from email import policy
from email.parser import BytesParser
import pandas as pd
import numpy as np
import re
import pickle
from bs4 import BeautifulSoup

import os
NLTK_DATA_DIR = os.path.join(os.getcwd(), 'nltk_data') 
if not os.path.exists(NLTK_DATA_DIR):
    os.makedirs(NLTK_DATA_DIR)
nltk.data.path.append(NLTK_DATA_DIR)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Lazy loading stopwords with fallback
def get_stopwords():
    try:
        return stopwords.words('english')
    except LookupError:
        logger.info(f"Downloading NLTK stopwords to {NLTK_DATA_DIR}...")
        nltk.download('stopwords', download_dir=NLTK_DATA_DIR)
        return stopwords.words('english')

# Lazy loading lemmatizer
def get_lemmatizer():
    try:
        return WordNetLemmatizer()
    except LookupError:
        logger.info(f"Downloading NLTK wordnet to {NLTK_DATA_DIR}...")
        nltk.download('wordnet', download_dir=NLTK_DATA_DIR)
        return WordNetLemmatizer()
    
stop_words = get_stopwords()
lem = get_lemmatizer()

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

end = round(time.time()-initial_start, 2)
logger.info(f'✔️ Libraries imported in {end}s')

start = time.time()

# Load pickle files with error handling
def load_pickle_file(filepath, description):
    try:
        with open(filepath, 'rb') as pickle_file:
            return pickle.load(pickle_file)
    except FileNotFoundError:
        logger.error(f"Required file not found: {filepath}")
        raise FileNotFoundError(f"Model file {filepath} not found. Please ensure all model files are uploaded.")
    except Exception as e:
        logger.error(f"Error loading {description}: {str(e)}")
        raise Exception(f"Error loading {description}: {str(e)}")

try:
    ohe = load_pickle_file('feature_encoders/one_hot_encoder.pkl', 'one-hot encoder')
    rec_cnt_scaler = load_pickle_file('scalers/rec_cnt_scaler.pkl', 'receiver count scaler')
    sub_char_scaler = load_pickle_file('scalers/sub_char_scaler.pkl', 'subject character scaler')
    content_char_scaler = load_pickle_file('scalers/content_char_scaler.pkl', 'content character scaler')
    tfidf_content = load_pickle_file('vectorizors/tfidf_content.pkl', 'content TF-IDF vectorizer')
    tfidf_subject = load_pickle_file('vectorizors/tfidf_subject.pkl', 'subject TF-IDF vectorizer')
    
    end = round(time.time()-start, 2)
    logger.info(f'✔️ Pickle files imported in {end}s')
except Exception as e:
    logger.error(f"Failed to load required model files: {str(e)}")
    raise

start = time.time()
try:
    model = tf.keras.models.load_model("dl_model/ann_model.h5")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    end = round(time.time()-start, 2)
    logger.info(f'✔️ DL model imported in {end}s')
except Exception as e:
    logger.error(f"Failed to load deep learning model: {str(e)}")
    raise

# Parse binary email file
def email_bytesparser(file, filepath=False):
    """Parse email from bytes or file path"""
    try:
        if filepath == False:
            parsed_email = BytesParser(policy=policy.default).parsebytes(file)
        else:
            with open(file, 'rb') as f:
                parsed_email = BytesParser(policy=policy.default).parse(f)
        return parsed_email
    except Exception as e:
        logger.error(f"Error parsing email: {str(e)}")
        raise Exception(f"Error parsing email file: {str(e)}")

# html parser
def parse_text(text):
    """Parse HTML content to extract text"""
    try:
        if text is None:
            return ""
        soup = BeautifulSoup(text, 'lxml')
        parsed_text = soup.get_text(strip=True)
        # Clean up whitespace
        parsed_text = re.sub(r'\s+', ' ', parsed_text)
        return parsed_text
    except Exception as ex:
        logger.warning(f"Error parsing HTML text: {str(ex)}")
        return ""

# Extract required details
def extract_to_df(parsed_email):
    """Extract email features into a DataFrame"""
    try:
        body = ""
        content_type = "none"
        
        if parsed_email.is_multipart():
            for part in parsed_email.iter_parts():
                part_content_type = part.get_content_type()
                if part_content_type == "text/plain":
                    content_type = "plain"
                    body = part.get_content()
                    break
                elif part_content_type == "text/html":
                    content_type = "html"
                    body = part.get_content()
        else:
            content_type = parsed_email.get_content_type()
            body = parsed_email.get_content()

        # Extract email fields with safe access
        receivers = parsed_email.get('To', '')
        subject = parsed_email.get('Subject', '')
        precedence = parsed_email.get('Precedence', '')
        list_unsub = parsed_email.get('List-Unsubscribe', '')
        replied_mail = parsed_email.get('Reply-To', '')
        list_sub = parsed_email.get('List-Subscribe', '')
        
        # Parse email content
        parsed_email_text = parse_text(body) if body else ""

        # Prepare the data as a DataFrame
        data = pd.DataFrame([{
            'To': receivers,
            'Subject': subject,
            'Content-type': content_type,
            'Precendence': precedence,
            'Full_content': parsed_email_text,
            'List-unsubscribe': list_unsub,
            'List-subscribe': list_sub,
            'In-reply-to': replied_mail
        }])

        return data
    except Exception as e:
        logger.error(f"Error extracting email data: {str(e)}")
        raise Exception(f"Error extracting email data: {str(e)}")

# Drop unwanted feature
def column_dropper(data, col):
    """Drop specified columns from DataFrame"""
    transformed_df = data.copy()
    if isinstance(col, list):
        transformed_df.drop(columns=col, inplace=True)
    else:
        transformed_df.drop(columns=[col], inplace=True)
    return transformed_df

# Text cleaners
extra_space_remover = lambda x: re.sub(r'\s+', ' ', str(x)).strip() if pd.notna(x) else np.nan

def clean_text(text):
    """Clean and normalize text"""
    if pd.isna(text) or text == '':
        return np.nan
    
    try:
        # Replaces URLs in a string with a single space
        url_pattern = re.compile(r'https?://\S+|www\.\S+|\b\S+\.(?:com|org|net|gov|edu|io|co|in|info)\b')
        no_links_text = url_pattern.sub(' ', str(text))

        # Remove HTML-like tags
        words = [word for word in no_links_text.split() if not ('>' in word or '<' in word or '=' in word)]
        no_tags_text = ' '.join(words)
        
        # Remove all punctuations, numbers and symbols
        cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', no_tags_text)
        
        if cleaned_text.strip() != '':
            return extra_space_remover(cleaned_text)
        else:
            return np.nan
    except Exception as e:
        logger.warning(f"Error cleaning text: {str(e)}")
        return np.nan

# Lemmatization and eliminating stopwords
def processing_text(text, stopwords=stop_words):
    """Process text with lemmatization and stopword removal"""
    if pd.isna(text) or text == '':
        return "blank"
    
    try:
        words = str(text).split()
        words = [lem.lemmatize(word.lower()) for word in words 
                if ((word.lower() not in stopwords) and (len(str(word)) > 1))]
        words = [word for word in words if len(word) > 1]
        return ' '.join(words) if words else "blank"
    except Exception as e:
        logger.warning(f"Error processing text: {str(e)}")
        return "blank"

def feature_engineering1(data):
    """Perform feature engineering on email data"""
    try:
        # Creating New bool feature 'Replied_mail'
        data['Replied_mail'] = np.where(pd.isna(data['In-reply-to']), 0, 1)
        data = column_dropper(data, 'In-reply-to')

        # Creating New numeric feature 'Recievers_count'
        data['Recievers_count'] = data['To'].apply(
            lambda x: len(str(x).split('@')) - 1 if pd.notna(x) else 0
        )
        data = column_dropper(data, 'To')

        # Creating New bool feature 'Sub_Unsub_link'
        data['Sub_Unsub_link'] = np.where(
            (pd.notna(data['List-subscribe']) & pd.notna(data['List-unsubscribe'])), 1, 0)
        data = column_dropper(data, ['List-subscribe', 'List-unsubscribe'])
        
        # Cleaning Content-Type feature
        def extract_cont_type(text):
            if pd.isna(text) or text == '':
                return "none"
            elif 'html' in str(text).lower():
                return 'html'
            elif 'plain' in str(text).lower():
                return 'plain'
            else:
                return "none"
        
        data['Content-type'] = data['Content-type'].apply(extract_cont_type)
        
        # Cleaning text features
        data['Full_content'] = data['Full_content'].apply(clean_text)
        data['Subject'] = data['Subject'].apply(clean_text)

        # Creating new numeric features "sub_char" & "content_char"
        data['content_char'] = data['Full_content'].apply(
            lambda x: len(str(x)) if pd.notna(x) else 0
        )
        data['sub_char'] = data['Subject'].apply(
            lambda x: len(str(x)) if pd.notna(x) else 0
        )
        
        return data
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise Exception(f"Error in feature engineering: {str(e)}")

# Encoding categorical features
def feature_encoding(data):
    """Encode categorical features"""
    try:
        encoded_content_type = ohe.transform(data[['Content-type']])
        encoded_df = pd.DataFrame(encoded_content_type, columns=ohe.get_feature_names_out(['Content-type']))

        data['Content_type_html'] = encoded_df['Content-type_html'].astype(int)
        data['Content_type_plain'] = encoded_df['Content-type_plain'].astype(int)
        data = column_dropper(data, 'Content-type')
        return data
    except Exception as e:
        logger.error(f"Error in feature encoding: {str(e)}")
        raise Exception(f"Error in feature encoding: {str(e)}")

# Scaling features with values > 1 or < 0
def feature_scaling(data):
    """Scale numerical features"""
    try:
        recievers_cnt = rec_cnt_scaler.transform(data[['Recievers_count']])
        data['Recievers_count'] = recievers_cnt

        sub_char = sub_char_scaler.transform(data[['sub_char']])
        data['sub_char'] = sub_char

        cont_char = content_char_scaler.transform(data[['content_char']])
        data['content_char'] = cont_char
        return data
    except Exception as e:
        logger.error(f"Error in feature scaling: {str(e)}")
        raise Exception(f"Error in feature scaling: {str(e)}")

# Applying lemmatization of text features
def apply_lemmatization(data):
    """Apply lemmatization to text features"""
    try:
        data['Subject'] = data['Subject'].apply(processing_text)
        data['Full_content'] = data['Full_content'].apply(processing_text)
        data.fillna("blank", inplace=True)
        return data
    except Exception as e:
        logger.error(f"Error in lemmatization: {str(e)}")
        raise Exception(f"Error in lemmatization: {str(e)}")

# Vectorization of text features
def vectorization(data):
    """Vectorize text features using TF-IDF"""
    try:
        content_vectors = tfidf_content.transform(data['Full_content']).toarray()
        subject_vectors = tfidf_subject.transform(data['Subject']).toarray()
        return content_vectors, subject_vectors, data
    except Exception as e:
        logger.error(f"Error in vectorization: {str(e)}")
        raise Exception(f"Error in vectorization: {str(e)}")
    
# Concatenating the vectors of texts and numeric features
def data_concatenation(content_vectors, subject_vectors, data):
    """Concatenate all features into final feature matrix"""
    try:
        X_1 = np.concatenate((content_vectors, subject_vectors), axis=1)
            
        X_2 = data[['content_char', 'sub_char', 'Replied_mail', 'Recievers_count',
                   'Sub_Unsub_link', 'Content_type_html', 'Content_type_plain']]
        X_2 = X_2.to_numpy()

        final_X = np.concatenate((X_1, X_2), axis=1)
        return final_X
    except Exception as e:
        logger.error(f"Error in data concatenation: {str(e)}")
        raise Exception(f"Error in data concatenation: {str(e)}")

def pipeline(file, filepath=False):
    """Complete preprocessing pipeline"""
    try:
        start = time.time()
        parsed_mail = email_bytesparser(file, filepath)
        end = round(time.time()-start, 2)
        logger.info(f'✔️ Email binary file parsing \t\t{end}s')
        
        start = time.time()
        data = extract_to_df(parsed_mail)
        end = round(time.time()-start, 2)
        logger.info(f'✔️ Extracting required features in tabular form \t\t{end}s')
        
        start = time.time()
        data = feature_engineering1(data)
        end = round(time.time()-start, 2)
        logger.info(f'✔️ Extracting numeric features \t\t{end}s')
        
        start = time.time()
        data = feature_encoding(data)
        end = round(time.time()-start, 2)
        logger.info(f'✔️ Encoding categorical features \t\t{end}s')
        
        start = time.time()
        data = feature_scaling(data)
        end = round(time.time()-start, 2)
        logger.info(f'✔️ Scaling features not in range (1,0) \t\t{end}s')
        
        start = time.time()
        data = apply_lemmatization(data)
        end = round(time.time()-start, 2)
        logger.info(f'✔️ Lemmatization of text \t\t{end}s')
        
        start = time.time()
        content_vectors, subject_vectors, data = vectorization(data)
        end = round(time.time()-start, 2)
        logger.info(f'✔️ Vectorization of text \t\t{end}s')
        
        start = time.time()
        data = data_concatenation(content_vectors, subject_vectors, data)
        end = round(time.time()-start, 2)
        logger.info(f'✔️ Concatenating vectors and numeric features \t\t{end}s')
        
        return data
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        raise Exception(f"Error in preprocessing pipeline: {str(e)}")

def predictor(data):
    """Make spam prediction using the trained model"""
    try:
        start = time.time()
        pred = model.predict(data, verbose=0)
        pred = float(pred[0][0])
        end = round(time.time()-start, 2)
        logger.info(f'✔️ Prediction \t\t{end}s')
        return pred
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise Exception(f"Error in prediction: {str(e)}")