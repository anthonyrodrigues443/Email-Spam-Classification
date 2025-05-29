# Email-Spam-Classification 📧🕵️⚠️
[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
![Flask](https://img.shields.io/badge/Flask-3.1.1-FF4B4B?logo=flask&logoColor=white)

This project implements a deep learning model to classify emails as spam or ham (not spam). It leverages natural language processing (NLP) techniques to clean(tokenize, lemmatize and vectorize) and extract features from email text and trains a classifier using neural networks. The application is built using Python, Flask for the web interface, and tensorflow's sequential model of dense layers and dropout layers for classifying emails.


## Overview 👁️
This project is a deep learning-based system for classifying email messages as Spam or Not Spam (Ham). It employs Natural Language Processing (NLP) techniques for preprocessing email text, followed by supervised deep learning classification using nueral networks. The primary goal is to build a reliable and interpretable model that automates spam detection, reducing manual filtering and enhancing communication security.


## Features 🚀
- **Real-time Email Classification**: Upload email files (.eml format) and get instant spam/legitimate predictions
- **Advanced NLP Pipeline**: Comprehensive text preprocessing including lemmatization, stopword removal, and TF-IDF vectorization
- **Deep Learning Model**: Artificial Neural Network (ANN) trained for binary classification
- **Web Interface**: User-friendly Flask-based web application (**CREDITS:** Frontend design fully developed by **"Claude ai"** and **"Gemini ai"**)
- **Feature Engineering**: Extracts multiple features including recipient count, content type, reply status, subscription links and subject and body characters count.


## Webpage
<img src="https://github.com/anthonyrodrigues443/Email-Spam-Classification/blob/main/webpage_ss/not_spam1.png" width="400px"><img src="https://github.com/anthonyrodrigues443/Email-Spam-Classification/blob/main/webpage_ss/not_spam2.png" width="400px"><img src="https://github.com/anthonyrodrigues443/Email-Spam-Classification/blob/main/webpage_ss/not_spam3.png" width="400px"><img src="https://github.com/anthonyrodrigues443/Email-Spam-Classification/blob/main/webpage_ss/spam1.png" width="400px"><img src="https://github.com/anthonyrodrigues443/Email-Spam-Classification/blob/main/webpage_ss/spam2.png" width="400px"><img src="https://github.com/anthonyrodrigues443/Email-Spam-Classification/blob/main/webpage_ss/spam3.png" width="400px">


## 📁 Project Structure

```
Email-Spam-Classification/
├── datasets/                    # Training and testing datasets
├── dl_model/                   # Trained deep learning model
├── feature_encoders/           # Pickle files for categorical encoding
├── inbox_test_mails/          # Sample email files for testing
├── scalers/                   # Feature scaling transformers
├── spam_email_project/        # Additional project files
├── templates/                 # HTML templates for web interface
├── vectorizors/              # TF-IDF vectorizers for text features
├── webpage_ss/               # Web application screenshots
├── app.py                    # Flask web application
├── preprocessor.py           # Data preprocessing pipeline
├── NLP (Email Spam Classification).ipynb  # Jupyter notebook with analysis
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```


## 🛠️ Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Email-Spam-Classification.git
   cd Email-Spam-Classification
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```


## 🏃‍♂️ Usage

### Web Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to `http://localhost:5000`

3. **Upload an email file** (.eml format) and get the classification result

### Programmatic Usage

```python
import preprocessor

# Process email file
file = r"C:\Users\fullpath\email_name.eml"
processed_data = preprocessor.pipeline(file=file, file_type="filepath")
prediction = preprocessor.predictor(processed_data)

print(f"Spam probability: {prediction}")
print(f"Classification: {'Spam' if prediction > 0.5 else 'Legitimate'}")
```

## 🔧 Technical Details

### Data Preprocessing Pipeline
The preprocessing pipeline includes the following steps:
1. **Email Parsing**: Extract email components using Python's email library
2. **Feature Extraction**:
   - Recipient count
   - Content type (HTML/Plain text)
   - Reply status
   - Subscription/Unsubscription links
   - Character counts for subject and content
3. **Text Cleaning**:
   - URL removal
   - HTML tag removal
   - Punctuation and number removal
   - Extra whitespace removal
4. **Text Processing**:
   - Stopword removal
   - Lemmatization using WordNet
5. **Feature Engineering**:
   - One-hot encoding for categorical features
   - Standard scaling for numerical features
   - TF-IDF vectorization for text features
6. **Data Concatenation**: Combine all features into final input vector

### Model Architecture
- **Type**: Artificial Neural Network (ANN)
- **Framework**: TensorFlow/Keras
- **Task**: Binary Classification (Spam vs Legitimate)
- **Architecture**: Sequential model with Dense → Dropout → Dense → Dropout → Dense layers
- **Regularization**: Dropout layers for preventing overfitting
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy


### Key Features Extracted
| Feature | Description |
|---------|-------------|
| `Replied_mail` | Boolean indicating if email is a reply |
| `Recievers_count` | Number of recipients |
| `Sub_Unsub_link` | Presence of subscription/unsubscription links |
| `Content_type_html` | Whether email content is HTML |
| `Content_type_plain` | Whether email content is plain text |
| `content_char` | Character count in email body |
| `sub_char` | Character count in subject line |
| `TF-IDF vectors` | Vectorized subject and content text |


## 📊 Performance
The model processes emails through a comprehensive pipeline that typically takes:
- Email parsing: ~0.01s
- Feature extraction: ~0.02s
- Text processing: ~0.05s
- Vectorization: ~0.03s
- Prediction: ~0.01s
**Total processing time**: ~0.12s per email


## 🧪 Testing
Sample email files are provided in the `inbox_test_mails/` directory for testing the application. These mails are classified by gmail and unseen by the Model.


## 📋 Dependencies
Key libraries used:
- Flask - Web framework
- TensorFlow - Deep learning model
- pandas, numpy - Data manipulation
- scikit-learn - Feature preprocessing
- NLTK - Natural language processing
- BeautifulSoup - HTML parsing
- pickle - Model serialization

See `requirements.txt` for complete list.


## 🤝 Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request


## 📝 License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/anthonyrodrigues443/Email-Spam-Classification/blob/main/LICENSE) file for details.


## 🙏 Acknowledgments
- **Claude AI** and **Gemini AI** for frontend development and web interface design

 
---

**Note**: Make sure to have all the required model files (`dl_model/`, `feature_encoders/`, `scalers/`, `vectorizors/`) in place before running the application, the dl model can be saved by running the jupyter notebook.
