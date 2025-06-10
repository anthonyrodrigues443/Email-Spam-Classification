from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import preprocessor
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for better frontend compatibility

# Set maximum file size (16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving index page: {str(e)}")
        return "Error loading page", 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Email Spam Classification API is running'
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if file is present in request
        if 'email_file' not in request.files:
            logger.warning("No file part in request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['email_file']
        
        # Check if file is selected
        if file.filename == '':
            logger.warning("No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file extension (optional - you can customize this)
        allowed_extensions = ['.eml', '.msg', '.txt', '']  # Allow files without extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        logger.info(f"Processing file: {file.filename}")
        
        # Read file content directly into memory
        file_content = file.read()
        
        # Check if file is empty
        if len(file_content) == 0:
            logger.warning("Empty file uploaded")
            return jsonify({'error': 'File is empty'}), 400
        
        # Process the file content
        logger.info("Starting email processing pipeline")
        processed_data = preprocessor.pipeline(file_content, filepath=False)
        
        # Make prediction
        logger.info("Making spam prediction")
        pred = preprocessor.predictor(processed_data)
        
        # Calculate probabilities
        spam_probability = float(pred)
        legitimate_probability = round(1.0 - pred, 2)
        
        # Determine label
        label = "Spam" if pred > 0.5 else "Legitimate"
        confidence = max(spam_probability, legitimate_probability)
        
        logger.info(f"Prediction completed: {label} (confidence: {confidence:.2f})")
        
        # Return JSON response for API calls or render template for web interface
        if request.headers.get('Content-Type') == 'application/json' or request.args.get('format') == 'json':
            return jsonify({
                'prediction': spam_probability,
                'label': label,
                'confidence': confidence,
                'probabilities': {
                    'spam': spam_probability,
                    'legitimate': legitimate_probability
                }
            })
        else:
            return render_template('result.html', 
                                 prediction=spam_probability, 
                                 label=label,
                                 confidence=confidence,
                                 spam_prob=spam_probability,
                                 legit_prob=legitimate_probability)
    
    except FileNotFoundError as e:
        logger.error(f"Required model files not found: {str(e)}")
        return jsonify({'error': 'Model files not found. Please check server setup.'}), 500
    
    except MemoryError as e:
        logger.error(f"Memory error processing file: {str(e)}")
        return jsonify({'error': 'File too large to process'}), 413
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict_api():
    return upload_file()

@app.errorhandler(413)
def too_large(e):
    logger.warning("File too large uploaded")
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug) 