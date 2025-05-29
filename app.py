from flask import Flask, render_template, request
import preprocessor

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['email_file']
    if file.filename == '':
        return "No selected file", 400
    
    # Read file content directly into memory
    file_content = file.read()
    
    # Process the file content directly (you'll need to modify your preprocessor)
    processed_data = preprocessor.pipeline(file_content)
    pred = preprocessor.predictor(processed_data)
    spam_val = pred
    not_spam_val = round(1.0 - pred, 2)

    label = "Spam" if pred > 0.5 else "Legitimate"
    return render_template('result.html', prediction=pred, label=label)

if __name__ == '__main__':
    app.run(debug=True)