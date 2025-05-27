from flask import Flask, request, render_template, send_from_directory
from email import policy
from email.parser import BytesParser
import preprocessor
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def index():
    email_data = {}
    attachments = []

    if request.method == 'POST':
        file = request.files['email_file']
        if file and file.filename.endswith('.eml'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            with open(filepath, 'rb') as f:
                msg = BytesParser(policy=policy.default).parse(f)
                print(msg)
                f.close()
        email_df = preprocessor.return_dicts(msg)
                

    return render_template('index.html', email=email_data, attachments=attachments)

def get_body(msg, html=False):
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if html and content_type == 'text/html':
                return part.get_payload(decode=True).decode()
            elif not html and content_type == 'text/plain':
                return part.get_payload(decode=True).decode()
    else:
        return msg.get_payload(decode=True).decode()
    return ""

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)