from flask import Flask, request, Response, render_template, redirect, url_for
import subprocess
import os
from werkzeug.utils import secure_filename
import pandas as pd
import uuid
from datetime import datetime
unique_id = str(uuid.uuid4()) 

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scrape')
def scrape():
    username = request.args.get('username')
    password = request.args.get('password')
    url = request.args.get('url')

    def generate_output():
        try:
            process = subprocess.Popen(
                ['python', 'sa.py', username, password, url],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                env={**os.environ, "PYTHONIOENCODING": "utf-8"}
            )

            for line in iter(process.stdout.readline, ''):
                yield f"data: {line.strip()}\n\n"
            process.stdout.close()

            process.wait()
            if process.returncode != 0:
                yield f"data: Error: {process.stderr.read()}\n\n"
            process.stderr.close()

        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return Response(generate_output(), content_type='text/event-stream')

@app.route('/analyze-sentiment')
def analyze_sentiment():
    return render_template('upload.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return "Không có tệp nào được gửi."
    
    file = request.files['file']
    
    if file.filename == '':
        return "Không có tệp nào được chọn."
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        unique_id = uuid.uuid4().hex  
        output_path = os.path.join(OUTPUT_FOLDER, f'output_{unique_id}.csv')

        try:
            subprocess.run(['python', 'analyze.py', upload_path, output_path], check=True)
            
            return redirect(url_for('result', result_path=output_path))
        except subprocess.CalledProcessError as e:
            return f"Đã xảy ra lỗi trong quá trình phân tích: {str(e)}", 500
        except Exception as e:
            return f"Đã xảy ra lỗi không xác định: {str(e)}", 500
    
    return "Tệp không hợp lệ.", 400

@app.route('/ana-result', methods=['GET'])
def result():
    result_path = request.args.get('result_path')
    
    if not result_path or not os.path.exists(result_path):
        return "Không tìm thấy kết quả phân tích.", 404
    
    try:
        desired_columns = ['content', 'react_list', 'comment', 'prediction'] 
        df = pd.read_csv(result_path, usecols=desired_columns)
        
        # Chuyển đổi DataFrame thành bảng HTML
        table_html = df.to_html(index=False, classes='table table-striped')
        
        # Hiển thị kết quả
        return render_template('result.html', result=table_html)
    except Exception as e:
        return f"Không thể đọc kết quả phân tích: {str(e)}", 500
    
    
@app.route('/statistics')
def statistics():
    return render_template('upload-sta.html')

@app.route('/statistics-result', methods=['POST'])
def analyze_statistics():
    if 'file' not in request.files:
        return "Không có tệp nào được gửi."
    
    file = request.files['file']
    
    if file.filename == '':
        return "Không có tệp nào được chọn."
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        try:
            result = subprocess.run(
                ['python', 'statistics.py', upload_path],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                env={**os.environ, "PYTHONIOENCODING": "utf-8"}
            )
            
            if result.returncode != 0:
                return f"Đã xảy ra lỗi khi chạy thống kê: {result.stderr}"
            
            output = result.stdout
            
            return render_template('statistics.html', output=output)
        except Exception as e:
            return f"Đã xảy ra lỗi: {str(e)}"
    
    return "Tệp không hợp lệ."

if __name__ == '__main__':
    app.run(debug=True)
