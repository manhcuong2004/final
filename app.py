from flask import Flask, request, Response, render_template, redirect, url_for
import subprocess
import os
from werkzeug.utils import secure_filename
import pandas as pd

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tạo thư mục nếu chưa tồn tại
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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

            if process.returncode != 0:
                yield f"data: Error: {process.stderr.read()}\n\n"
            process.stderr.close()

        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return Response(generate_output(), content_type='text/event-stream')

# Trang chủ hiển thị form upload
@app.route('/analyze-sentiment')
def indexa():
    return render_template('upload.html')

# Xử lý tải lên và phân tích tệp
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return "Không có tệp nào được gửi."
    
    file = request.files['file']
    
    if file.filename == '':
        return "Không có tệp nào được chọn."
    
    if file and allowed_file(file.filename):
        # Lưu file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        # Gọi script phân tích
        output_path = os.path.join(OUTPUT_FOLDER, 'output.csv')
        try:
            subprocess.run(['python', 'analyze.py', upload_path, output_path], check=True)
            return redirect(url_for('result', result_path=output_path))
        except Exception as e:
            return f"Đã xảy ra lỗi: {str(e)}"
    
    return "Tệp không hợp lệ."

# Hiển thị kết quả phân tích
@app.route('/result', methods=['GET'])
def result():
    result_path = request.args.get('result_path')
    if not os.path.exists(result_path):
        return "Không tìm thấy kết quả phân tích."
    
    df = pd.read_csv(result_path)
    return render_template('result.html', result=df.to_html(index=False))

if __name__ == '__main__':
    app.run(debug=True)