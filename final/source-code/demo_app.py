#!/usr/bin/env python3
"""
Ứng dụng demo cho đồ án Khai phá Dữ liệu DS317
Demo dự đoán mức độ tham gia của học viên MOOC

Sử dụng Flask để tạo giao diện web đơn giản
"""

from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import pickle
import os
from pathlib import Path

app = Flask(__name__)

# Đường dẫn đến mô hình và dữ liệu
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "results" / "phase6" / "phase6_best_model.pkl"

# Load mô hình và cột đặc trưng
try:
    with open(MODEL_PATH, "rb") as f:
        model_bundle = pickle.load(f)
    model = model_bundle["estimator"]
    feature_columns = model_bundle["feature_columns"]
    print("Đã tải mô hình và cột đặc trưng thành công")
except Exception as e:
    print(f"Cảnh báo: Không tìm thấy hoặc có lỗi khi tải mô hình: {e}. Vui lòng chạy thực nghiệm trước.")
    model = None
    feature_columns = None

# HTML template cho giao diện
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Demo Dự đoán Tham gia MOOC - DS317</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="number"] {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            width: 100%;
            font-size: 16px;
        }
        button:hover {
            background-color: #2980b9;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 4px;
            display: none;
        }
        .result.success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .result.error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .info {
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎓 Demo Dự đoán Mức độ Tham gia MOOC</h1>
        <div class="info">
            <h3>Đồ án Khai phá Dữ liệu - DS317</h3>
            <p>Ứng dụng demo dự đoán mức độ tham gia của học viên dựa trên các đặc trưng hoạt động.</p>
            <p><strong>Các mức độ tham gia:</strong></p>
            <ul>
                <li><strong>Low:</strong> Tham gia thấp</li>
                <li><strong>Medium:</strong> Tham gia trung bình</li>
                <li><strong>High:</strong> Tham gia cao</li>
            </ul>
        </div>

        <form id="predictionForm">
            <div class="form-group">
                <label for="total_clicks">Tổng số lần click:</label>
                <input type="number" id="total_clicks" name="total_clicks" min="0" required>
            </div>

            <div class="form-group">
                <label for="total_time">Tổng thời gian học (phút):</label>
                <input type="number" id="total_time" name="total_time" min="0" step="0.1" required>
            </div>

            <div class="form-group">
                <label for="avg_weekly_clicks">Trung bình click hàng tuần:</label>
                <input type="number" id="avg_weekly_clicks" name="avg_weekly_clicks" min="0" step="0.1" required>
            </div>

            <div class="form-group">
                <label for="avg_weekly_time">Trung bình thời gian hàng tuần (phút):</label>
                <input type="number" id="avg_weekly_time" name="avg_weekly_time" min="0" step="0.1" required>
            </div>

            <div class="form-group">
                <label for="weeks_active">Số tuần hoạt động:</label>
                <input type="number" id="weeks_active" name="weeks_active" min="1" max="52" required>
            </div>

            <div class="form-group">
                <label for="consistency_score">Điểm nhất quán (0-1):</label>
                <input type="number" id="consistency_score" name="consistency_score" min="0" max="1" step="0.01" required>
            </div>

            <button type="submit">Dự đoán mức độ tham gia</button>
        </form>

        <div id="result" class="result"></div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            const formData = new FormData(this);
            const data = Object.fromEntries(formData);

            // Convert to numbers
            Object.keys(data).forEach(key => {
                data[key] = parseFloat(data[key]);
            });

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                const resultDiv = document.getElementById('result');

                if (response.ok) {
                    resultDiv.className = 'result success';
                    resultDiv.innerHTML = `
                        <h3>Kết quả dự đoán:</h3>
                        <p><strong>Mức độ tham gia: ${result.prediction}</strong></p>
                        <p>Xác suất: ${result.probabilities ? Object.entries(result.probabilities).map(([k,v]) => `${k}: ${(v*100).toFixed(1)}%`).join(', ') : 'N/A'}</p>
                    `;
                } else {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `<p>Lỗi: ${result.error}</p>`;
                }

                resultDiv.style.display = 'block';
            } catch (error) {
                const resultDiv = document.getElementById('result');
                resultDiv.className = 'result error';
                resultDiv.innerHTML = `<p>Lỗi kết nối: ${error.message}</p>`;
                resultDiv.style.display = 'block';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Trang chủ với form nhập liệu"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint để dự đoán"""
    if model is None or feature_columns is None:
        return jsonify({'error': 'Mô hình chưa được tải. Vui lòng chạy thực nghiệm trước.'}), 500

    try:
        data = request.get_json()

        # Tạo DataFrame từ dữ liệu đầu vào
        input_df = pd.DataFrame([data])

        # Đảm bảo có tất cả cột đặc trưng cần thiết
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Điền giá trị mặc định

        # Chỉ giữ lại các cột đặc trưng
        input_df = input_df[feature_columns]

        # Dự đoán
        prediction = model.predict(input_df)[0]
        probabilities = None

        # Lấy xác suất nếu mô hình hỗ trợ
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(input_df)[0]
            class_names = ['Low', 'Medium', 'High']  # Giả sử có 3 lớp
            probabilities = {class_names[i]: float(probas[i]) for i in range(len(class_names))}

        return jsonify({
            'prediction': prediction,
            'probabilities': probabilities
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    """Endpoint kiểm tra sức khỏe"""
    model_status = "loaded" if model is not None else "not loaded"
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'feature_columns_count': len(feature_columns) if feature_columns else 0
    })

if __name__ == '__main__':
    print("🚀 Khởi động ứng dụng demo MOOC Engagement Prediction")
    print("📱 Truy cập: http://localhost:5000")
    print("ℹ️  Endpoint health: http://localhost:5000/health")
    app.run(debug=True, host='0.0.0.0', port=5000)