from flask import Flask, request, jsonify
import joblib
import pandas as pd

# تحميل النموذج والـ encoders و أسماء الأعمدة
model = joblib.load('model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
features = joblib.load('features.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # استلام البيانات بصيغة JSON
        data = request.get_json()

        # تحويل إلى DataFrame
        df = pd.DataFrame([data])

        # ترميز الأعمدة النصية بنفس الطريقة السابقة
        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])

        # التأكد من ترتيب الأعمدة
        df = df[features]

        # التنبؤ
        prediction = model.predict(df)[0]

        return jsonify({'prediction': round(float(prediction), 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
