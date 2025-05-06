from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model dan encoder yang sudah disimpan
model = joblib.load('model_rf.pkl')
encoder_provinsi = joblib.load('encoder_provinsi.pkl')
encoder_jenis_kelamin = joblib.load('encoder_jenis_kelamin.pkl')
scaler = joblib.load('scaler.pkl')

# Data provinsi untuk iterasi prediksi
provinsi_list = encoder_provinsi.classes_

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    provinsi = request.form.get('provinsi')
    tahun = int(request.form.get('tahun'))

    # Encode provinsi
    provinsi_encoded = encoder_provinsi.transform([provinsi])[0]

    # Persiapkan input untuk model
    input_data_male = pd.DataFrame({
        'Provinsi': [provinsi_encoded],
        'Tahun': [tahun],
        'Jenis Kelamin': [0]  # 0 untuk Laki-laki
    })

    input_data_female = pd.DataFrame({
        'Provinsi': [provinsi_encoded],
        'Tahun': [tahun],
        'Jenis Kelamin': [1]  # 1 untuk Perempuan
    })

    # Lakukan scaling
    input_data_male[['Provinsi', 'Tahun', 'Jenis Kelamin']] = scaler.transform(input_data_male[['Provinsi', 'Tahun', 'Jenis Kelamin']])
    input_data_female[['Provinsi', 'Tahun', 'Jenis Kelamin']] = scaler.transform(input_data_female[['Provinsi', 'Tahun', 'Jenis Kelamin']])

    # Prediksi untuk Laki-laki
    male_pred = model.predict(input_data_male)[0]
    # Prediksi untuk Perempuan
    female_pred = model.predict(input_data_female)[0]
    # Prediksi untuk Laki-laki dan Perempuan
    combined_pred = (male_pred + female_pred) / 2

    return render_template('index.html', provinsi=provinsi, tahun=tahun,
                           male_pred=male_pred, female_pred=female_pred, combined_pred=combined_pred)

@app.route('/api/highest', methods=['POST'])
def highest():
    try:
        data = request.json
        tahun = data.get('Tahun')

        if not isinstance(tahun, int):
            return jsonify({'error': 'Tahun harus berupa angka'}), 400

        highest_male = {'provinsi': None, 'persentase': -np.inf}
        highest_female = {'provinsi': None, 'persentase': -np.inf}
        highest_combined = {'provinsi': None, 'persentase': -np.inf}

        for provinsi in provinsi_list:
            provinsi_encoded = encoder_provinsi.transform([provinsi])[0]

            # Data untuk laki-laki
            input_data_male = pd.DataFrame({
                'Provinsi': [provinsi_encoded],
                'Tahun': [tahun],
                'Jenis Kelamin': [0]  # 0 untuk Laki-laki
            })
            input_data_male[['Provinsi', 'Tahun', 'Jenis Kelamin']] = scaler.transform(input_data_male[['Provinsi', 'Tahun', 'Jenis Kelamin']])
            male_pred = model.predict(input_data_male)[0]

            # Data untuk perempuan
            input_data_female = pd.DataFrame({
                'Provinsi': [provinsi_encoded],
                'Tahun': [tahun],
                'Jenis Kelamin': [1]  # 1 untuk Perempuan
            })
            input_data_female[['Provinsi', 'Tahun', 'Jenis Kelamin']] = scaler.transform(input_data_female[['Provinsi', 'Tahun', 'Jenis Kelamin']])
            female_pred = model.predict(input_data_female)[0]

            # Gabungan laki-laki dan perempuan
            combined_pred = (male_pred + female_pred) / 2

            # Update provinsi dengan prediksi tertinggi
            if male_pred > highest_male['persentase']:
                highest_male = {'provinsi': provinsi, 'persentase': male_pred}

            if female_pred > highest_female['persentase']:
                highest_female = {'provinsi': provinsi, 'persentase': female_pred}

            if combined_pred > highest_combined['persentase']:
                highest_combined = {'provinsi': provinsi, 'persentase': combined_pred}

        return jsonify({
            'Tahun': tahun,
            'Tertinggi Per Jenis Kelamin': {
                'Laki-laki': {
                    'Provinsi': highest_male['provinsi'],
                    'Persentase': round(highest_male['persentase'], 2)
                },
                'Perempuan': {
                    'Provinsi': highest_female['provinsi'],
                    'Persentase': round(highest_female['persentase'], 2)
                },
                'Gabungan': {
                    'Provinsi': highest_combined['provinsi'],
                    'Persentase': round(highest_combined['persentase'], 2)
                }
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
