import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np

# 1. Baca dataset
data = pd.read_excel("Persentase_Buta_Aksara.xlsx")

# Debug: Lihat nama kolom
print("Kolom dataset:", data.columns)

# Validasi keberadaan kolom
required_columns = ['Provinsi', 'Tahun', 'Jenis Kelamin', 'Persentase']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"Kolom '{col}' tidak ditemukan dalam dataset.")

# 2. Encoding fitur kategorikal
encoder_provinsi = LabelEncoder()
data['Provinsi'] = encoder_provinsi.fit_transform(data['Provinsi'])

encoder_jenis_kelamin = LabelEncoder()
data['Jenis Kelamin'] = encoder_jenis_kelamin.fit_transform(data['Jenis Kelamin'])

# Debug: Lihat hasil encoding
print("Data setelah encoding:")
print(data.head())

# 3. Pisahkan fitur dan target
X = data[['Provinsi', 'Tahun', 'Jenis Kelamin']]
y = data['Persentase']

# 4. Scaling fitur numerik
scaler = StandardScaler()
X[['Provinsi', 'Tahun', 'Jenis Kelamin']] = scaler.fit_transform(X[['Provinsi', 'Tahun', 'Jenis Kelamin']])

# Debug: Periksa tipe data X
print("Tipe data X setelah scaling:")
print(X.dtypes)

# 5. Split data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Bangun model Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Debug: Lihat fitur penting (feature importance)
print("Fitur penting model:", model.feature_importances_)

# 7. Evaluasi model
y_pred = model.predict(X_test)

# Hitung MAE (Mean Absolute Error)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae}")

# Hitung R-squared (Koefisien Determinasi)
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

# Hitung RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")

# 8. Simpan model dan encoder
joblib.dump(model, 'model_rf.pkl')
joblib.dump(encoder_provinsi, 'encoder_provinsi.pkl')
joblib.dump(encoder_jenis_kelamin, 'encoder_jenis_kelamin.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model dan encoder berhasil disimpan.")
