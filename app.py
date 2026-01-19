# ============================================================================
# ỨNG DỤNG DỰ ĐOÁN THỜI TIẾT ĐÀ NẴNG - STREAMLIT APP
# ============================================================================
# Giao diện đơn giản: Nhập input -> Dự đoán từ 3 mô hình
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CẤU HÌNH TRANG
# ============================================================================
st.set_page_config(
    page_title="Dự đoán Thời tiết Đà Nẵng",
    page_icon="🌤️",
    layout="wide"
)

# ============================================================================
# LOAD VÀ HUẤN LUYỆN MÔ HÌNH (CACHE)
# ============================================================================
@st.cache_resource
def load_and_train_models():
    """Load dữ liệu và huấn luyện 3 mô hình"""
    # Load data
    df = pd.read_csv('preprocessed_data.csv')
    
    # Tách features và target
    X = df.drop('weather_group', axis=1)
    y = df['weather_group']
    
    # Mã hóa nhãn
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Chia dữ liệu 80-20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # 1. Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # 2. AdaBoost
    ada_model = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
    ada_model.fit(X_train, y_train)
    
    # 3. Softmax Regression
    softmax_model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    softmax_model.fit(X_train, y_train)
    
    models = {
        'Random Forest': rf_model,
        'AdaBoost': ada_model,
        'Softmax Regression': softmax_model
    }
    
    return models, label_encoder, X.columns.tolist()

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Tiêu đề
    st.markdown("<h1 style='text-align: center; color: #1E88E5;'>🌤️ Dự đoán Thời tiết Đà Nẵng</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Nhập các thông số thời tiết và nhấn <b>Dự đoán</b> để xem kết quả từ 3 mô hình</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models
    models, label_encoder, feature_names = load_and_train_models()
    
    # ========================================================================
    # PHẦN NHẬP INPUT
    # ========================================================================
    st.header("📝 Nhập thông số thời tiết")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🌡️ Nhiệt độ & Độ ẩm")
        temperature = st.number_input("Nhiệt độ 2m (°C)", min_value=10.0, max_value=45.0, value=25.0, step=0.1)
        humidity = st.number_input("Độ ẩm tương đối (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
        dew_point = st.number_input("Điểm sương (°C)", min_value=0.0, max_value=35.0, value=21.0, step=0.1)
        humidity_change = st.number_input("Thay đổi độ ẩm", min_value=-20.0, max_value=20.0, value=0.0, step=0.5)
        hum_max_6h = st.number_input("Độ ẩm tối đa 6h (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
    
    with col2:
        st.subheader("🌬️ Áp suất & Gió")
        pressure = st.number_input("Áp suất bề mặt (hPa)", min_value=980.0, max_value=1040.0, value=1013.0, step=0.1)
        cloud_cover = st.number_input("Độ che phủ mây (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
        wind_speed = st.number_input("Tốc độ gió 10m (m/s)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
        pressure_trend = st.number_input("Xu hướng áp suất 6h", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
        press_std_12h = st.number_input("Độ lệch chuẩn áp suất 12h", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    
    with col3:
        st.subheader("⏰ Thời gian & Khác")
        hour = st.slider("Giờ trong ngày", min_value=0, max_value=23, value=12)
        month = st.slider("Tháng", min_value=1, max_value=12, value=6)
        temp_diff_3h = st.number_input("Chênh lệch nhiệt độ 3h (°C)", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
        
        # Tính sin/cos
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        st.info(f"hour_sin: {hour_sin:.4f}\nhour_cos: {hour_cos:.4f}\nmonth_sin: {month_sin:.4f}\nmonth_cos: {month_cos:.4f}")
    
    st.markdown("---")
    
    # Nút dự đoán
    predict_button = st.button("🔮 DỰ ĐOÁN", type="primary", use_container_width=True)
    
    st.markdown("---")
    
    # ========================================================================
    # PHẦN KẾT QUẢ DỰ ĐOÁN
    # ========================================================================
    if predict_button:
        # Tạo sample
        sample = pd.DataFrame({
            'temperature_2m': [temperature],
            'relative_humidity_2m': [humidity],
            'dew_point_2m': [dew_point],
            'surface_pressure': [pressure],
            'cloud_cover': [cloud_cover],
            'wind_speed_10m': [wind_speed],
            'hour_sin': [hour_sin],
            'hour_cos': [hour_cos],
            'month_sin': [month_sin],
            'month_cos': [month_cos],
            'humidity_change': [humidity_change],
            'pressure_trend_6h': [pressure_trend],
            'press_std_12h': [press_std_12h],
            'hum_max_6h': [hum_max_6h],
            'temp_diff_3h': [temp_diff_3h]
        })
        
        st.header("🎯 Kết quả dự đoán từ 3 mô hình")
        
        # Màu sắc và icon cho từng loại thời tiết
        weather_style = {
            'Cloudy': {'icon': '☁️', 'color': '#FFB74D', 'vn': 'Nhiều mây'},
            'Drizzle': {'icon': '🌧️', 'color': '#64B5F6', 'vn': 'Mưa phùn'},
            'Rain': {'icon': '🌧️', 'color': '#81C784', 'vn': 'Mưa'}
        }
        
        # Chia 3 cột cho 3 mô hình
        col1, col2, col3 = st.columns(3)
        
        model_cols = [col1, col2, col3]
        model_names = ['Random Forest', 'AdaBoost', 'Softmax Regression']
        model_icons = ['🌲', '🚀', '📈']
        
        for i, (col, name, icon) in enumerate(zip(model_cols, model_names, model_icons)):
            with col:
                model = models[name]
                
                # Dự đoán
                pred_encoded = model.predict(sample)[0]
                prediction = label_encoder.inverse_transform([pred_encoded])[0]
                probabilities = model.predict_proba(sample)[0]
                
                style = weather_style[prediction]
                
                # Hiển thị kết quả
                st.markdown(f"### {icon} {name}")
                
                st.markdown(f"""
                <div style="background-color: {style['color']}; padding: 1.5rem; 
                            border-radius: 15px; text-align: center; margin-bottom: 1rem;">
                    <h2 style="color: #333; margin: 0;">{style['icon']} {style['vn']}</h2>
                    <p style="color: #555; margin: 0.5rem 0 0 0;">({prediction})</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Bảng xác suất
                st.markdown("**Xác suất từng lớp:**")
                for j, cls in enumerate(label_encoder.classes_):
                    prob = probabilities[j] * 100
                    ws = weather_style[cls]
                    st.progress(prob / 100, text=f"{ws['icon']} {cls}: {prob:.1f}%")
                
                # Biểu đồ
                fig, ax = plt.subplots(figsize=(6, 3))
                colors = [weather_style[c]['color'] for c in label_encoder.classes_]
                bars = ax.barh(label_encoder.classes_, probabilities * 100, color=colors, edgecolor='black')
                
                # Highlight prediction
                for j, cls in enumerate(label_encoder.classes_):
                    if cls == prediction:
                        bars[j].set_edgecolor('red')
                        bars[j].set_linewidth(2)
                
                ax.set_xlabel('Xác suất (%)')
                ax.set_xlim(0, 100)
                ax.set_title(f'{name}', fontsize=10, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()


# ============================================================================
# CHẠY ỨNG DỤNG
# ============================================================================
if __name__ == "__main__":
    main()
