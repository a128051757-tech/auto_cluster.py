import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import urllib.request

# ================= 設定網頁外觀 =================
st.set_page_config(page_title="AOI 瑕疵辨識系統", page_icon="🏭", layout="wide")
st.title("🏭 AOI 瑕疵自動辨識系統")
st.write("請上傳機台截取的小圖，AI 將自動為您分類缺陷種類。")
st.markdown("---")

# ================= 1. 載入模型 (含自動下載機制) =================

@st.cache_resource
def load_model():
    # 🌟 這裡偷偷改個檔名，強迫伺服器重新下載！
    model_path = 'model_v2.h5' 
    
    # 🌟 請把剛才「按右鍵複製」的真正下載網址貼在這裡！(確認結尾是 .h5)
    MODEL_URL = 'https://github.com/你的帳號/你的專案/releases/download/v1.0/particle_classifier_opt.h5'
    
    # 如果雲端伺服器上沒有新檔名的模型，就自動下載
    if not os.path.exists(model_path):
        with st.spinner('首次載入，正在從雲端下載 AI 模型 (約 49MB)，請稍候...'):
            try:
                urllib.request.urlretrieve(MODEL_URL, model_path)
            except Exception as e:
                st.error(f"下載模型失敗，請檢查網址是否正確！錯誤訊息: {e}")
                st.stop()
            
    return tf.keras.models.load_model(model_path)
# ================= 2. 設定類別名稱 =================
# 必須與你訓練時的 dataset 資料夾名稱順序一致
class_names = ['other', 'pinhole_clear', 'semi_transparent', 'solid_block']

st.sidebar.info(f"**目前可辨識的類別：**\n" + "\n".join([f"- {name}" for name in class_names]))

# ================= 3. 圖片上傳區塊 =================
uploaded_files = st.file_uploader("請選擇要辨識的圖片 (可一次框選多張)", 
                                  type=['png', 'jpg', 'jpeg', 'bmp'], 
                                  accept_multiple_files=True)

# ================= 4. 辨識與結果顯示 =================
if uploaded_files:
    st.write(f"### 📊 辨識結果 (共 {len(uploaded_files)} 張)")
    
    # 建立 4 個欄位 (Columns) 來讓圖片並排顯示，畫面比較好看
    cols = st.columns(4)
    
    for idx, file in enumerate(uploaded_files):
        try:
            # 讀取圖片
            image = Image.open(file).convert('RGB')
            
            # 圖片預處理 (轉成 128x128 給 AI 看)
            img_resized = image.resize((128, 128))
            img_array = tf.keras.utils.img_to_array(img_resized)
            img_array = tf.expand_dims(img_array, 0)

            # 進行預測
            predictions = model.predict(img_array, verbose=0)
            score = tf.nn.softmax(predictions[0])
            predicted_class = class_names[np.argmax(score)]
            confidence = 100 * np.max(score)

            # 決定要放在哪一個欄位 (例如第 5 張圖會回到第 1 欄)
            col = cols[idx % 4]
            
            with col:
                # 顯示圖片
                st.image(image, use_container_width=True)
                
                # 根據信心度給予不同的顏色提示 (紅/黃/綠)
                if confidence > 80:
                    st.success(f"**{predicted_class}**\n\n信心度: {confidence:.1f}%")
                elif confidence > 50:
                    st.warning(f"**{predicted_class}**\n\n信心度: {confidence:.1f}%")
                else:
                    st.error(f"**{predicted_class}**\n\n信心度: {confidence:.1f}% (不確定)")
        except Exception as e:
            st.error(f"圖片處理失敗: {e}")
