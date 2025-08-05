import streamlit as st
import pandas as pd
import numpy as np
import re
import math
import os
import pickle
from datetime import datetime, timedelta
from typing import Union, Optional, Dict, Any, List, Tuple
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# Streamlit sayfa konfigürasyonu
st.set_page_config(
    page_title="💻 Akıllı Laptop Öneri Sistemi",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS ile özelleştirmeler
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    text-align: center;
    color: white;
    margin-bottom: 2rem;
}
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-left: 4px solid #667eea;
}
.deal-card {
    background: linear-gradient(135deg, #ff6b6b, #ff8e53);
    color: white;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}
.recommendation-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    background: #f9f9f9;
}
</style>
""", unsafe_allow_html=True)

class Config:
    """Streamlit için uyarlanmış konfigürasyon"""
    
    # Streamlit'te dosya yolları data klasöründe olacak
    DATASET_PATHS = [
        'data/vatan_laptop_data_cleaned.csv',
        'data/amazon_final.csv', 
        'data/cleaned_incehesap_data.csv'
    ]
    
    CACHE_FILE = 'data/laptop_cache.pkl'
    CACHE_DURATION = timedelta(hours=24)
    
    # GPU skorları (aynı)
    GPU_SCORES = {
        'rtx5090': 110, 'rtx5080': 105, 'rtx5070': 100, 'rtx5060': 85, 'rtx5050': 75,
        'rtx5000': 96, 'rtx4090': 100, 'rtx4080': 95, 'rtx3080': 88, 'rtx4070': 85,
        'rtx3070': 80, 'rtx4060': 75, 'rtx3060': 70, 'rtx4050': 60,
        'rtx3050': 55, 'rtx2060': 50, 'rtx': 45, 'gtx': 40,
        'mx550': 38, 'intel arc': 45, 'apple integrated': 35, 'intel uhd': 22,
        'intel iris xe graphics': 25, 'iris xe': 25, 'integrated': 20,
        'unknown': 30
    }
    DEFAULT_GPU_SCORE = 30
    
    # CPU skorları (aynı)
    CPU_SCORES = {
        'ultra 9 275hx': 98, 'ultra 9': 100, 'ultra 7 255h': 92, 'ultra 7': 90,
        'ultra 5 155h': 83, 'ultra 5': 80, 'core ultra 9': 98, 'core ultra 7': 90,
        'core ultra 5': 83, 'ryzen ai 9 hx370': 95, 'core 5 210h': 75,
        'i9': 95, 'i7': 85, 'i5': 75, 'i3': 60,
        'ryzen 9': 95, 'ryzen 7': 85, 'ryzen 5': 75, 'ryzen 3': 60,
        'm4 pro': 94, 'm4': 88, 'm3': 85, 'm2': 80, 'm1': 75,
        'snapdragon x': 78, 'unknown': 50
    }
    
    # Marka skorları (aynı)
    BRAND_SCORES = {
        'apple': 0.95, 'dell': 0.85, 'hp': 0.80, 'lenovo': 0.85,
        'asus': 0.82, 'msi': 0.80, 'acer': 0.75, 'monster': 0.70,
        'huawei': 0.78, 'samsung': 0.83, 'lg': 0.77, 'gigabyte': 0.76
    }
    
    # Puanlama ağırlıkları (aynı)
    WEIGHTS = {
        'price_fit': 15,
        'price_performance': 10,
        'purpose': {
            'base': 30,
            'oyun': {'dedicated': 1.0, 'integrated': 0.1, 'apple': 0.5},
            'taşınabilirlik': {'dedicated': 0.2, 'integrated': 1.0, 'apple': 0.9},
            'üretkenlik': {'dedicated': 0.6, 'integrated': 0.4, 'apple': 1.0},
            'tasarım': {'dedicated': 0.8, 'integrated': 0.5, 'apple': 1.0},
        },
        'user_preferences': {
            'performance': 12,
            'battery': 12,
            'portability': 8,
        },
        'specs': {
            'ram': 5,
            'ssd': 5,
        },
        'brand_reliability': 8,
    }
    
    SCREEN_CATEGORIES = {
        1: (13, 14),  # Kompakt
        2: (15, 16),  # Standart
        3: (17, 18),  # Büyük
        4: (0, 99)    # Farketmez
    }

class StreamlitDataHandler:
    """Streamlit için veri işleme"""
    
    def __init__(self, config: Config):
        self.config = config
    
    @st.cache_data
    def load_sample_data(_self):
        """Demo amaçlı örnek veri oluştur - DÜZELTME"""
        np.random.seed(42)
        
        brands = ['asus', 'hp', 'lenovo', 'dell', 'apple', 'msi', 'acer']
        gpus = ['rtx4060', 'rtx3050', 'integrated', 'apple integrated', 'rtx4070', 'mx550']
        cpus = ['i7', 'i5', 'ryzen 7', 'ryzen 5', 'm3', 'i9']
        
        n_samples = 500
        data = []
        
        for i in range(n_samples):
            brand = np.random.choice(brands)
            gpu = np.random.choice(gpus)
            cpu = np.random.choice(cpus)
            
            # Fiyat mantığına göre özellikler
            if gpu in ['rtx4070', 'rtx4060']:
                base_price = np.random.normal(35000, 8000)
                ram = np.random.choice([16, 32], p=[0.7, 0.3])
                ssd = np.random.choice([512, 1024], p=[0.6, 0.4])
            elif gpu == 'integrated':
                base_price = np.random.normal(20000, 5000)
                ram = np.random.choice([8, 16], p=[0.6, 0.4])
                ssd = np.random.choice([256, 512], p=[0.7, 0.3])
            else:
                base_price = np.random.normal(25000, 6000)
                ram = np.random.choice([8, 16], p=[0.5, 0.5])
                ssd = np.random.choice([512, 1024], p=[0.8, 0.2])
            
            price = max(10000, int(base_price))
            screen_size = np.random.choice([13.3, 14.0, 15.6, 17.3], p=[0.2, 0.3, 0.4, 0.1])
            
            # ÖNEMLİ: Tüm gerekli sütunları ekleyin
            data.append({
                'url': f'https://example.com/laptop-{i}',
                'name': f'{brand.title()} Laptop Model {i}',
                'brand': brand,
                'price': price,
                'screen_size': screen_size,
                'ssd_gb': ssd,
                'ram_gb': ram,
                'os': 'WINDOWS 11' if brand != 'apple' else 'MACOS',
                'gpu_clean': gpu,
                'cpu_clean': cpu,
                # Bu satırlar eksikti - EKLENDİ:
                'gpu_score': _self.config.GPU_SCORES.get(gpu, 30),
                'cpu_score': _self.config.CPU_SCORES.get(cpu, 50),
                'brand_score': _self.config.BRAND_SCORES.get(brand, 0.70),
                'is_apple': brand == 'apple',
                'has_dedicated_gpu': gpu not in ['integrated', 'apple integrated'],
                'is_price_outlier': False,
                'is_suspicious': False,
                'data_source': 'demo'
            })
        
        df = pd.DataFrame(data)
        return df
    
    @st.cache_data
    def load_data(_self):
        """Veri yükleme (gerçek veriler varsa onları, yoksa demo)"""
        try:
            # Önce gerçek verileri dene
            datasets = []
            for path in _self.config.DATASET_PATHS:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    datasets.append(df)
            
            if datasets:
                # Gerçek veriler bulundu
                combined_df = pd.concat(datasets, ignore_index=True)
                # Veri temizleme işlemleri burada yapılacak
                return combined_df
            else:
                # Demo veri kullan
                return _self.load_sample_data()
                
        except Exception as e:
            st.error(f"Veri yükleme hatası: {e}")
            return _self.load_sample_data()

class StreamlitScoringEngine:
    """Streamlit için puanlama motoru"""
    
    def __init__(self, config: Config):
        self.config = config
        self.weights = config.WEIGHTS
    
    def calculate_scores(self, df: pd.DataFrame, preferences: Dict[str, Any]) -> pd.DataFrame:
        """Puanlama hesapla"""
        # Filtreleri uygula
        filtered_df = self._apply_filters(df, preferences)
        
        if filtered_df.empty:
            return pd.DataFrame()
        
        # Her laptop için puan hesapla
        scores = []
        for _, row in filtered_df.iterrows():
            score = self._calculate_single_score(row, preferences)
            scores.append(score)
        
        filtered_df = filtered_df.copy()
        filtered_df['score'] = scores
        
        return filtered_df.sort_values('score', ascending=False)
    
    def _apply_filters(self, df: pd.DataFrame, preferences: Dict[str, Any]) -> pd.DataFrame:
        """Filtreleri uygula"""
        filtered_df = df.copy()
        
        # Fiyat filtresi
        filtered_df = filtered_df[
            (filtered_df['price'] >= preferences['min_budget']) &
            (filtered_df['price'] <= preferences['max_budget'])
        ]
        
        # Ekran boyutu filtresi
        if preferences.get('screen_preference', 4) != 4:
            min_size, max_size = self.config.SCREEN_CATEGORIES[preferences['screen_preference']]
            filtered_df = filtered_df[
                (filtered_df['screen_size'] >= min_size) &
                (filtered_df['screen_size'] <= max_size)
            ]
        
        # İşletim sistemi filtresi
        if preferences.get('os_preference', 3) == 1:  # Windows
            filtered_df = filtered_df[filtered_df['os'].str.contains('WINDOWS', na=False)]
        elif preferences.get('os_preference', 3) == 2:  # macOS
            filtered_df = filtered_df[filtered_df['is_apple'] == True]
        
        # Marka filtresi
        if preferences.get('brand_preference'):
            filtered_df = filtered_df[filtered_df['brand'] == preferences['brand_preference']]
        
        # Minimum donanım
        if preferences.get('min_ram', 8):
            filtered_df = filtered_df[filtered_df['ram_gb'] >= preferences['min_ram']]
        if preferences.get('min_ssd', 256):
            filtered_df = filtered_df[filtered_df['ssd_gb'] >= preferences['min_ssd']]
        
        return filtered_df
    
    def _calculate_single_score(self, row: pd.Series, preferences: Dict[str, Any]) -> float:
        """Tek laptop için puan hesapla"""
        total_score = 0
        
        # Fiyat uygunluğu
        price_diff = abs(row['price'] - preferences['ideal_price'])
        price_range = preferences['max_budget'] - preferences['min_budget']
        if price_range > 0:
            price_score = self.weights['price_fit'] * max(0, 1 - price_diff / (price_range / 2))
            total_score += price_score
        
        # Kullanım amacı puanı
        purpose_weights = self.weights['purpose'][preferences['purpose']]
        if row['is_apple']:
            multiplier = purpose_weights['apple']
        elif row['has_dedicated_gpu']:
            multiplier = purpose_weights['dedicated']
        else:
            multiplier = purpose_weights['integrated']
        
        combined_performance = (row['gpu_score'] * 0.7 + row['cpu_score'] * 0.3) / 100
        purpose_score = self.weights['purpose']['base'] * combined_performance * multiplier
        total_score += purpose_score
        
        # Donanım puanları
        ram_score = self.weights['specs']['ram'] * min(row['ram_gb'] / 16, 1.0)
        ssd_score = self.weights['specs']['ssd'] * min(row['ssd_gb'] / 1024, 1.0)
        total_score += ram_score + ssd_score
        
        # Marka güvenilirlik puanı
        brand_score = self.weights['brand_reliability'] * row['brand_score']
        total_score += brand_score
        
        return min(100, max(0, total_score))

class StreamlitTrendAnalyzer:
    """Streamlit için trend analizi"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
    
    def find_deals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fırsat ürünleri bul"""
        if len(df) < 10:
            return pd.DataFrame()
        
        # Performans skorunu hesapla
        df = df.copy()
        df['performance_score'] = (df['gpu_score'] * 0.6 + df['cpu_score'] * 0.4)
        
        # Anomali tespiti için özellikler
        features = df[['price', 'performance_score', 'ram_gb', 'ssd_gb']].fillna(0)
        
        # Anomali tespiti
        try:
            outliers = self.anomaly_detector.fit_predict(features)
            df['is_deal'] = outliers == -1  # Anomaliler fırsat
            
            deals = df[df['is_deal']].copy()
            
            if not deals.empty:
                # Fırsat skorunu hesapla
                deals['deal_score'] = self._calculate_deal_score(deals)
                return deals.sort_values('deal_score', ascending=False)
            
        except Exception:
            pass
        
        return pd.DataFrame()
    
    def _calculate_deal_score(self, deals_df: pd.DataFrame) -> pd.Series:
        """Fırsat skorunu hesapla"""
        # Basit fırsat skoru: performans/fiyat oranı
        performance_price_ratio = deals_df['performance_score'] / (deals_df['price'] / 10000)
        return performance_price_ratio * 10

def main():
    """Ana Streamlit uygulaması"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>💻 Akıllı Laptop Öneri Sistemi</h1>
        <p>Size özel laptop önerileri ve pazar analizleri</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Konfigürasyon ve veri yükleme
    config = Config()
    data_handler = StreamlitDataHandler(config)
    scoring_engine = StreamlitScoringEngine(config)
    trend_analyzer = StreamlitTrendAnalyzer()
    
    # Veriyi yükle
    with st.spinner('Veriler yükleniyor...'):
        df = data_handler.load_data()
    
    st.success(f"✅ {len(df)} laptop başarıyla yüklendi!")
    
    # Sidebar - Kullanıcı tercihleri
    st.sidebar.header("🎯 Tercihleriniz")
    
    # Ana sekmeleri oluştur
    tab1, tab2, tab3 = st.tabs(["🎯 Kişisel Öneri", "🔥 Fırsat Ürünleri", "📊 Pazar Analizi"])
    
    with tab1:
        # Kişisel öneri sekmesi
        st.header("Sizin İçin Özel Laptop Önerileri")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Bütçe")
            min_budget = st.number_input("Minimum bütçe (TL)", min_value=5000, max_value=200000, value=20000, step=1000)
            max_budget = st.number_input("Maksimum bütçe (TL)", min_value=min_budget, max_value=200000, value=50000, step=1000)
            
            st.subheader("Kullanım Amacı")
            purpose = st.selectbox(
                "Ana kullanım amacınız:",
                options=['oyun', 'taşınabilirlik', 'üretkenlik', 'tasarım'],
                format_func=lambda x: {
                    'oyun': '🎮 Oyun',
                    'taşınabilirlik': '🎒 Taşınabilirlik', 
                    'üretkenlik': '💼 Üretkenlik',
                    'tasarım': '🎨 Tasarım'
                }[x]
            )
            
            st.subheader("Önem Dereceleri")
            performance_importance = st.slider("Performans", 1, 5, 4)
            battery_importance = st.slider("Pil Ömrü", 1, 5, 3)
            portability_importance = st.slider("Taşınabilirlik", 1, 5, 3)
            
            # Gelişmiş filtreler
            with st.expander("🔧 Gelişmiş Filtreler"):
                screen_preference = st.selectbox(
                    "Ekran boyutu:",
                    options=[4, 1, 2, 3],
                    format_func=lambda x: {
                        1: "Kompakt (13-14\")",
                        2: "Standart (15-16\")", 
                        3: "Büyük (17\"+)",
                        4: "Farketmez"
                    }[x]
                )
                
                os_preference = st.selectbox(
                    "İşletim sistemi:",
                    options=[3, 1, 2],
                    format_func=lambda x: {
                        1: "Windows",
                        2: "macOS",
                        3: "Farketmez"
                    }[x]
                )
                
                brand_preference = st.selectbox(
                    "Marka tercihi:",
                    options=[None] + list(config.BRAND_SCORES.keys()),
                    format_func=lambda x: "Farketmez" if x is None else x.title()
                )
                
                min_ram = st.selectbox("Minimum RAM (GB):", [4, 8, 16, 32], index=1)
                min_ssd = st.selectbox("Minimum SSD (GB):", [128, 256, 512, 1024], index=1)
        
        with col2:
            if st.button("🔍 Laptop Önerilerini Getir", type="primary"):
                preferences = {
                    'min_budget': min_budget,
                    'max_budget': max_budget,
                    'ideal_price': (min_budget + max_budget) / 2,
                    'purpose': purpose,
                    'performance_importance': performance_importance,
                    'battery_importance': battery_importance,
                    'portability_importance': portability_importance,
                    'screen_preference': screen_preference,
                    'os_preference': os_preference,
                    'brand_preference': brand_preference,
                    'min_ram': min_ram,
                    'min_ssd': min_ssd
                }
                
                with st.spinner('Laptoplar analiz ediliyor...'):
                    recommendations = scoring_engine.calculate_scores(df, preferences)
                
                if not recommendations.empty:
                    st.success(f"✅ {len(recommendations)} laptop bulundu!")
                    
                    # En iyi 5 öneriyi göster
                    top_5 = recommendations.head(5)
                    
                    for i, (_, laptop) in enumerate(top_5.iterrows(), 1):
                        with st.container():
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <h4>{i}. {laptop['name']}</h4>
                                <div style="display: flex; justify-content: space-between; margin: 1rem 0;">
                                    <span><strong>Fiyat:</strong> {laptop['price']:,.0f} TL</span>
                                    <span><strong>Puan:</strong> {laptop['score']:.1f}/100</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Detaylar
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.write(f"**Ekran:** {laptop['screen_size']}\"")
                                st.write(f"**İşlemci:** {laptop['cpu_clean'].upper()}")
                            with col_b:
                                st.write(f"**RAM:** {int(laptop['ram_gb'])} GB")
                                st.write(f"**SSD:** {int(laptop['ssd_gb'])} GB")
                            with col_c:
                                st.write(f"**GPU:** {laptop['gpu_clean'].upper()}")
                                st.write(f"**Marka:** {laptop['brand'].title()}")
                            
                            # Link
                            st.markdown(f"[🔗 Ürüne Git]({laptop['url']})")
                            st.divider()
                    
                    # Karşılaştırma grafiği
                    if len(top_5) > 1:
                        st.subheader("📊 Karşılaştırma")
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=top_5['price'],
                            y=top_5['score'],
                            mode='markers+text',
                            text=top_5['name'].str[:20] + '...',
                            textposition="top center",
                            marker=dict(size=12, color=top_5['score'], colorscale='RdYlGn'),
                            name='Laptoplar'
                        ))
                        
                        fig.update_layout(
                            title="Fiyat vs Puan Karşılaştırması",
                            xaxis_title="Fiyat (TL)",
                            yaxis_title="Puan",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                else:
                    st.warning("❌ Kriterlere uygun laptop bulunamadı. Lütfen filtrelerinizi genişletin.")
    
    with tab2:
        # Fırsat ürünleri sekmesi
        st.header("🔥 Günün Fırsat Ürünleri")
        st.write("Piyasa analizi sonucu tespit edilen avantajlı fiyatlı laptoplar")
        
        # SADECE BUTONA TIKLANINCA ÇALIŞ
        if st.button("🔍 Fırsat Ürünlerini Analiz Et", type="primary"):
            with st.spinner('Fırsat ürünleri analiz ediliyor...'):
                deals = trend_analyzer.find_deals(df)
            
            if not deals.empty:
                st.success(f"🎯 {len(deals)} fırsat ürün tespit edildi!")
                
                for i, (_, deal) in enumerate(deals.head(5).iterrows(), 1):
                    st.markdown(f"""
                    <div class="deal-card">
                        <h4>🔥 {i}. {deal['name']}</h4>
                        <p><strong>Fiyat:</strong> {deal['price']:,.0f} TL | <strong>Fırsat Skoru:</strong> {deal['deal_score']:.1f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col_x, col_y = st.columns(2)
                    with col_x:
                        st.write(f"**Performans Skoru:** {deal['performance_score']:.1f}")
                        st.write(f"**RAM/SSD:** {int(deal['ram_gb'])}GB / {int(deal['ssd_gb'])}GB")
                    with col_y:
                        st.write(f"**GPU:** {deal['gpu_clean'].upper()}")
                        st.write(f"**Marka:** {deal['brand'].title()}")
                    
                    st.markdown(f"[🛒 Ürüne Git]({deal['url']})")
                    st.divider()
            else:
                st.info("📊 Şu anda belirgin fırsat ürün tespit edilemedi.")
        else:
            st.info("👆 Fırsat ürünleri analizi için butona tıklayın")
    
    with tab3:
        # Pazar analizi sekmesi
        st.header("📊 Pazar Analizi")
        
        col_1, col_2 = st.columns(2)
        
        with col_1:
            # Marka dağılımı
            brand_counts = df['brand'].value_counts()
            fig_brand = px.pie(
                values=brand_counts.values,
                names=brand_counts.index,
                title="Marka Dağılımı"
            )
            st.plotly_chart(fig_brand, use_container_width=True)
        
        with col_2:
            # Fiyat dağılımı
            fig_price = px.histogram(
                df,
                x='price',
                nbins=30,
                title="Fiyat Dağılımı",
                labels={'price': 'Fiyat (TL)', 'count': 'Adet'}
            )
            st.plotly_chart(fig_price, use_container_width=True)
        
        # RAM vs SSD analizi
        fig_specs = px.scatter(
            df,
            x='ram_gb',
            y='ssd_gb',
            color='price',
            size='gpu_score',
            hover_data=['brand', 'cpu_clean'],
            title="RAM vs SSD Analizi (Renk: Fiyat, Boyut: GPU Skoru)",
            labels={'ram_gb': 'RAM (GB)', 'ssd_gb': 'SSD (GB)'}
        )
        st.plotly_chart(fig_specs, use_container_width=True)
        
        # İstatistikler
        st.subheader("📈 Genel İstatistikler")
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("Toplam Laptop", len(df))
        with col_stat2:
            st.metric("Ortalama Fiyat", f"{df['price'].mean():,.0f} TL")
        with col_stat3:
            st.metric("En Düşük Fiyat", f"{df['price'].min():,.0f} TL")
        with col_stat4:
            st.metric("En Yüksek Fiyat", f"{df['price'].max():,.0f} TL")

if __name__ == "__main__":
    main()
