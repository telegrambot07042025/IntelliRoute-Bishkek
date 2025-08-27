import streamlit as st
import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
import folium
from streamlit_folium import st_folium
import pickle
import requests
from datetime import datetime, time
import os

# ==============================================================================
# 1. НАСТРОЙКА СТРАНИЦЫ И ЗАГРУЗКА АРТЕФАКТОВ
# ==============================================================================

# Настраиваем конфигурацию страницы
st.set_page_config(
    page_title="IntelliRoute Bishkek",
    page_icon="🗺️",
    layout="wide"
)

# Используем кэширование, чтобы не загружать модели и граф при каждом действии пользователя
@st.cache_resource
def load_artifacts():
    # Важно: В Streamlit Cloud нужно использовать относительные пути
    graph_filepath = "datasets/bishkek_drive_graph.graphml"
    models_filepath = "traffic_models.pkl"

    # Проверяем, есть ли файлы
    if not os.path.exists(graph_filepath) or not os.path.exists(models_filepath):
        st.error("Ошибка: Файлы моделей или графа не найдены! Убедитесь, что они загружены в репозиторий.")
        return None, None
        
    G_bishkek = ox.load_graphml(graph_filepath)
    with open(models_filepath, 'rb') as f:
        models = pickle.load(f)
    return G_bishkek, models

# --- Backend-функции из Ячейки 3 (без изменений) ---
# ... (здесь мы скопируем все наши функции get_weather_forecast, prepare_inference_data, и т.д.) ...
def get_weather_forecast(api_key, lat, lon, target_datetime):
    logging.info(f"Запрашиваем прогноз погоды на {target_datetime}...")
    try:
        url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        forecasts = data['list']
        closest_forecast = min(forecasts, key=lambda x: abs(target_datetime.timestamp() - x['dt']))
        weather_data = {
            'temp': closest_forecast['main'].get('temp', 0),
            'rhum': closest_forecast['main'].get('humidity', 0),
            'wspd': closest_forecast['wind'].get('speed', 0),
            'prcp': closest_forecast.get('rain', {}).get('3h', 0) / 3,
            'snow': closest_forecast.get('snow', {}).get('3h', 0) / 3,
        }
        logging.info(f"Прогноз погоды получен: {weather_data}")
        return weather_data
    except Exception as e:
        logging.error(f"Не удалось получить прогноз погоды: {e}. Используем значения по умолчанию.")
        return {'temp': 15, 'rhum': 50, 'wspd': 5, 'prcp': 0, 'snow': 0}

# --- Функция 2: Подготовка данных для инференса (ИСПРАВЛЕНА) ---
def prepare_inference_data(graph, target_datetime, weather_data):
    logging.info("Подготовка данных для инференса для каждого ребра графа...")
    edges_df = ox.graph_to_gdfs(graph, nodes=False)
    
    # Создаем временные признаки
    edges_df['hour'] = target_datetime.hour
    edges_df['weekday'] = target_datetime.weekday()
    edges_df['month'] = target_datetime.month
    edges_df['is_weekend'] = (edges_df['weekday'] >= 5).astype(int)
    
    # Добавляем погодные признаки
    for key, value in weather_data.items():
        edges_df[key] = value
        
    # --- ИСПРАВЛЕННЫЙ БЛОК ---
    # Создаем "справочники" координат X и Y для всех узлов графа
    nodes_df = ox.graph_to_gdfs(graph, edges=False)
    x_coords = nodes_df['x']
    y_coords = nodes_df['y']
    
    # Используем .map() для эффективного добавления координат
    edges_df['pickup_longitude'] = edges_df.index.get_level_values('u').map(x_coords)
    edges_df['pickup_latitude'] = edges_df.index.get_level_values('u').map(y_coords)
    edges_df['dropoff_longitude'] = edges_df.index.get_level_values('v').map(x_coords)
    edges_df['dropoff_latitude'] = edges_df.index.get_level_values('v').map(y_coords)
    # --- КОНЕЦ ИСПРАВЛЕННОГО БЛОКА ---
    
    edges_df.rename(columns={'length': 'distance_m'}, inplace=True)
    edges_df['distance_km'] = edges_df['distance_m'] / 1000.0

    edges_df['vendor_id'] = 1
    edges_df['passenger_count'] = 1
    
    features_cols = [
        'vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude', 'hour', 'weekday', 'month',
        'is_weekend', 'distance_km', 'temp', 'rhum', 'prcp', 'snow', 'wspd'
    ]
    return edges_df[features_cols]

# --- Функции 3, 4, 5 (без изменений) ---
def get_predictions(models, features_df):
    logging.info("Получаем предсказания от моделей...")
    preds_median_log = models['q_0.5'].predict(features_df)
    preds_upper_log = models['q_0.9'].predict(features_df)
    time_per_km_median = np.expm1(preds_median_log)
    time_per_km_upper = np.expm1(preds_upper_log)
    return time_per_km_median, time_per_km_upper

def calculate_edge_costs(features_df, preds_median, preds_upper, strategy):
    logging.info(f"Рассчитываем стоимость проезда для стратегии: {strategy}")
    travel_time_median = features_df['distance_km'] * preds_median
    travel_time_upper = features_df['distance_km'] * preds_upper
    if strategy == 'fastest':
        costs = travel_time_median
    else: # 'eco'
        smoothness_penalty = travel_time_upper - travel_time_median
        costs = travel_time_median + smoothness_penalty
    return costs, travel_time_median, travel_time_upper

def find_route(graph, start_coords, end_coords, attr_name='cost'):
    logging.info(f"Ищем маршрут от {start_coords} до {end_coords}")
    start_node = ox.nearest_nodes(graph, Y=start_coords[0], X=start_coords[1])
    end_node = ox.nearest_nodes(graph, Y=end_coords[0], X=end_coords[1])
    try:
        route = nx.shortest_path(graph, source=start_node, target=end_node, weight=attr_name)
        logging.info(f"Маршрут найден. Состоит из {len(route)} узлов.")
        return route
    except nx.NetworkXNoPath:
        logging.error("Маршрут между указанными точками не найден.")
        return None


# ==============================================================================
# 2. ИНТЕРФЕЙС ПРИЛОЖЕНИЯ (UI)
# ==============================================================================

st.title("IntelliRoute: Ваш Умный Навигатор по Бишкеку 🗺️")
st.markdown("Планируйте поездки заранее, чтобы избежать пробок! Модель прогнозирует трафик на основе времени, дня недели и погоды.")

# Загружаем артефакты
G_bishkek, models = load_artifacts()

if G_bishkek is None or models is None:
    st.stop() # Останавливаем выполнение, если файлы не загрузились

# Создаем две колонки для интерфейса
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ Параметры поездки")
    
    with st.form("trip_form"):
        # Координаты с значениями по умолчанию
        start_lat = st.number_input("Широта старта:", value=42.8759, format="%.4f")
        start_lon = st.number_input("Долгота старта:", value=74.5885, format="%.4f")
        end_lat = st.number_input("Широта финиша:", value=42.8781, format="%.4f")
        end_lon = st.number_input("Долгота финиша:", value=74.6133, format="%.4f")
        
        # Выбор даты и времени
        arrival_date = st.date_input("Дата прибытия:", value=datetime.now())
        arrival_time = st.time_input("Время прибытия:", value=time(10, 0))
        
        # Выбор стратегии
        strategy = st.radio("Выберите приоритет:", ('fastest', 'eco'), captions=["Максимально быстро", "Сэкономить топливо / Ехать спокойнее"])
        
        # Ключ API (используем секреты Streamlit)
        api_key = st.text_input("Ваш API ключ от OpenWeatherMap:", type="password", help="Получите бесплатно на openweathermap.org")
        
        # Кнопка для отправки формы
        submitted = st.form_submit_button("Рассчитать маршрут")

# ==============================================================================
# 3. ЛОГИКА РАСЧЕТА И ВЫВОД РЕЗУЛЬТАТА
# ==============================================================================

with col2:
    st.subheader("📍 Карта и Результат")
    
    # Создаем "пустое место" для карты
    map_container = st.container()
    
    with map_container:
        # Отображаем карту Бишкека по умолчанию
        m = folium.Map(location=[42.87, 74.6], zoom_start=13, tiles="cartodbpositron")
        st_folium(m, height=450, use_container_width=True)

if submitted:
    if not api_key:
        st.error("Пожалуйста, введите ваш API ключ от OpenWeatherMap.")
    else:
        with st.spinner('🧠 Думаю... Прогнозирую трафик и ищу лучший путь...'):
            # --- Запуск всей Backend-логики ---
            arrival_datetime = datetime.combine(arrival_date, arrival_time)
            
            weather = get_weather_forecast(api_key, start_lat, start_lon, arrival_datetime)
            features = prepare_inference_data(G_bishkek, arrival_datetime, weather)
            preds_med, preds_up = get_predictions(models, features)
            costs, travel_time_med, travel_time_up = calculate_edge_costs(features, preds_med, preds_up, strategy)

            # Обновляем атрибуты графа
            costs_dict = costs.to_dict()
            time_median_dict = travel_time_med.to_dict()
            time_upper_dict = travel_time_up.to_dict()
            nx.set_edge_attributes(G_bishkek, values=costs_dict, name='cost')
            nx.set_edge_attributes(G_bishkek, values=time_median_dict, name='time_median')
            nx.set_edge_attributes(G_bishkek, values=time_upper_dict, name='time_upper')
            
            route_nodes = find_route(G_bishkek, (start_lat, start_lon), (end_lat, end_lon), attr_name='cost')

            if route_nodes:
                # --- Обработка и вывод результата ---
                path_edges = list(zip(route_nodes, route_nodes[1:]))
                total_time_median = sum(G_bishkek.get_edge_data(u, v)[0]['time_median'] for u, v in path_edges)
                total_time_upper = sum(G_bishkek.get_edge_data(u, v)[0]['time_upper'] for u, v in path_edges)
                departure_time = arrival_datetime - pd.to_timedelta(total_time_upper, unit='s')
                
                # Обновляем левую колонку с результатами
                with col1:
                    st.success("✅ Ваш маршрут готов!")
                    st.metric("Рекомендуемое время выезда:", departure_time.strftime('%H:%M'))
                    st.metric("Прогноз времени в пути (минут):", f"{total_time_median/60:.0f} - {total_time_upper/60:.0f}")

                # Обновляем карту в правой колонке
                with map_container:
                    route_map = ox.plot_route_folium(G_bishkek, route_nodes, tiles='cartodbpositron')
                    folium.Marker(location=[start_lat, start_lon], popup="Старт", icon=folium.Icon(color='green')).add_to(route_map)
                    folium.Marker(location=[end_lat, end_lon], popup="Финиш", icon=folium.Icon(color='red')).add_to(route_map)
                    st_folium(route_map, height=450, use_container_width=True)
            else:
                st.error("Не удалось построить маршрут между указанными точками.")
