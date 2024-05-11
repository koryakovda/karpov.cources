from datetime import datetime
import hashlib

import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI
from loguru import logger
from src.schema import PostGet, Response
from sqlalchemy import create_engine

app = FastAPI()

# LOADING MODELS
logger.info("Loading SQL path")
# config_file = "./config.txt"
config_file = "src/config.txt"
with open(config_file, "r") as f:
    config_data = f.readlines()

config = {}
for line in config_data:
    key, value = line.strip().split("=")
    config[key] = value

connection_path = f"postgresql://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"


def batch_load_sql(query: str):
    engine = create_engine(connection_path)
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=200000):
        chunks.append(chunk_dataframe)
        logger.info(f"Got chunk: {len(chunk_dataframe)}")
        break
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_raw_features():
    # Загрузим уникальные записи post_id, user_id
    # Где был совершен лайк
    logger.info("loading liked posts")
    liked_posts_query = """
        SELECT distinct post_id, user_id
        FROM public.feed_data
        where action='like'"""
    liked_posts = batch_load_sql(liked_posts_query)
    logger.info(f"loaded liked posts = {liked_posts.shape}")

    # Загрузим фичи по юзерам
    logger.info("loading user features")
    user_features = pd.read_sql(
        """SELECT * FROM public.user_data""",
        con=connection_path)

    # Загрузим обработанные в juputer notebook и загруженные в БД
    # фичи по постам на основе tf-idf
    logger.info("loading posts features")
    posts_features = pd.read_sql(
        """SELECT * FROM public.koriakov_posts_info_features""",
        con=connection_path)

    # Загрузим обработанные в juputer notebook фичи по постам на основе dl
    logger.info("loading posts features dl")
    posts_features_dl = pd.read_sql(
        """SELECT * FROM public.koriakov_posts_info_features_dl""",
        con=connection_path
    )

    return [liked_posts, posts_features, posts_features_dl, user_features]


def get_model_path(model_version: str) -> str:
    model_path = (f"src/model_{model_version}")
    return model_path


def load_models(model_version: str):
    """Тут мы делаем загрузку модели из класса CatBoost так как у нас
    control или test модели это модели CatBoost"""

    model_path = get_model_path(model_version)
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)
    return loaded_model


# USER SPLITTING

"""
Основная часть, где мы реализуем функцию для разбиения пользователей.
В идеале соль мы должны не задавать константой, а где-то конфигурировать.
В том числе сами границы, но сделать для простоты мы как раз разбиваем
50/50
"""

SALT = "my_salt"


def get_user_group(id: int) -> str:
    value_str = str(id) + SALT
    value_num = int(hashlib.md5(value_str.encode()).hexdigest(), 16)
    percent = value_num % 100
    if percent < 50:
        return "control"
    elif percent < 100:
        return "test"
    return "unknown"


### Загружаем все таблички из SQL

features = load_raw_features()  # [liked_posts, posts_features, posts_features_dl, user_features]

# Теперь мы загружаем сразу 2 модели
model_control = load_models("control")
model_test = load_models("test")


# RECOMMENDATIONS


def calculate_features(id: int, time: datetime, group: str, features: list):
    """
    Тут мы готовим фичи, при этом в зависимости от группы это могут быть
    разные фичи под разные модели. Здесь это одни и те же фичи (то есть будто
    бы разница в самих моделях)
    """

    # Добавим 2 функции которые нам потребуются для генерации фитчей по пользователям

    def count_users_in_city(df):
        count = df['city'].value_counts()  # Считаем число пользователей в каждом городе
        df['users_in_city'] = df['city'].map(count)  # Map'им к DataFrame
        return df

    def users_average_age_per_city(df):
        av_age = df.groupby('city')['age'].mean()
        df['av_age_per_city'] = df['city'].map(av_age)
        return df

    def user_feature_creation(df):
        df = df.copy()  # Создаем копию DataFrame чтобы не изменять исходный df
        df = count_users_in_city(df)
        df = users_average_age_per_city(df)

        return df

    # features = load_raw_features(group)

    if group == "control":
        content = features[1][['post_id', 'text', 'topic']]
        # Загрузим фичи по постам
        logger.info("dropping columns")
        posts_features = features[1].drop(["index", "text"], axis=1)
    elif group == "test":
        content = features[2][['post_id', 'text', 'topic']]
        # Загрузим фичи по постам
        logger.info("dropping columns")
        posts_features = features[2].drop(["index", "text"], axis=1)
    else:
        raise ValueError("unknown group")

    liked_posts = features[0]

    # Загрузим фичи по пользователям
    logger.info(f"user_id: {id}")
    logger.info("reading features")
    user_features_ = features[3].loc[features[3].user_id == id]
    user_features = user_feature_creation(user_features_)
    user_features = user_features.drop("user_id", axis=1)

    # Объединим эти фичи
    logger.info("zipping everything")
    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    logger.info("assigning everything")
    user_posts_features = posts_features.assign(**add_user_features)
    user_posts_features = user_posts_features.set_index("post_id")

    # Добафим информацию о дате рекомендаций
    logger.info("add time info")
    user_posts_features["hour"] = time.hour
    user_posts_features["month"] = time.month
    user_posts_features["day_of_week"] = time.strftime("%A")

    return user_features, user_posts_features, liked_posts, content


def get_recommended_feed(id: int, time: datetime, limit: int) -> Response:
    # Выбираем группу пользователи
    user_group = get_user_group(id=id)
    logger.info(f"user group {user_group}")

    # Выбираем нужную модель
    if user_group == "control":
        model = model_control
    elif user_group == "test":
        model = model_test
    else:
        raise ValueError("unknown group")

    # Вычисляем фичи
    user_features, user_posts_features, liked_posts, content = calculate_features(
        id=id, time=time, group=user_group, features=features)

    # Сортируем колонки в инференсе в таком же порядке как при трейне
    user_posts_features = user_posts_features[model.feature_names_]

    # Сформируем предсказания вероятности лайкнуть пост для всех постов
    logger.info("predicting")
    predicts = model.predict_proba(user_posts_features)[:, 1]
    user_posts_features["predicts"] = predicts

    # Уберем записи, где пользователь ранее уже ставил лайк
    logger.info("deleting liked posts")
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_ = user_posts_features[~user_posts_features.index.isin(liked_posts)]

    # Рекомендуем топ-5 по вероятности постов
    recommended_posts = filtered_.sort_values("predicts")[-limit:].index

    return Response(
        recommendations=[
            PostGet(id=i, text=content[content.post_id == i].text.values[0],
                    topic=content[content.post_id == i].topic.values[0])
            for i in recommended_posts
        ],
        exp_group=user_group,
    )


@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(id: int, time: datetime, limit: int = 10) -> Response:
    return get_recommended_feed(id, time, limit)

# if __name__ == '__main__':
#     time = datetime(2024, 3, 15)
#     print(recommended_posts(202, time, 5))
#     print(recommended_posts(201, time, 5))
#     print(recommended_posts(300, time, 5))
