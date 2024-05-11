# Rec_sys_final_project

[Karpov Courses Start ML](https://karpov.courses/ml-start)

## Описание проекта

Целью этого проекта было создание готового сервиса рекомендательных систем на FAST API, способного предлагать персонализированные рекомендации постов на основе предпочтений пользователей с использованием 2х различных моделей в условной социальной сети.

## Структура:

```buildoutcfg
|   README.md                                  Файл с описанием проекта
|   requirements.txt                           Файл для настройки окружения
|   docker-compose.yaml                        Docker-compose файл для поднятия контейнера
|   Dockerfile                                 Dockerfile для создания образа
|   .dockerignore     
|   .gitignore
├───src                                        Готовый сервис
│       app.py                                 Приложение
│       model_control                          Модель контрольная
│       model_test                             Модель тестовая
│       config.txt                             Конфиг файл БД
│       schema.py                              Классы по которым валидировали выходы функция
│       __init__.py
│       request_examples.txt                   Примеры запросов
└───notebooks                                  Ноутбуки с обучением моделей
    ├───kc-fin-project-catboost.ipynb          Ноутбук с первой моделью
    ├───kc-fin-project-nn-catboost.ipynb       Ноутбук со второй моделью
    └───draft jupyter                          Черновики, используемые для финальных ноутбуков
        ├───kc-fin-project-catboost.ipynb      Черновики с первой моделью
        └───kc-fin-project-nn-catboost.ipynb   Черновики со второй моделью

```
### Исходные данные
##### Пользователи сервиса
| Поле      | Описание                        |
|-----------|---------------------------------|
| age       | Возраст                         |
| city      | Город                           |
| country   | Страна                          |
| exp_group | Экспериментальная группа        |
| gender    | Пол                             |
| id        | Идентификатор                   |
| os        | Операционная система устройства |
| source    | Источник трафика                |
Количество зарегистрированных пользователей: ~163 тысячи

##### Новостные посты
| Поле  | Описание      |
|-------|---------------|
| id    | Идентификатор |
| text  | Текст         |
| topic | Тематика      |
Количество постов: ~7 тысяч

##### Логи действий пользователей с постами ленты
| Поле      | Описание                                 |
|-----------|------------------------------------------|
| timestamp | Время                                    |
| user_id   | id пользователя                          |
| post_id   | id поста                                 |
| action    | Совершённое действие - просмотр или лайк |
| target    | Поставлен ли лайк **после** просмотра.   |
Количество записей: ~77 миллионов

##### Параметры запроса
user_id указывать в пределах от 200 до 168552, при этом около 5000 id не занято, при вводе несуществующего польлзователя получаем ошибку (так сделано осознанно т.к. по условию задания 
у нас нет новых пользователей, база неизменна, поэтому выдавать 'среднего' пользователя при отсутствии в базе user_id я не стал)

request_time указывать в пределах от 1.10.2021 до 29.12.2021 -
это тоже одно из принятых допущений задания

Подробнее как делать запрос описано в [файле](request_examples.txt).

| Поле         | Описание                  |
|--------------|---------------------------|
| user_id      | id пользователя           |
| request_time | Время запроса             |
| posts_limit  | Количество постов в ленте |

##### Параметры отклика (одного поста из ленты)
| Поле      | Описание    |
|-----------|-------------|
| exp_group | Группа пользователя  |
| id        | id поста    |
| text      | Текст поста |
| topic     | Тема поста  |


### Метрика
Оценка качества обученных алгоритмов будет замеряться по метрике hitrate@5 - есть ли хотя бы один лайк от пользователя в показанной ему ленте.

### Технические требования и принятые допущения
1. Время отклика сервиса на 1 запрос не должен превышать 0.5 секунд.
2. Сервис не должен занимать более 2 Гб памяти системы.
3. Сервис должен иметь минимум две модели - обученную методом классического ML и с использованием DL.
4. Набор юзеров фиксирован и никаких новых появляться не будет.
5. Временные рамки подаваемых сервису запросов ограничены предельными значениями в логах действий пользователей.
6. Модели не обучаются заново при использовании сервисов. Ожидается, что код будет импортировать уже обученную модель и применять ее.

### Перечень используемых методов и технологий с краткими описаниями

В проекте был использован контентный подход т.к. у нас очень много данных и мы не можем загрузить все 77млн взаимодейстыий пользователей с постами для построения коллабаративной фильтрации.
С целью уменьшить потребляемую память из базы данных было выгружено 6 миллионов записей.

Так как сервис должен иметь возможность давать рекомендации двумя различными моделями, 
необходимо сделать и обосновать выбор.
1. Рекомендационную модель на методе классического ML решено обучать с помощью CatboostClassifier.
Данный выбор обоснован тестированием различных моделей и необходимостью ранжировать полученные моделью предсказания,
в результате чего лучшие результаты показал именно CatboostClassifier.
Так как целевым показателем для нас является получение like хотя бы у одного поста в выдаче (согласно метрике hitrate@5),
ранжировать предсказания будем по вероятностям, полученными в классификаторе.
3. Вторая модель представляет собой тот же CatboostClassifier с обучением на полученных с помощью DL эмбеддингами
текстов постов. Эмбеддинги текстов получены с помощью трансформера DistilBertModel.

Таким образом отличие в моделях заключается в способе получени эмбеддингов для текстов: в первом случае это tf-idf, во втором эмбеддинги получены с помощью DistilBertModel

### Реализация сервиса
Сервис реализован с помощью FastAPI в виде endpoint "ручки":
1. По POST запросу принимается запрос от пользователя на выдачу ленты.
2. Полученный в запросе JSON обрабатывается, все необходимые признаки приводятся к требуемой для модели форме.
3. Модель делает предсказания для каждого поста и выбираются с наибольшей вероятностью получающие лайк.
4. Сервис возвращает отклик со списком рекомендованных постов.


## Пути улучшения полученного результата
Ввиду различных допущений и ограничений, а так же учебного характера проекта, укажем возможные пути улучшения сервиса:
1. Возможен более глубокий feature engineering (в итоговом варианте проекта для простоты и скорости работы были взяты не все фитчи из 'черновика': например можно добавить фитчу с топ 3 любимых категорий пользователя).
2. Обученные модели довольно просты. Возможна более тонкая настройка гиперпараметров использовавшегося CatboostClassifier.
3. Не исследован коллабаративный или смешанный подходы.
4. В DL модели эмбеддинги текстов можно получить и исползовав другие, более тяжелые, трансформеры BERT семейства.
5. Применить нейронные сети в качестве модели рекомендаций.

## Инструкция для запуска

#### Способ 1 (GitHub)
`git clone https://github.com/koryakovda/karpov.cources.git` <br />
`python -m pip install --upgrade pip` <br />
`pip install -r requirements.txt` <br />
`python -m uvicorn src.app:app ` <br />

#### Способ 2 (Docker-compose)
`git clone https://github.com/koryakovda/karpov.cources.git` <br />
`docker-compose up -d`

#### Способ 3 (DockerHub)
`docker pull koryakovda/posts_service` <br />
`docker run --rm -p 8000:8000 --name <имя_образа> koryakovda/posts_service`

Сервис доступен по http://127.0.0.1:8000/post/recommendations/, где задав параметры можно протестировать запрос на выдачу ленты. Примеры запросов есть [здесь](request_examples.txt).
