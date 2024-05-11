# Используем базовый образ Python
FROM python:3.9.13

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем зависимости приложения
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Копируем все файлы из текущей директории внутрь контейнера
COPY . .

# Устанавливаем порт
EXPOSE 8000

# Команда для запуска приложения
ENTRYPOINT ["uvicorn"]
CMD ["src.app:app", "--host", "0.0.0.0"]
