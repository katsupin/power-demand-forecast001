FROM python:3.10-slim

WORKDIR /app

# システムパッケージのインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 依存関係のコピーとインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードのコピー
COPY . .

# Streamlit設定
RUN mkdir -p ~/.streamlit/
RUN echo '\
[general]\n\
email = "demo@example.com"\n\
' > ~/.streamlit/credentials.toml
RUN echo '\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = 8080\n\
' > ~/.streamlit/config.toml

EXPOSE 8080

# Streamlitアプリケーション起動
CMD ["streamlit", "run", "streamlit_demo_app.py", "--server.port=8080", "--server.address=0.0.0.0"]