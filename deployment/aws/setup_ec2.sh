#!/bin/bash
# AWS EC2セットアップスクリプト

# システム更新
sudo apt update
sudo apt upgrade -y

# Python環境構築
sudo apt install python3.10 python3.10-venv python3-pip nginx apache2-utils -y

# nginxでBasic認証設定
sudo htpasswd -c /etc/nginx/.htpasswd demo_user

# Streamlitアプリケーション設定
cd /home/ubuntu
git clone https://github.com/your-repo/power-forecast-demo.git
cd power-forecast-demo

# 仮想環境作成
python3 -m venv power_forecast_env
source power_forecast_env/bin/activate
pip install -r requirements.txt

# Systemdサービス作成
sudo tee /etc/systemd/system/streamlit.service > /dev/null <<EOF
[Unit]
Description=Streamlit Power Forecast Demo
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/power-forecast-demo
Environment="PATH=/home/ubuntu/power-forecast-demo/power_forecast_env/bin"
ExecStart=/home/ubuntu/power-forecast-demo/power_forecast_env/bin/streamlit run streamlit_demo_app.py --server.port 8501
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# サービス開始
sudo systemctl enable streamlit
sudo systemctl start streamlit

# nginx設定
sudo cp deployment/aws/nginx.conf /etc/nginx/sites-available/streamlit
sudo ln -s /etc/nginx/sites-available/streamlit /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

echo "✅ セットアップ完了！"