#!/bin/bash
# 電力需給予測デモ環境セットアップスクリプト

echo "=== 電力需給予測デモ環境セットアップ開始 ==="

# 1. プロジェクトディレクトリの作成
PROJECT_DIR="power_forecast_demo"
if [ ! -d "$PROJECT_DIR" ]; then
    mkdir -p "$PROJECT_DIR"
    echo "✓ プロジェクトディレクトリを作成しました: $PROJECT_DIR"
else
    echo "! プロジェクトディレクトリは既に存在します: $PROJECT_DIR"
fi

cd "$PROJECT_DIR"

# 2. Pythonバージョンの確認
echo ""
echo "Pythonバージョンの確認..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Pythonが見つかりません。Pythonをインストールしてください。"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "✓ Python $PYTHON_VERSION を使用します"

# 3. 仮想環境の作成
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "仮想環境を作成中..."
    $PYTHON_CMD -m venv $VENV_DIR
    echo "✓ 仮想環境を作成しました"
else
    echo "! 仮想環境は既に存在します"
fi

# 4. 仮想環境の有効化
echo ""
echo "仮想環境を有効化中..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source $VENV_DIR/Scripts/activate
else
    # macOS/Linux
    source $VENV_DIR/bin/activate
fi

# 5. pipのアップグレード
echo ""
echo "pipをアップグレード中..."
pip install --upgrade pip

# 6. requirements.txtの作成
echo ""
echo "requirements.txtを作成中..."
cat > requirements.txt << EOF
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
prophet==1.1.4
plotly==5.17.0
xgboost==2.0.0
EOF
echo "✓ requirements.txtを作成しました"

# 7. 依存関係のインストール
echo ""
echo "依存関係をインストール中（数分かかる場合があります）..."
pip install -r requirements.txt

# 8. インストールの確認
echo ""
echo "インストール済みパッケージ:"
pip list | grep -E "streamlit|pandas|numpy|scikit-learn|prophet"

# 9. デモファイルのコピー
echo ""
echo "デモファイルをコピー中..."
if [ -f "../簡易デモ実装サンプル.py" ]; then
    cp "../簡易デモ実装サンプル.py" ./demo_app.py
    echo "✓ デモアプリをコピーしました: demo_app.py"
else
    echo "! デモファイルが見つかりません。手動でコピーしてください。"
fi

# 10. .gitignoreの作成
echo ""
echo ".gitignoreを作成中..."
cat > .gitignore << EOF
# Virtual Environment
venv/
env/
ENV/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
*.log
*.csv
!sample_data.csv
.streamlit/secrets.toml
EOF
echo "✓ .gitignoreを作成しました"

# 11. 実行スクリプトの作成
echo ""
echo "実行スクリプトを作成中..."
cat > run_demo.sh << 'EOF'
#!/bin/bash
# デモアプリ実行スクリプト

# 仮想環境の有効化
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Streamlitアプリの起動
echo "デモアプリを起動します..."
echo "ブラウザが自動的に開きます。開かない場合は http://localhost:8501 にアクセスしてください。"
streamlit run demo_app.py
EOF

chmod +x run_demo.sh
echo "✓ 実行スクリプトを作成しました: run_demo.sh"

# 12. README.mdの作成
echo ""
echo "README.mdを作成中..."
cat > README.md << EOF
# 電力需給予測デモアプリケーション

## 概要
過去の天気と電力データを元にAIで電力需給を予測するデモシステムです。

## 機能
- 予測精度の実証
- コスト削減シミュレーション
- リアルタイム予測

## セットアップ
\`\`\`bash
# 1. セットアップスクリプトの実行
bash setup_demo_env.sh

# 2. デモアプリの起動
bash run_demo.sh
\`\`\`

## 手動セットアップ
\`\`\`bash
# 1. 仮想環境の作成
python -m venv venv

# 2. 仮想環境の有効化
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. 依存関係のインストール
pip install -r requirements.txt

# 4. アプリの起動
streamlit run demo_app.py
\`\`\`

## トラブルシューティング
- Prophetのインストールが失敗する場合: \`pip install pystan==2.19.1.1\` を先に実行
- ポート8501が使用中の場合: \`streamlit run demo_app.py --server.port 8502\`
EOF
echo "✓ README.mdを作成しました"

echo ""
echo "=== セットアップ完了 ==="
echo ""
echo "次のステップ:"
echo "1. デモアプリを起動: bash run_demo.sh"
echo "2. または手動で起動: streamlit run demo_app.py"
echo ""
echo "仮想環境を無効化するには: deactivate"