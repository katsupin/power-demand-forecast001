@echo off
REM 電力需給予測デモ環境セットアップスクリプト (Windows用)

echo === 電力需給予測デモ環境セットアップ開始 ===

REM 1. プロジェクトディレクトリの作成
set PROJECT_DIR=power_forecast_demo
if not exist "%PROJECT_DIR%" (
    mkdir "%PROJECT_DIR%"
    echo √ プロジェクトディレクトリを作成しました: %PROJECT_DIR%
) else (
    echo ! プロジェクトディレクトリは既に存在します: %PROJECT_DIR%
)

cd "%PROJECT_DIR%"

REM 2. Pythonバージョンの確認
echo.
echo Pythonバージョンの確認...
python --version 2>NUL
if %ERRORLEVEL% neq 0 (
    echo × Pythonが見つかりません。Pythonをインストールしてください。
    echo   https://www.python.org/downloads/ からダウンロードできます。
    pause
    exit /b 1
)
echo √ Pythonを検出しました

REM 3. 仮想環境の作成
set VENV_DIR=venv
if not exist "%VENV_DIR%" (
    echo.
    echo 仮想環境を作成中...
    python -m venv %VENV_DIR%
    echo √ 仮想環境を作成しました
) else (
    echo ! 仮想環境は既に存在します
)

REM 4. 仮想環境の有効化
echo.
echo 仮想環境を有効化中...
call %VENV_DIR%\Scripts\activate.bat

REM 5. pipのアップグレード
echo.
echo pipをアップグレード中...
python -m pip install --upgrade pip

REM 6. requirements.txtの作成
echo.
echo requirements.txtを作成中...
(
echo streamlit==1.28.0
echo pandas==2.0.3
echo numpy==1.24.3
echo scikit-learn==1.3.0
echo matplotlib==3.7.2
echo seaborn==0.12.2
echo prophet==1.1.4
echo plotly==5.17.0
echo xgboost==2.0.0
) > requirements.txt
echo √ requirements.txtを作成しました

REM 7. 依存関係のインストール
echo.
echo 依存関係をインストール中（数分かかる場合があります）...
pip install -r requirements.txt

REM 8. インストールの確認
echo.
echo インストール済みパッケージ:
pip list | findstr /i "streamlit pandas numpy scikit-learn prophet"

REM 9. デモファイルのコピー
echo.
echo デモファイルをコピー中...
if exist "..\簡易デモ実装サンプル.py" (
    copy "..\簡易デモ実装サンプル.py" "demo_app.py"
    echo √ デモアプリをコピーしました: demo_app.py
) else (
    echo ! デモファイルが見つかりません。手動でコピーしてください。
)

REM 10. .gitignoreの作成
echo.
echo .gitignoreを作成中...
(
echo # Virtual Environment
echo venv/
echo env/
echo ENV/
echo.
echo # Python
echo __pycache__/
echo *.py[cod]
echo *$py.class
echo *.so
echo .Python
echo.
echo # IDE
echo .vscode/
echo .idea/
echo *.swp
echo *.swo
echo.
echo # OS
echo .DS_Store
echo Thumbs.db
echo.
echo # Project specific
echo *.log
echo *.csv
echo !sample_data.csv
echo .streamlit/secrets.toml
) > .gitignore
echo √ .gitignoreを作成しました

REM 11. 実行バッチファイルの作成
echo.
echo 実行バッチファイルを作成中...
(
echo @echo off
echo REM デモアプリ実行スクリプト
echo.
echo REM 仮想環境の有効化
echo call venv\Scripts\activate.bat
echo.
echo REM Streamlitアプリの起動
echo echo デモアプリを起動します...
echo echo ブラウザが自動的に開きます。開かない場合は http://localhost:8501 にアクセスしてください。
echo streamlit run demo_app.py
) > run_demo.bat
echo √ 実行バッチファイルを作成しました: run_demo.bat

echo.
echo === セットアップ完了 ===
echo.
echo 次のステップ:
echo 1. デモアプリを起動: run_demo.bat
echo 2. または手動で起動: streamlit run demo_app.py
echo.
echo 仮想環境を無効化するには: deactivate
echo.
pause