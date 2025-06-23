#!/usr/bin/env python3
"""
認証管理モジュール（簡素版）
シンプルなID・パスワード認証機能
"""

import streamlit as st
import hashlib
import time

class AuthManager:
    """認証管理クラス"""
    
    def __init__(self):
        # 認証情報（実際の運用では環境変数や設定ファイルから読み込み）
        self.credentials = {
            "ricrio": "simulation001"
        }
        
        # セッション状態の初期化
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
        if "login_attempts" not in st.session_state:
            st.session_state.login_attempts = 0
        if "last_attempt_time" not in st.session_state:
            st.session_state.last_attempt_time = 0
    
    def hash_password(self, password: str) -> str:
        """パスワードをハッシュ化"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_credentials(self, username: str, password: str) -> bool:
        """認証情報の確認"""
        if username in self.credentials:
            return self.credentials[username] == password
        return False
    
    def check_rate_limit(self) -> bool:
        """レート制限チェック（セキュリティ対策）"""
        current_time = time.time()
        
        # 5回失敗したら5分間ロック
        if st.session_state.login_attempts >= 5:
            if current_time - st.session_state.last_attempt_time < 300:  # 5分
                return False
            else:
                # ロック時間が過ぎたらリセット
                st.session_state.login_attempts = 0
        
        return True
    
    def login_form(self) -> bool:
        """ログインフォームの表示と処理"""
        
        # シンプルなページスタイル
        st.markdown("""
        <style>
        .login-header {
            text-align: center;
            color: #1f77b4;
            margin-bottom: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # ヘッダー
        st.markdown('<h2 class="login-header">⚡ 電力需給予測システム</h2>', unsafe_allow_html=True)
        st.markdown('<h3 class="login-header">認証が必要です</h3>', unsafe_allow_html=True)
        
        # レート制限チェック
        if not self.check_rate_limit():
            remaining_time = 300 - (time.time() - st.session_state.last_attempt_time)
            st.error(f"🚫 ログイン試行回数が上限に達しました。{int(remaining_time/60)}分{int(remaining_time%60)}秒後に再試行してください。")
            st.markdown('</div>', unsafe_allow_html=True)
            return False
        
        # ログインフォーム
        with st.form("login_form"):
            username = st.text_input("👤 ユーザー名", placeholder="ユーザー名を入力")
            password = st.text_input("🔒 パスワード", type="password", placeholder="パスワードを入力")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                login_button = st.form_submit_button("🔓 ログイン", use_container_width=True)
            
            if login_button:
                if self.verify_credentials(username, password):
                    st.session_state.authenticated = True
                    st.session_state.login_attempts = 0
                    st.success("✅ ログインに成功しました！")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.session_state.login_attempts += 1
                    st.session_state.last_attempt_time = time.time()
                    st.error(f"❌ ユーザー名またはパスワードが正しくありません。（試行回数: {st.session_state.login_attempts}/5）")
        
        # フッター
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
        🛡️ このシステムは認証によって保護されています<br>
        🤖 Powered by Streamlit & Claude Code
        </div>
        """, unsafe_allow_html=True)
        
        return False
    
    def logout(self):
        """ログアウト処理"""
        st.session_state.authenticated = False
        st.session_state.login_attempts = 0
        st.rerun()
    
    def is_authenticated(self) -> bool:
        """認証状態の確認"""
        return st.session_state.get("authenticated", False)
    
    def require_auth(func):
        """認証デコレータ（関数用）"""
        def wrapper(*args, **kwargs):
            auth_manager = AuthManager()
            if auth_manager.is_authenticated():
                return func(*args, **kwargs)
            else:
                auth_manager.login_form()
                return None
        return wrapper

# 使用例とテスト
if __name__ == "__main__":
    st.set_page_config(page_title="認証テスト", page_icon="🔐")
    
    auth_manager = AuthManager()
    
    if auth_manager.is_authenticated():
        st.success("認証済みです！")
        if st.button("ログアウト"):
            auth_manager.logout()
    else:
        auth_manager.login_form()