import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import Huggingface

# ボットトークンとソケットモードハンドラーを使ってアプリを初期化します
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))


@app.event("app_mention")
def ask_question(body, say):
    chatai = UsingHuggingface()
    say(chatai.talk(body["text"]))

# アプリを起動します
if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()