import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
load_dotenv()


# ボットトークンとソケットモードハンドラーを使ってアプリを初期化します
app = App(token=os.getenv("SLACK_BOT_TOKEN"),signing_secret=os.getenv("SLACK_SIGNING_SECRET"),)
# Hugging FaceのモデルIDを指定
model_id = "rinna/japanese-gpt-neox-3.6b-instruction-sft"
#スペックあるなら下のでも良いかも
#model_id = "CohereForAI/c4ai-command-r-plus"

# 指定されたモデルIDを使用してトークナイザーをロード
tokenizer = AutoTokenizer.from_pretrained(model_id)
# 指定されたモデルIDを使用して生成モデルをロード
model = AutoModelForCausalLM.from_pretrained(model_id).to("cpu")

@app.event("app_mention")
def ask_question(body, say):
	if "subtype" not in body["event"] or body["event"]["subtype"] != "bot_message":
		print(chatAI.talk(body["event"]["text"]))
		say(chatAI.talk(body["event"]["text"]))

def talk(content:str):
	# チャット形式のメッセージをリストとして定義
	messages = [{"role": "user", "content": content}]
	# メッセージをトークナイザーに渡し、トークン化された入力IDを生成
	input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
	# 入力IDに基づいて新しいテキストを生成
	gen_tokens = model.generate(input_ids,max_new_tokens=100,do_sample=True,temperature=0.3,)
	# 生成されたトークンをデコードして、人間が読めるテキストに変換
	gen_text = tokenizer.decode(gen_tokens[0])
	# 生成されたテキストを出力
	return gen_text

# アプリを起動します
SocketModeHandler(app, os.getenv["SLACK_APP_TOKEN"]).start()