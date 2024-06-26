from transformers import AutoTokenizer, AutoModelForCausalLM
import sys


class UsingHuggingface:

    # Hugging FaceのモデルIDを指定
    model_id = "CohereForAI/c4ai-command-r-plus"
    
    # モデルとトークナイザーを初期宣言
    tokenizer = None
    model = None

    def __init__(self,models = None):
        # 起動引数の数に応じて変数を設定
        if models != None:
            # モデルIDを指定する場合
            model_id = models

        # 指定されたモデルIDを使用してトークナイザーをロード
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # 指定されたモデルIDを使用して生成モデルをロード
        model = AutoModelForCausalLM.from_pretrained(model_id)

    def talk(content:str):
        # チャット形式のメッセージをリストとして定義
        messages = [{"role": "user", "content": content}]
        # メッセージをトークナイザーに渡し、トークン化された入力IDを生成
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

        # 入力IDに基づいて新しいテキストを生成
        gen_tokens = model.generate(
            input_ids, 
            max_new_tokens=100, 
            do_sample=True, 
            temperature=0.3,
        )

        # 生成されたトークンをデコードして、人間が読めるテキストに変換
        gen_text = tokenizer.decode(gen_tokens[0])

        # 生成されたテキストを出力
        return gen_text