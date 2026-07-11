import numpy as np
import urllib.request
import json
"""
سنقوم بعمل متغير يحمل اكبر تشابه 
ومتغير يحمل النص الخاص به 
ثم نبدا اللوب
في كل مرة ناخذ النص كويري_امبدد نقارنه مع نولدج_امبدد في الخطوة تلك
اذا وجدنا تشابه اكبر نضعه في المتغير الاول
اذا لم نجد ننتقل للخطوة التي بعدها حتى ينتهي اللوب
 ثم ناخذ جملة اكبر تشابه نطعمها للموديل وهو يجيب 
"""

def get_embedding(text):
    # 2. إعداد الطلب لخادم Ollama المحلي
    url = "http://localhost:11434/api/embeddings"
    data = json.dumps({"model": "nomic-embed-text", "prompt": text}).encode('utf-8')

    # 3. إرسال الطلب
    try:
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            
            # 4. استخراج الإحداثيات
            embedding = result["embedding"]
            
            return np.array(embedding)
            
    except Exception as e:
        print(f"Error: {e}")

# دالة التشابه التي كتبناها سابقاً (انسخها هنا)
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot_product / (norm1 * norm2)

from pathlib import Path

def load_data(folder_path):
    knowledge_base = []
    # Loop through all items in the folder
    for file_path in folder_path.iterdir():
        # Check if the item is a file (skips subfolders)
        if file_path.is_file():
            if(file_path.suffix == ".txt"):
                content = file_path.read_text(encoding="utf-8")
                knowledge_base.append((content, get_embedding(content)))

    return knowledge_base

folder_path = Path("knowledge")
knowledge_base = load_data(folder_path)

def get_closed_word(user_input):
    # 2. إحداثيات سؤال المستخدم الوهمي
    query_embed = get_embedding(user_input)

    # ---------------------------------------------------------
    # 3. منطقك الذي شرحته: المتغيرات، الـ Loop، والشرط
    max_similarity = -1.0
    closed_word = ""

    for knowledge in knowledge_base:
        similarity = cosine_similarity(query_embed,knowledge[1])
        if similarity > max_similarity:
            max_similarity = similarity
            closed_word = knowledge[0]
    return closed_word
    # ---------------------------------------------------------

import ollama

MODEL_NAME = 'qwen2.5:3b'


while True:
    user_input = input("👤 أنت:")
    if user_input == "exit": break
    closed_word = get_closed_word(user_input)

    messages = [
        {
            "role": "system",
            "content": "أنت مساعد ذكي. أجب على سؤال المستخدم بناءً على هذه الفقرة فقط. إذا لم تكن الإجابة في الفقرة قل لا أعرف.\n\nالفقرة: " + closed_word
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    response = ollama.chat(model=MODEL_NAME, messages=messages)
    model_response = response['message']['content']

    print(f"🤖 المساعد:\n{model_response}")
