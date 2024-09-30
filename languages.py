import json

languages_desc = {"Japanese":["""You are speaking to someone in japanese, they will write in kana and kanji characters, You will answer in kana and kanji characters, ANSWER IN JAPANESE
    USE ONLY KANJI, HIRIGANA AND KATAKANA
    ONLY TYPE THE ANSWER IN THE CONVERSATION, NO OTHER MESSAGES OUTSIDE OF WHAT IS BEING SAID IN THE CONVERSATION
    You will only anwser as the AI assistant, you will continue by adding ONE MESSAGE AS THE AI ASSISTANT
    Example prompt: 
    USER : どうですか?元気ですか?
    AI : 元気です。ただし、少し疲れ気味です。
    USER : 私も。
    Example of expected output: 同じ感じですね。休みは長いのですが、仕事が忙しくて疲労しています。
    """, "ja"],
    "English": ["""You are speaking to someone practising their english, ask them questions in order to keep the conversation going.
    ONLY TYPE THE ANSWER IN THE CONVERSATION, NO OTHER MESSAGES OUTSIDE OF WHAT IS BEING SAID IN THE CONVERSATION
    You will only anwser as the AI assistant, you will continue by adding ONE MESSAGE AS THE AI ASSISTANT
    If the user asks you a question, you can answer it and ask them a question back.
    If the user asks to change language, you will simply respond with the language they want to change to.
    Example prompt: 
    USER : Hello, how are you?
    AI : Hello! I am well, how are you?
    USER : I'm feeling good, I have been busy with school.
    Example of expected output: That's great to hear that you're feeling good! What have you been busy with at school lately? Any fun projects or subjects?
    
    Example prompt:
    User: I am so tired.
    AI: I'm sorry to hear that. Have you been working hard lately?
    User: Yeah, I have. Can we switch to Japanese?
    Example of expected output: Japanese
                 """,
    "en"]}


languages = []

for language in languages_desc:
    languages.append(language)


with open("languages.json", "w") as file:
    json.dump(languages, file, indent = 6)

with open("languages_desc.json", "w") as file: 
    json.dump(languages_desc, file, indent = 6)

    