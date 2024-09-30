import os
import openai
from openai import OpenAI

def gpt4(system_prompt, user_input):
    client = OpenAI(
        # this is also the default, it can be omitted
        api_key=os.environ['OPENAI_API_KEY'],
    )

    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )
    return completion.choices[0].message.content
