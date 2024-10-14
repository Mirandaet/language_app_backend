from openai import OpenAI


def llama(prompt, system_prompt):
    client = OpenAI(
        base_url='http://localhost:11434/v1/',

        # required but ignored
        api_key='ollama',
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {
                'role': 'user',
                'content': prompt,
            }
        ],
        model='llama3.2',
    )

    return chat_completion.choices[0].message.content
