import requests
import json

# The URL where the local server is running
url = "http://localhost:1234/v1/chat/completions"

# The headers to indicate that we are sending JSON data
headers = {
    "Content-Type": "application/json"
}

def llama(system_prompt, prompt):
    # The JSON data payload
    data = {
        "model": "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": False
    }

# Making the POST request to the local server
    response = requests.post(url, headers=headers, data=json.dumps(data))

    # Checking if the request was successful
    if response.status_code == 200:
        # Printing the response content
        response = (response.json()["choices"][0]["message"]["content"])
        return(response)
    else:
        print("Failed to get response:", response.status_code, response.text)