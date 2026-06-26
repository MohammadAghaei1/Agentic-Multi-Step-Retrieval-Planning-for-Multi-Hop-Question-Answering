from dotenv import load_dotenv
from groq import Groq

# Load API key from .env file
load_dotenv()

# Create Groq client
client = Groq()

# Send a simple test request
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": "Answer with one word: ready"
        }
    ],
)

# Print model response
print(response.choices[0].message.content)