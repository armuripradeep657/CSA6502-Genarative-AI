from google import genai

api_key = input("Enter your Gemini API key: ")

client = genai.Client(api_key=api_key)

print("===== FUNCTIONAL PROMPT GENERATOR =====")
print("1. Summarization")
print("2. Email Creation")
print("3. Content Generation")

choice = input("\nEnter your choice: ")

if choice == "1":

    text = input("\nEnter text to summarize: ")

    prompt = f"""
Summarize the following text in simple and clear language.
Keep the summary concise and include only the important points.

Text:
{text}
"""

elif choice == "2":

    purpose = input("\nEnter the purpose of the email: ")

    prompt = f"""
Write a professional email for the following purpose:

{purpose}

The email should have:
Subject
Greeting
Main message
Closing

Use a polite and professional tone.
"""

elif choice == "3":

    topic = input("\nEnter the content topic: ")

    prompt = f"""
Create informative and engaging content about the following topic:

{topic}

Use a clear title and well-organized paragraphs.
Keep the language simple and easy to understand.
"""

else:

    print("Invalid choice.")
    exit()

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt
)

print("\n===== GENERATED OUTPUT =====")
print(response.text)
