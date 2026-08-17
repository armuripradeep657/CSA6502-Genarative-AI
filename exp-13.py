from google import genai

api_key = input("Enter your Gemini API key: ")

client = genai.Client(api_key=api_key)

print("===== GEMINI AI APPLICATION =====")
print("Type 'exit' to stop.")

while True:

    question = input("\nEnter your question: ")

    if question.lower() == "exit":
        print("Program ended.")
        break

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=question
    )

    print("\nGemini Response:")
    print(response.text)
