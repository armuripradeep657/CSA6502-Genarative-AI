from google import genai

api_key = input("Enter your Gemini API key: ")

client = genai.Client(api_key=api_key)

print("===== STRUCTURED OUTPUT GENERATOR =====")
print("1. Generate Python Code")
print("2. Generate SQL Query")

choice = input("\nEnter your choice: ")

if choice == "1":

    requirement = input("Enter Python requirement: ")

    prompt = f"""
Generate valid Python code for the following requirement:

{requirement}

Return only the Python code.
Do not include explanations.
Do not use markdown code blocks.
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    print("\n===== GENERATED PYTHON CODE =====")
    print(response.text)

elif choice == "2":

    requirement = input("Enter SQL requirement: ")

    prompt = f"""
Generate a valid SQL query for the following requirement:

{requirement}

Return only the SQL query.
Do not include explanations.
Do not use markdown code blocks.
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    print("\n===== GENERATED SQL QUERY =====")
    print(response.text)

else:

    print("Invalid choice.")
