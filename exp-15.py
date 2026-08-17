from google import genai

api_key = input("Enter your Gemini API key: ")

client = genai.Client(api_key=api_key)

text = input("Enter a topic: ")

prompts = {
    "Zero-Shot": f"""
Explain the following topic in simple words:

{text}
""",

    "One-Shot": f"""
Example:

Topic: Artificial Intelligence
Explanation: Artificial Intelligence enables computers to perform tasks that normally require human intelligence.

Now explain this topic in a similar way:

Topic: {text}
""",

    "Few-Shot": f"""
Example 1:

Topic: Machine Learning
Explanation: Machine learning allows computers to learn patterns from data.

Example 2:

Topic: Cloud Computing
Explanation: Cloud computing provides computing services through the internet.

Example 3:

Topic: Cybersecurity
Explanation: Cybersecurity protects computers, networks and data from digital threats.

Now explain this topic using the same style:

Topic: {text}
"""
}

results = {}

for strategy, prompt in prompts.items():

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    results[strategy] = response.text

print("\n===== LLM PROMPT STRATEGY EVALUATION =====")

for strategy, result in results.items():

    print("\n" + "=" * 50)
    print(strategy)
    print("=" * 50)
    print(result)

print("\n===== COMPARISON =====")

print("""
Zero-Shot:
Uses only instructions without examples.

One-Shot:
Uses one example to guide the response format.

Few-Shot:
Uses multiple examples to provide stronger guidance.

Few-shot prompting generally provides more consistent
responses when the task requires a specific style or format.
""")
