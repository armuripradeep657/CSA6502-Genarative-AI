import google.generativeai as genai

genai.configure(api_key="YOUR_GEMINI_API_KEY")

model = genai.GenerativeModel("gemini-2.0-flash")

text = input("Enter a sentence to classify: ")

zero_shot_prompt = f"""
Classify the following sentence as Positive, Negative, or Neutral.

Sentence:
{text}

Give only the classification.
"""

one_shot_prompt = f"""
Classify the following sentences as Positive, Negative, or Neutral.

Example:
Sentence: I love this product.
Classification: Positive

Now classify:
Sentence: {text}

Give only the classification.
"""

few_shot_prompt = f"""
Classify each sentence as Positive, Negative, or Neutral.

Example 1:
Sentence: The movie was excellent.
Classification: Positive

Example 2:
Sentence: The service was very bad.
Classification: Negative

Example 3:
Sentence: The meeting is scheduled for tomorrow.
Classification: Neutral

Now classify:
Sentence: {text}

Give only the classification.
"""

zero_result = model.generate_content(zero_shot_prompt)
one_result = model.generate_content(one_shot_prompt)
few_result = model.generate_content(few_shot_prompt)

print("\n===== ZERO-SHOT RESULT =====")
print(zero_result.text)

print("\n===== ONE-SHOT RESULT =====")
print(one_result.text)

print("\n===== FEW-SHOT RESULT =====")
print(few_result.text)
