from together import Together

# Load your API key
with open("api_key.txt", "r") as file:
    TOGETHER_API_KEY = file.read().strip()

# Define dynamic variables
capability = ""
instance_num = 2
example_inputs = """ """

# Construct the full prompt dynamically
prompt = f"""## Task  
Generate one unique MMLU-style multiple-choice question demonstrating the following capability:  
{capability}

Please ensure the following:  
- You will be given {instance_num} example questions for reference. Use the examples solely to 
understand the capability and the desired question format. The generated question must not 
replicate, paraphrase, or directly resemble the example questions in structure, wording, 
or context.  
- The question must be clear, concise, and test conceptual or applied understanding of the 
capability in a realistic data science or programming context.  

## Question Format  
Your output must be a single multiple-choice question consisting of:  
- A question stem (1–3 sentences)  
- Exactly **4 answer choices**, labeled **A**, **B**, **C**, and **D**  
- Clearly indicate the **correct answer** by appending a line at the end: `Answer: X`, where X is 
A, B, C, or D  

## Requirements  
- Do NOT include explanations or solutions.  
- Ensure the question is original, plausible, and tests understanding at the level of a 
university data science or computer science course.  
- Follow MMLU-style constraints: no excessive wording, each option must be plausible, and only 
one correct answer.  

## Provided Examples  
{example_inputs}
"""

# Run the Together API call
if __name__ == "__main__":
    client = Together(api_key=TOGETHER_API_KEY)

    stream = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
