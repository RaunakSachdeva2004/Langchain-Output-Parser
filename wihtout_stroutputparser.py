from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

# 1. Use a valid text-generation model
# (Zephyr or Mistral are great for this)
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

# NOTE: We removed 'model = ChatHuggingFace(llm=llm)'
# We will just use 'llm' directly.

# 2. Define Prompts
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Write a 5 line summary on the following text.\n{text}',
    input_variables=['text']
)

# 3. Execution
print("Generating Report...")
# Invoke the template to get the string prompt
prompt1_str = template1.invoke({'topic': 'black hole'}) 
# Invoke the LLM directly
result_report = llm.invoke(prompt1_str) 
print(result_report) # Printing purely to see progress

print("\nGenerating Summary...")
prompt2_str = template2.invoke({'text': result_report})
result_summary = llm.invoke(prompt2_str)

print("\n--- FINAL SUMMARY ---\n")
print(result_summary)