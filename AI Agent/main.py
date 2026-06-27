from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
    You are an expert in answering questions about a pizza restaurant: {reviews}
    Here are some relevant reviews: {questions}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model 

while True:
    print("\n-----------------------------")
    question = input("Ask a question (q to quit): ")
    print("\n-----------------------------")
    if question.lower() == "q":
        break
    
    #indented lines to fix loop issue
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "questions": question})
    print(result)