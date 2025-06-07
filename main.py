import flask
from flask import render_template,Flask,request
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import pipeline
# import torch
from dotenv import load_dotenv 
import os
from huggingface_hub import InferenceClient

load_dotenv(dotenv_path="hftoken.env")

model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)
vectorstore = Chroma(persist_directory="./chroma_store", embedding_function=embeddings)
client = InferenceClient(api_key=os.getenv("hftoken"))


app=Flask(__name__)

@app.route("/",methods=["GET","POST"])
def chat():
    answer=""
    if request.method=="POST":
        user_input=request.form["message"]
        docs_store=vectorstore.similarity_search_with_relevance_scores(user_input,k=3)
        if not docs_store:
            context = "No relevant documents found."
            prompt = f"Context:\n{context}\n\nQuestion: {user_input}\nAnswer:"
        else:
            context="\n".join([doc.page_content for doc,score in docs_store])
            prompt = f"Context:\n{context}\n\nQuestion: {user_input}\nAnswer:"

          # Generate a response
        messages = [
                {"role": "system", "content":context},
                {"role": "user", "content": prompt}
                ]
        try:
    # Generate a response
    
            response = client.chat_completion(
            messages=messages,
            model="meta-llama/Llama-3.2-3B-Instruct",
            max_tokens=600,
            temperature=0.7
            )
            
           
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error during generation: {str(e)}"
        # Extract only the new model response after the prompt
    return render_template("chat.html", answer=answer)


if __name__ == "__main__":
    app.run(debug=True)
    