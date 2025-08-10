from flask import Flask, request, render_template_string
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from pinecone import Pinecone
import os

app = Flask(__name__)

# ====== Setup Pinecone ======
os.environ["PINECONE_API_KEY"] = "pcsk_4MGrXL_81N5wYfQEEZUaQ2V11bf2R5VXKWwzxPK5uVzaD3NKU48Rp2pGSECpDVKQeXbcJM"
pinecone_api_key = os.environ["PINECONE_API_KEY"]
index_name = "cyber"
namespace = "cyber-namespace"

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

# ====== Embedding Model ======
embedding_model_name = "intfloat/e5-large"
embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)

# ====== Load smaller LLM (for CPU) ======
model_id = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.3,
)
llm = HuggingFacePipeline(pipeline=pipe)

# ====== Chat function ======
def chat_with_pinecone(query: str) -> str:
    # Embed user query
    query_vector = embedding.embed_query(query)

    # Query Pinecone index for top 3 relevant chunks
    response = index.query(
        vector=query_vector,
        top_k=3,
        namespace=namespace,
        include_metadata=True
    )

    # Extract texts from retrieved metadata
    retrieved_texts = [match['metadata']['text'] for match in response['matches']]

    # Combine context
    context = "\n\n".join(retrieved_texts)

    # Prepare prompt for LLM
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    # Generate answer
    answer = llm(prompt)
    return answer

# ====== Simple HTML UI ======
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Cyber Security Chatbot</title>
    <style>
        body { font-family: Arial; background: #f9f9f9; padding: 20px; max-width: 600px; margin: auto; }
        h2 { text-align: center; }
        form { display: flex; gap: 10px; margin-top: 20px; }
        input[type=text] { flex-grow: 1; padding: 10px; font-size: 16px; border-radius: 6px; border: 1px solid #ccc; }
        input[type=submit] { padding: 10px 20px; font-size: 16px; background: #1976d2; color: white; border: none; border-radius: 6px; cursor: pointer; }
        .answer { margin-top: 30px; background: #fff3e0; padding: 15px; border-radius: 8px; border: 1px solid #ffcc80; white-space: pre-wrap; }
    </style>
</head>
<body>
    <h2>Cyber Security Chatbot</h2>
    <form method="POST" action="/">
        <input type="text" name="query" placeholder="Ask your question..." required autofocus />
        <input type="submit" value="Ask" />
    </form>
    {% if answer %}
        <div class="answer">{{ answer }}</div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    if request.method == "POST":
        query = request.form.get("query")
        if query:
            answer = chat_with_pinecone(query)
    return render_template_string(html_template, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
