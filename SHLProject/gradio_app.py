import gradio as gr
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the saved data and embeddings
df = pd.read_pickle("shl_data.pkl")
with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

# Load the model (same one used for embeddings)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define the recommendation function
def recommend_assessment(query):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-5:][::-1]
    results = df.iloc[top_indices]

    output = ""
    for _, row in results.iterrows():
        output += f"👉 **{row['Name']}**\n🔗 [Link]({row['URL']})\n🧠 Type: {row['Type']} | ⏱ Duration: {row['Duration']} mins\n\n---\n"

    return output

# Create the Gradio interface
iface = gr.Interface(
    fn=recommend_assessment,
    inputs=gr.Textbox(lines=2, placeholder="e.g. Test for pattern recognition or numerical skills"),
    outputs="markdown",
    title="🧠 SHL Assessment Recommender",
    description="Describe what kind of assessment you're looking for, and get matching SHL tests."
)

# ✅ Launch with public link
iface.launch(share=True)
