import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle

# Step 1: Load your Excel file
file_path = r"C:\Users\ramya\OneDrive\Documents\assessments.csv.xlsx"  # Change path if needed
df = pd.read_excel(file_path)

# Step 2: Clean column names
df.columns = df.columns.str.strip()

# Step 3: Check for required columns
required_columns = ['Name', 'URL', 'Type', 'Duration']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"❌ Column '{col}' not found in the Excel file.")

# Step 4: Combine fields for embedding
df['combined_text'] = (
    df['Name'].astype(str) + " " +
    df['Type'].astype(str) + " " +
    df['Duration'].astype(str)
)

# Step 5: Load the SentenceTransformer model
print("⚙️ Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 6: Generate embeddings
print("📈 Generating embeddings...")
embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True)

# Step 7: Save embeddings
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

# Step 8: Save the DataFrame
df.to_pickle("shl_data.pkl")

print("✅ All done! Files saved: 'embeddings.pkl' and 'shl_data.pkl'")
