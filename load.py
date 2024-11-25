import chromadb
import hashlib

client = chromadb.PersistentClient()
collection = client.get_or_create_collection("docs")

with open("./data.txt", "r", encoding="utf-8") as f:
    lines = [x.split("\n") for x in f.read().split("\n\n")]

for line in lines:
    line_id = hashlib.md5(line[0].encode()).hexdigest()
    collection.add(
        documents=[line[0]],
        ids=[line_id],
        metadatas=[{
            "question": line[0],
            "answer": line[1],
        }]
    )
