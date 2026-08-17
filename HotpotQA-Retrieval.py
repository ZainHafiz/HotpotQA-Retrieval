import numpy as np
from anthropic import Anthropic
import os
from dotenv import load_dotenv
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util.similarity import pairwise_cos_sim, pairwise_dot_score, pairwise_euclidean_sim, pairwise_manhattan_sim
import faiss

load_dotenv()

# Hotpot good wikipedia data to generate answers from, and compare those answers
ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
n = 10
comparison_indices = []
bridge_indices = []
i = 0
while len(bridge_indices) < n or len(comparison_indices) < n:
    if ds[i]["type"] == "bridge":
        if len(bridge_indices) < n:
            bridge_indices.append(i)
    else:
        if len(comparison_indices) < n:
            comparison_indices.append(i)
    i += 1

comparison_subset = ds.select(comparison_indices)

bridge_subset = ds.select(bridge_indices)

subset = ds.select(comparison_indices + bridge_indices)

# Build corpus as 1D array from dataset, this makes it easier for RAG to search through
corpus = []
meta = []
seen = set()

for row in subset:
    for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
        for sent_id, sentence in enumerate(sentences):
            key = (title, sent_id)
            if key in seen:
                continue
            seen.add(key)
            corpus.append(f"{title}: {sentence}")
            meta.append(key)

# miniLM-L6 has lowest computation cost, 384 token is enough tokens for our purpose
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode_document(corpus)  # vector database (numpy array)

# IndexFlat is simple enough for our purposes
dimensions = np.shape(embeddings)[1]
index = faiss.IndexFlatL2(dimensions)
index.add(embeddings)


def retrieve_indices(queries, k=10):
    query_embeddings = model.encode(queries).astype('float32')
    distances, indices = index.search(query_embeddings, k)
    return indices


def retrieve(queries, k=10):
    indices = retrieve_indices(queries, k)
    return [[corpus[j] for j in row] for row in indices]


def query(questions):
    context = retrieve(questions, k=10)
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    responses = []
    for i in range(len(questions)):
        prompt = f"""
Answer the question based on this context, keep your answer strictly between 1-6 words, answering yes/no questions with only either 'yes', 'no' or 'I don't know':
{chr(10).join(c for c in context[i])}

Question:
{questions[i]}
"""
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        for block in response.content:
            if block.type == "text":
                responses.append(block.text)
    return responses


def compare(answers, test_answers):
    answers_vectors = model.encode(answers)
    test_answers_vectors = model.encode(test_answers)
    similarities = [pairwise_cos_sim(answers_vectors, test_answers_vectors),
                    pairwise_dot_score(answers_vectors, test_answers_vectors),
                    pairwise_euclidean_sim(answers_vectors, test_answers_vectors),
                    pairwise_manhattan_sim(answers_vectors, test_answers_vectors)]
    return similarities


def recall_at_k(rows, k=10):
    queries = [row["question"] for row in rows]
    indices = retrieve_indices(queries, k)

    hits = total = 0
    per_question = []

    for row, idx_row in zip(rows, indices):
        sf = row["supporting_facts"]
        gold = set(zip(sf["title"], sf["sent_id"]))
        retrieved = {meta[j] for j in idx_row}

        found = len(gold & retrieved)
        hits += found
        total += len(gold)
        per_question.append(found / len(gold))

    return hits / total, per_question


# bridge_queries = [row["question"] for row in bridge_subset]
# comparison_queries = [row["question"] for row in comparison_subset]

# test_bridge_answers = [row["answer"] for row in bridge_subset]
# test_comparison_answers = [row["answer"] for row in comparison_subset]

# claude_bridge_answers = query(bridge_queries)
# claude_comparison_answers = query(comparison_queries)
# bridge_scores = compare(claude_bridge_answers, test_bridge_answers)
# comparison_scores = compare(claude_comparison_answers, test_comparison_answers)
# data_type = ("bridge", "comparison")
# questions = (bridge_queries, comparison_queries)
# test_answers = (test_bridge_answers, test_comparison_answers)
# claude_answers = (claude_bridge_answers, claude_comparison_answers)
# scores = (bridge_scores, comparison_scores)
# for i in range(2):
#     print(f"""
# Top {n} {data_type[i]} queries alongside database answers and claude answers:
# {chr(10).join([f"{j+1}. {q}\nAnswer: {test_answers[i][j]}\nClaude: {claude_answers[i][j]}" for j, q in enumerate(questions[i])])}
# Corresponding scores, listed by similarity score calculation method:
# Cosine: {scores[i][0]}
# Dot Product: {scores[i][1]}
# Euclidean: {scores[i][2]}
# Manhattan: {scores[i][3]}
# """)

for name, rows in [("bridge", bridge_subset), ("comparison", comparison_subset)]:
    micro, per_q = recall_at_k(rows, k=10)
    full = sum(1 for x in per_q if x == 1.0)
    print(f"{name}: recall@10 = {micro:.2f}, fully retrieved {full}/{len(per_q)}")