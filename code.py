import re
import sys
import math
import numpy as np
from stop_list import closed_class_stop_words

def clean_text(text, stop_words):
    """Clean text by removing punctuation, numbers, and stop words."""
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove digits
    words = text.lower().split()
    return [word for word in words if word not in stop_words]

def parse_file_with_sections(filename, stop_words):
    """
    Parse a file where each document starts with '.I <id>' and '.W' marks the beginning of content.
    Cleans and tokenizes the content, stores by ID.
    """
    results = {}
    with open(filename, 'r') as file:
        current_id = None
        content = []
        for line in file:
            if line.startswith('.I'):  # New document
                if current_id is not None:
                    results[current_id] = clean_text(' '.join(content), stop_words)
                current_id = int(line.split()[1])
                content = []
            elif line.startswith('.W'):
                continue  # Skip marker lines
            else:
                content.append(line.strip())
        if current_id is not None:
            results[current_id] = clean_text(' '.join(content), stop_words)
    return results

def compute_tf(doc):
    """Compute raw term frequency (TF) for a document."""
    tf = {}
    for word in doc:
        tf[word] = tf.get(word, 0) + 1
    return tf

def compute_idf(docs):
    """
    Compute Inverse Document Frequency (IDF) for all terms in a document set.
    IDF = log(N / df), where df = number of documents containing the word.
    """
    idf = {}
    total_docs = len(docs)
    all_words = set(word for content in docs.values() for word in content)
    for word in all_words:
        doc_freq = sum(word in docs[doc_id] for doc_id in docs)
        idf[word] = math.log(total_docs / doc_freq)
    return idf

def tf_idf(docs, idf):
    """Compute TF-IDF vectors for each document."""
    tfidf = {}
    for doc_id, words in docs.items():
        tf = compute_tf(words)
        tfidf[doc_id] = {word: tf[word] * idf.get(word, 0) for word in tf}
    return tfidf

def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two sparse TF-IDF vectors represented as dictionaries.
    """
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([val ** 2 for val in vec1.values()])
    sum2 = sum([val ** 2 for val in vec2.values()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    return numerator / denominator if denominator else 0

def rank_abstracts(query_tfidf, doc_tfidf):
    """
    For each query, compute cosine similarity to all documents and rank them.
    Returns: {query_id: [(doc_id, score), ...]}
    """
    results = {}
    for query_id, query_vec in query_tfidf.items():
        scores = [(doc_id, cosine_similarity(query_vec, doc_tfidf[doc_id])) for doc_id in doc_tfidf]
        results[query_id] = sorted(scores, key=lambda x: x[1], reverse=True)
    return results

def main():
    # Parse and clean queries and documents
    queries = parse_file_with_sections("cran.qry", closed_class_stop_words)
    abstracts = parse_file_with_sections("cran.all.1400", closed_class_stop_words)

    #  FIX: Use only document IDF to compute both doc and query TF-IDF
    idf = compute_idf(abstracts)

    query_tfidf = tf_idf(queries, idf)
    doc_tfidf = tf_idf(abstracts, idf)

    # Rank documents for each query
    ranked_results = rank_abstracts(query_tfidf, doc_tfidf)

    # Output results in TREC format
    with open("output.txt", "w") as output_file:
        for query_id, scores in ranked_results.items():
            for doc_id, score in scores:
                if score > 0:
                    output_file.write(f"{query_id} {doc_id} {score:.6f}\n")
    print("Ranking results have been saved to output.txt.")

if __name__ == "__main__":
    main()
