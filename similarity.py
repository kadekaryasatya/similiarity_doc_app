import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sentence_transformers import SentenceTransformer

# Function to calculate total similarity
def calculate_total_similarity(total_similarity_data, num_columns):
    total_similarities = []
    num_samples = len(total_similarity_data)
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            total_similarity = total_similarity_data[i][j] / num_columns
            total_similarities.append(total_similarity)
    return total_similarities

# Tf-IDF
def calculate_similarity_tfidf(documents):
    columns = ['pemrakarsa', 'level_peraturan', 'konten_penimbang', 'peraturan_terkait', 
               'konten_peraturan', 'kategori_peraturan', 'topik_peraturan', 'struktur_peraturan']
    similarity_data = {col: [] for col in columns}
    total_similarity_data = [[0] * len(documents) for _ in range(len(documents))]

    for col in columns:
        contents = [" ".join(doc[col].split()) for doc in documents]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(contents)
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = similarity_matrix[i, j] * 100
                similarity_data[col].append({
                    'Dokumen 1': documents[i]['title'],
                    'Dokumen 2': documents[j]['title'],
                    'Keterkaitan (%)': similarity
                })
                total_similarity_data[i][j] += similarity

    return similarity_data, total_similarity_data

# N-Gram
def calculate_similarity_ngram(documents, n=1):
    columns = ['pemrakarsa', 'level_peraturan', 'konten_penimbang', 'peraturan_terkait', 
               'konten_peraturan', 'kategori_peraturan', 'topik_peraturan', 'struktur_peraturan']
    similarity_data = {col: [] for col in columns}
    total_similarity_data = [[0] * len(documents) for _ in range(len(documents))]

    for col in columns:
        contents = [" ".join(doc[col].split()) for doc in documents]
        vectorizer = CountVectorizer(ngram_range=(1, n))
        ngram_matrix = vectorizer.fit_transform(contents)
        similarity_matrix = cosine_similarity(ngram_matrix, ngram_matrix)

        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = similarity_matrix[i, j] * 100
                similarity_data[col].append({
                    'Dokumen 1': documents[i]['title'],
                    'Dokumen 2': documents[j]['title'],
                    'Keterkaitan (%)': similarity
                })
                total_similarity_data[i][j] += similarity

    return similarity_data, total_similarity_data

# Tf-IDF and N-Gram
def calculate_similarity_tfidf_ngram(documents, n=1):
    columns = ['pemrakarsa', 'level_peraturan', 'konten_penimbang', 'peraturan_terkait', 
               'konten_peraturan', 'kategori_peraturan', 'topik_peraturan', 'struktur_peraturan']
    similarity_data = {col: [] for col in columns}
    total_similarity_data = [[0] * len(documents) for _ in range(len(documents))]

    for col in columns:
        contents = [" ".join(doc[col].split()) for doc in documents]
        vectorizer = TfidfVectorizer(ngram_range=(1, n))
        tfidf_matrix = vectorizer.fit_transform(contents)
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = similarity_matrix[i, j] * 100
                similarity_data[col].append({
                    'Dokumen 1': documents[i]['title'],
                    'Dokumen 2': documents[j]['title'],
                    'Keterkaitan (%)': similarity
                })
                total_similarity_data[i][j] += similarity

    return similarity_data, total_similarity_data

# Word Count
def calculate_similarity_word_count(documents):
    columns = ['pemrakarsa', 'level_peraturan', 'konten_penimbang', 'peraturan_terkait', 
               'konten_peraturan', 'kategori_peraturan', 'topik_peraturan', 'struktur_peraturan']
    similarity_data = {col: [] for col in columns}
    total_similarity_data = [[0] * len(documents) for _ in range(len(documents))]

    for col in columns:
        contents = [" ".join(doc[col].split()) for doc in documents]
        vectorizer = CountVectorizer()
        word_count_matrix = vectorizer.fit_transform(contents)
        similarity_matrix = cosine_similarity(word_count_matrix, word_count_matrix)

        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = similarity_matrix[i, j] * 100
                similarity_data[col].append({
                    'Dokumen 1': documents[i]['title'],
                    'Dokumen 2': documents[j]['title'],
                    'Keterkaitan (%)': similarity
                })
                total_similarity_data[i][j] += similarity

    return similarity_data, total_similarity_data

# Word Count and Tf-IDF
def calculate_similarity_word_count_tfidf(documents):
    columns = ['pemrakarsa', 'level_peraturan', 'konten_penimbang', 'peraturan_terkait', 
               'konten_peraturan', 'kategori_peraturan', 'topik_peraturan', 'struktur_peraturan']
    similarity_data = {col: [] for col in columns}
    total_similarity_data = [[0] * len(documents) for _ in range(len(documents))]

    for col in columns:
        contents = [doc[col] for doc in documents]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(contents)
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = similarity_matrix[i, j] * 100
                similarity_data[col].append({
                    'Dokumen 1': documents[i]['title'],
                    'Dokumen 2': documents[j]['title'],
                    'Keterkaitan (%)': similarity
                })
                total_similarity_data[i][j] += similarity

    return similarity_data, total_similarity_data

# Word Count and N-Gram
def calculate_similarity_word_count_ngram(documents, n=1):
    columns = ['pemrakarsa', 'level_peraturan', 'konten_penimbang', 'peraturan_terkait', 
               'konten_peraturan', 'kategori_peraturan', 'topik_peraturan', 'struktur_peraturan']
    similarity_data = {col: [] for col in columns}
    total_similarity_data = [[0] * len(documents) for _ in range(len(documents))]

    for col in columns:
        contents = [doc[col] for doc in documents]
        vectorizer = CountVectorizer(ngram_range=(1, n))
        ngram_matrix = vectorizer.fit_transform(contents)
        similarity_matrix = cosine_similarity(ngram_matrix, ngram_matrix)

        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = similarity_matrix[i, j] * 100
                similarity_data[col].append({
                    'Dokumen 1': documents[i]['title'],
                    'Dokumen 2': documents[j]['title'],
                    'Keterkaitan (%)': similarity
                })
                total_similarity_data[i][j] += similarity

    return similarity_data, total_similarity_data

# Word Count and Tf-IDF + N-Gram
def calculate_similarity_word_count_tfidf_ngram(documents, n=1):
    columns = ['pemrakarsa', 'level_peraturan', 'konten_penimbang', 'peraturan_terkait', 
               'konten_peraturan', 'kategori_peraturan', 'topik_peraturan', 'struktur_peraturan']
    similarity_data = {col: [] for col in columns}
    total_similarity_data = [[0] * len(documents) for _ in range(len(documents))]

    for col in columns:
        contents = [doc[col] for doc in documents]
        vectorizer = TfidfVectorizer(ngram_range=(1, n))
        tfidf_ngram_matrix = vectorizer.fit_transform(contents)
        similarity_matrix = cosine_similarity(tfidf_ngram_matrix, tfidf_ngram_matrix)

        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = similarity_matrix[i, j] * 100
                similarity_data[col].append({
                    'Dokumen 1': documents[i]['title'],
                    'Dokumen 2': documents[j]['title'],
                    'Keterkaitan (%)': similarity
                })
                total_similarity_data[i][j] += similarity

    return similarity_data, total_similarity_data

# Bag Of Words
def calculate_similarity_bow(documents):
    columns = ['pemrakarsa', 'level_peraturan', 'konten_penimbang', 'peraturan_terkait', 
               'konten_peraturan', 'kategori_peraturan', 'topik_peraturan', 'struktur_peraturan']
    similarity_data = {col: [] for col in columns}
    total_similarity_data = [[0] * len(documents) for _ in range(len(documents))]

    for col in columns:
        contents = [doc[col] for doc in documents]
        vectorizer = TfidfVectorizer(use_idf=False, norm=None)
        bow_matrix = vectorizer.fit_transform(contents)
        similarity_matrix = cosine_similarity(bow_matrix, bow_matrix)

        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = similarity_matrix[i, j] * 100
                similarity_data[col].append({
                    'Dokumen 1': documents[i]['title'],
                    'Dokumen 2': documents[j]['title'],
                    'Keterkaitan (%)': similarity
                })
                total_similarity_data[i][j] += similarity

    return similarity_data, total_similarity_data

# Bag Of Words and Tf-IDF
def calculate_similarity_bow_tfidf(documents):
    columns = ['pemrakarsa', 'level_peraturan', 'konten_penimbang', 'peraturan_terkait', 
               'konten_peraturan', 'kategori_peraturan', 'topik_peraturan', 'struktur_peraturan']
    similarity_data = {col: [] for col in columns}
    total_similarity_data = [[0] * len(documents) for _ in range(len(documents))]

    for col in columns:
        contents = [doc[col] for doc in documents]
        vectorizer = TfidfVectorizer()
        bow_matrix = vectorizer.fit_transform(contents)
        similarity_matrix = cosine_similarity(bow_matrix, bow_matrix)

        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = similarity_matrix[i, j] * 100
                similarity_data[col].append({
                    'Dokumen 1': documents[i]['title'],
                    'Dokumen 2': documents[j]['title'],
                    'Keterkaitan (%)': similarity
                })
                total_similarity_data[i][j] += similarity

    return similarity_data, total_similarity_data

# Bag Of Words and N-Gram
def calculate_similarity_bow_ngram(documents, n=1):
    columns = ['pemrakarsa', 'level_peraturan', 'konten_penimbang', 'peraturan_terkait', 
               'konten_peraturan', 'kategori_peraturan', 'topik_peraturan', 'struktur_peraturan']
    similarity_data = {col: [] for col in columns}
    total_similarity_data = [[0] * len(documents) for _ in range(len(documents))]

    for col in columns:
        contents = [doc[col] for doc in documents]
        vectorizer = CountVectorizer(ngram_range=(1, n))
        bow_matrix = vectorizer.fit_transform(contents)
        similarity_matrix = cosine_similarity(bow_matrix, bow_matrix)

        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = similarity_matrix[i, j] * 100
                similarity_data[col].append({
                    'Dokumen 1': documents[i]['title'],
                    'Dokumen 2': documents[j]['title'],
                    'Keterkaitan (%)': similarity
                })
                total_similarity_data[i][j] += similarity

    return similarity_data, total_similarity_data

# Bag Of Words and Tf-IDF + N-Gram
def calculate_similarity_bow_tfidf_ngram(documents, n=1):
    columns = ['pemrakarsa', 'level_peraturan', 'konten_penimbang', 'peraturan_terkait', 
               'konten_peraturan', 'kategori_peraturan', 'topik_peraturan', 'struktur_peraturan']
    similarity_data = {col: [] for col in columns}
    total_similarity_data = [[0] * len(documents) for _ in range(len(documents))]

    for col in columns:
        contents = [doc[col] for doc in documents]
        vectorizer = TfidfVectorizer(ngram_range=(1, n))
        bow_matrix = vectorizer.fit_transform(contents)
        similarity_matrix = cosine_similarity(bow_matrix, bow_matrix)

        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = similarity_matrix[i, j] * 100
                similarity_data[col].append({
                    'Dokumen 1': documents[i]['title'],
                    'Dokumen 2': documents[j]['title'],
                    'Keterkaitan (%)': similarity
                })
                total_similarity_data[i][j] += similarity

    return similarity_data, total_similarity_data

# Word Embedding
def calculate_similarity_word_embedding(documents):
    columns = ['pemrakarsa', 'level_peraturan', 'konten_penimbang', 'peraturan_terkait', 
               'konten_peraturan', 'kategori_peraturan', 'topik_peraturan', 'struktur_peraturan']
    similarity_data = {col: [] for col in columns}
    total_similarity_data = [[0] * len(documents) for _ in range(len(documents))]

    # Load pre-trained word embeddings
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def document_vector(doc):
        return model.encode(doc)

    for col in columns:
        contents = [doc[col] for doc in documents]
        vectors = [document_vector(doc) for doc in contents]
        similarity_matrix = cosine_similarity(vectors)

        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = similarity_matrix[i, j] * 100
                similarity_data[col].append({
                    'Dokumen 1': documents[i]['title'],
                    'Dokumen 2': documents[j]['title'],
                    'Keterkaitan (%)': similarity
                })
                total_similarity_data[i][j] += similarity

    return similarity_data, total_similarity_data

# Word Embedding and Tf-IDF
def calculate_similarity_word_embedding_tfidf(documents):
    columns = ['pemrakarsa', 'level_peraturan', 'konten_penimbang', 'peraturan_terkait', 
               'konten_peraturan', 'kategori_peraturan', 'topik_peraturan', 'struktur_peraturan']
    similarity_data = {col: [] for col in columns}
    total_similarity_data = [[0] * len(documents) for _ in range(len(documents))]

    # Load pre-trained word embeddings
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def document_vector(doc):
        return model.encode(doc).tolist()

    for col in columns:
        contents = [" ".join(doc[col].split()) for doc in documents]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(contents)  # tfidf_matrix is used to fit the vectorizer
        embeddings = [document_vector(content) for content in contents]
        similarity_matrix = cosine_similarity(embeddings)

        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = similarity_matrix[i, j] * 100
                similarity_data[col].append({
                    'Dokumen 1': documents[i]['title'],
                    'Dokumen 2': documents[j]['title'],
                    'Keterkaitan (%)': similarity
                })
                total_similarity_data[i][j] += similarity

    return similarity_data, total_similarity_data

# Word Embedding and N-Gram
def calculate_similarity_word_embedding_ngram(documents, n=1):
    columns = ['pemrakarsa', 'level_peraturan', 'konten_penimbang', 'peraturan_terkait', 
               'konten_peraturan', 'kategori_peraturan', 'topik_peraturan', 'struktur_peraturan']
    similarity_data = {col: [] for col in columns}
    total_similarity_data = [[0] * len(documents) for _ in range(len(documents))]

    # Load pre-trained word embeddings
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def document_vector(doc):
        return model.encode(doc).tolist()

    for col in columns:
        contents = [" ".join(doc[col].split()) for doc in documents]
        vectorizer = CountVectorizer(ngram_range=(1, n))
        ngram_matrix = vectorizer.fit_transform(contents)
        embeddings = [document_vector(content) for content in contents]
        similarity_matrix = cosine_similarity(embeddings)

        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = similarity_matrix[i, j] * 100
                similarity_data[col].append({
                    'Dokumen 1': documents[i]['title'],
                    'Dokumen 2': documents[j]['title'],
                    'Keterkaitan (%)': similarity
                })
                total_similarity_data[i][j] += similarity

    return similarity_data, total_similarity_data

# Word Embedding and Tf-IDF + N-Gram
def calculate_similarity_word_embedding_tfidf_ngram(documents, n=1):
    columns = ['pemrakarsa', 'level_peraturan', 'konten_penimbang', 'peraturan_terkait', 
               'konten_peraturan', 'kategori_peraturan', 'topik_peraturan', 'struktur_peraturan']
    similarity_data = {col: [] for col in columns}
    total_similarity_data = [[0] * len(documents) for _ in range(len(documents))]

    # Load pre-trained word embeddings
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def document_vector(doc):
        return model.encode(doc).tolist()

    for col in columns:
        contents = [" ".join(doc[col].split()) for doc in documents]
        vectorizer = TfidfVectorizer(ngram_range=(1, n))
        tfidf_ngram_matrix = vectorizer.fit_transform(contents)
        embeddings = [document_vector(content) for content in contents]
        similarity_matrix = cosine_similarity(embeddings)

        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = similarity_matrix[i, j] * 100
                similarity_data[col].append({
                    'Dokumen 1': documents[i]['title'],
                    'Dokumen 2': documents[j]['title'],
                    'Keterkaitan (%)': similarity
                })
                total_similarity_data[i][j] += similarity

    return similarity_data, total_similarity_data