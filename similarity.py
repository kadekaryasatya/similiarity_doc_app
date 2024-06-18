import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# import gensim.downloader as api
import pandas as pd
from sentence_transformers import SentenceTransformer



def calculate_similarity_tfidf(documents):
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

def calculate_similarity_word_count(documents):
    columns = ['pemrakarsa', 'level_peraturan', 'konten_penimbang', 'peraturan_terkait', 
               'konten_peraturan', 'kategori_peraturan', 'topik_peraturan', 'struktur_peraturan']
    similarity_data = {col: [] for col in columns}
    total_similarity_data = [[0] * len(documents) for _ in range(len(documents))]

    for col in columns:
        contents = [" ".join(doc[col].split()) for doc in documents]  # Menggunakan jumlah kata dari teks yang dipisahkan
        vectorizer = CountVectorizer()  # Menggunakan CountVectorizer untuk menghitung jumlah kata
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


def calculate_similarity_ngram(documents, n=2):
    columns = ['pemrakarsa', 'level_peraturan', 'konten_penimbang', 'peraturan_terkait', 
               'konten_peraturan', 'kategori_peraturan', 'topik_peraturan', 'struktur_peraturan']
    similarity_data = {col: [] for col in columns}
    total_similarity_data = [[0] * len(documents) for _ in range(len(documents))]

    for col in columns:
        contents = [doc[col] for doc in documents]
        valid_contents = []
        valid_indices = []

        for idx, content in enumerate(contents):
            if len(content.split()) >= n:
                valid_contents.append(content)
                valid_indices.append(idx)

        if not valid_contents:
            continue  # Skip if no valid contents

        vectorizer = CountVectorizer(ngram_range=(n, n), stop_words=None)
        ngram_matrix = vectorizer.fit_transform(valid_contents)
        similarity_matrix = cosine_similarity(ngram_matrix, ngram_matrix)

        for i, idx_i in enumerate(valid_indices):
            for j, idx_j in enumerate(valid_indices):
                if idx_i != idx_j:
                    similarity = similarity_matrix[i, j] * 100
                    similarity_data[col].append({
                        'Dokumen 1': documents[idx_i]['title'],
                        'Dokumen 2': documents[idx_j]['title'],
                        'Keterkaitan (%)': similarity
                    })
                    total_similarity_data[idx_i][idx_j] += similarity

    return similarity_data, total_similarity_data


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