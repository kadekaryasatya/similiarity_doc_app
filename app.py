import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from database import get_documents, save_document, delete_document
from preprocessing import preprocess_text1, preprocess_text2, preprocess_text3, preprocess_text4, preprocess_text5, pdf_to_text
from extract_item import extract_details, extract_title
from similarity import calculate_similarity_tfidf, calculate_similarity_word_count, calculate_similarity_bow, calculate_similarity_ngram, calculate_similarity_word_embedding, calculate_similarity_tfidf_ngram, calculate_similarity_word_count_tfidf, calculate_similarity_word_count_ngram, calculate_similarity_word_count_tfidf_ngram, calculate_similarity_bow_tfidf, calculate_similarity_bow_ngram, calculate_similarity_bow_tfidf_ngram, calculate_similarity_word_embedding_tfidf, calculate_similarity_word_embedding_ngram, calculate_similarity_word_embedding_tfidf_ngram
from clustering import perform_clustering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np


st.set_page_config(page_title="Similarity Docs Apps",
                   page_icon="📃", layout="wide")

def home_page():
    st.title("Sistem Analisa Tingkat Keterkaitan Antara Naskah Peraturan")
    
    st.subheader("Dokumen yang Tersimpan")
    documents = get_documents()
    
    if documents:
        for doc in documents:
            st.divider()
            col1, col2, col3 = st.columns([6, 2, 2])
            with col1:
                st.write(doc['title'])
            with col2:
                if st.button("Lihat Isi", key=f"view_{doc['id']}"):
                    st.session_state.page = 'view'
                    st.session_state.current_doc = doc
            with col3:
                if st.button("Hapus", key=f"delete_{doc['id']}"):
                    delete_document(doc['id'])
                    st.experimental_rerun()
            
    else:
       st.divider()
       st.warning("Tidak ada dokumen yang tersimpan.")


    st.divider()
    if st.button("Tambah Dokumen Baru"):
       st.session_state.page = 'input'

def view_document_page():
    doc = st.session_state.get('current_doc', None)
    if doc:
        st.title(doc['title'])
        st.divider()

        st.subheader("Pemrakarsa")
        st.write(doc['pemrakarsa'])
        
        st.subheader("Level Peraturan")
        st.write(doc['level_peraturan'])
        
        st.subheader("Konten Penimbang")
        st.write(doc['konten_penimbang'])
        
        st.subheader("Peraturan Terkait")
        st.write(doc['peraturan_terkait'])
        
        st.subheader("Konten Peraturan")
        st.write(doc['konten_peraturan'])
        
        st.subheader("Kategori Peraturan")
        st.write(doc['kategori_peraturan'])
        
        st.subheader("Topik Peraturan")
        st.write(doc['topik_peraturan'])
        
        st.subheader("Struktur Peraturan")
        st.write(doc['struktur_peraturan'])
        
        if st.button("Kembali"):
            st.session_state.page = 'home'

def input_document_page():
    st.title("Tambah Dokumen Baru")

    # title_input = st.text_input("Masukkan Judul Dokumen")
    uploaded_file = st.file_uploader("Unggah File PDF", type="pdf")
    
    if uploaded_file is not None:
        content = pdf_to_text(uploaded_file)
        title = extract_title(content)
        if st.button("Simpan"):
            st.write(content)
            details_content = extract_details(content)
            save_document(title, details_content)
            st.success("Dokumen berhasil disimpan!")

def similarity_page():
    st.title("Keterkaitan Dokumen")
    
    documents = get_documents()
    num_samples = len(documents)
    
    if num_samples < 2:
        st.warning("Untuk Menghitung Keterkaitan, diperlukan setidaknya 2 dokumen.")
        st.divider()

        if st.button("Tambah Dokumen Baru"):
            st.session_state.page = 'input'
        return

    n_value = 2  
    if num_samples > 2:
        num_clusters = st.number_input("Jumlah Cluster", min_value=2, max_value=num_samples-1, value=2, step=1)
    
    st.markdown("<h5 style='text-align:center;'>", unsafe_allow_html=True)

    if st.button("Hitung Keterkaitan"):
        st.header("Nilai Keterkaitan Antar Naskah:")
        columns = ['pemrakarsa', 'level_peraturan', 'konten_penimbang', 'peraturan_terkait', 
                   'konten_peraturan', 'kategori_peraturan', 'topik_peraturan', 'struktur_peraturan']
        
        preprocessed_documents1 = [
            {
                'title': doc['title'],
                'pemrakarsa': preprocess_text1(doc['pemrakarsa']),
                'level_peraturan': preprocess_text1(doc['level_peraturan']),
                'konten_penimbang': preprocess_text1(doc['konten_penimbang']),
                'peraturan_terkait': preprocess_text1(doc['peraturan_terkait']),
                'konten_peraturan': preprocess_text1(doc['konten_peraturan']),
                'kategori_peraturan': preprocess_text1(doc['kategori_peraturan']),
                'topik_peraturan': preprocess_text1(doc['topik_peraturan']),
                'struktur_peraturan': preprocess_text1(doc['struktur_peraturan'])
            }
            for doc in documents
        ]

        preprocessed_documents2 = [
            {
                'title': doc['title'],
                'pemrakarsa': preprocess_text2(doc['pemrakarsa']),
                'level_peraturan': preprocess_text2(doc['level_peraturan']),
                'konten_penimbang': preprocess_text2(doc['konten_penimbang']),
                'peraturan_terkait': preprocess_text2(doc['peraturan_terkait']),
                'konten_peraturan': preprocess_text2(doc['konten_peraturan']),
                'kategori_peraturan': preprocess_text2(doc['kategori_peraturan']),
                'topik_peraturan': preprocess_text2(doc['topik_peraturan']),
                'struktur_peraturan': preprocess_text2(doc['struktur_peraturan'])
            }
            for doc in documents
        ]

        preprocessed_documents3 = [
            {
                'title': doc['title'],
                'pemrakarsa': preprocess_text3(doc['pemrakarsa']),
                'level_peraturan': preprocess_text3(doc['level_peraturan']),
                'konten_penimbang': preprocess_text3(doc['konten_penimbang']),
                'peraturan_terkait': preprocess_text3(doc['peraturan_terkait']),
                'konten_peraturan': preprocess_text3(doc['konten_peraturan']),
                'kategori_peraturan': preprocess_text3(doc['kategori_peraturan']),
                'topik_peraturan': preprocess_text3(doc['topik_peraturan']),
                'struktur_peraturan': preprocess_text3(doc['struktur_peraturan'])
            }
            for doc in documents
        ]

        preprocessed_documents4 = [
            {
                'title': doc['title'],
                'pemrakarsa': preprocess_text4(doc['pemrakarsa']),
                'level_peraturan': preprocess_text4(doc['level_peraturan']),
                'konten_penimbang': preprocess_text4(doc['konten_penimbang']),
                'peraturan_terkait': preprocess_text4(doc['peraturan_terkait']),
                'konten_peraturan': preprocess_text4(doc['konten_peraturan']),
                'kategori_peraturan': preprocess_text4(doc['kategori_peraturan']),
                'topik_peraturan': preprocess_text4(doc['topik_peraturan']),
                'struktur_peraturan': preprocess_text4(doc['struktur_peraturan'])
            }
            for doc in documents
        ]

        preprocessed_documents5 = [
            {
                'title': doc['title'],
                'pemrakarsa': preprocess_text5(doc['pemrakarsa']),
                'level_peraturan': preprocess_text5(doc['level_peraturan']),
                'konten_penimbang': preprocess_text5(doc['konten_penimbang']),
                'peraturan_terkait': preprocess_text5(doc['peraturan_terkait']),
                'konten_peraturan': preprocess_text5(doc['konten_peraturan']),
                'kategori_peraturan': preprocess_text5(doc['kategori_peraturan']),
                'topik_peraturan': preprocess_text5(doc['topik_peraturan']),
                'struktur_peraturan': preprocess_text5(doc['struktur_peraturan'])
            }
            for doc in documents
        ]

        similarity_methods = {
            "TF-IDF": calculate_similarity_tfidf,
            "TF-IDF + N-Gram": lambda docs: calculate_similarity_tfidf_ngram(docs, n=n_value),
            "N-Gram": lambda docs: calculate_similarity_ngram(docs, n=n_value),
            "Word Count": calculate_similarity_word_count,
            "Word Count (TF-IDF)": calculate_similarity_word_count_tfidf,
            "Word Count (N-Gram)": lambda docs: calculate_similarity_word_count_ngram(docs, n=n_value),
            "Word Count (TF-IDF + N-Gram)": lambda docs: calculate_similarity_word_count_tfidf_ngram(docs, n=n_value),
            "Bag of Words": calculate_similarity_bow,
            "Bag of Words (TF-IDF)": lambda docs: calculate_similarity_bow_tfidf(docs),
            "Bag of Words (N-Gram)": lambda docs: calculate_similarity_bow_ngram(docs, n=n_value),
            "Bag of Words (TF-IDF + N-Gram)": lambda docs: calculate_similarity_bow_tfidf_ngram(docs, n=n_value),
            "Word Embedding": calculate_similarity_word_embedding,
            "Word Embedding (TF-IDF)": lambda docs: calculate_similarity_word_embedding_tfidf(docs),
            "Word Embedding (N-Gram)": lambda docs: calculate_similarity_word_embedding_ngram(docs, n=n_value),
            "Word Embedding (TF-IDF + N-Gram)": lambda docs: calculate_similarity_word_embedding_tfidf_ngram(docs, n=n_value)
        }

        tab_names = list(similarity_methods.keys())
        tabs = st.tabs(tab_names)

        for i, method_name in enumerate(tab_names):
            with tabs[i]:
                st.subheader(method_name)

                similarity_data1, total_similarity_data1 = similarity_methods[method_name](preprocessed_documents1)
                similarity_data2, total_similarity_data2 = similarity_methods[method_name](preprocessed_documents2)
                similarity_data3, total_similarity_data3 = similarity_methods[method_name](preprocessed_documents3)
                similarity_data4, total_similarity_data4 = similarity_methods[method_name](preprocessed_documents4)
                similarity_data5, total_similarity_data5 = similarity_methods[method_name](preprocessed_documents5)

                # Similarity Metode 1
                expander1 = st.expander(f"Similarity (case folding, tokenizing)")
                with expander1:
                    for col in columns:
                        expander1.write(f"Item: {col}")
                        col_similarity_df = pd.DataFrame(similarity_data1[col])
                        col_styler = col_similarity_df.style.set_properties(**{'text-align': 'center'}).format({'Keterkaitan (%)': "{:.4f}"}).hide(axis='index')
                        expander1.write(col_styler.to_html(), unsafe_allow_html=True)
                        expander1.divider()

                    st.header("Total Nilai Keterkaitan Antar Naskah (Metode 1):")
                    total_similarity_results1 = []
                    for i in range(num_samples):
                        for j in range(i + 1, num_samples):
                            total_similarity = total_similarity_data1[i][j] / len(columns)
                            total_similarity_results1.append({
                                'Dokumen 1': documents[i]['title'],
                                'Dokumen 2': documents[j]['title'],
                                'Total Keterkaitan (%)': total_similarity
                            })

                    total_similarity_df1 = pd.DataFrame(total_similarity_results1)
                    total_styler1 = total_similarity_df1.style.set_properties(**{'text-align': 'center'}).format({'Total Keterkaitan (%)': "{:.4f}"}).hide(axis='index')
                    st.write(total_styler1.to_html(), unsafe_allow_html=True)

                    st.divider()
                
                # Cluster Metode 1
                expander_cluster1 = st.expander(f"Clustering (case folding, tokenizing)")
                with expander_cluster1:
                    if num_samples == 2:
                        st.warning("Clustering tidak dapat dilakukan dengan hanya dua dokumen. Silakan tambahkan lebih banyak dokumen.")
                    else:
                        combined_contents1 = [
                            doc['pemrakarsa'] + " " + doc['level_peraturan'] + " " + doc['konten_penimbang'] + " " + 
                            doc['peraturan_terkait'] + " " + doc['konten_peraturan'] + " " + doc['kategori_peraturan'] + " " +
                            doc['topik_peraturan'] + " " + doc['struktur_peraturan']
                            for doc in preprocessed_documents1
                        ]

                        if method_name == "Word Embedding":
                            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                            embeddings1 = np.array([model.encode(content) for content in combined_contents1])
                            similarity_matrix1 = cosine_similarity(embeddings1)
                        else:
                            if method_name == "TF-IDF":
                                vectorizer1 = TfidfVectorizer()
                            elif method_name == "Word Count":
                                vectorizer1 = CountVectorizer()
                            elif method_name == "N-Gram" or method_name == "TF-IDF + N-Gram":
                                vectorizer1 = CountVectorizer(ngram_range=(1, n_value))
                            elif method_name.startswith("Bag of Words"):
                                if "TF-IDF" in method_name:
                                    vectorizer1 = TfidfVectorizer(use_idf=True)
                                elif "N-Gram" in method_name:
                                    vectorizer1 = CountVectorizer(ngram_range=(1, n_value))
                                else:
                                    vectorizer1 = CountVectorizer()

                            tfidf_matrix1 = vectorizer1.fit_transform(combined_contents1)
                            similarity_matrix1 = cosine_similarity(tfidf_matrix1)

                        silhouette_avg1, labels1 = perform_clustering(similarity_matrix1, num_clusters)
                        st.write(f"Silhouette Coefficient (Metode 1): {silhouette_avg1}")

                        cluster_data1 = {f"Cluster {i + 1}": [] for i in range(num_clusters)}
                        for cluster_id in range(num_clusters):
                            cluster_docs1 = [documents[i]['title'] for i, label in enumerate(labels1) if label == cluster_id]
                            cluster_data1[f"Cluster {cluster_id + 1}"] = cluster_docs1
                        
                        cluster_df1 = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in cluster_data1.items()])).fillna('')
                        cluster_styler1 = cluster_df1.style.set_properties(**{'text-align': 'center'}).hide(axis='index')
                        st.write(cluster_styler1.to_html(), unsafe_allow_html=True)

                        st.divider()
                
                # Similarity Metode 2
                expander2 = st.expander(f"Similarity (case folding, tokenizing, stemming)")
                with expander2:
                    for col in columns:
                        expander2.write(f"Item: {col}")
                        col_similarity_df = pd.DataFrame(similarity_data2[col])
                        col_styler = col_similarity_df.style.set_properties(**{'text-align': 'center'}).format({'Keterkaitan (%)': "{:.4f}"}).hide(axis='index')
                        expander2.write(col_styler.to_html(), unsafe_allow_html=True)
                        expander2.divider()

                    st.header("Total Nilai Keterkaitan Antar Naskah (Metode 2):")
                    total_similarity_results2 = []
                    for i in range(num_samples):
                        for j in range(i + 1, num_samples):
                            total_similarity = total_similarity_data2[i][j] / len(columns)
                            total_similarity_results2.append({
                                'Dokumen 1': documents[i]['title'],
                                'Dokumen 2': documents[j]['title'],
                                'Total Keterkaitan (%)': total_similarity
                            })

                    total_similarity_df2 = pd.DataFrame(total_similarity_results2)
                    total_styler2 = total_similarity_df2.style.set_properties(**{'text-align': 'center'}).format({'Total Keterkaitan (%)': "{:.4f}"}).hide(axis='index')
                    st.write(total_styler2.to_html(), unsafe_allow_html=True)

                    st.divider()
                
                # Cluster Metode 2
                expander_cluster2 = st.expander(f"Clustering (case folding, tokenizing, stemming)")
                with expander_cluster2:
                    if num_samples == 2:
                        st.warning("Clustering tidak dapat dilakukan dengan hanya dua dokumen. Silakan tambahkan lebih banyak dokumen.")
                    else:
                        combined_contents2 = [
                            doc['pemrakarsa'] + " " + doc['level_peraturan'] + " " + doc['konten_penimbang'] + " " + 
                            doc['peraturan_terkait'] + " " + doc['konten_peraturan'] + " " + doc['kategori_peraturan'] + " " +
                            doc['topik_peraturan'] + " " + doc['struktur_peraturan']
                            for doc in preprocessed_documents2
                        ]

                        if method_name == "Word Embedding":
                            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                            embeddings2 = np.array([model.encode(content) for content in combined_contents2])
                            similarity_matrix2 = cosine_similarity(embeddings2)
                        else:
                            if method_name == "TF-IDF":
                                vectorizer2 = TfidfVectorizer()
                            elif method_name == "Word Count":
                                vectorizer2 = CountVectorizer()
                            elif method_name == "N-Gram" or method_name == "TF-IDF + N-Gram":
                                vectorizer2 = CountVectorizer(ngram_range=(1, n_value))
                            elif method_name.startswith("Bag of Words"):
                                if "TF-IDF" in method_name:
                                    vectorizer2 = TfidfVectorizer(use_idf=True)
                                elif "N-Gram" in method_name:
                                    vectorizer2 = CountVectorizer(ngram_range=(1, n_value))
                                else:
                                    vectorizer2 = CountVectorizer()

                            tfidf_matrix2 = vectorizer2.fit_transform(combined_contents2)
                            similarity_matrix2 = cosine_similarity(tfidf_matrix2)

                        silhouette_avg2, labels2 = perform_clustering(similarity_matrix2, num_clusters)
                        st.write(f"Silhouette Coefficient (Metode 2): {silhouette_avg2}")

                        cluster_data2 = {f"Cluster {i + 1}": [] for i in range(num_clusters)}
                        for cluster_id in range(num_clusters):
                            cluster_docs2 = [documents[i]['title'] for i, label in enumerate(labels2) if label == cluster_id]
                            cluster_data2[f"Cluster {cluster_id + 1}"] = cluster_docs2
                        
                        cluster_df2 = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in cluster_data2.items()])).fillna('')
                        cluster_styler2 = cluster_df2.style.set_properties(**{'text-align': 'center'}).hide(axis='index')
                        st.write(cluster_styler2.to_html(), unsafe_allow_html=True)

                        st.divider()

                # Similarity Metode 3
                expander3 = st.expander(f"Similarity (case folding, tokenizing, filtering)")
                with expander3:
                    for col in columns:
                        expander3.write(f"Item: {col}")
                        col_similarity_df = pd.DataFrame(similarity_data3[col])
                        col_styler = col_similarity_df.style.set_properties(**{'text-align': 'center'}).format({'Keterkaitan (%)': "{:.4f}"}).hide(axis='index')
                        expander3.write(col_styler.to_html(), unsafe_allow_html=True)
                        expander3.divider()

                    st.header("Total Nilai Keterkaitan Antar Naskah (Metode 3):")
                    total_similarity_results3 = []
                    for i in range(num_samples):
                        for j in range(i + 1, num_samples):
                            total_similarity = total_similarity_data3[i][j] / len(columns)
                            total_similarity_results3.append({
                                'Dokumen 1': documents[i]['title'],
                                'Dokumen 2': documents[j]['title'],
                                'Total Keterkaitan (%)': total_similarity
                            })

                    total_similarity_df3 = pd.DataFrame(total_similarity_results3)
                    total_styler3 = total_similarity_df3.style.set_properties(**{'text-align': 'center'}).format({'Total Keterkaitan (%)': "{:.4f}"}).hide(axis='index')
                    st.write(total_styler3.to_html(), unsafe_allow_html=True)

                    st.divider()

                # Cluster Metode 3
                expander_cluster3 = st.expander(f"Clustering (case folding, tokenizing, filtering)")
                with expander_cluster3:
                    if num_samples == 2:
                        st.warning("Clustering tidak dapat dilakukan dengan hanya dua dokumen. Silakan tambahkan lebih banyak dokumen.")
                    else:
                        combined_contents3 = [
                            doc['pemrakarsa'] + " " + doc['level_peraturan'] + " " + doc['konten_penimbang'] + " " + 
                            doc['peraturan_terkait'] + " " + doc['konten_peraturan'] + " " + doc['kategori_peraturan'] + " " +
                            doc['topik_peraturan'] + " " + doc['struktur_peraturan']
                            for doc in preprocessed_documents3
                        ]

                        if method_name == "Word Embedding":
                            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                            embeddings3 = np.array([model.encode(content) for content in combined_contents3])
                            similarity_matrix3 = cosine_similarity(embeddings3)
                        else:
                            if method_name == "TF-IDF":
                                vectorizer3 = TfidfVectorizer()
                            elif method_name == "Word Count":
                                vectorizer3 = CountVectorizer()
                            elif method_name == "N-Gram" or method_name == "TF-IDF + N-Gram":
                                vectorizer3 = CountVectorizer(ngram_range=(1, n_value))
                            elif method_name.startswith("Bag of Words"):
                                if "TF-IDF" in method_name:
                                    vectorizer3 = TfidfVectorizer(use_idf=True)
                                elif "N-Gram" in method_name:
                                    vectorizer3 = CountVectorizer(ngram_range=(1, n_value))
                                else:
                                    vectorizer3 = CountVectorizer()

                            tfidf_matrix3 = vectorizer3.fit_transform(combined_contents3)
                            similarity_matrix3 = cosine_similarity(tfidf_matrix3)

                        silhouette_avg3, labels3 = perform_clustering(similarity_matrix3, num_clusters)
                        st.write(f"Silhouette Coefficient (Metode 3): {silhouette_avg3}")

                        cluster_data3 = {f"Cluster {i + 1}": [] for i in range(num_clusters)}
                        for cluster_id in range(num_clusters):
                            cluster_docs3 = [documents[i]['title'] for i, label in enumerate(labels3) if label == cluster_id]
                            cluster_data3[f"Cluster {cluster_id + 1}"] = cluster_docs3
                        
                        cluster_df3 = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in cluster_data3.items()])).fillna('')
                        cluster_styler3 = cluster_df3.style.set_properties(**{'text-align': 'center'}).hide(axis='index')
                        st.write(cluster_styler3.to_html(), unsafe_allow_html=True)

                        st.divider()
                
                # Similarity Metode 4
                expander4 = st.expander(f"Similarity (case folding, tokenizing, filtering, stemming)")
                with expander4:
                    for col in columns:
                        expander4.write(f"Item: {col}")
                        col_similarity_df = pd.DataFrame(similarity_data4[col])
                        col_styler = col_similarity_df.style.set_properties(**{'text-align': 'center'}).format({'Keterkaitan (%)': "{:.4f}"}).hide(axis='index')
                        expander4.write(col_styler.to_html(), unsafe_allow_html=True)
                        expander4.divider()

                    st.header("Total Nilai Keterkaitan Antar Naskah (Metode 4):")
                    total_similarity_results4 = []
                    for i in range(num_samples):
                        for j in range(i + 1, num_samples):
                            total_similarity = total_similarity_data4[i][j] / len(columns)
                            total_similarity_results4.append({
                                'Dokumen 1': documents[i]['title'],
                                'Dokumen 2': documents[j]['title'],
                                'Total Keterkaitan (%)': total_similarity
                            })

                    total_similarity_df4 = pd.DataFrame(total_similarity_results4)
                    total_styler4 = total_similarity_df4.style.set_properties(**{'text-align': 'center'}).format({'Total Keterkaitan (%)': "{:.4f}"}).hide(axis='index')
                    st.write(total_styler4.to_html(), unsafe_allow_html=True)

                    st.divider()

                # Cluster Metode 4
                expander_cluster4 = st.expander(f"Clustering (case folding, tokenizing, filtering, stemming)")
                with expander_cluster4:
                    if num_samples == 2:
                        st.warning("Clustering tidak dapat dilakukan dengan hanya dua dokumen. Silakan tambahkan lebih banyak dokumen.")
                    else:
                        combined_contents4 = [
                            doc['pemrakarsa'] + " " + doc['level_peraturan'] + " " + doc['konten_penimbang'] + " " + 
                            doc['peraturan_terkait'] + " " + doc['konten_peraturan'] + " " + doc['kategori_peraturan'] + " " +
                            doc['topik_peraturan'] + " " + doc['struktur_peraturan']
                            for doc in preprocessed_documents4
                        ]

                        if method_name == "Word Embedding":
                            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                            embeddings4 = np.array([model.encode(content) for content in combined_contents4])
                            similarity_matrix4 = cosine_similarity(embeddings4)
                        else:
                            if method_name == "TF-IDF":
                                vectorizer4 = TfidfVectorizer()
                            elif method_name == "Word Count":
                                vectorizer4 = CountVectorizer()
                            elif method_name == "N-Gram" or method_name == "TF-IDF + N-Gram":
                                vectorizer4 = CountVectorizer(ngram_range=(1, n_value))
                            elif method_name.startswith("Bag of Words"):
                                if "TF-IDF" in method_name:
                                    vectorizer4 = TfidfVectorizer(use_idf=True)
                                elif "N-Gram" in method_name:
                                    vectorizer4 = CountVectorizer(ngram_range=(1, n_value))
                                else:
                                    vectorizer4 = CountVectorizer()

                            tfidf_matrix4 = vectorizer4.fit_transform(combined_contents4)
                            similarity_matrix4 = cosine_similarity(tfidf_matrix4)

                        silhouette_avg4, labels4 = perform_clustering(similarity_matrix4, num_clusters)
                        st.write(f"Silhouette Coefficient (Metode 4): {silhouette_avg4}")

                        cluster_data4 = {f"Cluster {i + 1}": [] for i in range(num_clusters)}
                        for cluster_id in range(num_clusters):
                            cluster_docs4 = [documents[i]['title'] for i, label in enumerate(labels4) if label == cluster_id]
                            cluster_data4[f"Cluster {cluster_id + 1}"] = cluster_docs4
                        
                        cluster_df4 = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in cluster_data4.items()])).fillna('')
                        cluster_styler4 = cluster_df4.style.set_properties(**{'text-align': 'center'}).hide(axis='index')
                        st.write(cluster_styler4.to_html(), unsafe_allow_html=True)

                        st.divider()
                
                # Similarity Metode 5
                expander5 = st.expander(f"Similarity (part of speech, lemmatization, chunking)")
                with expander5:
                    for col in columns:
                        expander5.write(f"Item: {col}")
                        col_similarity_df = pd.DataFrame(similarity_data5[col])
                        col_styler = col_similarity_df.style.set_properties(**{'text-align': 'center'}).format({'Keterkaitan (%)': "{:.4f}"}).hide(axis='index')
                        expander5.write(col_styler.to_html(), unsafe_allow_html=True)
                        expander5.divider()

                    st.header("Total Nilai Keterkaitan Antar Naskah (Metode 5):")
                    total_similarity_results5 = []
                    for i in range(num_samples):
                        for j in range(i + 1, num_samples):
                            total_similarity = total_similarity_data5[i][j] / len(columns)
                            total_similarity_results5.append({
                                'Dokumen 1': documents[i]['title'],
                                'Dokumen 2': documents[j]['title'],
                                'Total Keterkaitan (%)': total_similarity
                            })

                    total_similarity_df5 = pd.DataFrame(total_similarity_results5)
                    total_styler5 = total_similarity_df5.style.set_properties(**{'text-align': 'center'}).format({'Total Keterkaitan (%)': "{:.4f}"}).hide(axis='index')
                    st.write(total_styler5.to_html(), unsafe_allow_html=True)

                    st.divider()

                # Cluster Metode 5
                expander_cluster5 = st.expander(f"Clustering (part of speech, lemmatization, chunking)")
                with expander_cluster5:
                    if num_samples == 2:
                        st.warning("Clustering tidak dapat dilakukan dengan hanya dua dokumen. Silakan tambahkan lebih banyak dokumen.")
                    else:
                        combined_contents5 = [
                            doc['pemrakarsa'] + " " + doc['level_peraturan'] + " " + doc['konten_penimbang'] + " " + 
                            doc['peraturan_terkait'] + " " + doc['konten_peraturan'] + " " + doc['kategori_peraturan'] + " " +
                            doc['topik_peraturan'] + " " + doc['struktur_peraturan']
                            for doc in preprocessed_documents5
                        ]

                        if method_name == "Word Embedding":
                            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                            embeddings5 = np.array([model.encode(content) for content in combined_contents5])
                            similarity_matrix5 = cosine_similarity(embeddings5)
                        else:
                            if method_name == "TF-IDF":
                                vectorizer5 = TfidfVectorizer()
                            elif method_name == "Word Count":
                                vectorizer5 = CountVectorizer()
                            elif method_name == "N-Gram" or method_name == "TF-IDF + N-Gram":
                                vectorizer5 = CountVectorizer(ngram_range=(1, n_value))
                            elif method_name.startswith("Bag of Words"):
                                if "TF-IDF" in method_name:
                                    vectorizer5 = TfidfVectorizer(use_idf=True)
                                elif "N-Gram" in method_name:
                                    vectorizer5 = CountVectorizer(ngram_range=(1, n_value))
                                else:
                                    vectorizer5 = CountVectorizer()

                            tfidf_matrix5 = vectorizer5.fit_transform(combined_contents5)
                            similarity_matrix5 = cosine_similarity(tfidf_matrix5)

                        silhouette_avg5, labels5 = perform_clustering(similarity_matrix5, num_clusters)
                        st.write(f"Silhouette Coefficient (Metode 5): {silhouette_avg5}")

                        cluster_data5 = {f"Cluster {i + 1}": [] for i in range(num_clusters)}
                        for cluster_id in range(num_clusters):
                            cluster_docs5 = [documents[i]['title'] for i, label in enumerate(labels5) if label == cluster_id]
                            cluster_data5[f"Cluster {cluster_id + 1}"] = cluster_docs5
                        
                        cluster_df5 = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in cluster_data5.items()])).fillna('')
                        cluster_styler5 = cluster_df5.style.set_properties(**{'text-align': 'center'}).hide(axis='index')
                        st.write(cluster_styler5.to_html(), unsafe_allow_html=True)

                        st.divider()
              
        st.markdown("</h5>", unsafe_allow_html=True)

def navigation():
    st.sidebar.title("Navigasi")
    if st.sidebar.button("Home"):
        st.session_state.page = 'home'
    if st.sidebar.button("Input Dokumen"):
        st.session_state.page = 'input'
    if st.sidebar.button("Similarity"):
        st.session_state.page = 'similarity'
        
def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    navigation()
    
    if st.session_state.page == 'home':
        home_page()
    elif st.session_state.page == 'input':
        input_document_page()
    elif st.session_state.page == 'similarity':
        similarity_page()
    elif st.session_state.page == 'view':
        view_document_page()
    else:
        st.error("Halaman tidak ditemukan")

if __name__ == "__main__":
    main()
