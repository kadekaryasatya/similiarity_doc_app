import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from database import get_documents, save_document, delete_document
from preprocessing import preprocess_text, preprocess_text_new, pdf_to_text
from extract_item import extract_details, extract_title
from similarity import calculate_similarity_tfidf, calculate_similarity_word_count, calculate_similarity_bow, calculate_similarity_ngram, calculate_similarity_word_embedding
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
        preprocessed_content = preprocess_text(content)
        title = extract_title(preprocessed_content)
        if st.button("Simpan"):
            st.write(preprocessed_content)
            details_content = extract_details(preprocessed_content)
            save_document(title, details_content)
            st.success("Dokumen berhasil disimpan!")

def input_document_page_new():
    st.title("Tambah Dokumen Baru v2")

    # title_input = st.text_input("Masukkan Judul Dokumen")
    uploaded_file = st.file_uploader("Unggah File PDF", type="pdf")
    
    if uploaded_file is not None:
        content = pdf_to_text(uploaded_file)
        preprocessed_content = preprocess_text_new(content)
        title = extract_title(preprocessed_content)
        if st.button("Simpan"):
            st.write(preprocessed_content)
            details_content = extract_details(preprocessed_content)
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

    st.subheader("Pilih Metode Perhitungan:")
    similarity_method = st.selectbox("Metode:", ["TF-IDF", "Word Count", "N-Gram", "Bag of Words", "Word Embedding"])

    n_value = 2  
    if similarity_method == "N-Gram":
        n_value = st.number_input("Pilih nilai N untuk N-Gram:", min_value=1, max_value=5, value=2, step=1)

    if num_samples > 2:
        num_clusters = st.number_input("Jumlah Cluster", min_value=2, max_value=num_samples-1, value=2, step=1)
    
    st.markdown("<h5 style='text-align:center;'>", unsafe_allow_html=True)

    if st.button("Hitung Keterkaitan"):
        st.header("Nilai Keterkaitan Antar Naskah:")
        columns = ['pemrakarsa', 'level_peraturan', 'konten_penimbang', 'peraturan_terkait', 
                   'konten_peraturan', 'kategori_peraturan', 'topik_peraturan', 'struktur_peraturan']

        if similarity_method == "TF-IDF":
            similarity_data, total_similarity_data = calculate_similarity_tfidf(documents)
        elif similarity_method == "Word Count":
            similarity_data, total_similarity_data = calculate_similarity_word_count(documents)
        elif similarity_method == "N-Gram":
            similarity_data, total_similarity_data = calculate_similarity_ngram(documents, n=n_value)
        elif similarity_method == "Bag of Words":
            similarity_data, total_similarity_data = calculate_similarity_bow(documents)
        else:
            similarity_data, total_similarity_data = calculate_similarity_word_embedding(documents)


        for col in columns:
            st.write(f"Item: {col}")
            col_similarity_df = pd.DataFrame(similarity_data[col])
            col_styler = col_similarity_df.style.set_properties(**{'text-align': 'center'}).format({'Keterkaitan (%)': "{:.4f}"}).hide(axis='index')
            st.write(col_styler.to_html(), unsafe_allow_html=True)
            st.divider()

        st.header("Total Nilai Keterkaitan Antar Naskah:")
        total_similarity_results = []
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                total_similarity = total_similarity_data[i][j] / len(columns)  # Rata-rata keterkaitan
                total_similarity_results.append({
                    'Dokumen 1': documents[i]['title'],
                    'Dokumen 2': documents[j]['title'],
                    'Total Keterkaitan (%)': total_similarity
                })

        total_similarity_df = pd.DataFrame(total_similarity_results)
        total_styler = total_similarity_df.style.set_properties(**{'text-align': 'center'}).format({'Total Keterkaitan (%)': "{:.4f}"}).hide(axis='index')
        st.write(total_styler.to_html(), unsafe_allow_html=True)

        st.divider()

        st.header("Hasil Clustering:")

        if num_samples == 2:
            st.warning("Clustering tidak dapat dilakukan dengan hanya dua dokumen. Silakan tambahkan lebih banyak dokumen.")
            st.divider()
        else:
            combined_contents = [
                doc['pemrakarsa'] + " " + doc['level_peraturan'] + " " + doc['konten_penimbang'] + " " + 
                doc['peraturan_terkait'] + " " + doc['konten_peraturan'] + " " + doc['kategori_peraturan'] + " " +
                doc['topik_peraturan'] + " " + doc['struktur_peraturan']
                for doc in documents
            ]

            if similarity_method == "Word Embedding":
                model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                embeddings = np.array([model.encode(content) for content in combined_contents])
                similarity_matrix = cosine_similarity(embeddings)
                silhouette_avg, labels = perform_clustering(similarity_matrix, num_clusters)
                st.write(f"Silhouette Coefficient: {silhouette_avg}")
                cluster_data = {f"Cluster {i + 1}": [] for i in range(num_clusters)}
                for cluster_id in range(num_clusters):
                    cluster_docs = [documents[i]['title'] for i, label in enumerate(labels) if label == cluster_id]
                    cluster_data[f"Cluster {cluster_id + 1}"] = cluster_docs
                
                cluster_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in cluster_data.items()])).fillna('')
                
                cluster_styler = cluster_df.style.set_properties(**{'text-align': 'center'}).hide(axis='index')
                st.write(cluster_styler.to_html(), unsafe_allow_html=True)
            else:
                if similarity_method == "TF-IDF":
                    vectorizer = TfidfVectorizer()
                elif similarity_method == "Word Count":
                    vectorizer = CountVectorizer()
                elif similarity_method == "N-Gram":
                    vectorizer = CountVectorizer(ngram_range=(1, n_value))
                elif similarity_method == "Bag of Words":
                    vectorizer = TfidfVectorizer(use_idf=False, norm=None)

                tfidf_matrix = vectorizer.fit_transform(combined_contents)
                similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

                silhouette_avg, labels = perform_clustering(similarity_matrix, num_clusters)
                st.write(f"Silhouette Coefficient: {silhouette_avg}")
                cluster_data = {f"Cluster {i + 1}": [] for i in range(num_clusters)}
                for cluster_id in range(num_clusters):
                    cluster_docs = [documents[i]['title'] for i, label in enumerate(labels) if label == cluster_id]
                    cluster_data[f"Cluster {cluster_id + 1}"] = cluster_docs
                
                cluster_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in cluster_data.items()])).fillna('')
                
                cluster_styler = cluster_df.style.set_properties(**{'text-align': 'center'}).hide(axis='index')
                st.write(cluster_styler.to_html(), unsafe_allow_html=True)

def navigation():
    st.sidebar.title("Navigasi")
    if st.sidebar.button("Home"):
        st.session_state.page = 'home'
    if st.sidebar.button("Input Dokumen"):
        st.session_state.page = 'input'
    if st.sidebar.button("Input Dokumen New"):
        st.session_state.page = 'input_new'
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
    elif st.session_state.page == 'input_new':
        input_document_page_new()
    elif st.session_state.page == 'similarity':
        similarity_page()
    elif st.session_state.page == 'view':
        view_document_page()
    else:
        st.error("Halaman tidak ditemukan")

if __name__ == "__main__":
    main()
