import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from database import get_documents, save_document, delete_document
from preprocessing import preprocess_text,extract_details, pdf_to_text, extract_title
from clustering import perform_clustering
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Similiarity Apps",
                   page_icon="📈", layout="wide")

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
       st.markdown("<h5 style='text-align:center;'>Tidak ada dokumen yang tersimpan.</h5>", unsafe_allow_html=True)

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

    title_input = st.text_input("Masukkan Judul Dokumen")
    uploaded_file = st.file_uploader("Unggah File PDF", type="pdf")
    
    if uploaded_file is not None:
        content = pdf_to_text(uploaded_file)
        title = title_input.strip() if title_input else extract_title(content)
        if st.button("Simpan"):
            preprocessed_content = preprocess_text(content)
            details_content = extract_details(preprocessed_content)
            save_document(title, details_content)
            st.success("Dokumen berhasil disimpan!")

def similarity_page():
    st.title("Keterkaitan Dokumen")
    
    documents = get_documents()
    num_samples = len(documents)
    
    if num_samples < 2:
        st.warning("Untuk Menghitung Keterkaitan, diperlukan setidaknya 2 dokumen.")
        st.write("Silakan tambahkan dokumen lebih dari 1.")
        if st.button("Tambah Dokumen Baru"):
           st.session_state.page = 'input'
        return

    contents = [
        doc['pemrakarsa'] + " " + doc['level_peraturan'] + " " + doc['konten_penimbang'] + " " + 
        doc['peraturan_terkait'] + " " + doc['konten_peraturan'] + " " + doc['kategori_peraturan'] + " " +
        doc['topik_peraturan'] + " " + doc['struktur_peraturan']
        for doc in documents
    ]    
    min_clusters = 2
    max_clusters = min(2, num_samples - 1)  # Update the maximum clusters
    
    if num_samples == 2:
        num_clusters = 2
    else:
        num_clusters = st.number_input("Jumlah Cluster", min_value=min_clusters, max_value=max_clusters, value=min_clusters, step=1)
        
    if st.button("Hitung Keterkaitan"):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(contents)
        
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        st.header("Nilai Keterkaitan Antar Naskah:")
        similarity_data = []
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                similarity = similarity_matrix[i, j] * 100  
                similarity_data.append({
                    'Dokumen 1': documents[i]['title'],
                    'Dokumen 2': documents[j]['title'],
                    'Keterkaitan (%)': similarity   # Ubah nama kolom
                })
        similarity_df = pd.DataFrame(similarity_data)
        styler = similarity_df.style.set_properties(**{'text-align': 'center'}).format({'Keterkaitan': "{:.4f}"}).hide(axis='index')
        st.write(styler.to_html(), unsafe_allow_html=True)
        
        st.divider()
        silhouette_avg, labels = perform_clustering(similarity_matrix, num_clusters)
        st.write(f"Silhouette Coefficient: {silhouette_avg}")

        st.header("Hasil Clustering:")

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
    if st.sidebar.button("Similiarity"):
        st.session_state.page = 'similiarity'
        

def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    navigation()
    
    if st.session_state.page == 'home':
        home_page()
    elif st.session_state.page == 'input':
        input_document_page()
    elif st.session_state.page == 'similiarity':
        similarity_page()
    elif st.session_state.page == 'view':
        view_document_page()
    else:
        st.error("Halaman tidak ditemukan")

if __name__ == "__main__":
    main()
