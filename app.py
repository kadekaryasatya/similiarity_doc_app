import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from database import get_documents, save_document, delete_document
from preprocessing import preprocess_text, pdf_to_text, extract_title
from clustering import perform_clustering
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Similiarity Apps",
                   page_icon="📈", layout="wide")

# Fungsi untuk menampilkan halaman home
def home_page():
    st.title("Sistem Pengelompokan Dokumen Kerjasama atau Peraturan Pemerintahan")
    
    st.subheader("Dokumen yang Tersimpan")
    documents = get_documents()
    # if documents:
    #     for doc in documents:
    #         st.subheader(doc['title'])
    #         if st.button("Lihat Isi", key=f"view_{doc['id']}"):
    #             st.write(doc['content'])
    #         if st.button("Hapus", key=f"delete_{doc['id']}"):
    #             delete_document(doc['id'])
    #             st.experimental_rerun()
    
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
        st.write("Tidak ada dokumen yang tersimpan.")

    if st.button("Tambah Dokumen Baru"):
        st.session_state.page = 'input'



# Fungsi untuk menampilkan halaman view dokumen
def view_document_page():
    doc = st.session_state.get('current_doc', None)
    if doc:
        st.title(doc['title'])
        st.write(doc['content'])
        if st.button("Kembali"):
            st.session_state.page = 'home'



# Fungsi untuk menampilkan halaman input dokumen
def input_document_page():
    st.title("Tambah Dokumen Baru")

    title_input = st.text_input("Masukkan Judul Dokumen")
    uploaded_file = st.file_uploader("Unggah File PDF", type="pdf")
    
    if uploaded_file is not None:
        content = pdf_to_text(uploaded_file)
        title = title_input.strip() if title_input else extract_title(content)
        if st.button("Simpan"):
            preprocessed_content = preprocess_text(content)
            save_document(title, preprocessed_content)
            st.success("Dokumen berhasil disimpan!")


# Fungsi untuk menampilkan halaman clustering
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

    contents = [doc['content'] for doc in documents]
    
    min_clusters = 2
    max_clusters = min(2, num_samples - 1)  # Update the maximum clusters
    
    if num_samples == 2:
        num_clusters = 2
    else:
        num_clusters = st.number_input("Jumlah Cluster", min_value=min_clusters, max_value=max_clusters, value=min_clusters, step=1)
        
    if st.button("Hitung Keterkaitan"):
        # Membuat matriks TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(contents)
        
        # Hitung keterkaitan antar naskah
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

         # Tampilkan nilai keterkaitan antar naskah
        st.header("Nilai Keterkaitan Antar Naskah:")
        similarity_data = []
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                similarity = similarity_matrix[i, j] * 100  # Ubah menjadi persen
                similarity_data.append({
                    'Dokumen 1': documents[i]['title'],
                    'Dokumen 2': documents[j]['title'],
                    'Keterkaitan (%)': similarity   # Ubah nama kolom
                })
        similarity_df = pd.DataFrame(similarity_data)
        # st.table(similarity_df)
        styler = similarity_df.style.set_properties(**{'text-align': 'center'}).format({'Keterkaitan': "{:.4f}"}).hide(axis='index')
        st.write(styler.to_html(), unsafe_allow_html=True)
        
        st.divider()
        # Lakukan clustering
        silhouette_avg, labels = perform_clustering(similarity_matrix, num_clusters)
        st.write(f"Silhouette Coefficient: {silhouette_avg}")

        st.header("Hasil Clustering:")

        cluster_data = {f"Cluster {i + 1}": [] for i in range(num_clusters)}
        for cluster_id in range(num_clusters):
            cluster_docs = [documents[i]['title'] for i, label in enumerate(labels) if label == cluster_id]
            cluster_data[f"Cluster {cluster_id + 1}"] = cluster_docs
        
        cluster_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in cluster_data.items()])).fillna('')
        
        # Style the DataFrame to hide index
        cluster_styler = cluster_df.style.set_properties(**{'text-align': 'center'}).hide(axis='index')
        st.write(cluster_styler.to_html(), unsafe_allow_html=True)

# Fungsi untuk menampilkan navigasi
def navigation():
    st.sidebar.title("Navigasi")
    if st.sidebar.button("Home"):
        st.session_state.page = 'home'
    if st.sidebar.button("Input Dokumen"):
        st.session_state.page = 'input'
    if st.sidebar.button("Similiarity"):
        st.session_state.page = 'similiarity'
        

# Fungsi utama untuk routing halaman
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
