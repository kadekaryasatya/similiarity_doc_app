import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from database import get_documents, save_document, delete_document
from preprocessing import preprocess_text1, preprocess_text2, preprocess_text3, preprocess_text4, preprocess_text5, pdf_to_text, preprocess_documents,convert_df_to_pdf
from extract_item import extract_details, extract_title
from similarity import calculate_similarity_tfidf, calculate_similarity_word_count, calculate_similarity_bow, calculate_similarity_ngram, calculate_similarity_word_embedding, calculate_similarity_tfidf_ngram, calculate_similarity_word_count_tfidf, calculate_similarity_word_count_ngram, calculate_similarity_word_count_tfidf_ngram, calculate_similarity_bow_tfidf, calculate_similarity_bow_ngram, calculate_similarity_bow_tfidf_ngram, calculate_similarity_word_embedding_tfidf, calculate_similarity_word_embedding_ngram, calculate_similarity_word_embedding_tfidf_ngram, calculate_total_similarity
from clustering import perform_clustering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
import pdfkit
import tempfile
import base64

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
            # st.write(content)
            details_content = extract_details(content)
            save_document(title, details_content)
            st.success("Dokumen berhasil disimpan!")

def convert_html_to_pdf(html_content, output_path):
    path_to_wkhtmltopdf = 'C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe'
    config = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)
    pdfkit.from_string(html_content, output_path, configuration=config)


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
        num_cluster = 2

    st.markdown("<h5 style='text-align:center;'>", unsafe_allow_html=True)

    if st.button("Hitung Keterkaitan"):

        st.header("Nilai Keterkaitan Antar Naskah:")
        columns = ['pemrakarsa', 'level_peraturan', 'konten_penimbang', 'peraturan_terkait', 
                   'konten_peraturan', 'kategori_peraturan', 'topik_peraturan', 'struktur_peraturan']

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

        preprocess_titles = [
            "case folding, tokenizing",
            "case folding, tokenizing, stemming",
            "case folding, tokenizing, filtering",
            "case folding, tokenizing, filtering, stemming",
            "part of speech, lemmatization, chunking"
        ]
   
        preprocess_functions = [
            preprocess_text1, 
            preprocess_text2,
            preprocess_text3,
            preprocess_text4,
            preprocess_text5
        ]
        
        tab_names = list(similarity_methods.keys())
        tabs = st.tabs(tab_names)
        
        for i, method_name in enumerate(tab_names):
            with tabs[i]:
                st.subheader(method_name)

                similarity_data = {preprocess_titles[k]: {} for k in range(len(preprocess_functions))}
                total_similarity_data = []

                preprocessed_documents = []
                for preprocess_func in preprocess_functions:
                    preprocessed_documents.append(preprocess_documents(documents, preprocess_func))

                for k, preprocessed_docs in enumerate(preprocessed_documents):
                    similarity_data_method, total_similarity_data_method = similarity_methods[method_name](preprocessed_docs)
                    similarity_data[preprocess_titles[k]] = similarity_data_method
                    total_similarity_data.append(total_similarity_data_method)

                all_html_content = ""

                for i in range(num_samples):
                    for j in range(i + 1, num_samples):
                        doc1 = documents[i]['title']
                        doc2 = documents[j]['title']

                        st.write(f"Document 1 : {doc1}")
                        st.write(f"Document 2 : {doc2}")

                        similarity_data_per_doc = {preprocess_titles[k]: {} for k in range(len(preprocess_functions))}

                        index = i * num_samples + j - ((i + 1) * (i + 2)) // 2
                        valid_index = True
                        for k in range(len(preprocess_functions)):
                            try:
                                if index >= len(similarity_data[preprocess_titles[k]][columns[0]]):
                                    st.write(f"Indeks di luar jangkauan untuk pra-pemrosesan {k+1}, indeks {index}")
                                    valid_index = False
                                    break
                            except KeyError:
                                st.write(f"KeyError: Pra-pemrosesan {k+1} tidak memiliki data untuk kolom {columns[0]}")
                                valid_index = False
                                break

                        if valid_index:
                            for col in columns:
                                for k in range(len(preprocess_functions)):
                                    similarity_data_per_doc[preprocess_titles[k]][col] = similarity_data[preprocess_titles[k]][col][index]['Keterkaitan (%)']

                            df = pd.DataFrame(similarity_data_per_doc)

                            if not df.empty:
                                df.index = columns

                                total_similarity_row = pd.Series({
                                    preprocess_titles[k]: total_similarity_data[k][i][j] / len(columns)
                                    for k in range(len(preprocess_functions))
                                }, name='Total Keterkaitan (%)')

                                df = pd.concat([df, total_similarity_row.to_frame().T])

                                df_styled = df.style.set_properties(**{'text-align': 'center'}).format("{:.2f}%").set_table_styles(
                                    [{'selector': 'table, th, td', 'props': [('border', '1px solid black')]}])

                                st.table(df_styled)
                                st.divider()

                                df_styled_html = df_styled.to_html()
                                all_html_content += f"<h2>{method_name}</h2><h3>Document 1: {doc1}</h3><h3>Document 2: {doc2}</h3>" + df_styled_html

                            else:
                                st.write("DataFrame kosong, tidak bisa mengatur indeks")
                        else:
                            st.write("Indeks di luar jangkauan, tidak bisa membuat DataFrame untuk pasangan dokumen ini.")

                if num_samples == 2:
                    st.warning("Clustering tidak dapat dilakukan dengan hanya dua dokumen. Silakan tambahkan lebih banyak dokumen.")
                else:
                    combined_contents = [
                        doc['pemrakarsa'] + " " + doc['level_peraturan'] + " " + doc['konten_penimbang'] + " " + 
                        doc['peraturan_terkait'] + " " + doc['konten_peraturan'] + " " + doc['kategori_peraturan'] + " " +
                        doc['topik_peraturan'] + " " + doc['struktur_peraturan']
                        for doc in documents
                    ]

                    if method_name == "Word Embedding":
                        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                        embeddings = np.array([model.encode(content) for content in combined_contents])
                        similarity_matrix = cosine_similarity(embeddings)
                    else:
                        if method_name == "TF-IDF":
                            vectorizer = TfidfVectorizer()
                        elif method_name == "Word Count":
                            vectorizer = CountVectorizer()
                        elif method_name == "N-Gram" or method_name == "TF-IDF + N-Gram":
                            vectorizer = CountVectorizer(ngram_range=(1, n_value))
                        elif method_name.startswith("Bag of Words"):
                            if "TF-IDF" in method_name:
                                vectorizer = TfidfVectorizer(use_idf=True)
                            elif "N-Gram" in method_name:
                                vectorizer = CountVectorizer(ngram_range=(1, n_value))
                            else:
                                vectorizer = CountVectorizer()

                        tfidf_matrix = vectorizer.fit_transform(combined_contents)
                        similarity_matrix = cosine_similarity(tfidf_matrix)

                    silhouette_avg, labels = perform_clustering(similarity_matrix, 2)  # Default to 2 clusters
                    
                    st.subheader("Clustering")

                    st.write(f"Silhouette Coefficient: {silhouette_avg}")

                    cluster_data = {f"Cluster {i + 1}": [] for i in range(2)}
                    for cluster_id in range(2):
                        cluster_docs = [documents[i]['title'] for i, label in enumerate(labels) if label == cluster_id]
                        cluster_data[f"Cluster {cluster_id + 1}"] = cluster_docs
                    
                    cluster_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in cluster_data.items()])).fillna('')
                    cluster_styler = cluster_df.style.set_properties(**{'text-align': 'center'}).hide(axis='index')
                    st.write(cluster_styler.to_html(), unsafe_allow_html=True)

                if all_html_content:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        convert_html_to_pdf(all_html_content, tmp_file.name)
                        st.markdown("<br>", unsafe_allow_html=True)  # Space above the download button
                        
                    st.markdown("<div style='text-align:right'>", unsafe_allow_html=True)
                    st.markdown(f'<a href="data:application/pdf;base64,{base64.b64encode(open(tmp_file.name, "rb").read()).decode()}" target="_blank" download="similarity_{method_name}.pdf">Download PDF</a>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

       
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
