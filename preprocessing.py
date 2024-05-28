import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import fitz  # PyMuPDF

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    stop_words = set(stopwords.words('indonesian'))
    word_tokens = word_tokenize(text)
    
    # Hilangkan tanda baca atau karakter yang tidak diperlukan
    clean_tokens = [word for word in word_tokens if word.isalnum()]
    
    # Hilangkan kata-kata yang merupakan stop words
    filtered_text = [word for word in clean_tokens if word.lower() not in stop_words]
    
    # Gabungkan kata-kata yang telah difilter kembali menjadi satu teks
    cleaned_text = " ".join(filtered_text)
    
    return cleaned_text

def pdf_to_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_title(text):
    lines = text.split('\n')
    title = lines[0].strip() if lines else "Judul Tidak Ditemukan"
    return title
