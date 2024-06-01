import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import fitz  

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    stop_words = set(stopwords.words('indonesian'))
    word_tokens = word_tokenize(text)
    
    clean_tokens = [word for word in word_tokens if word.isalnum()]
    
    filtered_text = [word for word in clean_tokens if word.lower() not in stop_words]
    
    cleaned_text = " ".join(filtered_text)
    
    return cleaned_text

def extract_details(text):
    pemrakarsa = re.findall(r'(?:MENTERI|LEMBAGA NON KEMENTERIAN) [A-Z ]+', text)
    level_peraturan = re.findall(r'(?:PERATURAN|UNDANG-UNDANG|KEPUTUSAN) [A-Z ]+', text)
    konten_penimbang = re.search(r'Menimbang\s*:\s*(.*?)\s*Mengigat\s*:', text, re.DOTALL)
    peraturan_terkait = re.findall(r'Nomor\s+\d+\s+\d{4}', text)
    konten_peraturan = re.search(r'Menetapkan\s*:\s*(.*?)\s*MEMUTUSKAN', text, re.DOTALL)
    kategori_peraturan = re.findall(r'KATEGORI\s*:\s*(.*?)\s*\n', text)
    topik_peraturan = re.findall(r'TOPIK\s*:\s*(.*?)\s*\n', text)
    struktur_peraturan = re.findall(r'Pasal\s+\d+', text)

    pemrakarsa = pemrakarsa if pemrakarsa else ["Pemrakarsa tidak ditemukan"]
    level_peraturan = level_peraturan if level_peraturan else ["Level Peraturan tidak ditemukan"]
    konten_penimbang = konten_penimbang.group(1).strip() if konten_penimbang else "Penimbang tidak ditemukan"
    peraturan_terkait = peraturan_terkait if peraturan_terkait else ["Peraturan Terkait tidak ditemukan"]
    konten_peraturan = konten_peraturan.group(1).strip() if konten_peraturan else "Konten peraturan tidak ditemukan"
    kategori_peraturan = kategori_peraturan if kategori_peraturan else ["Kategori Peraturan tidak ditemukan"]
    topik_peraturan = topik_peraturan if topik_peraturan else ["Topik Peraturan tidak ditemukan"]
    struktur_peraturan = struktur_peraturan if struktur_peraturan else ["Struktur Peraturan tidak ditemukan"]
    
    return {
        "Pemrakarsa": ", ".join(pemrakarsa),
        "Level Peraturan": ", ".join(level_peraturan),
        "Konten Penimbang": konten_penimbang,
        "Peraturan Terkait": ", ".join(peraturan_terkait),
        "Konten Peraturan": konten_peraturan,
        "Kategori Peraturan": ", ".join(kategori_peraturan),
        "Topik Peraturan": ", ".join(topik_peraturan),
        "Struktur Peraturan": ", ".join(struktur_peraturan)
    }

def pdf_to_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

import re

import re

def extract_title(text):
    title_pattern = r'(?:Peraturan|Undang-Undang)\s+[A-Za-z0-9\s\-/,.()]+'
    title_match = re.search(title_pattern, text)
    
    if title_match:
        title = title_match.group().strip()
        title_words = title.split()
        if len(title_words) > 8:
            title = ' '.join(title_words[:8]) 
    else:
        title = "Judul Tidak Ditemukan"
    
    return title
