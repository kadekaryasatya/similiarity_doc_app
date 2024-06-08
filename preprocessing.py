import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import fitz  

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # stop_words = set(stopwords.words('indonesian'))
    word_tokens = word_tokenize(text)
    
    clean_tokens = [word for word in word_tokens if word.isalnum()]

    filtered_text = [word.lower() for word in clean_tokens if word.lower()]

    cleaned_text = " ".join(filtered_text)
    
    return cleaned_text

def extract_details(text):
    pemrakarsa_pattern = r'\b(?:Presiden|Wakil Presiden|Lembaga Pemerintah Non-Kementerian|Dewan Perwakilan Rakyat(?: \(DPR\))?|Majelis Permusyawaratan Rakyat(?: \(MPR\))?|Mahkamah Agung(?: \(MA\))?|Mahkamah Konstitusi(?: \(MK\))?|Badan Pemeriksa Keuangan(?: \(BPK\))?|Bank Indonesia(?: \(BI\))?|Otoritas Jasa Keuangan(?: \(OJK\))?|Badan Pengawas Pemilihan Umum(?: \(Bawaslu\))?|Komisi Pemilihan Umum(?: \(KPU\))?|Komisi Pemberantasan Korupsi(?: \(KPK\))?|Gubernur|Dewan Perwakilan Rakyat Daerah Provinsi(?: \(DPRD Provinsi\))?|Bupati|Walikota|Dewan Perwakilan Rakyat Daerah Kabupaten/Kota(?: \(DPRD Kabupaten/Kota\))?|Kepala Desa|Lurah|Badan Permusyawaratan Desa(?: \(BPD\))?|Komisi Yudisial(?: \(KY\))?|Lembaga Negara Independen|Komnas HAM|Komisi Informasi|Menteri Dalam Negeri|Menteri Luar Negeri|Menteri Pertahanan|Menteri Hukum dan Hak Asasi Manusia|Menteri Keuangan|Menteri Pendidikan dan Kebudayaan|Menteri Riset dan Teknologi|Menteri Agama|Menteri Ketenagakerjaan|Menteri Energi dan Sumber Daya Mineral|Menteri Perindustrian|Menteri Perdagangan|Menteri Pertanian|Menteri Lingkungan Hidup dan Kehutanan|Menteri Kelautan dan Perikanan|Menteri Desa, Pembangunan Daerah Tertinggal, dan Transmigrasi|Menteri Perencanaan Pembangunan Nasional|Menteri Pendayagunaan Aparatur Negara dan Reformasi Birokrasi|Menteri Pekerjaan Umum dan Perumahan Rakyat|Menteri Kesehatan|Menteri Sosial|Menteri Pariwisata|Menteri Komunikasi dan Informatika|Menteri Koordinator Bidang Politik, Hukum, dan Keamanan|Menteri Koordinator Bidang Perekonomian|Menteri Koordinator Bidang Pembangunan Manusia dan Kebudayaan|Menteri Koordinator Bidang Kemaritiman dan Investasi|Menteri Badan Usaha Milik Negara|Menteri Koperasi dan Usaha Kecil dan Menengah|Menteri Pemuda dan Olahraga|Menteri Perhubungan|Menteri Agraria dan Tata Ruang/Badan Pertanahan Nasional|Menteri Perumahan Rakyat|Menteri Percepatan Pembangunan Daerah Tertinggal|Menteri Perencanaan Pembangunan Nasional/Bappenas|Menteri Sekretaris Negara|Menteri Sekretariat Kabinet|Menteri(?: [A-Za-z]+)*|Kementerian(?: [A-Za-z]+)*)\b'
    pemrakarsa_match = re.search(pemrakarsa_pattern, text, re.IGNORECASE)
    pemrakarsa = pemrakarsa_match.group(0).strip() if pemrakarsa_match else "Pemrakarsa tidak ditemukan"
    
    level_peraturan_pattern = r'\b(?:Undang-Undang Dasar 1945|Ketetapan Majelis Permusyawaratan Rakyat|Undang-Undang|Peraturan Pemerintah Pengganti Undang-Undang|Peraturan Pemerintah|Keputusan Presiden|Peraturan Menteri|Peraturan Gubernur|Peraturan Bupati)\b'
    level_peraturan_match = re.search(level_peraturan_pattern, text, re.IGNORECASE)
    if level_peraturan_match:
        level_peraturan = level_peraturan_match.group(0).strip()
        if level_peraturan.lower() == "peraturan gubernur":
            level_peraturan = "peraturan daerah provinsi"
        elif level_peraturan.lower() == "peraturan bupati":
            level_peraturan = "peraturan daerah kabupaten/Kota"
    else:
        level_peraturan = "Level Peraturan tidak ditemukan"

    penimbang_pattern = r'Menimbang\s*(.*?)(?=Mengingat|$)'
    penimbang_match = re.search(penimbang_pattern, text, re.DOTALL| re.IGNORECASE)
    konten_penimbang = penimbang_match.group(1).strip() if penimbang_match else "Penimbang tidak ditemukan"

    peraturan_terkait_pattern = r'Mengingat\s*(.*?)(?=Memutuskan|$)'
    peraturan_terkait_match = re.search(peraturan_terkait_pattern, text, re.DOTALL| re.IGNORECASE)
    peraturan_terkait = peraturan_terkait_match.group(1).strip() if peraturan_terkait_match else "Peraturan Terkait tidak ditemukan"
    
    konten_peraturan_pattern = r'Memutuskan\s*(.*?)$'
    konten_peraturan_match = re.search(konten_peraturan_pattern, text, re.IGNORECASE | re.DOTALL)
    konten_peraturan = (konten_peraturan_match.group(1)[:255]).strip() if konten_peraturan_match else "Konten peraturan tidak ditemukan"

    if level_peraturan == "Level Peraturan tidak ditemukan":
       kategori_peraturan = "peraturan biasa"
    else:
       kategori_peraturan = "peraturan perundang-undangan"
    

    topik_kata_kunci = {
        "pendidikan": ["sekolah", "kurikulum", "pengajaran", "siswa", "guru", "pendidikan tinggi", "universitas", "beasiswa"],
        "kesehatan": ["rumah sakit", "dokter", "obat-obatan", "penyakit menular", "vaksinasi", "pelayanan kesehatan", "asuransi kesehatan"],
        "lingkungan hidup": ["polusi udara", "polusi air", "limbah", "konservasi", "hutan", "energi terbarukan", "pengelolaan sampah"],
        "pertanian": ["tanaman", "peternakan", "lahan pertanian", "irigasi", "pupuk", "pestisida", "perlindungan tanaman"],
        "ketenagakerjaan": ["tenaga kerja", "upah", "keamanan kerja", "hak-hak pekerja", "serikat pekerja", "perlindungan sosial"],
        "perpajakan": ["pajak penghasilan", "pajak pertambahan nilai", "tarif pajak", "penghindaran pajak", "insentif pajak"],
        "investasi": ["pasar modal", "saham", "obligasi", "regulasi investasi", "perlindungan investor", "modal ventura"],
        "transportasi": ["jalan", "transportasi umum", "kendaraan bermotor", "bandara", "pelabuhan", "transportasi massal"],
        "keuangan": ["perbankan", "asuransi", "pasar keuangan", "regulasi keuangan", "inflasi", "suku bunga", "kebijakan moneter"]
    }

    topik_peraturan_text = None
    for topik, kata_kunci in topik_kata_kunci.items():
        for kata in kata_kunci:
            if kata in text.lower():
                topik_peraturan_text = topik
                break
        if topik_peraturan_text:
            break

    if topik_peraturan_text is None:
        topik_peraturan_text = "Topik tidak ditemukan"
    
    judul =  extract_title(text)

    struktur_peraturan_pattern = {
    "Judul": judul,
    "Pembukaan": r'(Dengan Rahmat Tuhan Yang Maha Esa.*?)(?=Menimbang)',
    "Batang Tubuh": konten_penimbang + "\n\n" + peraturan_terkait + "\n\n" + konten_peraturan,
    "Penjelasan": r'(Penjelasan atas.*?)(?=Ditetapkan Di)',
    "Penutup": r'(?:Ditetapkan Di).*?(?=Lampiran|$)',
    "Lampiran": r'(Lampiran.*?$)'
}

    struktur_peraturan = {}

    for bagian, pola in struktur_peraturan_pattern.items():
        if bagian == "Batang Tubuh":
            struktur_peraturan[bagian] = pola
        else:
            matches = re.findall(pola, text, re.DOTALL | re.IGNORECASE)
            if matches:
                if bagian == "Lampiran":
                    struktur_peraturan[bagian] = matches[-1].strip()
                else:
                    struktur_peraturan[bagian] = matches[0].strip()
            else:
                struktur_peraturan[bagian] = "Tidak ditemukan"

    struktur_peraturan_mix = "\n\n".join([f"{bagian}: {konten}" for bagian, konten in struktur_peraturan.items()])

    return {
        "Pemrakarsa": pemrakarsa,
        "Level Peraturan": level_peraturan,
        "Konten Penimbang": konten_penimbang,
        "Peraturan Terkait": peraturan_terkait,
        "Konten Peraturan": konten_peraturan,
        "Kategori Peraturan": kategori_peraturan,
        "Topik Peraturan": topik_peraturan_text,
        "Struktur Peraturan": struktur_peraturan_mix
    }
    
def pdf_to_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_title(text):
    title_pattern = r'\b(Peraturan|Undang-Undang)\b\s+[A-Za-z0-9\s\-/,.()]+'
    title_match = re.search(title_pattern, text, re.IGNORECASE)
    
    if title_match:
        title = title_match.group().strip()
        
        title_words = title.split()
        if len(title_words) > 8:
            title = ' '.join(title_words[:8]) 
    else:
        title = "Judul Tidak Ditemukan"
    
    return title


