import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import fitz  
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

# case folding, tokenizing
def preprocess_text1(text):
    # Tokenisasi
    word_tokens = word_tokenize(text)
    
    # Menghapus token non-alfanumerik
    clean_tokens = [word for word in word_tokens if word.isalnum()]

    # Lowercase dan Menghapus stop words 
    filtered_text = [word.lower() for word in clean_tokens if word.lower()]

    cleaned_text = " ".join(filtered_text)

    return cleaned_text

# case folding, tokenizing, stemming
def preprocess_text2(text):

    # Tokenisasi
    word_tokens = word_tokenize(text)
    
    # Menghapus token non-alfanumerik
    clean_tokens = [word for word in word_tokens if word.isalnum()]

    # Lowercase dan Menghapus stop words 
    filtered_text = [word.lower() for word in clean_tokens if word.lower()]

    cleaned_text = " ".join(filtered_text)
    
    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_text]

    cleaned_text = " ".join(stemmed_tokens)

    return cleaned_text

# case folding, tokenizing, filtering
def preprocess_text3(text):
    stop_words = set(stopwords.words('indonesian'))

    # Tokenisasi
    word_tokens = word_tokenize(text)
    
    # Menghapus token non-alfanumerik
    clean_tokens = [word for word in word_tokens if word.isalnum()]

    # Lowercase dan Menghapus stop words 
    filtered_text = [word.lower() for word in clean_tokens if word.lower() not in stop_words]

    cleaned_text = " ".join(filtered_text)
    
    return cleaned_text

# case folding, tokenizing, filtering, stemming
def preprocess_text4(text):
    stop_words = set(stopwords.words('indonesian'))

    # Tokenisasi
    word_tokens = word_tokenize(text)
    
    # Menghapus token non-alfanumerik
    clean_tokens = [word for word in word_tokens if word.isalnum()]

    # Lowercase dan Menghapus stop words 
    filtered_text = [word.lower() for word in clean_tokens if word.lower() not in stop_words]

    cleaned_text = " ".join(filtered_text)
    
    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_text]

    cleaned_text = " ".join(stemmed_tokens)

    return cleaned_text

# part of speech, lemmatization, chunking
def preprocess_text5(text):
    # Tokenisasi
    word_tokens = word_tokenize(text)
    
    # POS Tagging
    pos_tagged = pos_tag(word_tokens)
    
    # Lematisasi
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for word, tag in pos_tagged:
        wordnet_pos = get_wordnet_pos(tag) or wordnet.NOUN
        lemmatized_tokens.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    
    # Lowercase ,Menghapus stop words dan token non-alfanumerik
    stop_words = set(stopwords.words('indonesian'))
    filtered_tokens = [word.lower() for word in lemmatized_tokens if word.isalnum() and word.lower() not in stop_words]
    
    # Chunking
    chunked = ne_chunk(pos_tag(filtered_tokens))
    
    terms = []
    for subtree in chunked:
        if hasattr(subtree, 'label'):
            terms.append(' '.join([token for token, pos in subtree.leaves()]))
        else:
            terms.append(subtree[0])
    
    # Menggabungkan istilah menjadi satu string
    processed_text = " ".join(terms)
    
    return processed_text

def get_wordnet_pos(treebank_tag):
    """
    Mengubah tag treebank menjadi tag bagian dari ucapan WordNet.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def pdf_to_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    
    text = ""
    for page in doc:
        text += page.get_text()
    
    # Tokenisasi
    word_tokens = word_tokenize(text)
    
    filtered_tokens = [word for word in word_tokens if word.isalnum() or '-' in word]
    
    cleaned_text = " ".join(filtered_tokens)

    return cleaned_text