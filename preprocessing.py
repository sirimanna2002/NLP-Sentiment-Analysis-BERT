import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords
nltk.download('stopwords')

def clean_text(text):
    """
    clean Text 
    """
    #remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # remove @mention and #hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # remove special letters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # All Lowercase
    text = text.lower()
    
    # Remove Stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    
    # Stemming
    stemmer = PorterStemmer()
    words = text.split()
    words = [stemmer.stem(word) for word in words]
    text = ' '.join(words)
    
    return text