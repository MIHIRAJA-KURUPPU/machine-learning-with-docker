import re

def cleanse_text(text):
    """
    Preprocess text by removing special characters, lowercasing,
    and removing common English stopwords.
    
    This version doesn't require NLTK resources.
    """
    if not isinstance(text, str) or not text:
        return ""
    
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Simple tokenization by splitting on whitespace
    tokens = text.split()
    
    # Simple list of English stopwords
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 
                 'while', 'of', 'to', 'in', 'for', 'on', 'by', 'with', 'about', 'against', 
                 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 
                 'from', 'up', 'down', 'is', 'are', 'were', 'was', 'am', 'been', 'being', 
                 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'i', 'me', 
                 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
                 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 
                 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 
                 'themselves', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 
                 'would', 'should', 'could', 'ought', 'im', 'youre', 'hes', 'shes', 'were', 
                 'theyre', 'ive', 'youve', 'weve', 'theyve', 'cant', 'dont', 'wont', 'not'}
    
    # Filter out stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Simple stemming (just cutting off common endings)
    stemmed_tokens = []
    for word in tokens:
        if len(word) > 3:
            if word.endswith('ing'):
                word = word[:-3]
            elif word.endswith('ed'):
                word = word[:-2]
            elif word.endswith('s') and not word.endswith('ss'):
                word = word[:-1]
            elif word.endswith('ly'):
                word = word[:-2]
            elif word.endswith('ment'):
                word = word[:-4]
        stemmed_tokens.append(word)
    
    return ' '.join(stemmed_tokens)
