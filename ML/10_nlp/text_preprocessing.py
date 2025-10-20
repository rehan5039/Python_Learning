"""
Text Preprocessing Implementation
===============================

This module demonstrates various text preprocessing techniques essential for NLP tasks.
It covers tokenization, cleaning, normalization, and other preprocessing steps.

Key Concepts:
- Text Cleaning and Normalization
- Tokenization Techniques
- Stop Word Removal
- Stemming and Lemmatization
- Handling Special Cases
"""

import re
import string
from collections import Counter


class TextPreprocessor:
    """
    A comprehensive text preprocessor for NLP tasks.
    
    Parameters:
    -----------
    lowercase : bool, default=True
        Whether to convert text to lowercase
    remove_punctuation : bool, default=True
        Whether to remove punctuation
    remove_numbers : bool, default=False
        Whether to remove numbers
    remove_whitespace : bool, default=True
        Whether to remove extra whitespace
    expand_contractions : bool, default=True
        Whether to expand contractions
    remove_stopwords : bool, default=False
        Whether to remove stop words
    stopword_list : list, optional
        Custom list of stop words
    stem_words : bool, default=False
        Whether to apply stemming
    lemmatize_words : bool, default=False
        Whether to apply lemmatization
    """
    
    def __init__(self, lowercase=True, remove_punctuation=True, remove_numbers=False,
                 remove_whitespace=True, expand_contractions=True, remove_stopwords=False,
                 stopword_list=None, stem_words=False, lemmatize_words=False):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_whitespace = remove_whitespace
        self.expand_contractions = expand_contractions
        self.remove_stopwords = remove_stopwords
        self.stem_words = stem_words
        self.lemmatize_words = lemmatize_words
        
        # Default stop words (simplified)
        self.default_stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
            'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above',
            'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
        }
        
        # Use provided stopword list or default
        self.stopwords = set(stopword_list) if stopword_list else self.default_stopwords
        
        # Contraction mappings
        self.contractions = {
            "ain't": "am not", "aren't": "are not", "can't": "cannot",
            "could've": "could have", "couldn't": "could not",
            "didn't": "did not", "doesn't": "does not", "don't": "do not",
            "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
            "he'd": "he had", "he'll": "he will", "he's": "he is",
            "i'd": "i had", "i'll": "i will", "i'm": "i am", "i've": "i have",
            "isn't": "is not", "it'd": "it had", "it'll": "it will",
            "it's": "it is", "let's": "let us", "might've": "might have",
            "mightn't": "might not", "must've": "must have", "mustn't": "must not",
            "shan't": "shall not", "she'd": "she had", "she'll": "she will",
            "she's": "she is", "should've": "should have", "shouldn't": "should not",
            "that's": "that is", "there's": "there is", "they'd": "they had",
            "they'll": "they will", "they're": "they are", "they've": "they have",
            "wasn't": "was not", "we'd": "we had", "we'll": "we will",
            "we're": "we are", "we've": "we have", "weren't": "were not",
            "what'll": "what will", "what're": "what are", "what's": "what is",
            "what've": "what have", "where's": "where is", "who'll": "who will",
            "who's": "who is", "who've": "who have", "won't": "will not",
            "would've": "would have", "wouldn't": "would not", "you'd": "you had",
            "you'll": "you will", "you're": "you are", "you've": "you have"
        }
    
    def expand_contractions(self, text):
        """
        Expand contractions in text.
        
        Parameters:
        -----------
        text : str
            Input text
            
        Returns:
        --------
        text : str
            Text with expanded contractions
        """
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text
    
    def remove_punctuation(self, text):
        """
        Remove punctuation from text.
        
        Parameters:
        -----------
        text : str
            Input text
            
        Returns:
        --------
        text : str
            Text without punctuation
        """
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def remove_numbers(self, text):
        """
        Remove numbers from text.
        
        Parameters:
        -----------
        text : str
            Input text
            
        Returns:
        --------
        text : str
            Text without numbers
        """
        return re.sub(r'\d+', '', text)
    
    def remove_extra_whitespace(self, text):
        """
        Remove extra whitespace from text.
        
        Parameters:
        -----------
        text : str
            Input text
            
        Returns:
        --------
        text : str
            Text with normalized whitespace
        """
        return ' '.join(text.split())
    
    def tokenize(self, text):
        """
        Tokenize text into words.
        
        Parameters:
        -----------
        text : str
            Input text
            
        Returns:
        --------
        tokens : list
            List of tokens
        """
        return text.split()
    
    def remove_stopwords(self, tokens):
        """
        Remove stop words from tokens.
        
        Parameters:
        -----------
        tokens : list
            List of tokens
            
        Returns:
        --------
        filtered_tokens : list
            Tokens without stop words
        """
        return [token for token in tokens if token.lower() not in self.stopwords]
    
    def simple_stem(self, word):
        """
        Simple stemming implementation.
        
        Parameters:
        -----------
        word : str
            Input word
            
        Returns:
        --------
        stemmed_word : str
            Stemmed word
        """
        # Very simple stemming rules
        if word.endswith('ing'):
            return word[:-3]
        elif word.endswith('ed'):
            return word[:-2]
        elif word.endswith('ly'):
            return word[:-2]
        elif word.endswith('s') and len(word) > 3:
            return word[:-1]
        return word
    
    def stem_tokens(self, tokens):
        """
        Apply stemming to tokens.
        
        Parameters:
        -----------
        tokens : list
            List of tokens
            
        Returns:
        --------
        stemmed_tokens : list
            Stemmed tokens
        """
        return [self.simple_stem(token) for token in tokens]
    
    def preprocess(self, text):
        """
        Apply all preprocessing steps to text.
        
        Parameters:
        -----------
        text : str
            Input text
            
        Returns:
        --------
        processed_text : str or list
            Preprocessed text (string if returning text, list if returning tokens)
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Store original text for reference
        original_text = text
        
        # Apply preprocessing steps
        if self.lowercase:
            text = text.lower()
        
        if self.expand_contractions:
            text = self.expand_contractions(text)
        
        if self.remove_punctuation:
            text = self.remove_punctuation(text)
        
        if self.remove_numbers:
            text = self.remove_numbers(text)
        
        if self.remove_whitespace:
            text = self.remove_extra_whitespace(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stop words if specified
        if self.remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Apply stemming if specified
        if self.stem_words:
            tokens = self.stem_tokens(tokens)
        
        # Return tokens or joined text
        return tokens if self.stem_words or self.remove_stopwords else ' '.join(tokens)
    
    def preprocess_batch(self, texts):
        """
        Preprocess a batch of texts.
        
        Parameters:
        -----------
        texts : list
            List of texts
            
        Returns:
        --------
        processed_texts : list
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]


class AdvancedTextPreprocessor:
    """
    Advanced text preprocessing with additional features.
    """
    
    def __init__(self):
        pass
    
    def detect_language(self, text):
        """
        Simple language detection based on character frequency.
        
        Parameters:
        -----------
        text : str
            Input text
            
        Returns:
        --------
        language : str
            Detected language ('english' or 'unknown')
        """
        # Very simple language detection
        english_chars = sum(1 for c in text.lower() if c in 'abcdefghijklmnopqrstuvwxyz')
        total_chars = len(text)
        if total_chars > 0 and english_chars / total_chars > 0.8:
            return 'english'
        return 'unknown'
    
    def handle_special_characters(self, text):
        """
        Handle special characters and encoding issues.
        
        Parameters:
        -----------
        text : str
            Input text
            
        Returns:
        --------
        text : str
            Cleaned text
        """
        # Remove or replace common special characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
        text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)  # Remove control characters
        return text
    
    def normalize_unicode(self, text):
        """
        Normalize unicode characters.
        
        Parameters:
        -----------
        text : str
            Input text
            
        Returns:
        --------
        text : str
            Normalized text
        """
        # Replace common unicode characters
        replacements = {
            '"': '"', '"': '"', ''': "'", ''': "'",
            '–': '-', '—': '-', '…': '...', '€': 'EUR'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text


# Example usage and demonstration
if __name__ == "__main__":
    # Sample texts for demonstration
    sample_texts = [
        "Hello, world! This is a sample text with contractions like don't and can't.",
        "The quick brown fox jumps over the lazy dog. It's a beautiful day!",
        "I'm going to the store to buy some groceries. Won't you join me?",
        "Machine learning is amazing! It can do so much for us."
    ]
    
    print("Text Preprocessing Demonstration")
    print("=" * 50)
    
    # Basic preprocessing
    print("\n1. Basic Preprocessing:")
    preprocessor = TextPreprocessor()
    
    for i, text in enumerate(sample_texts[:2]):
        processed = preprocessor.preprocess(text)
        print(f"Original: {text}")
        print(f"Processed: {processed}")
        print()
    
    # Advanced preprocessing with stop word removal
    print("\n2. Preprocessing with Stop Word Removal:")
    preprocessor_stopwords = TextPreprocessor(remove_stopwords=True)
    
    text = "The quick brown fox jumps over the lazy dog. It's a beautiful day!"
    processed = preprocessor_stopwords.preprocess(text)
    print(f"Original: {text}")
    print(f"Processed: {processed}")
    print(f"Tokens: {processed}")  # When remove_stopwords=True, returns list
    
    # Preprocessing with stemming
    print("\n3. Preprocessing with Stemming:")
    preprocessor_stem = TextPreprocessor(remove_stopwords=True, stem_words=True)
    
    text = "The children are playing happily in the beautiful gardens"
    processed = preprocessor_stem.preprocess(text)
    print(f"Original: {text}")
    print(f"Stemmed tokens: {processed}")
    
    # Batch preprocessing
    print("\n4. Batch Preprocessing:")
    processed_batch = preprocessor.preprocess_batch(sample_texts)
    for i, (original, processed) in enumerate(zip(sample_texts, processed_batch)):
        print(f"{i+1}. {original}")
        print(f"   -> {processed}")
        print()
    
    # Advanced preprocessing
    print("\n5. Advanced Preprocessing:")
    advanced_preprocessor = AdvancedTextPreprocessor()
    
    text_with_unicode = "Café résumé naïve façade"
    print(f"Original with unicode: {text_with_unicode}")
    normalized = advanced_preprocessor.normalize_unicode(text_with_unicode)
    print(f"Normalized: {normalized}")
    
    text_with_special = "Hello\x00World\x01Test"
    print(f"Text with special chars: {repr(text_with_special)}")
    cleaned = advanced_preprocessor.handle_special_characters(text_with_special)
    print(f"Cleaned: {repr(cleaned)}")
    
    # Language detection
    english_text = "This is an English sentence."
    french_text = "Ceci est une phrase française."
    
    print(f"\nLanguage detection:")
    print(f"'{english_text}' -> {advanced_preprocessor.detect_language(english_text)}")
    print(f"'{french_text}' -> {advanced_preprocessor.detect_language(french_text)}")
    
    # Performance comparison
    print("\n" + "="*50)
    print("Preprocessing Pipeline Comparison")
    print("="*50)
    
    import time
    
    # Large text for performance testing
    large_text = " ".join(sample_texts * 100)  # Repeat sample texts
    
    # Time basic preprocessing
    start_time = time.time()
    basic_preprocessor = TextPreprocessor()
    basic_result = basic_preprocessor.preprocess(large_text)
    basic_time = time.time() - start_time
    
    # Time advanced preprocessing
    start_time = time.time()
    advanced_result = advanced_preprocessor.handle_special_characters(
        advanced_preprocessor.normalize_unicode(large_text.lower())
    )
    advanced_time = time.time() - start_time
    
    print(f"Basic preprocessing time: {basic_time:.4f} seconds")
    print(f"Advanced preprocessing time: {advanced_time:.4f} seconds")
    print(f"Text length: {len(large_text)} characters")
    
    # Common preprocessing challenges
    print("\n" + "="*50)
    print("Common Preprocessing Challenges")
    print("="*50)
    print("1. Handling Domain-Specific Text:")
    print("   - Medical texts, legal documents, social media slang")
    print("   - Specialized vocabulary and abbreviations")
    
    print("\n2. Dealing with Noise:")
    print("   - HTML tags, URLs, email addresses")
    print("   - Emojis, hashtags, mentions")
    
    print("\n3. Multilingual Text:")
    print("   - Mixed language content")
    print("   - Language-specific preprocessing needs")
    
    print("\n4. Context Preservation:")
    print("   - Maintaining semantic meaning")
    print("   - Handling negations and modifiers")
    
    # Best practices
    print("\n" + "="*50)
    print("Best Practices for Text Preprocessing")
    print("="*50)
    print("1. Understand Your Data:")
    print("   - Analyze text characteristics before preprocessing")
    print("   - Identify domain-specific requirements")
    
    print("\n2. Preserve Important Information:")
    print("   - Don't over-clean and lose meaning")
    print("   - Keep context where possible")
    
    print("\n3. Consistency:")
    print("   - Apply same preprocessing to training and test data")
    print("   - Document all preprocessing steps")
    
    print("\n4. Evaluation:")
    print("   - Test impact of different preprocessing steps")
    print("   - Validate that preprocessing improves model performance")
    
    print("\n5. Efficiency:")
    print("   - Optimize for large-scale processing")
    print("   - Consider parallel processing for large datasets")