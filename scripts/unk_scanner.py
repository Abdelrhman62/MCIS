from sklearn.feature_extraction.text import CountVectorizer

def scan_unknown_tokens(train_texts, test_texts):
    """
    Identifies words in the test set that do not exist in the training set.
    """
    print("Scanning for Out-of-Vocabulary (OOV) / UNK tokens...")
    
    # Extract vocabulary from training set
    vectorizer_train = CountVectorizer(lowercase=True).fit(train_texts)
    train_vocab = set(vectorizer_train.vocabulary_.keys())
    
    # Extract vocabulary from test set
    vectorizer_test = CountVectorizer(lowercase=True).fit(test_texts)
    test_vocab = set(vectorizer_test.vocabulary_.keys())
    
    # Find difference
    unk_tokens = test_vocab - train_vocab
    
    print(f"Total Unique Tokens in Train: {len(train_vocab)}")
    print(f"Total Unique Tokens in Test: {len(test_vocab)}")
    print(f"Total UNK Tokens (in test but not train): {len(unk_tokens)}")
    
    # Calculate percentage of test vocabulary that is unknown
    unk_percentage = (len(unk_tokens) / len(test_vocab)) * 100
    print(f"Percentage of Test Vocab that is UNK: {unk_percentage:.2f}%\n")
    
    return list(unk_tokens)

# Example usage:
# unk_words = scan_unknown_tokens(train_df['text'], val_df['text'])
# print("Sample UNK words:", unk_words[:10])