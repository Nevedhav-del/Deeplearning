from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Flatten, Dropout
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import re
import string
import pytest
import random
import itertools

nltk.download('stopwords', quiet=True)

# Load the text document
with open("D:\\code\\user.txt", 'r') as file:
    text = file.read()

# Split the text into paragraphs
paragraphs = text.split("\n\n")

# Tokenize the text into sentences
sentences = sent_tokenize(text)

# Create bag of words representation
vectorizer = CountVectorizer()
bow = vectorizer.fit_transform([text]).toarray()

# Define the model
model = Sequential()
model.add(Embedding(bow.shape[1], 100, input_length=bow.shape[0]))
model.add(LSTM(100))
model.add(Dropout(0.2))  # Add a dropout layer with dropout rate 0.2
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(bow.shape[1], activation='sigmoid'))
model.summary()

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(bow, bow, epochs=10)


# Define a list to store all the extracted keywords
all_keywords = []

# Load stopwords from a text file
stopwords_file = "D:\\code\\stopwords (1).txt"

with open(stopwords_file, 'r') as file:
    stopwords_text = file.read()

# Split the stopwords text into individual stopwords
custom_stopwords = stopwords_text.split('\n')

# Remove stopwords from the set
stop_words = set(stopwords.words('english'))
stop_words -= set(custom_stopwords)

# Load input words from a text file
with open("D:\\code\\input words.txt", 'r') as file:
    input_words = [line.strip() for line in file.readlines()]
    
# Load input words from a text file
with open("D:\\code\\output word.txt", 'r') as file:
    output_words = [line.strip() for line in file.readlines()]
    
# Load condition words from a text file
with open("D:\\code\\condition word.txt", 'r') as file:
    condition_words = [line.strip() for line in file.readlines()]

def check(sentences, words):
    matching_phrases = []
    matching_sentences = []

    for sentence in sentences:
        if any(word.lower() in sentence.lower() for word in words):
            matching_phrases.append(sentence)
            if sentence not in matching_sentences:
                matching_sentences.append(sentence)

    return matching_phrases, matching_sentences

# Iterate over the paragraphs (user stories)
for i, paragraph in enumerate(paragraphs):
    keyword_phrases = []

    # Tokenize the paragraph into sentences
    sentences = sent_tokenize(paragraph)

    # Extract keywords for each sentence
    for sentence in sentences:
        keywords = []
        words = word_tokenize(sentence)
        meaningful_keywords = [word for word in words if word.lower() not in stop_words and word.lower() not in string.punctuation]
        num_keywords = len(meaningful_keywords)
        step_size = 5
        
        # Ensure at least 3 words are present for a keyphrase
        if num_keywords >= 5:
            keyphrases = [" ".join(meaningful_keywords[i:i+7]) for i in range(3, num_keywords, step_size)]
            keyword_phrases.extend(keyphrases)
        else:
            keyword_phrases.append(" ".join(meaningful_keywords))

    # Store the extracted keywords for the current user story
    all_keywords.append(keyword_phrases)

    # Print user story and keywords only if the list is not empty
    if keyword_phrases:
        print("---------------------")
        print(f"User Story {i+1}:")
        print(paragraph)
        print("---------------------")
        print("Keywords:")
        print(keyword_phrases)
        # Check if any input words match with the keyword phrases
        matching_phrases, matching_sentences = check(keyword_phrases, input_words)
        if matching_phrases:
            print("---------------------")
            print("Input sentences:")
            print("---------------------")
            for i, phrase in enumerate(matching_phrases, start=1):
                print(f"{i}. {phrase}")
                # Perform part-of-speech tagging on the matched phrase
                tagged_words = nltk.pos_tag(word_tokenize(phrase))
                # Extract nouns from the tagged words
                nouns = [word for word, pos in tagged_words if pos.startswith('N')]
                # Print the extracted nouns in a straight line
                print(" Input words:", end=" ")
                print("[" + ", ".join(nouns) + "]")
            

        # Print output sentences
        matching_phrases_1, matching_sentences_1 = check(keyword_phrases, output_words)
        output_sentences = [sentence for sentence in matching_sentences_1 if sentence not in phrase]
        if output_sentences:
            print("---------------------")
            print("Output sentences:")
            print("---------------------")
            for i, sentence in enumerate(output_sentences, start=1):
                print(f"{i}. {sentence}")
                # Perform part-of-speech tagging on the matched phrase
                tagged_words = nltk.pos_tag(word_tokenize(sentence))
                # Extract nouns from the tagged words
                nouns = [word for word, pos in tagged_words if pos.startswith('N')]
                # Print the extracted nouns in a straight line
                print(" output words:", end=" ")
                print("[" + ", ".join(nouns) + "]")
                a = (nouns)
               
        
        # Check if any input words match with the keyword phrases
        matching_phrases, matching_sentences = check(keyword_phrases, condition_words)
        if matching_phrases:
            print("---------------------")
            print("condition sentences:")
            print("---------------------")
            phrases_list = [f"{i}. {phrase}" for i, phrase in enumerate(matching_phrases, start=1)]
            print("[" + ", ".join(phrases_list) + "]")
           
 
        
       

     
        

      
       
    

        

       