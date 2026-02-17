# --- Extractive Text Summarizer using spaCy ---
# This script summarizes a long piece of text by identifying the most important sentences.
# It works by calculating the frequency of each word and then scoring each sentence
# based on the importance of the words it contains.

# Step 1: Install necessary libraries
# In your terminal, run the following commands:
# pip install spacy
# python -m spacy download en_core_web_sm

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

def extractive_summarizer(text, num_sentences=3):
    """
    Summarizes a text by extracting the most important sentences.
    :param text: The input text to summarize.
    :param num_sentences: The number of sentences the final summary should have.
    :return: A string containing the summarized text.
    """
    # Load the spaCy English model
    # 'en_core_web_sm' is a small English model, perfect for this task.
    nlp = spacy.load('en_core_web_sm')

    # Process the text with the spaCy pipeline
    doc = nlp(text)

    # --- Word Frequency Calculation ---
    # We will calculate the frequency of each word that isn't a stop word or punctuation.
    stopwords = list(STOP_WORDS)
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1

    # Normalize the word frequencies
    # We divide each frequency by the maximum frequency to get a score between 0 and 1.
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / max_frequency)

    # --- Sentence Scoring ---
    # We score each sentence based on the frequencies of the words it contains.
    sentence_tokens = [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]

    # --- Select Top Sentences ---
    # We use nlargest to get the top 'num_sentences' sentences with the highest scores.
    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    
    # Convert the selected sentence tokens back to a single string
    final_summary = [sent.text for sent in summary_sentences]
    summary = ' '.join(final_summary)

    return summary

# --- Example Usage ---
if __name__ == '__main__':
    # You can replace this text with any long article or text you want to summarize.
    long_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. 
    Colloquially, the term "artificial intelligence" is often used to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving".
    As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. 
    A quip in Tesler's Theorem says "AI is whatever hasn't been done yet." For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology. 
    Modern machine capabilities generally classified as AI include successfully understanding human speech, competing at the highest level in strategic game systems (such as chess and Go), autonomously operating cars, and intelligent routing in content delivery networks and military simulations.
    """

    print("--- Original Text ---")
    print(long_text)
    print("\n" + "="*50 + "\n")

    # Generate the summary
    summary = extractive_summarizer(long_text, 2)

    print("--- Summarized Text ---")
    print(summary)
