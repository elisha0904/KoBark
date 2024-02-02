import requests

def get_sentence_from_book(book, sentence_index):
    with open(book, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    if sentence_index < len(sentences):
        return sentences[sentence_index]
    else:
        return None