import streamlit as st

import spacy 
from textblob import TextBlob
from gensim.summarization import summarize
import nltk
nltk.download('punkt')

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


def text_analyzer(my_text):
	nlp = spacy.load('en')
	docx = nlp(my_text)

	tokens = [token.text for token in docx]
	allData = ['"Tokens":{},\n"Lemmatized":{}'.format(token.text,token.lemma_) for token in docx]
	return allData


def entity_analyzer(my_text):
	nlp = spacy.load('en')
	docx = nlp(my_text)
	tokens = [token.text for token in docx]
	entities = [(entity.text,entity.label_) for entity in docx.ents]
	allData = ['"Tokens":{},\n"Entities":{}'.format(tokens,entities)]
	return allData


def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summar = LexRankSummarizer()
	summary = lex_summar(parser.document,3)
	summary_list = [str(sent) for sent in summary]
	result = ' '.join(summary_list)
	return result



def main():

	st.title("NLP with Streamlit")
	st.subheader("Natural Language Processsing using various tools")


	#Tokenization
	if st.checkbox("Show Tokens and Lemmatization"):
		st.subheader("Tokenize Your Text")
		message = st.text_area("Enter Your Text")
		if st.button("Analyze"):
			nlp_result = text_analyzer(message)
			st.json(nlp_result)


	# Named Entity
	if st.checkbox("Show Named Entities"):
		st.subheader("Extract Entities From Your Text")
		message = st.text_area("Enter Your Text","Type here")
		if st.button("Extract"):
			nlp_result = entity_analyzer(message)
			st.json(nlp_result)
			

	# Sentiment Analysis
	if st.checkbox("Show Sentiment Analysis"):
		st.subheader("Sentiment of Your Text")
		message = st.text_area("Enter Your Text")
		if st.button("Analyze"):
			blob = TextBlob(message)
			result_sentiment = blob.sentiment
			st.success(result_sentiment)

   
	if st.checkbox("Show Text Summerization"):
		st.subheader("summarize Your Text")
		message = st.text_area("Enter Your Text for summarization")
		summary_options = st.selectbox("Choise Your Summerizer",("gensim","sumy"))
		if st.button("Summerize"):
			if summary_options == 'gensim':
				st.text("Using Gensim..")
				summary_result = summarize(message)
			elif summary_options == 'sumy':
				st.text("Using Sumy..")
				summary_result = sumy_summarizer(message)

			else:
				st.warning("Using Default Summarizer")
				st.text("Using Gensim..")
				summary_result = Summerize(message)

			st.success(summary_result)



if __name__ == '__main__':
	main()