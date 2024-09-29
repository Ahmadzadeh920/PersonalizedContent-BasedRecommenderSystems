<div align="center">
<h1 align="center">News Personalized Content-Based Recommender Systems </h1>
<h3 align="center">by Using NLP, Deep Learning RNN and CNN Approaches</h3>
</div>
<p align="center">
<a href="https://machinelearning.org.in/" target="_blank"> <img src="https://img.icons8.com/?size=100&id=JdKL8UDIBykm&format=png&color=000000" alt="machine learning" width="40" height="40"/> </a>
<a href="https://www.nlp-techniques.org/" target="_blank"> <img src="https://img.icons8.com/?size=100&id=Fc8jfmNVFcMq&format=png&color=000000" alt="NLP" width="40" height="40"/> </a>
<a href="https://www.deeplearningbook.org/" target="_blank"> <img src="https://img.icons8.com/?size=100&id=97384&format=png&color=000000" alt="deep neural networks" width="40" height="40"/> </a>
<a href="https://en.wikipedia.org/wiki/Recommender_system" target="_blank"> <img src="https://img.icons8.com/?size=100&id=Vebh7PFFXQZL&format=png&color=000000" alt="recommender systems" width="40" height="40"/> </a>




</p>




# Introduction
 . Autonomous recommender systems are the systems that help users find and select interest items. One of the main strategies to make identification of user interests in a news portal is the read news content.  This project offers a personalized news recommender system in which user interests can be predicted based on the contents of news read by their users, extracting keywords, and finding semantic relations between them. One of the topic modeling algorithms is Latent Dirichlet Allocation (LDA), which scores terms based on repetition of terms. Deep neural networks have a great success in many NLP tasks, which are based on RNN, CNN, LSTM, and recently attention mechanisms inspired by neuroscience. The experimental work is done in Python language and finally compares six kinds of DNN in the prediction of interest news which proposed a structure that involves CNN to extract local information, an attention layer to extract semantic information from critical words, and LSTM to extract dependencies between words. This proposed DNN possesses a higher F1 measure in comparison with others. The input of DNN is a term-topic matrix and the output is user-topic for each document.


# Proposed Model

- [Data Description](#development-usage)
- [Data Preprocessin](#Data-Preprocessing)
- [Analyze](#Analyze)
- [Experimental Results](Experimental-Results)
  -[Training Section](Training-Section)
  -[Testing Section](Testing-Section)
- [Evaluation](Evaluation)



# Data Description

The data has been collected from three Irish online news sources (RTE, Irish Times and Irish Independent
Table 1. Data Statistics
Data Collections	
31		User
3857	News
2009/7/8 – 2009/7/31	Time Collected Data

Table 2. Example of Transaction in Dataset
User_id	article_id
31	6
3857	7
2009/7/8 – 2009/7/31	13

Table 3. Example Article of Term Index and Frequency Pairs
term_indexN:freqN	…	term_index2:freq2	term_index1:freq1	article data	article_id
01:270	…	06:01	04:147	08/07/2009	2
01:353	….	01:752	01:351	21/07/2009	2237
01:459	….	03:01	01:460	31/07/2009	3686

Table 4. List of some Unique Terms
Term_id	Term
20	Swedish
753	told
