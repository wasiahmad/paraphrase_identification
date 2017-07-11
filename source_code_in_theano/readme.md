## Deep Bidirectional Long Short Term Memory for Paraphrase Identification

<p align="justify">
In this work, we present an approach to compute similarity between sentences for the paraphrase identification task. The proposed method 
uses a deep bidirectional Long-Short Term Memory (DBLSTM) network which sequentially read words from sentences and then generate a context 
vectors representing each word of a sentence. Max pooling on all pair similarity of generated context vectors resulted in features which 
were used to train a binary classifier afterwards for paraphrase prediction. We evaluated our proposed model on Microsoft Research 
Paraphrase Corpus (MSRP). The results are not promising but we sincerely believe training our model on a large dataset should results 
in a good performance.
<p align="justify">

<p align="justify">
Our proposed model is shown below.
<p align="justify">

<p align="center">
<img src="http://i.imgur.com/aOolTRg.png" width="85%">
<p align="center">
