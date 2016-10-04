
## [Paraphrase Identification (State of the art)](https://aclweb.org/aclwiki/index.php?title=Paraphrase_Identification_(State_of_the_art))

### Summary of the state-of-art techniques

#### [Support Vector Machines for Paraphrase Identification and Corpus Construction](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/I05-50015B15D.pdf)

<p align="justify">
**Essence**: This paper describes the extraction of parallel corpora from clustered news articles using annotated seed corpora and an SVM classifier, demonstrating that large parallel corpora can be induced by a classifier that includes morphological and synonymy features derived from both statis and dynamic resources.
<p align="justify">
<p align="justify">
This work actually refines the output of the second heuristic proposed by [Dolan, et al. (2004)](http://www.aclweb.org/anthology/C04-1051.pdf) which assumes that the early sentences of a news article will tend to summarize the whole article and are thus likely to contain the same information as other early sentences of other articles in the cluster. This heuristic is a text-feature-based heuristic in which the first two sentences of each article in a cluster are cross-matched with each other to find out paraphrasing sentences.
<p align="justify">
<p align="justify">
For SVM, they have used the implementation of the **Sequential Minimal Optimization (SMO)** algorithm described in [Platt (1999)](http://www.cs.utsa.edu/~bylander/cs6243/smo-book.pdf). SMO offers the benefit of relatively short training times over very large feature sets, and in particular, appears well suited to handling the sparse features encountered in natural language classification tasks.
<p align="justify">

**Expermental Dataset**: [Microsoft Research Paraphrase Corpus](https://github.com/wasiahmad/Paraphrase-Identification-Task/tree/master/Dataset/MSRParaphraseCorpus).

**Bibliography**
```
@inproceedings{brockett2005support,
  title={Support vector machines for paraphrase identification and corpus construction},
  author={Brockett, Chris and Dolan, William B},
  booktitle={Proceedings of the 3rd International Workshop on Paraphrasing},
  pages={1--8},
  year={2005}
}
```

#### [Dynamic Pooling and Unfolding Recursive Autoencoders for Paraphrase Detection](http://papers.nips.cc/paper/4204-dynamic-pooling-and-unfolding-recursive-autoencoders-for-paraphrase-detection.pdf)
<p align="justify">
**Essence**: This paper leveraged and extended the method described in Recursive Autoencoder describen in [Socher et. al (EMNLP.11)](http://dl.acm.org/citation.cfm?id=2145450). Prior feeding data to RAE, they build binary parse tree from test corpus. Recursive autoencoder is a recursive neural network, which recursively learns representation of words as well as other non terminals in the parse tree. They extended RAE as Unfolding RAE, which decodes the non-terminal down to the terminal level.
<p align="justify">
<p align="justify">
They introduced dynamic pooling approach which generates fixed sized similarity matrix between words and non-terminals from variable sized matrix. They used the similarity matrix along with 3 additional features for classifying paraphrases. Those additional features are, The ﬁrst is 1 if two sentences contain exactly the same numbers or no number and 0 otherwise, the second is 1 if both sentences contain the same numbers and the third is 1 if the set of numbers in one sentence is a strict subset of the numbers in the other sentence. 
<p align="justify">

**Experimental Dataset & Result**: They used [MSRP Dataset describe inD Dolan et. al.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/para_coling2004.pdf) and their accuracy is 76.8% in terms of accuracy. 

**More Details on this work**: [Blog Link](http://www.socher.org/index.php/Main/DynamicPoolingAndUnfoldingRecursiveAutoencodersForParaphraseDetection), [Code](https://github.com/jeremysalwen/ParaphraseAutoencoder-octave)

**Bibliography**
````
@incollection{SocherEtAl2011:PoolRAE,
  title = {{Dynamic Pooling and Unfolding Recursive Autoencoders for Paraphrase Detection}},
  author = {{Richard Socher and Eric H. Huang and Jeffrey Pennington and Andrew Y. Ng and Christopher D. Manning}},
  booktitle = {{Advances in Neural Information Processing Systems 24}},
  year = {2011}
}
````


#### [Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks](http://aclweb.org/anthology/D/D15/D15-1181.pdf)

I am reading this paper and update with summary shortly. 

#### [Corpus-based and Knowledge-based Measures of Text Semantic Similarity](http://www.aaai.org/Papers/AAAI/2006/AAAI06-123.pdf)

<p align="justify">
**Essence**: This paper demonstrates the effectiveness of two corpus-based and six knowledge-based measures for text semantic similarity. Main idea to measure semantic similarity of texts by exploiting the information that can be drawn from the similarity of the component words. Two corpus-based measures are [Pointwise Mutual Information](https://en.wikipedia.org/wiki/Pointwise_mutual_information) and [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis). Six knowledge-based measures are [Leacock & Chodorow](http://www.aclweb.org/anthology/J98-1006.pdf) similarity, [Lesk](http://dl.acm.org/citation.cfm?id=318728) similarity, [Wu and Palmer](http://dl.acm.org/citation.cfm?id=981751) similarity, [Resnik](https://arxiv.org/abs/cmp-lg/9511007) similarity, [Lin](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.55.1832&rep=rep1&type=pdf) similarity and [Jiang & Conrath](https://arxiv.org/abs/cmp-lg/9709008) similarity.
<p align="justify">

**Expermental Dataset**: [Microsoft Research Paraphrase Corpus](https://github.com/wasiahmad/Paraphrase-Identification-Task/tree/master/Dataset/MSRParaphraseCorpus).

**Bibliography**
```
@inproceedings{mihalcea2006corpus,
  title={Corpus-based and knowledge-based measures of text semantic similarity},
  author={Mihalcea, Rada and Corley, Courtney and Strapparava, Carlo},
  booktitle={AAAI},
  volume={6},
  pages={775--780},
  year={2006}
}
```

#### [Re-examining Machine Translation Metrics for Paraphrase Identification](http://www.aclweb.org/anthology/N12-1019.pdf)



#### [Discriminative Improvements to Distributional Sentence Similarity](http://www.aclweb.org/anthology/D/D13/D13-1090.pdf)

<p align="justify">
**Essence**: The main contribution of this work is proposing a new term-weighting metric called TF-KLD which includes term frequency and KL-divergence. TF-KLD measures the discriminability of a feature and the newly reweighted feature-context matrix factorization yields better semantic relatedness between a pair of paraphrased sentences. Moreover, they have converted latent representation of a pair of sentences into a sample vector by concatenating the element-wise sum and absolute difference of vector representations. This representation is further used in supervised classification of sentence paraphrasing. They have experimented from two perspectives, namely, similarity-based classification and supervised classification. For similarity-based classification, they have used TF-KLD weighting with SVD (Singular Value Decomposition) and NMF (Non-negative Matrix Factorization) and found that NMF is performing slightly better than SVD. For supervised classification, they have used Support Vector Machines. For all of the experiments, they have used two different distribution feature sets. First one included only unigrams while the second one also includes bigrams and unlabeled dependency pairs obtained from [MaltParser](http://stp.lingfil.uu.se/~nivre/docs/nivre_hall_2005.pdf).
<p align="justify">

**Expermental Dataset**: [Microsoft Research Paraphrase Corpus](https://github.com/wasiahmad/Paraphrase-Identification-Task/tree/master/Dataset/MSRParaphraseCorpus).

**Bibliography**
```
@inproceedings{ji2013discriminative,
  title={Discriminative Improvements to Distributional Sentence Similarity.},
  author={Ji, Yangfeng and Eisenstein, Jacob},
  booktitle={EMNLP},
  pages={891--896},
  year={2013}
}
```

#### [Syntax-Aware Multi-Sense Word Embeddings for Deep Compositional Models of Meaning](http://www.aclweb.org/anthology/D/D15/D15-1177.pdf)



