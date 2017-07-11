## [Paraphrase Identification (State of the art)](https://aclweb.org/aclwiki/index.php?title=Paraphrase_Identification_(State_of_the_art))

### Summary of the state-of-art techniques
 1. [Support Vector Machines for Paraphrase Identification and Corpus Construction](#support-vector-machines-for-paraphrase-identification-and-corpus-construction)
 2. [Dynamic Pooling and Unfolding Recursive Autoencoders for Paraphrase Detection](#dynamic-pooling-and-unfolding-recursive-autoencoders-for-paraphrase-detection)
 3. [Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks](#multi-perspective-sentence-similarity-modeling-with-convolutional-neural-networks)
 4. [Corpus-based and Knowledge-based Measures of Text Semantic Similarity](#corpus-based-and-knowledge-based-measures-of-text-semantic-similarity)
 5. [Re-examining Machine Translation Metrics for Paraphrase Identification](#re-examining-machine-translation-metrics-for-paraphrase-identification)
 6. [Discriminative Improvements to Distributional Sentence Similarity](#discriminative-improvements-to-distributional-sentence-similarity)
 7. [Syntax-Aware Multi-Sense Word Embeddings for Deep Compositional Models of Meaning](#syntax-aware-multi-sense-word-embeddings-for-deep-compositional-models-of-meaning)
 8. [Using Dependency-Based Features to Take the “Para-farce” out of Paraphrase](#using-dependency-based-features-to-take-the-para-farce-out-of-paraphrase)
 9. [A Semantic Similarity Approach to Paraphrase Detection](#a-semantic-similarity-approach-to-paraphrase-detection)
 10. [Paraphrase recognition via dissimilarity significance classification](#paraphrase-recognition-via-dissimilarity-significance-classification)
 
---

#### [Support Vector Machines for Paraphrase Identification and Corpus Construction](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/I05-50015B15D.pdf)

<p align="justify">
This paper describes the extraction of parallel corpora from clustered news articles using annotated seed corpora and an SVM classifier, demonstrating that large parallel corpora can be induced by a classifier that includes morphological and synonymy features derived from both statis and dynamic resources.
<p align="justify">

<p align="justify">
This work actually refines the output of the second heuristic proposed by <a href="http://www.aclweb.org/anthology/C04-1051.pdf">Dolan, et al. (2004)</a> which assumes that the early sentences of a news article will tend to summarize the whole article and are thus likely to contain the same information as other early sentences of other articles in the cluster. This heuristic is a text-feature-based heuristic in which the first two sentences of each article in a cluster are cross-matched with each other to find out paraphrasing sentences.
<p align="justify">

<p align="justify">
For SVM, they have used the implementation of the <b>Sequential Minimal Optimization (SMO)</b> algorithm described in <a href="http://www.cs.utsa.edu/~bylander/cs6243/smo-book.pdf">Platt (1999)</a>. SMO offers the benefit of relatively short training times over very large feature sets, and in particular, appears well suited to handling the sparse features encountered in natural language classification tasks.
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

---

#### [Dynamic Pooling and Unfolding Recursive Autoencoders for Paraphrase Detection](http://papers.nips.cc/paper/4204-dynamic-pooling-and-unfolding-recursive-autoencoders-for-paraphrase-detection.pdf)

<p align="justify">
This paper leveraged and extended the method described in Recursive Autoencoder by <a href="http://dl.acm.org/citation.cfm?id=2145450">Socher et. al</a>. Prior feeding data to RAE, they build binary parse tree from test corpus. Recursive autoencoder is a recursive neural network, which recursively learns representation of words as well as other non terminals in the parse tree. They extended RAE as Unfolding RAE, which decodes the non-terminal down to the terminal level.
<p align="justify">

<p align="justify">
They introduced dynamic pooling approach which generates fixed sized similarity matrix between words and non-terminals from variable sized matrix. They used the similarity matrix along with 3 additional features for classifying paraphrases. Those additional features are, The ﬁrst is 1 if two sentences contain exactly the same numbers or no number and 0 otherwise, the second is 1 if both sentences contain the same numbers and the third is 1 if the set of numbers in one sentence is a strict subset of the numbers in the other sentence. 
<p align="justify">

**Experimental Dataset & Result**: [Microsoft Research Paraphrase Corpus](https://github.com/wasiahmad/Paraphrase-Identification-Task/tree/master/Dataset/MSRParaphraseCorpus). Accuracy is 76.8% and F1 score is 83.6%. 

**More Details on this work**: [Blog Link](http://www.socher.org/index.php/Main/DynamicPoolingAndUnfoldingRecursiveAutoencodersForParaphraseDetection), [Code](https://github.com/jeremysalwen/ParaphraseAutoencoder-octave)

**Bibliography**
```
@inproceedings{socher2011dynamic,
  title={Dynamic pooling and unfolding recursive autoencoders for paraphrase detection},
  author={Socher, Richard and Huang, Eric H and Pennin, Jeffrey and Manning, Christopher D and Ng, Andrew Y},
  booktitle={Advances in Neural Information Processing Systems},
  pages={801--809},
  year={2011}
}
```

---

#### [Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks](http://aclweb.org/anthology/D/D15/D15-1181.pdf)

<p align="justify">
In this work, they claim that their algorithm will do better with low amount of data, as it investigates intrinsic features in different granularity. But they did not provide any substantial proof. They used convolutional-neural-networks to obtain sentence embeddings. They used convolutional-neural-networks in multiple granularity. Every word in a sentence is represented as a word embedding vector. They used convolution on both whole word embedding and every dimension of the embedding. Based on that, they grouped convolutions into 2 groups.
<p align="justify">

<p align="justify">
GroupA for whole word embedding and GroupB for per-dimension convolution. They used 3 types of pooling function (i.e. Max, Min. Mean) for group1 and 2 types (Max, Min) for GroupB. Although, they used identical convolutionNN for each of the pooling, but they maintained different NN instead. They also incorporated multiple window sizes for different filters including W<sub>s</sub>(window size) to be infinite (which, infact, considers the whole sentence). On top of the pooling layer, they have a similarity measurement layer which uses a specific similarity measurement of 2 sentence representation and passes it to a fully connected softmax layer. 
<p align="justify">

**Experimental Dataset & Result**: They used the following three datasets.
 1. [Microsoft Research Paraphrase Corpus](https://github.com/wasiahmad/Paraphrase-Identification-Task/tree/master/Dataset/MSRParaphraseCorpus): Accuracy 78.60%, F1 score 84.73%
 2. [SICK dataset](http://clic.cimec.unitn.it/composes/sick.html): Recall 86.86%, Precision 80.47%, Mean Squared Error 26.06%
 3. [MSRVID](https://www.cs.york.ac.uk/semeval-2012/task6/data/uploads/datasets/): Recall 90.90%

For MSRP, they used hinge loss function as trainning and for other 2, they used Regularized KL-divergence loss.

**More Details on this work**: [Code](https://github.com/hohoCode/textSimilarityConvNet)

**Bibliography**
```
@inproceedings{he2015multi,
  title={Multi-perspective sentence similarity modeling with convolutional neural networks},
  author={He, Hua and Gimpel, Kevin and Lin, Jimmy},
  booktitle={Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing},
  pages={1576--1586},
  year={2015}
}
```

---

#### [Corpus-based and Knowledge-based Measures of Text Semantic Similarity](http://www.aaai.org/Papers/AAAI/2006/AAAI06-123.pdf)

<p align="justify">
This paper demonstrates the effectiveness of two corpus-based and six knowledge-based measures for text semantic similarity. Main idea to measure semantic similarity of texts by exploiting the information that can be drawn from the similarity of the component words. Two corpus-based measures are <a href="https://en.wikipedia.org/wiki/Pointwise_mutual_information">Pointwise Mutual Information</a> and <a href="https://en.wikipedia.org/wiki/Latent_semantic_analysis">Latent Semantic Analysis</a>. Six knowledge-based measures are <a href="http://www.aclweb.org/anthology/J98-1006.pdf">Leacock & Chodorow</a> similarity, <a href="http://dl.acm.org/citation.cfm?id=318728">Lesk</a> similarity, <a href="http://dl.acm.org/citation.cfm?id=981751">Wu and Palmer</a> similarity, <a href="https://arxiv.org/abs/cmp-lg/9511007">Resnik</a> similarity, <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.55.1832&rep=rep1&type=pdf">Lin</a> similarity and <a href="https://arxiv.org/abs/cmp-lg/9709008">Jiang & Conrath</a> similarity.
<p align="justify">

**Expermental Dataset & Result**: [Microsoft Research Paraphrase Corpus](https://github.com/wasiahmad/Paraphrase-Identification-Task/tree/master/Dataset/MSRParaphraseCorpus). Accuracy is 70.3% and F1 score is 81.3%.

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

---

#### [Re-examining Machine Translation Metrics for Paraphrase Identification](http://www.aclweb.org/anthology/N12-1019.pdf)

<p align="justify">
The goal of this paper was to determine whether approaches developed for the related but different task of Machine Translation (MT) evaluation can be as competitive as approaches developed specifically for the task of paraphrase identification. It is reported that a meta-classifier trained using only MT metrics outperforms all previous approaches for the MSRP corpus. They used a simple meta-classifier that uses the average of the unweighted probability estimates from the constituent classifiers to make its final decision. They used three constituent classifiers: Logistic regression, the SMO implementation of a support vector machine and a lazy, instance-based classifier that extends the nearest neighbor algorithm.
<p align="justify">

<p align="justify">
They explored eight most sophisticated MT metrics of the last few years that claim to go beyond simple n-gram overlap and edit distance. The metrices are <a href="http://www.aclweb.org/anthology/P02-1040.pdf">BLEU</a>, <a href="http://dl.acm.org/citation.cfm?id=1289273">NIST</a>, <a href="https://www.cs.umd.edu/~snover/pub/amta06/ter_amta.pdf">TER</a>, <a href="http://link.springer.com/article/10.1007/s10590-009-9062-9">TERp</a>, <a href="http://www.cs.cmu.edu/~alavie/papers/meteor-naacl2010.pdf">METEOR</a>, <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.219.4894&rep=rep1&type=pdf">SEPIA</a>, <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.219.6503&rep=rep1&type=pdf">BADGER</a> and <a href="https://www.comp.nus.edu.sg/~nght/pubs/acl08.pdf">MAXSIM</a>. They have identified TERp, METEOR, BADGER and SEPIA as the best four metrices and also given good examples to demonstrate their effectiveness. They have done rigorous error analysis and presented top 5 & 3 sources of errors in the MSRP and MT-Metrics-Paraphrase corpus respectively. They suggested few improvements through incorporating world knowledge, anophora resolution system or giving more weights on the differences in proper names and their variants.
<p align="justify">

**Expermental Dataset & Result**: [Microsoft Research Paraphrase Corpus](https://github.com/wasiahmad/Paraphrase-Identification-Task/tree/master/Dataset/MSRParaphraseCorpus) and [MT-Metrics-Paraphrase-Corpus](https://github.com/wasiahmad/Paraphrase-Identification-Task/tree/master/Dataset/MT-Metrics-Paraphrase-Corpus). Accuracy is 77.4% and F1 score is 84.1%.

**Bibliography**
```
@inproceedings{madnani2012re,
  title={Re-examining machine translation metrics for paraphrase identification},
  author={Madnani, Nitin and Tetreault, Joel and Chodorow, Martin},
  booktitle={Proceedings of the 2012 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages={182--190},
  year={2012},
  organization={Association for Computational Linguistics}
}
```

---

#### [Discriminative Improvements to Distributional Sentence Similarity](http://www.aclweb.org/anthology/D/D13/D13-1090.pdf)

<p align="justify">
The main contribution of this work is proposing a new term-weighting metric called TF-KLD which includes term frequency and KL-divergence. TF-KLD measures the discriminability of a feature and the newly reweighted feature-context matrix factorization yields better semantic relatedness between a pair of paraphrased sentences. Moreover, they have converted latent representation of a pair of sentences into a sample vector by concatenating the element-wise sum and absolute difference of vector representations. This representation is further used in supervised classification of sentence paraphrasing. 
<p align="justify">

<p align="justify">
They have experimented from two perspectives, namely, similarity-based classification and supervised classification. For similarity-based classification, they have used TF-KLD weighting with SVD (Singular Value Decomposition) and NMF (Non-negative Matrix Factorization) and found that NMF is performing slightly better than SVD. For supervised classification, they have used Support Vector Machines. For all of the experiments, they have used two different distribution feature sets. First one included only unigrams while the second one also includes bigrams and unlabeled dependency pairs obtained from <a href="http://stp.lingfil.uu.se/~nivre/docs/nivre_hall_2005.pdf">MaltParser</a>.
<p align="justify">

**Expermental Dataset & Result**: [Microsoft Research Paraphrase Corpus](https://github.com/wasiahmad/Paraphrase-Identification-Task/tree/master/Dataset/MSRParaphraseCorpus). Accuracy is 80.4% and F1 score is 85.9%.

**More Details on this work**: [Code](https://github.com/jiyfeng/tfkld)

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

---

#### [Syntax-Aware Multi-Sense Word Embeddings for Deep Compositional Models of Meaning](http://www.aclweb.org/anthology/D/D15/D15-1177.pdf)

<p align="justify">
This work proposes an architecture for jointly training a compositional model and a set of word embeddings, in a way that imposes dynamic word sense induction for each word during the learning process. The learning takes places in the context of a RecNN (or an RNN), and both word embeddings and parameters of the compositional layer are optimized against a generic objective function that uses a hinge loss function. Novelty of their work is an additional layer on top of the compositional layer which scores the linguistic plausibility of the composed sentence or phrase vector with regard to both syntax and semantics.
<p align="justify">

<p align="justify">
They extended their model to address lexical ambiguity by applying a gated architecture, similar to the one used in the multi-sense model of <a href="https://arxiv.org/pdf/1504.06654.pdf">Neelakantan et al.</a>, but advancing the main idea to the compositional setting. They have adopted <a href="https://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf">siamese architecture</a> for paraphrase detection. They have considered both L2 norm variation and cosine similarity to compare sentence vectors produced by RecNN or RNN.
<p align="justify">

**Expermental Dataset & Result**: [Microsoft Research Paraphrase Corpus](https://github.com/wasiahmad/Paraphrase-Identification-Task/tree/master/Dataset/MSRParaphraseCorpus), [PPDB: The Paraphrase Database](http://www.cis.upenn.edu/~ccb/ppdb/). Accuracy is 78.6% and F1 score is 85.3%.

**Bibliography**
```
@article{cheng2015syntax,
  title={Syntax-aware multi-sense word embeddings for deep compositional models of meaning},
  author={Cheng, Jianpeng and Kartsaklis, Dimitri},
  journal={arXiv preprint arXiv:1508.02354},
  year={2015}
}
```

---

#### [Using Dependency-Based Features to Take the “Para-farce” out of Paraphrase](http://www.alta.asn.au/events/altw2006/proceedings/swan-final.pdf)

<p align="justify">
This work investigates whether features based on syntactic dependencies can aid in paraphrase identification. This work proposes a machine learning approach based on syntactic dependency information to filter out false paraphrases. This work explored 17 different features of 4 kinds, namely, N-gram Overlap, Dependency Relation Overlap, Dependency Tree-Edit Distance and Surface features.
<p align="justify">

<p align="justify">
For the N-gram Overlap features, they used precision, recall on unigrams, lemmatised unigrams, Bleu, lemmatised Bleu and fmeasure. For Dependency Relation Overlap features, they used precision, recall on dependency relation and lemmatised dependency relation. For Dependency Tree-Edit Distance features, they used an ordered tree-edit distance algorithm based on dynamic programming. For Surface features, they used the difference in length of two sentences.
<p align="justify">

**Expermental Dataset & Result**: [Microsoft Research Paraphrase Corpus](https://github.com/wasiahmad/Paraphrase-Identification-Task/tree/master/Dataset/MSRParaphraseCorpus). Accuracy is 75.6% and F1 score is 83%. 

**Bibliography**
```
@inproceedings{wan2006using,
  title={Using dependency-based features to take the “para-farce” out of paraphrase},
  author={Wan, Stephen and Dras, Mark and Dale, Robert and Paris, C{\'e}cile},
  booktitle={Proceedings of the Australasian Language Technology Workshop},
  volume={2006},
  year={2006}
}
```

---

#### [A Semantic Similarity Approach to Paraphrase Detection](http://staffwww.dcs.shef.ac.uk/people/S.Fernando/pubs/clukPaper.pdf)

<p align="justify">
This work presents an algorithm for paraphrase identification which makes extensive use of word similarity information derived from WordNet. This work uses all word-to-word similarities in a pair of sentences. To compute word-to-word similarity, they considered six different similarity metric, namely <a href="http://www.aclweb.org/anthology/J98-1006.pdf">Leacock and Chodorow</a> similarity, <a href="http://dl.acm.org/citation.cfm?id=318728">Lesk</a> similarity, <a href="http://dl.acm.org/citation.cfm?id=981751">Wu and Palmer</a> similarity, <a href="https://arxiv.org/abs/cmp-lg/9511007">Resnik</a> similarity, <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.55.1832&rep=rep1&type=pdf">Lin</a> similarity and <a href="https://arxiv.org/abs/cmp-lg/9709008">Jiang and Conrath</a> similarity.
<p align="justify">

**Expermental Dataset & Result**: [Microsoft Research Paraphrase Corpus](https://github.com/wasiahmad/Paraphrase-Identification-Task/tree/master/Dataset/MSRParaphraseCorpus). Accuracy is 74.1% and F1 score is 82.4%. 

**Bibliography**
```
@inproceedings{fernando2008semantic,
  title={A semantic similarity approach to paraphrase detection},
  author={Fernando, Samuel and Stevenson, Mark},
  booktitle={Proceedings of the 11th Annual Research Colloquium of the UK Special Interest Group for Computational Linguistics},
  pages={45--52},
  year={2008},
  organization={Citeseer}
}
```

---

#### [Paraphrase Recognition via Dissimilarity Significance Classification](https://www.comp.nus.edu.sg/~kanmy/papers/paraphrase_emnlp2006.pdf)

<p align="justify">
This work proposes a two-phase framework emphasizing dissimilarity classification between a pair of sentence. In the first phase, they pair up tuples (predicate, argument) in a <i>greedy</i> manner. In the second phase, a dissimilarity classification module uses the lexical head of the predicates and the tuples' path of attachment as features to decide whether such tuples are barriers to paraphrase. The key idea is, for a pair of sentences to be a paraphrase, they must possess two attributes. (1) They share a substantial amount of information nuggets and (2) if extra information in the sentences exists, the effect of its removal is not significant.
<p align="justify">

<p align="justify">
A pair of sentences first fed to a syntactic parser and then passed to a semantic role labeler to label predicate argument tuples. Then normalized tuple similarity scores are computed over the tuple pairs using a metric that accounts for similarities in both syntactic structure and content of each tuple. Then most similar predicate argument tuples are greedily paired together. Any remaining unpaired tuples represent extra information and passed to a dissimilarity classifier to decide whether such information is significant. The dissimilarity classifier uses supervised machine learning to make such a decision. 
<p align="justify">

**Expermental Dataset & Result**: [Microsoft Research Paraphrase Corpus](https://github.com/wasiahmad/Paraphrase-Identification-Task/tree/master/Dataset/MSRParaphraseCorpus) and [PASCAL Recognizing Textual Entailment (RTE) Challenge Corpus](https://tac.nist.gov//2011/RTE/index.html). Accuracy is 72.0% and F1 score is 81.6%. 

**Bibliography**
```
@inproceedings{qiu2006paraphrase,
  title={Paraphrase recognition via dissimilarity significance classification},
  author={Qiu, Long and Kan, Min-Yen and Chua, Tat-Seng},
  booktitle={Proceedings of the 2006 Conference on Empirical Methods in Natural Language Processing},
  pages={18--26},
  year={2006},
  organization={Association for Computational Linguistics}
}
```

