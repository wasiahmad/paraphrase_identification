
## [Paraphrase Identification (State of the art)](https://aclweb.org/aclwiki/index.php?title=Paraphrase_Identification_(State_of_the_art))

### Summary of the state-of-art techniques

#### [Support Vector Machines for Paraphrase Identification and Corpus Construction](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/I05-50015B15D.pdf)

**Essence**: This paper describes the extraction of parallel corpora from clustered news articles using annotated seed corpora and an SVM classifier, demonstrating that large parallel corpora can be induced by a classifier that includes morphological and synonymy features derived from both statis and dynamic resources.

This work actually refines the output of the second heuristic proposed by [Dolan, et al. (2004)](http://www.aclweb.org/anthology/C04-1051.pdf) which assumes that the early sentences of a news article will tend to summarize the whole article and are thus likely to contain the same information as other early sentences of other articles in the cluster. This heuristic is a text-feature-based heuristic in which the first two sentences of each article in a cluster are cross-matched with each other to find out paraphrasing sentences.

For SVM, they have used the implementation of the **Sequential Minimal Optimization (SMO)** algorithm described in [Platt (1999)](http://www.cs.utsa.edu/~bylander/cs6243/smo-book.pdf). SMO offers the benefit of relatively short training times over very large feature sets, and in particular, appears well suited to handling the sparse features encountered in natural language classification tasks.

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



#### [Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks](http://aclweb.org/anthology/D/D15/D15-1181.pdf)



#### [Corpus-based and Knowledge-based Measures of Text Semantic Similarity](http://www.aaai.org/Papers/AAAI/2006/AAAI06-123.pdf)



#### [Re-examining Machine Translation Metrics for Paraphrase Identification](http://www.aclweb.org/anthology/N12-1019.pdf)



#### [Discriminative Improvements to Distributional Sentence Similarity](http://www.aclweb.org/anthology/D/D13/D13-1090.pdf)



#### [Syntax-Aware Multi-Sense Word Embeddings for Deep Compositional Models of Meaning](http://www.aclweb.org/anthology/D/D15/D15-1177.pdf)



