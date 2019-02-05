
## List of dataset used in state-of-art techniques
#### [Quora Question Pairs](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)
<p align='justify'>
Quora released a new dataset in January 2017. The dataset consists of over 400K potential duplicate question pairs.
<p align='justify'>
 
#### [TwitterPPDB corpus](https://languagenet.github.io/)

<p align='justify'>
The initial corpus contains 51,524 human annotated sentence pairs: 42200 for training and 9324 for testing. Authors have released data collected over 1 year which consists of 2,869,657 candidate pairs. 
<p align='justify'>

#### [Microsoft Research Paraphrase Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52398).

<p align='justify'>
This dataset contains 5,801 pairs of sentences with 4,076 for training and the remaining 1,725 for testing. The training set contains 2753 true paraphrase pairs and 1323 false paraphrase pairs; the test set contains 1147 and 578 pairs, respectively.
<p align='justify'>

#### [Machine Translation Metrics Paraphrase Corpus](http://www.aclweb.org/anthology/N12-1019.pdf)

<p align='justify'>
The training set contains 5000 true paraphrase pairs and 5000 false paraphrase pairs; the test set contains 1500 and 1500 pairs, respectively. The test collection from the <a href="http://pan.webis.de/clef10/pan10-web/plagiarism-detection.html">PAN 2010 plagiarism detection competition</a> was used to generate the sentence-level PAN dataset. PAN 2010 dataset consists of 41,233 text documents from Project Gutenberg in which 94,202 cases of plagiarism have been inserted. The plagiarism was created either by using an algorithm or by explicitly asking Turkers to paraphrase passages from the original text. Only on the human created plagiarism instances were used here.
<p align='justify'>

<p align='justify'>
To generate the sentence-level PAN dataset, a heuristic alignment algorithm is used to find corresponding pairs of sentences within a passage pair linked by the plagiarism relationship. The alignment algorithm utilized only bag-of-words overlap and length ratios and no MT metrics. For negative evidence, sentences were sampled from the same document and extracted sentence pairs that have at least 4 content words in common. Then from both the positive and negative evidence files, training set of 10,000 sentence pairs and a test set of 3,000 sentence pairs were created through random sampling.
<p align='justify'>

#### [Microsoft Video Paraphrase Corpus (MSRVID)](https://www.cs.york.ac.uk/semeval-2012/task6/data/uploads/datasets/)

<p align='justify'>
In this dataset, each sentence pair has a relatedness score &isin; [0, 5], with higher scores indicating the two sentences are more closely-related. The dataset comprises pairs of sentences drawn from publicly available datasets which are given below.
<p align='justify'>

 - [Microsoft Research Paraphrase Corpus](http://research.microsoft.com/en-us/downloads/607d14d9-20cd-47e3-85bc-a2f65cd28042/): 750 pairs of sentences.
 - [Microsoft Research Video Description Corpus](http://research.microsoft.com/en-us/downloads/38cf15fd-b8df-477e-a4e4-a4680caa75af/): 750 pairs of sentences. 
 - [SMTeuroparl: WMT2008 develoment dataset (Europarl section)](http://www.statmt.org/wmt08/shared-evaluation-task.html): 734 pairs of sentences.

#### [Image Annotation with Descriptive Sentences](http://dl.acm.org/citation.cfm?id=1866717)
 - [Pascal Dataset](http://nlp.cs.illinois.edu/HockenmaierGroup/pascal-sentences/index.html): 1000 images with 5 different sentences describing the corresponding image. 
 - [Flicker8k](http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html): 7678 images from Flicker with 5 different sentences describing the corresponding image.
 - [Flicker30k](http://shannon.cs.illinois.edu/DenotationGraph/): An image caption corpus consisting of 158,915 crowd-sourced captions describing 31,783 images.
 - [MSCOCO](http://cocodataset.org/#overview): 328,000 images with 5 different sentences describing the corresponding image.

#### [Video Annotation with Descriptive Sentences]()
 - [MSR-VTT Dataset](http://ms-multimedia-challenge.com/2017/challenge): Comprised of 10,000 videos with 20 sentences each describing the videos.
 
#### [Sentences Involving Compositional Knowledge (SICK) dataset](http://clic.cimec.unitn.it/composes/sick.html)

<p align='justify'>
This dataset consists of 9,927 sentence pairs with 4,500 for training, 500 as a development set, and the remaining 4,927 in the test set. The sentences are drawn from image video descriptions. Each sentence pair is annotated with a relatedness score &isin; [1, 5], with higher scores indicating the two sentences are more closely-related.
<p align='justify'>

#### [PPDB: The Paraphrase Database](http://www.cis.upenn.edu/~ccb/ppdb/)

<p align='justify'>
The PPDB contains more than 220 million paraphrase pairs of which 73 million are phrasal paraphrases and 140 million are paraphrase patterns that capture syntactic transformations of sentences.
<p align='justify'>
  
#### [WikiAnswers Paraphrase Corpus](http://knowitall.cs.washington.edu/oqa/data/wikianswers/)

<p align='justify'>
The WikiAnswers corpus contains clusters of questions tagged by WikiAnswers users as paraphrases. Each cluster optionally contains an answer provided by WikiAnswers users. There are 30,370,994 clusters containing an average of 25 questions per cluster. 3,386,256 (11%) of the clusters have an answer.
<p align='justify'>

<p align='justify'>
The data can be downloaded from: http://knowitall.cs.washington.edu/oqa/data/wikianswers/. The corpus is split into 40 gzip-compressed files. The total compressed filesize is 8GB; the total decompressed filesize is 40GB. Each file contains one cluster per line. Each cluster is a tab-separated list of questions and answers. Questions are prefixed by q: and answers are prefixed by a:. Here is an example cluster (tabs replaced with newlines):
<p align='justify'>

  ```
  q:How many muslims make up indias 1 billion population?
  q:How many of india's population are muslim?
  q:How many populations of muslims in india?
  q:What is population of muslims in india?
  a:Over 160 million Muslims per Pew Forum Study as of October 2009.
  ```

  Reference: [https://github.com/afader/oqa#wikianswers-corpus](https://github.com/afader/oqa#wikianswers-corpus)<br/>
  Related Corpus: [Paralex: Paraphrase-Driven Learning for Open Question Answering](http://knowitall.cs.washington.edu/paralex/)
