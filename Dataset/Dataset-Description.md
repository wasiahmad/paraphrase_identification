
## List of dataset used in state-of-art techniques
1. [Microsoft Research Paraphrase Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52398).

  <p align="justify">
  **Description**: This dataset contains **5,801** pairs of sentences with **4,076** for training and the remaining **1,725** for testing. The training set contains **2753** true paraphrase pairs and **1323** false paraphrase pairs; the test set contains **1147** and **578** pairs, respectively.
  <p align="justify">

2. [Machine Translation Metrics Paraphrase Corpus](http://www.aclweb.org/anthology/N12-1019.pdf)

  <p align="justify">
  **Description**: The training set contains **5000** true paraphrase pairs and **5000** false paraphrase pairs; the test set contains **1500** and **1500** pairs, respectively. The test collection from the [PAN 2010 plagiarism detection competition](http://pan.webis.de/clef10/pan10-web/plagiarism-detection.html) was used to generate the sentence-level PAN dataset. PAN 2010 dataset consists of **41,233** text documents from Project Gutenberg in which **94,202** cases of plagiarism have been inserted. The plagiarism was created either by using an algorithm or by explicitly asking Turkers to paraphrase passages from the original text. Only on the human created plagiarism instances were used here.
  <p align="justify">
  <p align="justify">
  To generate the sentence-level PAN dataset, a heuristic alignment algorithm is used to find corresponding pairs of sentences within a passage pair linked by the plagiarism relationship. The alignment algorithm utilized only bag-of-words overlap and length ratios and no MT metrics. For negative evidence, sentences were sampled from the same document and extracted sentence pairs that have at least 4 content words in common. Then from both the positive and negative evidence files, training set of **10,000** sentence pairs and a test set of **3,000** sentence pairs were created through random sampling.
  <p align="justify">

3. [Microsoft Video Paraphrase Corpus (MSRVID)](https://www.cs.york.ac.uk/semeval-2012/task6/data/uploads/datasets/)
  <p align="justify">
  **Description**: In this dataset, each sentence pair has a relatedness score &isin; [0, 5], with higher scores indicating the two sentences are more closely-related. The dataset comprises pairs of sentences drawn from publicly available datasets which are given below.
 - [Microsoft Research Paraphrase Corpus](http://research.microsoft.com/en-us/downloads/607d14d9-20cd-47e3-85bc-a2f65cd28042/): **750** pairs of sentences.
 - [Microsoft Research Video Description Corpus](http://research.microsoft.com/en-us/downloads/38cf15fd-b8df-477e-a4e4-a4680caa75af/): **750** pairs of sentences. 
 - [SMTeuroparl: WMT2008 develoment dataset (Europarl section)](http://www.statmt.org/wmt08/shared-evaluation-task.html): **734** pairs of sentences.
  <p align="justify">

4. [Image Annotation with Descriptive Sentences](http://dl.acm.org/citation.cfm?id=1866717)
 - [Pascal Dataset](http://nlp.cs.illinois.edu/HockenmaierGroup/pascal-sentences/index.html): **1000** Images with **5** different sentences describing the corresponding image. 
 - [Flicker Dataset](http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html): **7678** Images from Flicker with **5** different sentences describing the corresponding image.
 
5. [Sentences Involving Compositional Knowledge (SICK) dataset](http://clic.cimec.unitn.it/composes/sick.html)
  <p align="justify">
  **Description**: This dataset consists of **9,927** sentence pairs with **4,500** for training, **500** as a development set, and the remaining **4,927** in the test set. The sentences are drawn from image video descriptions. Each sentence pair is annotated with a relatedness score &isin; [1, 5], with higher scores indicating the two sentences are more closely-related.
  <p align="justify">

6. [PPDB: The Paraphrase Database](http://www.cis.upenn.edu/~ccb/ppdb/)

