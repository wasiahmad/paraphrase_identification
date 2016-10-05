
## List of dataset used in state-of-art techniques
1. [Microsoft Research Paraphrase Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52398). The training set contains **2753** true paraphrase pairs and **1323** false paraphrase pairs; the test set contains **1147** and **578** pairs, respectively.
2. [Machine Translation Metrics Paraphrase Corpus](http://www.aclweb.org/anthology/N12-1019.pdf). The training set contains **5000** true paraphrase pairs and **5000** false paraphrase pairs; the test set contains **1500** and **1500** pairs, respectively.
  
  <p align="justify">
  **Description**: The test collection from the [PAN 2010 plagiarism detection competition](http://pan.webis.de/clef10/pan10-web/plagiarism-detection.html) was used to generate the sentence-level PAN dataset. PAN 2010 dataset consists of **41,233** text documents from Project Gutenberg in which **94,202** cases of plagiarism have been inserted. The plagiarism was created either by using an algorithm or by explicitly asking Turkers to paraphrase passages from the original text. Only on the human created plagiarism instances were used here.
  <p align="justify">
  <p align="justify">
  To generate the sentence-level PAN dataset, a heuristic alignment algorithm is used to find corresponding pairs of sentences within a passage pair linked by the plagiarism relationship. The alignment algorithm utilized only bag-of-words overlap and length ratios and no MT metrics. For negative evidence, sentences were sampled from the same document and extracted sentence pairs that have at least 4 content words in common. Then from both the positive and negative evidence files, training set of **10,000** sentence pairs and a test set of **3,000** sentence pairs were created through random sampling.
  <p align="justify">
  
  
3. [MSRVID data](https://www.cs.york.ac.uk/semeval-2012/task6/data/uploads/datasets/): The dataset comprises pairs of sentences drawn from publicly available datasets which are given below.
 - [Microsoft Research Paraphrase Corpus](http://research.microsoft.com/en-us/downloads/607d14d9-20cd-47e3-85bc-a2f65cd28042/): 750 pairs of sentences.
 - [Microsoft Research Video Description Corpus](http://research.microsoft.com/en-us/downloads/38cf15fd-b8df-477e-a4e4-a4680caa75af/): 750 pairs of sentences. 
 - [SMTeuroparl: WMT2008 develoment dataset (Europarl section)](http://www.statmt.org/wmt08/shared-evaluation-task.html): 734 pairs of sentences.
 The sentence pairs have been manually tagged with a number from 0 to 5, as defined below (cf. Gold Standard section).

4. [Image Annotation](http://dl.acm.org/citation.cfm?id=1866717)
 - [Pascal Dataset](http://nlp.cs.illinois.edu/HockenmaierGroup/pascal-sentences/index.html): 1000 Images with 5 different sentences describing the corresponding image. 
 - [Flicker Dataset](http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html): 7678 Images from Flicker with 5 different sentences describing the corresponding image.
