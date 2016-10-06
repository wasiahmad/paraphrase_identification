# Paraphrase-Identification-Task
Paraphrase detection is the task of examining two text entities (ex. sentence) and determining whether they have the same meaning. In order to obtain high accuracy on this task, thorough syntactic and semantic analysis of the two text entities is required.

## What is Paraphrase?
In simple words, paraphrase is just an alternative representation of the same meaning.

## Classification of Paraphrases
According to granularity, paraphrases are of two types.
  * Surface Paraphrases
    - Lexical level
        * Example - ***solve*** and ***resolve***
    - Phrase level
        * Example - ***look after*** and ***take care of***
    - Sentence level
        * Example - ***The table was set up in the carriage shed*** and ***The table was laid under the cart-shed***
    - Discourse level
  * Structural paraphrases
   - Pattern level
        * Example - ***[X] considers [Y]*** and ***[X] takes [Y] into consideration***
   - Collocation level
        * Example - ***(turn on, OBJ ligth)*** and ***(switch on, OBJ light)***

According to paraphrase style, they can be classified into five types.
  * Trivial Change
      * Example - ***all the members of*** and ***all members of***
  * Phrase replacement
      * Example - ***There will be major cuts in the salaries of high-level civil servants*** and ***There will be major cuts in the salaries of senior officials***
  * Phrase reordering
      * Example - ***Last night, I saw TOM in the shopping mall*** and ***I saw Tom in the shopping mall last night***
  * Sentence split & merge
      * Example - ***He baught a computer which is very expensive*** and ***(1) He bought a computer. (2) The computer is very expensive.***
  * Complex paraphrase
      * Example - ***He said there will be major cuts in the salaries of high-level civil servants*** and ***He claimed to implement huge salary cut to senior civil servants***

## Applications of Paraphrase Identification
  * Machine Translation
    - Simplify input sentences
    - Alleviate data sparseness
  * Question Answering
    - Question reformulation
  * Information Extraction
    - IE pattern expansion
  * Information Retrieval
    - Query reformulation
  * Summarization
    - Sentence clustering
    - Automatic evaluation
  * Natural Language Generation
    - Sentence rewriting
  * Others
    - Changing writing style
    - Text simplification
    - Identifying plagiarism

## Relevant Research Topic
  * Textual Entailment
  * Semantic Textual Similarity

## Research on Paraphrasing
  * Paraphrase identification
  * Paraphrase extraction
  * Paraphrase generation
  * Paraphrase applications
  
## Paraphrase Identification
  * Specially refers to sentential paraphrase identification
    - Given any pair of sentences, automatically identifies whether these two sentences are paraphrases

## Overview of Paraphrase Identification Methods
  * Classification based methods
    * Reviewed as a binary classification problem
    * Compute the similarities between two sentences at different levels which are then used as classification features
    * Previous works: [Brockett and Dolan, 2005](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/I05-50015B15D.pdf), [Finch et al., 2005](http://www.aclweb.org/anthology/I05-5003), [Malakasiotis, 2009](http://www.aclweb.org/anthology/P09-3004)
  * Alignment based methods
    * Align the two sentences and score the pair based on the alignment results
    * Previous works: [Wu, 2005](http://dl.acm.org/citation.cfm?id=1631867), [Das and Smith, 2009](https://www.aclweb.org/anthology/P/P09/P09-1053.pdf)

More discussion on the previous works are documented [here](https://github.com/wasiahmad/Paraphrase-Identification-Task/blob/master/state-of-art-details.md).
## Reference
  * [Paraphrases and Application - Association for Computational Linguistics](http://www.aclweb.org/anthology/C10-4001)
