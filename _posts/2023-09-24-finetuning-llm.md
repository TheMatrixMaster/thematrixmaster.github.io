---
layout: post
title: LLM Finetuning w/ SMILES-BERT
date: 2023-09-24
tags: machine-learning llm nlp finetuning distill
giscus_comments: true
---

## Motivation
In a previous blog [post](/blog/2023/embedding-gpt), I introduced the concept of semantic embedding search to improve the performance of a large language model (llm), and provided the source code implenetation for a conversational retrieval Q&A agent in LangChain. Today, I want to explore the alternative method of improving llm performance through finetuning on a downstream sequence classification task. This method is arguably more powerful than prompt tuning since finetuning can modify the model's weights which parametrize its learned distribution over the dataset.

### SMILES: language of chemical structure
In the last blog post on [network analysis](/blog/2023/network-analysis-p1), I was working on the [FooDB](https://foodb.ca/) dataset which documents a large collection of compounds found in common foods. Each compound has a chemical structure represented by a data format known as [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system), short for simplified molecular-input line-entry system. In formal language theory, SMILES is a context free language and can be modelled by a context free grammar (CFG) operating on finite sets of non-terminal, terminal, and start symbols. Alternatively, SMILES can also be interpreted as the string obtained from a depth-first tree traversal of a chemical graph. This graph is preprocessed into a traversable spanning tree by removing its hydrogen atoms and breaking its cycles. Numeric suffix labels are added to the symbols where the cycle was broken, and parentheses are used to indicate points of branching in the tree. Thus, the same chemical graph may have multiple valid SMILES representations, depending on our choice of where to break cycles, the starting atom for DFS, and the branching order. By adding constraints to some of these choices, algorithms have been developed to output **canonical** SMILES formats which can be converted back into its molecular graph.

<div class="fake-img">
  {% include figure.html path="assets/img/blog/2023-09-24-finetuning-smiles/smiles-example.jpeg" class="img-fluid rounded z-depth-0" zoomable=true %}
</div>
<div class="caption">
    One possible SMILES string for 3-cyanoanisole. Notice that the choice of depth-first traversal highlighted on the right is one of many possible choices (Image source: <a href="https://www.researchgate.net/publication/261258149_Methods_for_Similarity-based_Virtual_Screening">Kristensen</a>, 2013)
</div>

### SMILES-BERT
Given the representational power of SMILES, attempts have been made to model this language through representation learning to capture the deep semantic knowledge contained in chemical graphs. Given that structure and function are intrinsically related to each other in biology, one would posit that learning the structural distribution of molecules could enable us to discover new relationships to function and other properties of interest. This leads us to [SMILES-BERT](https://dl.acm.org/doi/10.1145/3307339.3342186), a BERT-like model trained on SMILES strings through a masked recovery task and finetuned on three downstream tasks of chemical property prediction. 

<div class="fake-img">
  {% include figure.html path="assets/img/blog/2023-09-24-finetuning-smiles/smiles-bert.jpeg" class="img-fluid rounded z-depth-0" zoomable=true %}
</div>
<div class="caption">
    SMILES-BERT model architecture (Wang, 2019)
</div>

### Finetuning SMILES-BERT on FooDB compound library
Given my recent work on the FooDB compound library, which contains SMILES representations, I thought that it would be interesting to finetune the SMILES-BERT model on this dataset to further study the chemical properties tracked by the database. 

To start, I decided to run the zero-shot SMILES-BERT model over the sequences and compress the embeddings using PCA to visualize the amount of inherent partitioning learned by the masked pretraining task. Here are the results when I color the projected embeddings by **superclass**, **flavor**, **health effect**, and **ontology (function)**:

<div class="fake-img">
    {% include figure.html path="assets/img/blog/2023-09-24-finetuning-smiles/pca_compound_superklass.png" class="img-fluid rounded z-depth-0" zoomable=true %}
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/2023-09-24-finetuning-smiles/pca_compound_flavor_base5.png" class="img-fluid rounded z-depth-0" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/2023-09-24-finetuning-smiles/pca_compound_health_effects_top10.png" class="img-fluid rounded z-depth-0" zoomable=true %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/2023-09-24-finetuning-smiles/pca_compound_ontology_level4top10.png" class="img-fluid rounded z-depth-0" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/2023-09-24-finetuning-smiles/pca_compound_ontology_level5top10.png" class="img-fluid rounded z-depth-0" zoomable=true %}
    </div>
</div>

As you can see, SMILES-BERT clearly learns to differentiate some properties such as superclass and flavor, but struggles on some other categories like ontology.

Next, I wanted to actually finetune the model on these downstream classification tasks, then re-evaluate the model embedding projections on the held out test set to see if we can learn some non-linear transformations that actually enable us to better linearly separate these classes.

Notably, I used this [model](https://huggingface.co/seyonec/PubChem10M_SMILES_BPE_450k) from the huggingface modelhub and separated my dataset into (80, 10, 10) training, validation, and testing splits. I ran the finetuning over 5 epochs with a batch size of 16, learning rate of 2e-5, with 0.01 weight decay, and AdamW optimizer. The complete code can be found in the following github [repository](https://github.com/TheMatrixMaster/foodb-analysis).

### Results
As expected, the finetuning yields better results on the categories that already had good embedding separation from the PCA results we saw earlier. From the training and eval loss curves plotted below, we can see that the model tends to overfit on the health effects and flavors categories.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/2023-09-24-finetuning-smiles/train_loss_all.png" class="img-fluid rounded z-depth-0" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/2023-09-24-finetuning-smiles/eval_loss_all.png" class="img-fluid rounded z-depth-0" zoomable=true %}
    </div>
</div>

The best results I obtained were on the superklass category, and I think that this is because this categorization had the largest data to num_labels ratio, as well being fairly reliant on molecule structure which is captured by the SMILES strings. Here are the learning curves for this run where I obtained ~98% true accuracy on the held out test set.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/2023-09-24-finetuning-smiles/eval_loss_superklass.png" class="img-fluid rounded z-depth-0" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/2023-09-24-finetuning-smiles/eval_acc_superklass.png" class="img-fluid rounded z-depth-0" zoomable=true %}
    </div>
</div>

Finally, I reran this finetuned model on the raw SMILES strings to obtain embeddings and down-projected them using PCA. As you can see below, the model has clearly now learnt an embedding space that better separates the categories defined by the superklass label.

<div class="fake-img">
    {% include figure.html path="assets/img/blog/2023-09-24-finetuning-smiles/pca_compound_superklass_finetuned_top7.png" class="img-fluid rounded z-depth-0" zoomable=true %}
</div>

Compare this to the zero-shot embeddings below:

<div class="fake-img">
    {% include figure.html path="assets/img/blog/2023-09-24-finetuning-smiles/pca_compound_superklass.png" class="img-fluid rounded z-depth-0" zoomable=true %}
</div>
