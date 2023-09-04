---
layout: distill
title: A Practical Foray into Network Analysis [Part 1]
date: 2023-09-03
tags: machine-learning network graph visualization distill
giscus_comments: false

authors:
  - name: Stephen Lu
    url: "https://matrixmaster.me"
    affiliations:
      name: McGill University

toc:
  - name: Motivation
  - name: Food-Centric View
    subsections:
        - name: Graph Visualization
        - name: Post Analysis
  - name: Compound-Centric View
  - name: Conclusion

bibliography: 2023-09-03-networks.bib
---

## Motivation
A majority of biological data is most suitably modelled as graphs. From the atom-bond model of small molecules to the residue-backbone structure of proteins to the complex interaction networks of signal transduction pathways, the possible configurations are endless. 

Recently, I've been particularly interested by these research areas incorporating network science and graph representation learning to tackle problems in biology and chemistry such as drug design <d-cite key="bengio2021flow"></d-cite> and material science engineering <d-cite key="duval2023faenet"></d-cite>.

In this series of blog posts, my goal is to document my learning process from the most basic principles in network science to advanced topics in graph neural networks through a hands-on application of techniques to interesting datasets that I can get my hands on.

In this part 1, I focus on graph data visualization using the Canadian [FooDB](www.foodb.ca) dataset covering detailed compositional, biochemical and physiological information about common food items.

## Food-Centric View
Given that the database documents the absolute enrichment of many compounds in common foods, a natural first approach is to take the food-centric view and ask whether certain compounds are especially enriched in some categories of food. To address this question in a visual way, let's try to construct a food-centric graph. Let $$G=(V,E)$$ denote our graph where each node $$v \in V$$ represents a food item (ex: strawberries) and $$\exists (u,v) \in E \iff$$ the foods $$(u,v)$$ share at least one compound. Then, for each undirected edge $$(u,v) \in E$$, we can add a normalized edge weight using the following function $$f(u,v)=\sum_{k\in K}{\frac{1}{S_k}}$$ where $$K$$ is the set of all compounds shared between foods $$(u,v)$$ and $$S_{k}$$ is the number of foods in which compound $$k$$ is found. This function $$f$$ penalizes the weight of compounds that are shared by too many foods. Finally, we make node size proportional to node degree which highlights foods that are highly interconnected in the graph by sharing many compounds with other foods.

### Graph Visualization
For visualization, let's use the network layout algorithm [ForceAtlas](10.1371/journal.pone.0098679) in the Gephi<d-cite key="ICWSM09154"></d-cite> software that takes into account edge weight to produce a graph where strongly connected components are clustered together. 

However, I quickly realized that doing this on the entire compound dataset yielded highly connected inexplicable graphs because there were many highly shared compounds that were practically found in all food items. Further, there were also a number of compounds that were only measured in a single food item, so discarding these wouldn't affect our visualization either. After removing these compounds, I still had too much data to process so I chose to only keep the observations for the 1500 least abundant compounds that were shared by at least one pair of food items. This yielded the following graph on all 1024 food items.

<div class="fake-img">
    {% include figure.html path="assets/img/blog/2023-09-03-network-analysis/by_food.svg" class="img-fluid rounded z-depth-0" zoomable=false %}
</div>
<div class="caption">
    Fig 1. Food-centric undirected graph where food item vertices and edges representing a normalized count of shared compounds between foods
</div>

As you can expect, the layout yields pretty well clusters for some major food groups such as fruits, spices, animal foods (meat products), and vegetables. I also learned that pulses are the seeds of dry legumes such as beans, lentils, and peas.

Looking at this graph, I saw that vegetables seem to cluster into two distinct families, so I decided to repeat the experiment, but only include foods from the vegetable family. This yielded the following graph:

<div class="fake-img">
    {% include figure.html path="assets/img/blog/2023-09-03-network-analysis/by_vegetable.svg" class="img-fluid rounded z-depth-0" zoomable=false %}
</div>
<div class="caption">
    Fig 2. Food-centric undirected graph for the vegetable food category
</div>

This graph shows that root vegetables, leaf vegetables, and tubers cluster together while fruit vegetables (tomato, pepper, etc.), onions, and mushrooms form their own distinct clusters.

### Post Analysis
Now that we have these food-centric graphs that seem to cluster food categories by their composition, I'd like to know which compounds are actually more enriched in which kinds of foods. To accomplish this, I'll group the foods by their category, then look for compounds that are relatively enriched in a given category. To choose an **enrichment** metric, I wanted to factor in the following considerations:

- Breadth: a compound should have higher enrichment value if it is found in a larger proportion of the individual foods of a food family
- Depth: a compound should have higher enrichment value if it has a higher concentration in a given food item of the family
- Normalization: a compound's enrichment in a family should be computed with respect to its enrichment in the other food families. In other words, enrichment is relative and not absolute.

To accomplish this, I first compute a local enrichment score $$S_{F,c}$$ for each compound $$c$$ in each food family $$F$$ using the following formula:

$$
S_{F,c} = \frac{1}{|F|} \sum_{f}[c]_f
$$

where $$\|F\|$$ is the number of foods in a food family, and $$[c]_f$$ is the concentration of compound $$c$$ in food $$f$$.

Then, I compute the final relative enrichment score $$S^r_{F,c}$$ by normalizing the local enrichment score against the scores of compound $$c$$ accross all food families $$F$$:

$$
S^r_{F,c} = \frac{S_{F,c}}{\sum_F S_{F,c}}
$$

Computing this metric accross all food categories, then sorting the values in non-ascending order yielded a list of the most relatively enriched compounds in each food family. I've summarized the most enriched compound in each food category in the table below. A value of 1.0 means the compound was only found in that food category.

| Food Category     | Most Enriched Compound    | $$S^r_{F,c}$$ |
| ----------------- | -------------------------:| -------------:|
| Animal Foods      | 4-Hydroxyproline          | 0.446359      | 
| Aquatic Foods     | Eicosapentaenoic acid     | 0.926140      |
| Baby Foods        | beta-Lactose              | 0.945600      |
| Baking Goods      | Caffeic acid ethyl ester  | 1.0           |
| Cocoa Products    | Theophylline              | 0.988216      |
| Coffee Products   | 4-Feruloylquinic acid     | 0.999425      |
| Eggs              | Arachidonic acid          | 0.730055      |
| Fats and oils     | Vaccenic acid             | 0.916681      |
| Fruits            | Cyanidin 3                | 1.0           |
| Gourds            | Kynurenine                | 0.971037      |
| Herbs and Spices  | Luteolin 7                | 1.0           |
| Milk Products     | D-Tryptophan              | 0.997252      |
| Nuts              | N-Dodecane                | 1.0           |
| Snack foods       | D-Galactose               | 0.387740      |
| Soy               | Formononetin              | 0.998967      |
| Tea               | Theaflavin                | 1.0           |
| Vegetables        | Isoorientin               | 1.0           |

### Relative Compound Enrichment at the Food Item Level 
Now that we've identified some commonly enriched compounds, it seemed interesting to me to flip the perspective and identify the individual foods that are most relatively enriched with respect to a set of target compounds of interest. To visualize this information, I once again built a per-compound graph where each node represents a food item, but this time I decided to draw an undirected edge between two food items if they belong to the same food category. Finally, node size is representative of the relative compound enrichment in each food item. This visualization allows us to quickly see which food items and food families are relatively enriched in each target compound.

Here is the graph for sugar compounds. As you would expect, the largest nodes correspond to foods such as *chocolate*, *candies* and some *fruits*. 
<div class="fake-img">
    {% include figure.html path="assets/img/blog/2023-09-03-network-analysis/by_sugars.svg" class="img-fluid rounded z-depth-0" zoomable=false %}
</div>
<div class="caption">
    Fig 3. Relative food enrichment graph for sugar compounds
</div>

Below, I've provided download links to the food enrichment graphs of some other compounds that I tested.

| Compound      | Link                                                                               | 
| ------------- | ----------------------------------------------------------------------------------:|
| Cholesterol   | [Download Link](/assets/img/blog/2023-09-03-network-analysis/by_cholesterol.svg)   |
| Lactose       | [Download Link](/assets/img/blog/2023-09-03-network-analysis/by_lactose.svg)       |
| Maltose       | [Download Link](/assets/img/blog/2023-09-03-network-analysis/by_maltose.svg)       |
| Nitrogen      | [Download Link](/assets/img/blog/2023-09-03-network-analysis/by_nitrogen.svg)      |
| Retinol       | [Download Link](/assets/img/blog/2023-09-03-network-analysis/by_retinol.svg)       |
| Sodium        | [Download Link](/assets/img/blog/2023-09-03-network-analysis/by_sodium.svg)        |
| Sucrose       | [Download Link](/assets/img/blog/2023-09-03-network-analysis/by_sucrose.svg)       |


## Compound-Centric View
Now that we've analyzed the food-centric view, the next step is to look at the compound-centric view where we consider networks where each node represents a different compound. One interesting graph that we can build is a compound signature graph $$G=(V,E)$$ for each food item $$f$$ where each vertex $$v\in V$$ corresponds to a compound that is measured in $$f$$ with vertex size proportional to the concentration/enrichment of compound $$v$$ in $$f$$, denoted $$[v]_f$$. Then, for each pair of vertices $$(u,v) \in V \times V$$, we define an edge $$(u,v,w) \in E$$ where $$w$$ is a weight metric that measures the co-occurence of compounds $$(u,v)$$ in different food items. Formally, 

$$
w_{uv} = \log_2 (S_{uv}+1)
$$

where $$S_{uv}$$ is the number of food items in which the compounds $$(u,v)$$ co-occur. Naturally, $$w_{uv}$$ is lower bounded by 0 in the above equation. Before visualizing the graph for each food item, we prune the vertices that have no expression in the food which leaves us with a fully connected graph. However, edges with weight 1 in this graph are not very interesting to keep because it simply tells us that this edge joins two compounds that only co-occur in the current food item. So, to create a better sparse visualization, we also prune edges with unit weight.

Here is the outcome for the **beer** food item.
<div class="fake-img">
    {% include figure.html path="assets/img/blog/2023-09-03-network-analysis/beer.svg" class="img-fluid rounded z-depth-0" zoomable=false %}
</div>
<div class="caption">
    Fig 4. Absolute compound enrichment in beer
</div>

Here are some additional graphs for **rice, chicken, and cow milk**. I personally think that this provides a neat way to rapidly visualize a compound signature for every food item. A cool project idea to do next would be to compute a measure of graph edit distance between these structures to get a sense of *how different two food items are based on their composition*.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/2023-09-03-network-analysis/rice.svg" class="img-fluid rounded z-depth-0" zoomable=false %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/2023-09-03-network-analysis/chicken.svg" class="img-fluid rounded z-depth-0" zoomable=false %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/2023-09-03-network-analysis/cow_milk.svg" class="img-fluid rounded z-depth-0" zoomable=false %}
    </div>
</div>
<div class="caption">
    Fig 5. Absolute compound enrichment in rice, chicken, and cow milk
</div>

## Conclusion
And that concludes the visualization work in this first part of the network analysis series. All I used to perform this analysis is some basic python code with the pandas and networkx libraries, as well as graph visualizations in Gephi. Although these graphs don't provide very rigourous answers to some of the questions that we asked, I think that their main value lies in their capacity to convey meaningful information about the relationships between a very large amount of data points in a way that is natural for us to reason about. This large scale view of the dataset yields many insights for further analysis that I will explore in the next part of this series.