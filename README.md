# Pool-based active learning for crowdsourcing word-sense disambiguation tasks

Word-sense disambiguation task is a task to resolve ambiguity: find out which of the possible meanings the phrase has in a particular context. An example of disambiguation task:

> Its use should be postponed in patients with **Sardinella siccus** affecting the stomach or gut.

> Does Sardinella siccus in this text mean a type of disorder or a living being?

There are 190 000 cases of ambiguous terms produced by automated text annotation tool. The goal is to resolve all of them. To train a classifier to perform such tasks labeled data is needed. A project is conducted at Computational Linguistics Lab of UZH to use crowdsourcing: Amazon Mechanical Turk workers are asked to solve such tasks:

![mantracrowd survey](http://davtyan.org/pml/screenshot.png)

As of now, tasks are being randomly picked from a pool of 190 000 ambiguous cases. Each of them is solved by at least 3 different workers. The goal of the project would be to implement active learning:

1. Have a classifier to predict phrase meaning from context (solve disambiguation tasks) 
1. Request MTurk workers to solve tasks which are the most informative for training the classifier

## Data

Unlabeled data: ~195 000 disambiguation tasks

Labeled data: 

* 821 answers to 255 tasks (taken out of these 195 000) by MTurk workers. More answers can be easily retrieved if needed.
* Up to 16 million non-ambiguous annotations, which can be viewed as tasks with known answers to train the initial classifier

## Resources

![Resouces](http://davtyan.org/pml/resources.png)

1. Applying active learning to supervised word sense disambiguation in MEDLINE. Chen et al., 2012
2. Active Learning with Amazon Mechanical Turk. Laws et at., EMNLP, 2011 (link)
3. Adaptive Submodularity: Theory and Applications in Active Learning and Stochastic Optimization. Golovin and Krause, 2011
4. Near-optimal Batch Mode Active Learning and Adaptive Submodular Optimization. Chen and Krause, 2013

## Roadmap

1. Train the Naïve Bayes classifier to solve disambiguation tasks on non-ambiguous labeled data. Use bag of words or bag of bigrams on a context window around the ambiguous phrase. Experiment on what features work best. 
2. Simulate active learning setting:
  1. Pretend labeled data from MTurk is unlabeled and treat it a pool from which labels can be queried.
  2. Abstract from the fact that we have multiple answers to one task: take majority vote of answers. Only consider answers with agreement level larger than 50%. This way we pretend for every task we can query a definite (correct) label.
3. Use (adaptive) submodularity to implement pool-based active learning. Analyze classifier performance.
4. Conditional: Implement batch-mode version of the algorithm
5. Conditional: Switch from simulation to querying actual MTurk workers. Analyze classifier performance.

## Results

See the constantly updated [summary of results](results.org).
