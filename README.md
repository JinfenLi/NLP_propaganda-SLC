# propaganda-SLC
*SHARED TASK ON FINE-GRAINED PROPAGANDA DETECTION @NLP4IF 2019*
> task link: https://propaganda.qcri.org/nlp4if-shared-task/index.html

> check our paper: https://www.aclweb.org/anthology/D19-5017.pdf

## Background
We refer to propaganda whenever information is purposefully shaped to foster a predetermined agenda. Propaganda uses psychological and rhetorical techniques to reach its purpose. Such techniques include the use of logical fallacies and appealing to the emotions of the audience. Logical fallacies are usually hard to spot since the argumentation, at first sight, might seem correct and objective. However, a careful analysis shows that the conclusion cannot be drawn from the premise without the misuse of logical rules. Another set of techniques makes use of emotional language to induce the audience to agree with the speaker only on the basis of the emotional bond that is being created, provoking the suspension of any rational analysis of the argumentation. All of these techniques are intended to go unnoticed to achieve maximum effect.

## Our Approach
Various propaganda techniques are used to manipulate peoples perspectives in order to foster a predetermined agenda such as by the use of logical fallacies or appealing to the emotions of the audience. In this paper, we develop a Logistic Regression-based tool that automatically classifies whether a sentence is propagandistic or not. We utilize features like TF-IDF, BERT vector, sentence length, readability grade level, emotion feature, LIWC feature and emphatic content feature to help usdifferentiate these two categories. The linguistic and semantic features combination results in 66.16% of F1 score, which outperforms the baseline hugely.

## How to Run
main function: configure the main function of logistic.py. Train our model first and evaluate on the dev or test dataset by configuring 'test' or 'dev'.
