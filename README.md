# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
"This dataset contains data about telemarketing in bank, and we seek to predict the most effective channels

The best performing model was given by a AutoML

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

I have chosen the pipeline that was in the introductory sample notebooks.
First I chose the standard compute cluster with 20 minutes time-out.  Then I assigned random batch sampler (x25 sized). The early termination policy is bandit policy with the evaluation interval of 2.
Estimator is a standard one suitable for compute target.
Regarding hyperparameter tuning I should probably choose the channels with most people coming to use banking.


## AutoML
I was note able to fully set up the Scikit-learn Model so I did not use AutoML, though I predict it's easier to set-up and all I will need to do is look up the best model in the UI.
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
I am pretty sure considering my Scikit learn random sampling set up and compute cluster time-out the AutoML decision will be more effective. But not necessarily  much better.
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
For Scikit-learn model you need a better understanding of business problem.
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
I was not able to finish the code so the compute cluster was not working. 
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**