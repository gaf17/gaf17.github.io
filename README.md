Discovered potential biomarkers for Uterine and Cervical Cancers using classification algorithms and feature importance via SHAP

Capstone Report for MS Data Science & Artificial Intelligence

https://gaf17.github.io/
 
Discovering Potential Biomarkers for Uterine and Cervical Cancers with Machine Learning

Advisor: Dr. Giri Narasimhan
Mentor: Dr. Ananda Mondal

Abstract

Genetic information holds the power to give insight into the human condition- even revealing potential for illness in the future. Oncogenes, or cancer-causing genes, and tumor suppressing genes have been found to play a major role in the development of cancer. Though the scientific and medical communities are aware of several biomarkers for cervical and uterine cancers, this does not indicate the impossibility of others existing. In this study, three datasets of copy number alterations (CNA), gene expression, and microRNA (miRNA) were used, as well as one combined dataset consisting of overlapping participants in all three datasets. These data were used to find previously discovered biomarkers as well as new potential biomarkers via the feature importance of several different types of classification models. The important features were determined by SHapley Additive exPlanations, or SHAP values. The values were indicative of the overall impact of each feature for the model. Overall, it was found that among the three datasets, the most promising evidence was found in that of the miRNA, HtSeq, and Combined datasets. hsa-mir-224, ENSG00000269899.1, ENSG00000274501.1, ENSG00000215267.7, ENSG00000215030.5, ENSG00000225131.2, and ENSG00000128228.4 were found as potential candidates for biomarkers. While these are currently biomarkers for either other cancers or no cancers at all, further research may be able to reveal their impact on the cancers evaluated in this study, uterine and cervical.

Introduction

Though there have been a number of breakthroughs in recent years regarding cancer treatment, research must continue in order to find more effective methods of treatment and eventually a cure. While oncogenes were discovered over 50 years ago [36], they still remain a vital subject of interest in the search for better cancer treatments and prevention. A particular method of study involves the use of genetic information in order to determine a person’s predisposition for developing specific types of cancer. This predisposition for disease was coined as a ‘biomarker’. In machine learning, classification models can be used to represent the difference between classes in a dataset.

Machine learning models are able to recognize patterns in data that would not have been discoverable with the simple naked eye. They can then use this large amount of data to make predictions. Features refer to the columns in a dataset, or the categories of data of a sample. Though all of the features are considered when training a machine learning model, some contribute to the model in different strengths. Therefore by determining the feature importance of each feature in these genetic datasets, potential biomarkers for each cancer are revealed. These feature importances were determined using SHAP values, an explainable AI approach to feature importance [28]. These values are calculated using a game theory technique that calculates the contribution of each feature to the final prediction. This technique is particularly valuable in that it is model-agnostic [28], meaning that it can be used to interpret any machine learning model, regardless of the type. Though many models are black-box, SHAP is able to give insight into what is actually going on behind the scenes.

Cervical cancer (CESC) and endometrial uterine cancer (UCEC) are two cancers that negatively impact the lives of women across the globe. As with most cancers, a diagnosis such as this results in a complete upheaval of one’s life. From surgery, to radiation, to even chemotherapy, the detriment to a person’s well-being is impossible to deny. In the US alone, there were 11542 new cases of cervical cancer and 4272 deaths from 2016 to 2020. Within that same time frame, 54744 people were diagnosed with and 11995 died of uterine cancer. Learning more information about cancers affecting the female reproductive system could lead to a mitigation of death by early detection or total prevention via biomarkers. Because these cancers have high rates of metastasis — the spreading of cancer to another part of the body — with each other due to proximity, they were chosen as the subjects of study for this experiment.

Packages Used

• pandas

• numpy

• matplotlib.pyplot

• SHAP

• tensorflow.keras

• scikit-learn

Classification Algorithms Used

• Logistic Regression

• Linear Discriminant Analysis

• Classification and Regression Tree

• eXtreme Gradient Boosting Classifier

• Adaptive Boosting Classifier

• Extra Trees Classifier

• Passive Aggressive Classifier

• C-Support Vector Classification

• Stochastic Gradient Descent Classifier

• Random Forest Classifier

• Bagging Classifier

• K Neighbors Classifier

• Gaussian Naive Bayes Classifier

• Calibrated Classifier

• Multi-layer Perceptron Classifier

• Ridge Classifier

• Nearest Centroid Classifier

Code
https://github.com/gaf17

Datasets

Three types of data were used in this study. Gene expression data, copy number alteration data (CNA), & microRNA (miRNA) data, were all sourced via the XenaBrowser, which collected the data from The Cancer Genome Atlas (TCGA) that was published in 2019 [33]. The gene expression data was in the form of high-throughput sequencing data, or Htseq [18]. There were 892 samples and 60483 features. The features in this dataset represented the coding, non-coding, and pseudogenes. The CNA data was in the form of binary GISTIC values where ‘-1’ represented significant evidence of a copy number deletion, ‘0’ represented no evidence of a copy number alteration, and ‘1’ represented significant evidence of a copy number amplification. There were 845 samples and 19729 columns. The columns represented the number of only protein-coding genes that we have in our genome as humans. The miRNA values were quantified by signal intensity of the stem loop expression. There were 887 samples and 1881 columns. The miRNA dataset was quantified by the signal intensity of miRNA only, with no other non-coding genes. Finally, the last dataset consisted of the ‘combined’ data that existed in all three categories. There were a total of 824 samples and 82093 concatenated features of all datasets in this final dataset.

Methodology

In order to accomplish the goal of finding the biomarkers, several processing steps were implemented before beginning. Originally, the datasets were separated by cancer type. The uterine cancer dataset and cervical cancer datasets were originally separate, but were merged by adding a target feature label column to each sample in the datasets. The cervical cancer (CESC) target variable was labeled with a ‘1’ & the uterine cancer (UCEC) target variable was labeled with a ‘0’. The two datasets were then merged to represent both cancer types. This ensured binary simplicity for the classification task. After undergoing this process, the datasets were ready to begin being manipulated for the study. To start, each of the features were required to be standardized in order for the feature importance values to be used. This ensures that the model is not being influenced solely by values that have a large range. By giving the values of each feature a position on the same scale, we ensure that these can be justly compared down the line. This way, the feature importance is not overly influenced by large values.

. Machine Learning Analysis:The next step involved the use of a feature selection method called LASSO in order to reduce the number of features in the dataset. As the datasets given by TCGA contained thousands- and in some cases tens of thousands- of features, it was necessary to reduce that number in order to prevent the models from overfitting. Overfitting refers to a model becoming too tuned to the data that it has been trained on, which results in the model being unable to accurately predict results when given new, unseen data. LASSO reduces the number of features by identifying strongly correlation features that are redundant or irrelevant, without a significant loss of information when they are removed. This is done via linear regression that uses shrinkage, or the reduction of data values towards the mean (in this case, zero) in order to reduce the impact of unimportant features to nothing. In the case of this study, LASSO resulted in a feature reduction that left 131 out of 1881 features for the miRNA dataset, 77 out of 19729 features for the GISTIC dataset, 118 out of 60483 features for the Htseq dataset, and 152 out of 82093 features for the combined dataset. This significant reduction was intended to contribute to a lessening of overfitting for all models evaluated. Another control for overfitting was cross validation. In this study, a ten-fold cross validation was employed. Ten fold cross validation is a resampling method that takes ten different splits of the training data and testing data. The data is split into ten equal groups, or folds, where each fold takes a turn of being the one testing data fold, while the other nine folds are used as the training data. This is repeated until every fold has had an opportunity to be used as the sole testing data fold. The model is evaluated for each of the ten splits and the final accuracy is the average accuracy of all of the splits. This was repeated for each algorithm in the study, which are listed below.

Results

Each dataset was fit to the models with the stratified training set and evaluated using the testing set. The accuracies of each fold in the 10 fold cross validation were recorded and evaluated. By looking at the average accuracies for the ten folds, it was clear to see the overall effectiveness of the algorithm in predicting cancer types. For the HtSeq dataset, all algorithms performed either with perfect accuracy or near perfect accuracy. The classification algorithms in the GISTIC dataset had average accuracies in the 70s, with some dropping all the way to the high 60s. The miRNA dataset had average accuracies around the mid 90s for all classification algorithms. Finally, the Combined dataset was similar to the HtSeq dataset in that it had near perfect accuracy.

By looking at the boxplots, it was clear to see the underperforming algorithms. Though all of the classification models for the HtSeq and Combined datasets performed at around the same level, the miRNA and GISTIC datasets had underperformers. For the miRNA dataset, the Naive Bayes (NB) had a median accuracy of about 0.85, while several others were performing at least in the 90s. For this reason, it was determined that the value would be omitted for the remainder of the study. For the GISTIC dataset, the underperformer was the Passive Aggressive Classifier, or PAC. For almost all of the classification tasks, the model performed under 50% accuracy, which would be worse than even just random guessing.

Because of the variation in accuracies in the GISTIC and miRNA datasets, only the top performer was chosen to be analyzed for feature importance for the remainder of the study. On the other hand, because of the consistency in accuracy for the HtSeq and Combined datasets, it was determined that the feature importance would be decided by taking the average feature importance values for all of the classification algorithms.

Conclusions

Though there were several genes that were found as influential in the study, how does one decide which ones are actually impacting the cancers? It first starts with the evaluation. As displayed in the results, the accuracies for the GISTIC dataset were far from perfect and very clearly underperformed despite all of the cautions taken in order to minimize overfitting. Therefore, it would be improper to assume that the feature importances discovered from this dataset actually reflect the copy number alteration data in cancer patients. Though it is not ideal to have to ignore a full dataset, it is unlikely that there is valuable information to be gathered in that area regarding biomarkers. In the future, a larger dataset with more samples may lead to better results, even without any modification to the code.

Inversely, the other datasets did not have this problem. The highest-performing datasets in terms of accuracy were the HtSeq and Combined dataset, with perfect accuracies in nearly every model in all the cross validation folds. The miRNA dataset had relatively similar, but slightly lower accuracies around 99%. Given this performance, there is potential for all of the genes that have not been determined as biomarkers for UCEC or CESC to have some sort of impact on the illnesses. The most prominent in terms of future research would be hsa-mir-224, ENSG00000269899.1, ENSG00000274501.1, ENSG00000215267.7, ENSG00000215030.5, ENSG00000225131.2, and ENSG00000128228.4 due to their high SHAP values.
