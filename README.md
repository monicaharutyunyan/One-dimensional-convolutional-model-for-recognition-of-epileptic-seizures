# Recognition of Epileptic Events in the Electroencephalogram

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://github.com/monicaharutyunyan/One-dimensional-convolutional-model-for-recognition-of-epileptic-seizures.git)


## Abstract

Millions of people throughout the world suffer with epilepsy, a common neurological illness characterized by unpredictable and repeated seizures. Electroencephalogram (EEG) data must be manually analyzed, which takes time, is subjective, and is not standardized. In this study proposal, we offer a unique method for automatically identifying epileptic seizures in rat EEG recordings using a one-dimensional convolutional neural network (CNN).

EEG data from rat models of epilepsy are imported and preprocessed as part of the research effort. To get the data ready for the CNN model, it will be transformed. The CNN architecture will be created to learn directly from the unprocessed time series data, doing away with the requirement for manual feature extraction and utilizing deep learning's capability.

The preprocessed EEG data will be used to train and fine-tune the CNN model. The effectiveness of its operation will be assessed by contrasting the outcomes of automated seizure detection with those of manual inspection. The suggested study uses CNNs' capability for sequence classification to get around the drawbacks of existing approaches.

A reliable and effective technique for the automatic detection of epileptic seizures in rodent EEG recordings will be made available if the CNN model is implemented successfully. This invention will greatly speed up and standardize preclinical investigations in epilepsy research, enabling the assessment of possible antiepileptic medications and the creation of cutting-edge intervention strategies.

This study is innovative in that it applies CNNs exclusively to the detection of seizures in mouse EEG data. Human EEG analysis could be used in conjunction with the study's findings and techniques to enhance epilepsy diagnosis and treatment in clinical settings.

The overall goal of this research effort is to advance the field of epilepsy by presenting a novel automated seizure detection method. This research has the potential to influence the development of new treatment approaches and enhance understanding of epilepsy in both animal models and human patients by increasing the effectiveness and dependability of preclinical studies.
## Introduction and Background

According to the World Health Organization (WHO), epilepsy affects roughly 65 million people globally. Recurrent and unpredictable seizures characterize this chronic disorder, which poses serious difficulties for patients and their families and lowers quality of life. Epilepsy is thought to affect 1% of people worldwide, and both inherited and acquired factors might contribute to its development.

It is well known that effective management and therapy of epilepsy depend on understanding its genetic and physiological roots, even if the precise etiology of the disorder is still unclear in roughly 70% of instances. Analyzing epilepsy phenomena on biological models, particularly animal models such as rats and mice, provides valuable insights into the anomalies underlying the development of seizures and epileptogenesis at a systemic level.

Preclinical testing of possible antiepileptic medications, as well as the development and assessment of new antiepileptogenic agents and intervention techniques, are crucial components of investigating experimental models of epilepsy. The identification of spontaneous seizures and the evaluation of the acute and long-term consequences of therapies are both made possible by long-term continuous EEG recording, which is essential to this procedure. An automated technique is required since manual analysis of EEG data is time-consuming, prone to mistakes, and lacking in standardization.

Despite the development of various automated techniques for EEG classification and seizure detection, most of them are created for use with human EEG analysis. Only a few techniques are expressly designed for rats, and they are all model-specific. Due to the shortcomings of current automated methods, which include challenges in parameter selection, high false detection rates, and the sporadic omission of actual seizures, many investigators still rely on manual review.

The goal of this research proposal is to create a more trustworthy algorithm for automatically identifying epileptic seizures in EEG recordings in order to overcome these difficulties. Since convolutional neural networks (CNNs) can learn directly from unprocessed time series data without the requirement for manual feature extraction, they provide a potential method. CNNs may be modified to work with one-dimensional data sequences like EEG and have demonstrated good performance in complex activity recognition challenges.

### Data characteristics:

- time step: 3.90625000000000E-0003 s (frequency sampling 256 Hz)
- data size (in counts): Non_Seizures - (924222, 1), Seizures (1056939, 1)
- window size: 1599 samples


> The initial version of the network showed an unsatisfactory result of 67% correct recognition. However, after applying the procedure for standardizing the source data and selecting the optimal network learning rate, this indicator improved significantly. The resulting 50 model performs with an average accuracy of 96%, and the maximum accuracy is over 98%, which is an excellent result. Thus, our proposed and tested model based on a one-dimensional convolutional neural network allows us to distinguish epileptic seizures with a high degree of reliability.

