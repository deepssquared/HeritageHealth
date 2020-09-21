# Predicting Hospital Admissions using Claims Data

#### *Please note that this project is still in progress*

## Introduction

Health informatics can carry significant impact with regards to costs and availability of services. The Heritage Health Competition was a past data competition hosted on Kaggle. Participants use available patient data to predict which patients are more likely to experience readmission. 

In this project, I use the past datasets to conduct data cleaning, exploratory data analysis, modeling, and appropriate predictive analysis.


## Data Wrangling 

The datasets were released via Kaggle in CSV formats. They contain many instances of incomplete cases and require extensive cleaning. The tables were pulled from a relational database, in which the member id is the primary field linking tables. Therefore, joins are required; the **members** and **target** tables have one-to-one relationships, they can be merged using left and/or inner joins. The **drugs** and **labs** tables have a one-to-many relationship with the member table, as they contain records on a yearly basis.


## Feature Selection

One aspect of this project, which may differ from how other participants approached the challenge, entails my experience as a hospital volunteer, a public health student, and later a research assistant. Based on this, rather than employing forward or backward stepwise model building, I will be deliberately selecting features that have documented impacts on health. 

One feature that I will be constructing is an SES categorical variable, derived from the pay delay field. Pay delays can be the result of financial hardship, as I've learned through first hand experience. Socioeconomic status is a key determinant of health and will therefore be included in model building.

