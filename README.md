# Salary Predictions Based on Job Descriptions
## Part 1 - DEFINE
### ---- 1 Define the problem ----

Salary of a given position which is characterised through several features e.g. job type, required degree, 
years of experience etc. As it is expected that there are correlations and associations between these features 
and the salary of the corresponding position, it is possible to develop a model to predict salary value based on
given related features' values. 

To develop a model these are key issues relating to the data and models needed to be considered:
1. Are there correlation or association between given features and salary value? 
   How can we illustrated these ones? - through correlation coefficients incase of numerical 
   (what should be check in case of categorial data) / charts (scatter/line?)
2. Content and quality of data: which types of data are available (format, scale, numerical or categorical etc.)
   / consideration of required transformation (if applicable) or approach to handle missing values or technical 
   collecting problem / appropriate feature engineering techniques for these corresponding inputs - outputs.
3. Which models are approrpiate to this regression problems, especially in terms of limited computational resources 
   and large dataset, as well as the issues of hyperparameter tuning?
 
 Given data including three files:
 1. train_features.csv:this file contains eight characteristics of one million employed positions, 
                       including jobId, companyId, jobType, degree, major, industry, yearsExperience 
                       and milesFromMetropolis. jobId is used as an identification number of a position. 
 2. train_salaries: this file contains the salary of the corresponding one million positions in train_features.csv file.
                    There are two columns including jobId and salary. jobId is also used as an identification number of
                    a position.
 3. test_features.csv: this file share the same structure as the train_features.csv file with eight columns for eight
                       characteristics but with different one million positions. The final objective of this assignment is
                       to predict the salary of these positions based on given information in this file.

## Part 2 - DISCOVER

## Part 3 - DEVELOP
