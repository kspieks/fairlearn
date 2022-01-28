# Project Summary

During this microinternship, I investigated two main questions:

1) Is the ACSIncome dataset a good replacement for UCI Adult?
2) How to mitigation techniques impact model structure?

The sections below summarize the results. Additional detail can be found in the corresponding notebook. For more detail about ACSIncome, see the [documentation](https://fairlearn.org/main/user_guide/datasets/acs_income.html).

Note: the scope of this project focused on disparities between sex. Future work could explore other sensitive features, such as race.

# 1 Data Analysis
ACSIncome offers a few improvements over UCI Adult, such as 
- providing more datapoints (1,664,500 vs. 48,842)
- providing more recent data (2018 vs. 1994)
- disparities are present in the dataset, which give opportunities to apply unfairness mitigation techniques

One example of a disparity is that when holding occupation constant and examining just elementary and middle school teachers, the distribution of age, hours worked per week, and education level were nearly identical between males and females. Despite these similarities, males earned 18% more on average. It is unclear why this disparity exists, but it offers opportunities to apply and evaluate mitigation techniques.

## 1.1 Data Preparation
These notebooks prepared data for the ML models. The scope of this project examined the following features: age, hours worked per week, education level, marital status, occupation, and sex. Future work could explore use of the remaining features, like class of worker, place of birth, relationship to householder, and race.

`1.1_preprocess_data.ipynb` prepares data for LightGBMs and EBMs.
`1.1_preprocess_data_onehot.ipynb` creates one hot vector representation of the categorical variables in preparation for Logistic Regression.


# 2 Training Models
The task was binary classification to predict whether a person's income was above a given threshold, $35,000 in this case. Future work should explore how the results change when using different thresholds. Three different models were explored: EBMs, LightGBMs, and Logistic Regression. Data was split into 70% training, 15% validation, 15% testing. The validation set was used to determine the optimal probability threshold. Model interpretation is done on the test set. A random seed is used to ensure that the same data splits are used for all models.

## 2.1 Baseline Models
These notebooks trained models without any mitigation. Logistic Regression had the lowest disparity in the selection rate for the sensitive group (sex). However, all models show a large disparity relative to the models in other sections, which is expected since no mitigation was applied here. The models simply fit the historical bias i.e. being male gave a slight boost to the probability that a person's income was above the threshold while being female reduced the probability.

## 2.2 Models with Mitigation: Exponentiated Gradient
These notebooks used the demographic parity constraint with exponentiated gradient to train each of the three models. In all cases, mitigation worked as expected by reducing the disparity in selection rate between males and females. When examining how mitigation changed model structure, we observe that the correlation between sex and target variable has been reversed. Now, being male reduces the probability of belonging to the positive class while being female boosts the probability. More complex changes happened with the importance assigned to occupation, which was a result of certain occupations being dominated by either males or females.

## 2.3 Model with Mitigation: Dropping Sensitive Feature During Inference
This notebook trained a Logistic Regression model using sex as one of the inputs. After training, the weight corresponding to the input for sex was removed. The purpose of this mitigation approach is to allow the model to first fit the pattern of historical bias (as in Section 2.1) and then try to mitigate the unfairness by removing the bias associated with sex. Results show that the disparity in selection rate is smaller than the models from Section 2.1 and 2.4, but larger than the models in Section 2.2. This result is expected since exponentiated gradient (especially with a tight tolerance) reverses the importance of male and female to meet the constraint, while this approach cannot reverse the importance to make male have negative contribution and female have a positive contribution. Instead, this approach attempts to make the model neutral to sex by removing any associated bias during inference.

## 2.4 Models Trained Without Sensitive Feature
These notebooks trained models without using the sensitive feature of sex. Although the disparity in selection rate is smaller than the baseline models from Section 2.1, the disparity is larger than models from Section 2.2 and 2.3. This result is expected since the variables are somewhat correlated so the model can recover some of the historical bias from the other inputs. For example, teachers have 2.3x as many females as as males; unsurprisingly, the model lowers the probability of belonging to the positive class for this group. The effect is more pronounced for occupations with even higher percentages of females, like health services which have 4.9x as many females as males.

