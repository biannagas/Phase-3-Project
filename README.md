# Predicting Vaccination Status Using Machine Learning Models
![image](https://github.com/biannagas/Phase-3-Project/assets/131709766/0291233b-5dc7-4a1c-97d4-0f38d433e240)
By Bianna Gas

## Business and Data Understanding
According to the Duke Global Health Institute, the probability of a pandemic with similar impact to COVID-19 is about 2% in any given year, and is predicted to grow three-fold in the next few decades.

With the risk of subsequent global pandemics and vaccine misinformation on the rise,  immunization campaigns are crucial to preventing loss of life and  financial impact. Analysis of New York City's Vaccine for All campaign during COVID found that the campaign saved around $28 billion in healthcare expenses and significantly reduced strain on the healthcare system. 

To help the New York State Department of Health best prepare future targeted vaccination campaigns, I used predictive modeling to explore what population factors have the strongest relationship with vaccination status. 

## Data 
In late 2009 and early 2010, the United States conducted the National 2009 H1N1 Flu Survey. This phone survey asked respondents whether they had received the H1N1 and seasonal flu vaccines, in conjunction with questions about themselves. These additional questions covered their social, economic, and demographic background, opinions on risks of illness and vaccine effectiveness, and behaviors towards mitigating transmission. The target population was people 6 months or older living in the United States, and NCHS surveyed 26,700 people.

https://www.drivendata.org/competitions/66/flu-shot-learning/page/211/

For all binary variables: 0 = No; 1 = Yes.

* h1n1_concern - Level of concern about the H1N1 flu.
* h1n1_knowledge - Level of knowledge about H1N1 flu.
* behavioral_antiviral_meds - Has taken antiviral medications. (binary)
* behavioral_avoidance - Has avoided close contact with others with flu-like symptoms. (binary)
* behavioral_face_mask - Has bought a face mask. (binary)
* behavioral_wash_hands - Has frequently washed hands or used hand sanitizer. (binary)
* behavioral_large_gatherings - Has reduced time at large gatherings. (binary)
* behavioral_outside_home - Has reduced contact with people outside of own household. (binary)
* behavioral_touch_face - Has avoided touching eyes, nose, or mouth. (binary)
* doctor_recc_h1n1 - H1N1 flu vaccine was recommended by doctor. (binary)
* doctor_recc_seasonal - Seasonal flu vaccine was recommended by doctor. (binary)
* chronic_med_condition - Has any of the following chronic medical conditions: asthma or an other lung condition, diabetes, a heart condition, a kidney condition, sickle cell anemia or other anemia, a neurological or neuromuscular condition, a liver condition, or a weakened immune system caused by a chronic illness or by medicines taken for a chronic illness. (binary)
* child_under_6_months - Has regular close contact with a child under the age of six months. (binary)
* health_worker - Is a healthcare worker. (binary)
* health_insurance - Has health insurance. (binary)
* opinion_h1n1_vacc_effective - Respondent's opinion about H1N1 vaccine effectiveness.
* opinion_h1n1_risk - Respondent's opinion about risk of getting sick with H1N1 flu without vaccine.
* opinion_h1n1_sick_from_vacc - Respondent's worry of getting sick from taking H1N1 vaccine.
* opinion_seas_vacc_effective - Respondent's opinion about seasonal flu vaccine effectiveness.
* opinion_seas_risk - Respondent's opinion about risk of getting sick with seasonal flu without vaccine.
* opinion_seas_sick_from_vacc - Respondent's worry of getting sick from taking seasonal flu vaccine.
* age_group - Age group of respondent.
* education - Self-reported education level.
* race - Race of respondent.
* sex - Sex of respondent.
* income_poverty - Household annual income of respondent with respect to 2008 Census poverty thresholds.
* marital_status - Marital status of respondent.
* rent_or_own - Housing situation of respondent.
* employment_status - Employment status of respondent.
* hhs_geo_region - Respondent's residence using a 10-region geographic classification defined by the U.S. Dept. of Health and Human Services. Values are represented as short random character strings.
* census_msa - Respondent's residence within metropolitan statistical areas (MSA) as defined by the U.S. Census.
* household_adults - Number of other adults in household, top-coded to 3.
* household_children - Number of children in household, top-coded to 3.
* employment_industry - Type of industry respondent is employed in. Values are represented as short random character strings.
* employment_occupation - Type of occupation of respondent. Values are represented as short random character strings.

## Exploratory Data Analysis 
<img width="336" alt="image" src="https://github.com/biannagas/Phase-3-Project/assets/131709766/be2c189d-528a-4d30-8c6a-6bb11e634004">

We can see that roughly only 20% of all survey respondents received the H1N1 vaccine.

<img width="299" alt="image" src="https://github.com/biannagas/Phase-3-Project/assets/131709766/b60cdf5a-b602-497a-b2a3-b6157ae90e19">

As the level of concern and level of knowledge grows for an individual, so does the likelyhood that they received the vaccine.

<img width="311" alt="image" src="https://github.com/biannagas/Phase-3-Project/assets/131709766/466a478b-35dd-4041-870c-7ac9aae45a1a">

Those with health insurance were significantly more likely to become vaccinated.

<img width="359" alt="image" src="https://github.com/biannagas/Phase-3-Project/assets/131709766/63f59043-4a3e-491e-b810-c0d28aeb4d03">

Employed individuals were more likley to be vaccinated compared to unemployed individuals and those not currently in the labor force.

## Modeling
The model with the highest ROC-AUC curve score and the highest precision score was a Gradient Boosting Model with tuned hyperparameters and oversampling of the majority class using SMOTE. 

<img width="247" alt="image" src="https://github.com/biannagas/Phase-3-Project/assets/131709766/ca587545-cdca-49cb-a995-9d6498296f05">


The AUC-ROC curve measures the ability of the classifier model to distinguish between classes. The final model's AUC score was 86%.

<img width="316" alt="image" src="https://github.com/biannagas/Phase-3-Project/assets/131709766/397b0e66-3be0-4e21-a910-927354639bd6">


Precision is the measure of how many patients were correctly predicted as being vaccinated out of all of the patients that were actually vaccinated. Precision was chosen as one of the metrics because it is far worse for the model to flag someone as vaccinated when they were not, as opposed to flagging them as unvaccinated when they were. The final precision scores were 88% for unvaccinated and 68% for vaccinated.

<img width="326" alt="image" src="https://github.com/biannagas/Phase-3-Project/assets/131709766/296705ec-d075-469f-afc1-ec158f5adbe6">

## Feature Importance 
The model determined determined the factors most responsible for predicting if an individual will receive an H1N1 vaccine were:
* Employment Status
* Income Level
* Education Level 
* MSA (the individuals residence within an area that has a population of at least 100,000)
* Whether or not the individual has health insurance
* Level of knowledge about the H1N1 flu
* Level of concern about the H1N1 flu

<img width="334" alt="image" src="https://github.com/biannagas/Phase-3-Project/assets/131709766/eaee3c9a-6630-494a-9adb-6bb27df0f04a">

## Recommendations 

1.  Gear future information campaigns and provide affordable access to vaccines for individuals from the following demographic groups:
     * Unemployed Individuals
     * Income below $75,000
     * No college degree
     * Non-MSA Resident
     * No Health Insurance
2.  Increase awareness about the risks of future virus's and effectiveness of vaccines to increase knowledge of the virus and level of concern surrounding the virus.

## Next Steps
* The models were built using survey data from 2009-2010. More recent data from the COVID pandemic would make the model more relevant and reflect how attitudes towards vaccination has changed over time.
* The dataset is extremely imbalanced in terms of race with nearly 80% of all respondents being White. Underepresentation of people of color has important implications as the COVID-19 pandemic had a disproportionate impact on communities of color. Future data collection efforts should seek to address this imbalance.
* Certain features like employment industry, employment occuation and hhs_geo_region were encoded, so it's uncertain if these features could have had an effect on the models or provided any usefull insights.











  
