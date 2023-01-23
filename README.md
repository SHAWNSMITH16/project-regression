# ------Determining Prices

### ------Project Description

Using the zillow database given, there are numerous factors that could affect the proposed value of a home.  

### ------Project Goal
Discover drivers of determining home values.
Use drivers to develop a machine learning model to predict home values.
This information could be used to determine how much a house could or should cost based on current market

### ------Initial Thoughts
My initial hypothesis is that the size of homes will affect how much the home is valued at. 

### ------The Plan
Aquire data from SQL Server

Prepare data

Create Engineered columns from existing data
research and familiarize myself with the data
determine columns that do not fit into my initial exploratory thoughts and get rid of them
check for null values
tidy the data (make sure it fits the 4 key components of tidy data - tabular/one value per cell/
               each observation is one and only one row/each variable is one and only one column)


##### Answer the following initial questions
- Does the amount of bedrooms affect the value of a house?

- Does the amount of bathrooms affect the value of a house?

- Does the amount of square footage affect the value of a house?

Use drivers identified in explore to build predictive models of different types
Evaluate models on train and validate data
Select the best model based on highest accuracy
Evaluate the best model on test data
Draw conclusions

### ------Data Dictionary
- Feature -               - Definition -
bath                      - tells how many half and full bathrooms in the house
bed                       - tells how many bedrooms in the house
sqft                      - tells the total square footage of the house itself
fin_sqft                  - tells finished square footage of the house itself
fips                      - tells the county code of the property 
full_bath                 - tells the amount of full bathrooms only
lotsize                   - tells the size of the entire property lot in square feet
zip                       - tells what zip code the house is located within the county
rooms                     - tells how many total rooms there are in the house
yearbuilt                 - tells you the year the house was built
taxvaluedollarcnt         - tells you the value of the home

### ------Steps to Reproduce
Clone this repo.
Acquire the data from Codeup SQL server
Put the data in the file containing the cloned repo.
Run notebook.

### ------Takeaways and Conclusions
* The final model outperformed the baseline by about $40K. 

* Common sense might have you believe that size alone could be a good indicator of how much a home could be valued at. While this was the best indicator here, there are numerous other factors to be considered. 

### ------Recommendations

* Don't solely look at the size of the home as a way to gauge the value of homes. It can serve as a good launching pad but would be best utilized when coupled with other factors.

* Quantities of bedrooms and bathrooms kind of coincides with teh size of the home as you have to have the space avaialable to fit the rooms, but having more bathrooms did have a slight indication that homes were more valuable the more bathrooms it had


### Next Steps
* I would combine some features and create new columns such as combining garage, hottub, pool, fireplace, single story and explore from there

