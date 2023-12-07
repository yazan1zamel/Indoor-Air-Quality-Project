Hello! Welcome to the Exploratory Data Analysis Section! This section is focused on four different scripts: 
1. Visualization.pynb
2. Exploratory Data Analysis - Correlations.ipynb
3. Exploratory Data Analysis - NO2, CO2, & PM2.5.ipynb
4. Exploratory Data Analysis - Occupants' Activity.ipynb

For the visualization script, it focuses on developing visualizations based on zip code and location; these pieces of information are available in the dataset. 
There are 10 visualization utilized in the notebook:
- Relative Blue pinpoints for each zipcode based on Price Per Sq ft
- PM2.5 vs Price per Sq/ft & # CO2 vs Price per Sq/ft & NO2 vs Price per Sq/ft & T_ETC vs Price per Sq/ft
- PM2.5 vs Total_Price & CO2 vs Total_Price & NO2 vs Total_Price & T_ETC vs Total_Price
- An interactive with all 70 house points with  markers to have info about it.

For the Correlations script, this section aims to characterize each home based on the IAQ parameters, namely PM2.5, CO2, NO2, Temperature, and ventilation rates in terms of infiltration rates and air change rate. This characterization is completed through the following:
- Use of python's computational features to compute average or mean of each of the 6 parameters for each of the 70 homes over a period of two years. The data is consulted from IAQ_monitoring and Airflow datasets.
   
- Data from A is summarized in a comprehensive table for future data analysis, as seen below.
   
   <img width="579" alt="Screen Shot 2023-12-06 at 8 26 14 AM" src="https://github.com/yazan1zamel/Indoor-Air-Quality-Project/assets/117310888/aba7e453-597c-4146-bfd2-47d76f9c119a">
   
- After computing the mean for each of the parameters, a correlation matrix has been developed on python. The output can be seen below.
   
   ![1](https://github.com/yazan1zamel/Indoor-Air-Quality-Project/assets/117310888/3f14eb9d-6d53-40d7-bd83-a2a060e53e6c)

- Use of python's computational features to determine outliers in the datasets; the mean was computed over 6 months instead of annually to account for seasonal variations.

   <img width="261" alt="Screen Shot 2023-12-06 at 9 54 52 PM" src="https://github.com/yazan1zamel/Indoor-Air-Quality-Project/assets/117310888/110f27ce-0ed2-488b-86e8-5fed2e5c4c73">

For the NO2, CO2, & PM2.5 script, the goal is to link the "Occupant_Activity" dataset with the IAQ dataset, namely PM2.5, NO2, & CO2. "Occupant_Activity" is a source of data that relates to various occupant activities, such as cooking. The goal of this python script is to extract the dates and home ID from the "occupant_Activity" dataset and link this with IAQ data from IAQ_Monitoring dataset at that same timestamp. In this way, for every timestamp present in "Occupant_Activity" dataset, there are three datapoints for IAQ: PM2.5, NO2, and CO2. This step is completed to allow the analysis at step 4. 

For the fourth script in the Data Exploratory Analysis folder, 


   
