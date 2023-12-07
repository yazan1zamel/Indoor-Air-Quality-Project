Hello! Welcome to the Exploratory Data Analysis Section! This section is focused on four different scripts: 
1. Visualization.pynb
2. Exploratory Data Analysis - Correlations.ipynb
3. Exploratory Data Analysis - NO2, CO2, & PM2.5.ipynb
4. Exploratory Data Analysis - Occupants' Activity.ipynb

For the visualization script, it focuses on developing visualizations based on zip code and location; these pieces of information are available in the "Home_Characteristic" dataset. 
There are 10 visualization utilized in the notebook:
- Relative Blue pinpoints for each zipcode based on price per square feet. 
- PM2.5 vs Price per Sq/ft & CO2 concentration vs Price per Sq/ft & NO2 concentration vs Price per Sq/ft & T_ETC vs Price per Sq/ft
- PM2.5 vs Total_Price & CO2 vs Total_Price & NO2 vs Total_Price & T_ETC vs Total_Price
- An interactive with all 70 house points with  markers to have useful information in it; this information include real estate pricing, price per square feet, and concentration of CO2, NO2, and PM2.5.

  <img width="1276" alt="Screen Shot 2023-12-06 at 8 39 45 PM" src="https://github.com/yazan1zamel/Indoor-Air-Quality-Project/assets/117310888/9a305485-ac04-4b7d-a94f-aee41b3f0e4b">

For the Correlations script, this section aims to characterize each home based on the IAQ parameters, namely PM2.5, CO2, NO2, Temperature, and ventilation rates in terms of infiltration rates and air change rate. This characterization is completed through the following:
- Use of python's computational features to compute average or mean of each of the 6 parameters for each of the 70 homes over a period of two years. The data is consulted from IAQ_monitoring and Airflow datasets.
   
- Data from the above step is summarized in a comprehensive table for future data analysis, as seen below.
   
   <img width="579" alt="Screen Shot 2023-12-06 at 8 26 14 AM" src="https://github.com/yazan1zamel/Indoor-Air-Quality-Project/assets/117310888/aba7e453-597c-4146-bfd2-47d76f9c119a">
   
- After computing the mean for each of the parameters, a correlation matrix has been developed on python. The output can be seen below.
   
   ![1](https://github.com/yazan1zamel/Indoor-Air-Quality-Project/assets/117310888/3f14eb9d-6d53-40d7-bd83-a2a060e53e6c)

- Use of python's computational features to determine outliers in the datasets; the mean was computed over 6 months instead of annually to account for seasonal variations.

   <img width="261" alt="Screen Shot 2023-12-06 at 9 54 52 PM" src="https://github.com/yazan1zamel/Indoor-Air-Quality-Project/assets/117310888/110f27ce-0ed2-488b-86e8-5fed2e5c4c73">

For the NO2, CO2, & PM2.5 script, the goal is to link the "Occupant_Activity" dataset with the IAQ dataset, namely PM2.5, NO2, & CO2. "Occupant_Activity" is a source of data that relates to various occupant activities, such as cooking. The goal of this python script is to extract the dates and home ID from the "occupant_Activity" dataset and link this with IAQ data from IAQ_Monitoring dataset at that same timestamp. In this way, for every timestamp present in "Occupant_Activity" dataset, there are three datapoints for IAQ: PM2.5, NO2, and CO2. This step is completed to allow the analysis at step 4. The output is called average_concentration_results.csv. 

For the fourth script in the Data Exploratory Analysis folder, the focus of this file is to analyze the "occupant_activity" results effectively. The outcome is to determine the impacts of duration of various occupants behavior, such as cooking, on the level on IAQ. 

- To do so, cleaning the dataset is vital. To provide context, in this dataset, there are 4,600 data points available, where each point provides information on occupancy and information on occupant behaviour such as time spent cooking, vacuuming, opening the window, grilling on a BBQ, using the oven, or using a humidifier, air purifier, or incense. The data available corresponds to a specific timeframe on a selected day for a specific home (e.g. 15/07/2017 from 7 am  to 11 am for HomeID 001). To illustrate the dataset we had available, the raw data in the Excel file can be seen below. 

<img width="1418" alt="Screen Shot 2023-12-05 at 11 19 28 AM" src="https://github.com/yazan1zamel/Indoor-Air-Quality-Project/assets/117310888/1af2daa1-7dfd-4e0e-a906-6f8d32d12f7c">

- To analyze the data effectively, the data is cleaned by removing rows that are at least 50% blank to avoid skewing the data. In addition, the timeframes are categorized into categories A (7 am –11 am ), B (11 am –1 pm), C (1 pm - 5 pm), D (5 pm - 9 pm), E (9 pm - 12 am), F (12 am - 7 am). This is because some of the timings were presented inconsistently, such as 1 pm at times or 13:00 at other times. 

- The timeframes are also originally presented in columns, as illustrated in the Figure above, and the data associated with those timeframes is challenging to analyze. The issue with this is that each timeframe and data associated with each timeframe does not have its own row, and accordingly, it is challenging to determine correlation in the data patterns. The data is transposed such that they are presented in rows, and the timeframes are now added as a separate column. The timeframes are presented as separate rows instead of adjacent columns as in the case of raw data. In addition, information on CO2 indoor concentrations and NO2 indoor concentrations are added to the dataset, effectively linking the data from IAQ CSV files.
  
- Furthermore, the additional events column was changed to a categorical column to assist in data analysis; events such as "humidifier", "air purifier", "air cleaning" have been labelled as "yes" to reflect positive implications on IAQ, where events such as "smoke", "wildfires", "fireplace" are labelled as "no" to denote negative implications on IAQ. The cleaner dataset can be seen below: 

<img width="1138" alt="Screen Shot 2023-12-05 at 11 26 10 AM" src="https://github.com/yazan1zamel/Indoor-Air-Quality-Project/assets/117310888/42e55860-60fa-4eaa-b834-4c3929544393">

- With this newly clean dataset, various visualizations can be performed via python to represent the data and highlight the impacts of occupant activities. A total of 18 visualizations were produced. One comprehensive visual outcome can be seen below: 

![compiled data 2](https://github.com/yazan1zamel/Indoor-Air-Quality-Project/assets/117310888/06021eb8-2d91-45ee-9e26-68d0469499d0)

- In addition, computational analysis was performed to identify mean and median for various occupants activity to assist in identifying impacts of occupant activity on indoor air quality. An example output can be seen below:
     - Example of mean:
       
  <img width="531" alt="Screen Shot 2023-12-06 at 9 31 06 PM" src="https://github.com/yazan1zamel/Indoor-Air-Quality-Project/assets/117310888/f1c899dd-0c2d-4661-837e-ee41d9f13e47">
  
     - Example of median:
       
     <img width="533" alt="Screen Shot 2023-12-06 at 9 30 43 PM" src="https://github.com/yazan1zamel/Indoor-Air-Quality-Project/assets/117310888/6292c2cd-ed5e-4338-a72b-42eb7bd53986">
