# Indoor-Air-Quality-Project
This is a repository for our project on "Breathable Homes, Affordable Zones: Unveiling the Secrets of Indoor Air Quality and Housing Costs"
Thank you for taking the time to review our project!

Introduction 

Indoor air quality (IAQ) is a critical aspect of our living environments, and it has significant implications for our health, comfort, and well-being. Indoor air quality can be influenced by human activities, building materials, and level of ventilation. IAQ is assessed by analyzing various parameters, such as concentrations of carbon dioxide, nitrogen dioxide, and PM2.5.

Data

In this Data Science Project, we decided to investigate indoor air quality data for 70 homes in the State of California, published by a group researcher at the Lawrence Berkeley National Laboratory and Wichita State University. Our group has taken a step further to perform web scraping to gather data on the costs of the 70 houses under examination by using the zip code for each home and information pertaining to the size and year that each house was built. 

21,500 data points that relate to the measured indoor air quality over a period of two years, 2016–2017, are available; those data points are placed across 70 CSV files, each file corresponding to a single home. Additional 18,100 measure points that relate to ventilation and airflow rates in the areas where IAQ data are collected are available; the data points are also placed in 70 CSV files. Information on the building characteristics (e.g. zip code, house size, house type), occupant activity (e.g. cooking, vacuuming, opening windows), and occupant survey are displayed in two separate Excel files. The survey contains information about the occupants' behavioural patterns, level of satisfaction from IAQ in different seasons, and racial information. There is a CSV file that includes information on outdoor air quality. 10% of the data was blank, and hence this data was cleaned.

Goals & Objectives

From an exploratory data analysis perspective, the goal is to:
- Identify the factors that affect indoor air quality. 
- Highlight the impact of ventilation rates on indoor air quality.
- Examine the impacts of various occupant activities, such as cooking, vacuuming, grilling, and occupancy on the indoor air quality. 
- Identify recommendations to enhance indoor air quality.
  
The web scrapped data and the data available are used in a machine learning model to:
- Develop a classification model to predict the cost category of a house (expensive, medium, and low) based on information on the indoor air quality and house characteristics.
- Develop a linear regression model to predict the indoor air quality of houses based on occupant perception of IAQ and housing prices.

Importance of the project

The project is useful since it brings insights on how do different factors affect indoor air quality, and it sheds light on ways to predict indoor air quality parameters by examining the housing prices and occupant perception of IAQ. 

How does the project work? 

There are three key folders uploaded in our repository: 
1. Exploratory Data Analysis: Investigates the correlation between different IAQ, ventilation, and building attributes; highlights which parameters impact IAQ dominantly 
2. Machine Learning Models: A classification and a linear regression model 
3. Web Scrapping Data: Housing Provides a new set of data to estimate the pricing of the 70 homes in the dataset; the pricing is a parameter that is not presented in our data. The real estate pricing is used in the machine learning model.
4. Data: The dataset encompasses important information gathered from 70 houses, covering their mechanical equipment, participant survey results, air leakage, and airflow measurements. Additionally, it includes concentrations of pollutants measured by time-integrated passive samplers inside and outside the homes, details about cooktop and oven usage, the open state of external doors, and time-series data for air pollutants and environmental indicators measured both within and outside the houses.

There is a readme page presented in each of the four folders to assist you in further understanding the codes, visualizations, and work presented. 
