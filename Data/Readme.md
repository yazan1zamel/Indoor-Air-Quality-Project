The dataset encompasses important information gathered from 70 houses, covering their mechanical equipment, participant survey results, air leakage, and airflow measurements. Additionally, it includes concentrations of pollutants measured by time-integrated passive samplers inside and outside the homes, details about cooktop and oven usage, the open state of external doors, and time-series data for air pollutants and environmental indicators measured both within and outside the houses.

Source: Chan, Wanyu et al. (2020). Data from: Indoor air quality in California homes with code-required mechanical ventilation [Dataset]. Dryad. https://doi.org/10.7941/D1ZS7X

Organization of Dataset 

Airflow:
- Contains time series data for monitored mechanical ventilation equipment, air infiltration rate estimates, and overall air exchange rate.
Each home is represented by a separate CSV file. Refer to "HENGH_Airflow_ReadMe" for additional details.
Ambient_PM:
- Includes a summary of PM2.5 data reported by ambient air monitoring stations closest to each study home.
The EXCEL file consolidates PM2.5 data from up to three regulatory monitoring sites, with a composite estimate calculated using an inverse distance weighing method.

Home_Equipment_Data:
- Comprises data about the house, encompassing basic characteristics, air leakage test results, and measured airflow rates of mechanical ventilation equipment.
- The EXCEL file contains data for all homes, accompanied by ReadMe information addressing the provided data and a note about data quality issues related to exhaust airflow measurements of over-the-range microwaves.

IAQ_Monitoring:
- Holds time-resolved air quality data, including estimated PM2.5 measured by photometry, carbon dioxide (CO2), nitrogen dioxide (NO2), formaldehyde (FRM), temperature (T), and relative humidity (RH).
- Each home is represented by a CSV file with 1-minute time-series data. Refer to "HENGH_IAQ_Monitoring_ReadMe" for data header definitions and information about data issues.

Occupant_Activity:
- Contains tabulated information derived from daily activity logs provided by study participants.
An EXCEL file, transcribed and independently spot-checked by staff, represents the data. The corresponding PDF file outlines the daily activity log used.

Occupant_Survey:
- Holds results from a survey about occupants, their general activities related to ventilation, and IAQ satisfaction.
An EXCEL file, transcribed by a staff member, includes survey data. For two homes that did not complete surveys, the data files indicate "No survey." Questions for the occupant surveys are available in both MS Word and PDF formats.
