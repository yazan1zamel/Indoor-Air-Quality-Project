import pandas as pd

# File paths
housing_price_file = r"C:\Users\Windows\OneDrive\Documents\Research\New folder\Dataset1\Housingprice.xlsx"
home_characteristics_file = r"C:\Users\Windows\OneDrive\Documents\Research\New folder\Dataset1\Home_Equipment_Data\HENGH_Study_Home_Characteristics.xlsx"

# Read data
housing_data = pd.read_excel(housing_price_file)
home_characteristics_data = pd.read_excel(home_characteristics_file, sheet_name='HENGH_House_Data')

# Ensure correct column names are used
zip_column_housing = 'ZIP Codes' if 'ZIP Codes' in housing_data.columns else 'ZIPCodes'
zip_column_home_char = 'Zipcode' if 'Zipcode' in home_characteristics_data.columns else 'Zip Codes'
city_column_home_char = 'City' if 'City' in home_characteristics_data.columns else 'City Name'

# Process 'ZIP Codes' in housing_data: Extract first three digits
housing_data[zip_column_housing] = housing_data[zip_column_housing].apply(lambda x: str(x).split(',')[0][:3])

# Function to find price per sq/ft for a given city and zipcode
def find_price(city, zipcode):
    matches = housing_data[(housing_data[zip_column_housing].str.startswith(zipcode)) & (housing_data['City'] == city)]
    if not matches.empty:
        return matches['Price per Sq/ft'].iloc[0]
    return None

# Apply the function to each row in home_characteristics_data
home_characteristics_data['Price per Sq/ft'] = home_characteristics_data.apply(
    lambda row: find_price(row[city_column_home_char], str(row[zip_column_home_char])),
    axis=1
)

# Calculate total price
home_characteristics_data['Total_Price'] = home_characteristics_data['Price per Sq/ft'] * home_characteristics_data['FloorArea_sqft']

# Select required columns
final_data = home_characteristics_data[['Home_ID', zip_column_home_char, 'FloorArea_sqft', 'Price per Sq/ft', 'Total_Price']]

# Save to a new Excel file
output_file = r"C:\Users\Windows\OneDrive\Documents\Research\New folder\Dataset1\Merged_Housing_Data.xlsx"
final_data.to_excel(output_file, index=False)