import pandas as pd

# Load the dataset
data = pd.read_csv("Refinitiv ESG Final Data for Analysis.csv")

# Delete rows containing the value 'Unknown' 
data = data[~data.eq('Unknown').any(axis=1)]

# Columns to clean
columns_to_clean = [
    'CSR Strategy Score', 'Community Score', 'ESG Combined Score', 'ESG Controversies Score', 'ESG Score',
    'Emissions Score', 'Environment Pillar Score', 'Environmental Innovation Score', 'Governance Pillar Score',
    'Human Rights Score', 'Management Score', 'Product Responsibility Score', 'Resource Use Score',
    'Shareholders Score', 'Social Pillar Score', 'TRDIR Controversies Score', 'TRDIR Diversity Score',
    'TRDIR Inclusion Score', 'TRDIR People Development Score', 'TRDIR Score', 'Workforce Score',
    'Total Returns', 'Standard Deviation'
]

# Convert all columns in columns_to_clean to float type
for col in columns_to_clean:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Averaging 'Total Returns' and other scores for each unique value in the 'Symbol' column
columns_to_average = [
    'CSR Strategy Score', 'Community Score', 'ESG Combined Score', 'ESG Controversies Score', 'ESG Score',
    'Emissions Score', 'Environment Pillar Score', 'Environmental Innovation Score', 'Governance Pillar Score',
    'Human Rights Score', 'Management Score', 'Product Responsibility Score', 'Resource Use Score',
    'Shareholders Score', 'Social Pillar Score', 'TRDIR Controversies Score', 'TRDIR Diversity Score',
    'TRDIR Inclusion Score', 'TRDIR People Development Score', 'TRDIR Score', 'Workforce Score',
    'Total Returns', 'Standard Deviation'
]

# Group by 'Symbol' and calculate the mean for the specified columns
averaged_data = data.groupby('Symbol').agg({
    'Name': 'first',
    **{col: 'mean' for col in columns_to_average}
}).reset_index()

# Rename 'Total Returns' to 'Predicted Total Returns' and multiply values by 10 and add %
averaged_data['Predicted Total Returns'] = averaged_data['Total Returns'] * 10
averaged_data['Predicted Total Returns'] = averaged_data['Predicted Total Returns'].apply(lambda x: f"{x:.2f}%")

df = pd.DataFrame(averaged_data)
df.to_csv('preprocessed.csv', index=False)