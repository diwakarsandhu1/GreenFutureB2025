import pandas as pd
import sys

import os

# Set pandas options to display all rows and columns without truncation
pd.set_option('display.max_rows', None)  # To display all rows
pd.set_option('display.max_columns', None)  # To display all columns
pd.set_option('display.max_colwidth', None)  # To display full column content without truncation
pd.set_option('display.expand_frame_repr', False)  # To avoid breaking large dataframes into multiple lines

data = pd.read_csv("data_science/preprocess.csv")

environmental = 1
social = 1
governance = 1

if(len(sys.argv) == 4):
    environmental = float(sys.argv[1])
    social = float(sys.argv[2])
    governance = float(sys.argv[3])


# User responses to the questionnaire 
user_preferences = {
    'Environmental': environmental,  # Based on Question 1 and 3
    'Social': social,         # Based on Question 2 and 4
    'Governance': governance      # Based on Question 5
}

# Normalizes user preferences (so they sum to 1)
total_weight = sum(user_preferences.values())
weights = {key: value / total_weight for key, value in user_preferences.items()}

# Function to calculate ESG weight multiplier based on user preference
def calculate_esg_weight_multiplier(user_preference):
    if user_preference == 3:
        return 0
    else:
        return user_preference - 2 

# Calculates the weights for each category
esg_weight_multipliers = {
    'Environmental': calculate_esg_weight_multiplier(user_preferences['Environmental']),
    'Social': calculate_esg_weight_multiplier(user_preferences['Social']),
    'Governance': calculate_esg_weight_multiplier(user_preferences['Governance'])
}

# Calculates a weighted ESG risk score for each company based on user responses to questionnaire
data['Weighted ESG Risk Score'] = (
    data['Environment Risk Score'] * weights['Environmental'] * esg_weight_multipliers['Environmental'] +
    data['Social Risk Score'] * weights['Social'] * esg_weight_multipliers['Social'] +
    data['Governance Risk Score'] * weights['Governance'] * esg_weight_multipliers['Governance']
)

# Calculates a final score based on both the ESG risk score and the growth estimate
# Better (higher scores) involve higher growth estimates and lower risk
data['Final Score'] = (data['Growth Estimate (+5 years)']*10) - (data['Weighted ESG Risk Score'])

# Sorts the companies by the final score 
sorted_data = data.sort_values(by='Final Score', ascending=False)

# Shows the top companies that match user preferences
print(sorted_data[['Symbol', 'Name', 'Growth Estimate (+5 years)', 'Final Score']].head(10))

sys.stdout.flush()
