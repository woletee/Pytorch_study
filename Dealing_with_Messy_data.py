#in this code shows how we can deel with the Exploration data analysis process(EDA)
import pandas as pd
data=pd.read_csv("energydata_complete.csv")
data=data.drop(columns=["date"])
cols=data.columns
num_cols=data._get_numeric_data().columns
list(set(cols)-set(num_cols))
data.isnull().sum()
import pandas as pd

# Create an empty dictionary to store the outlier percentages
outliers = {}

# Assuming 'data' is your DataFrame

for column_name in data.columns:
    # Calculate the mean and standard deviation for the current column
    mean = data[column_name].mean()
    std = data[column_name].std()
    
    # Define the lower and upper thresholds for outliers
    min_t = mean - (3 * std)
    max_t = mean + (3 * std)
    
    # Count the number of outliers in the current column
    count = ((data[column_name] < min_t) | (data[column_name] > max_t)).sum()
    
    # Calculate the percentage of outliers
    percentage = (count / data.shape[0]) * 100
    
    # Store the percentage in the 'outliers' dictionary
    outliers[column_name] = round(percentage, 3)

# Now 'outliers' contains the percentage of outliers for each column

# Display the outlier percentages for each feature
for column_name, percentage in outliers.items():
    print(f"Percentage of outliers in {column_name}: {percentage}%")

    #the above resulting dictionary displays a list of the featurs in the data set along with their outlirs
#from the above we can see that it is not important to care with the outliers since the % is less than 5 
#with his we hahve finished the data exploration phase
data.head()
#in the next step we will try to scale our data 
#the first step is to separate the featurs from the target
X=data.iloc[:,1:]
Y=data.iloc[:,0]
