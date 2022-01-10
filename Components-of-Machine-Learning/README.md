# Jupyter Notebooks

**Pandas Data Frames**

```
import pandas as pd

# Create dictionary
mydict = {'animal':['cat', 'dog','mouse','rat', 'cat'],
         'name':['Fluffy','Chewy','Squeaky','Spotty', 'Diablo'],
         'age, years': [3,5,0.5,1,8]}

# Create dataframe from dictionary
df = pd.DataFrame(mydict, index=['id1','id2','id3','id4','id5'])

# Access row by name with .loc 
print(df.loc['id1'])

# Access row by index with .iloc 
print('\n', df.iloc[0])

# Access column by name with .loc 
print(df.loc[:,'animal'])

# Accsss column by name without .loc 
print('\n', df['animal'])

# Access column by index with .iloc 
print('\n', df.iloc[:,0])

# Loading from .csv file by using DataFrame
df = pd.read_csv('R0_data/Data.csv')

print("Shape of the dataframe: ",df.shape)
print("Number of dataframe rows: ",df.shape[0])
print("Number of dataframe columns: ",df.shape[1])

# Print first 5 rows 
df.head()


```



