df.head(5)

df.describe()
df['col'].describe()

df['col'].value_counts() # count number for each unique value

df.corr() # correlation matrix
df[["col1", "col2", "col3"]].corr() # correlation matrix