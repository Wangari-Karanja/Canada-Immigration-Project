#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[ ]:





# In[2]:


df_can = pd.read_csv(r'C:\Users\HP\Documents\canadian_immegration_data.csv')
df_can


# In[3]:


df_can.info


# In[4]:


df_can.info(verbose=False)


# In[5]:


df_can.columns


# In[6]:


df_can.index


# In[7]:


df_can.columns.tolist()
df_can.index.tolist()


# In[8]:


print(type(df_can.columns.tolist()))
print(type(df_can.index.tolist()))


# In[9]:


#To view the dimensions of the dataframe

df_can.shape


# In[10]:


#We will also add a 'Total' column that sums up the total immigrants by country over the entire period 1980 - 2013, as follows:

df_can['Total'] = df_can.sum(axis=1)


# In[11]:


df_can.head(2)


# In[12]:


#Check how many null objects we have

df_can.isnull().sum()


# In[13]:


#Summary of a dataframe

df_can.describe()


# In[14]:


df_can.describe(include='all')


# In[15]:


df_can.Country


# In[16]:


#set index

df_can.set_index('Country', inplace=True)


# In[17]:


df_can.head(2)


# In[18]:


df_can.iloc[90]


# In[19]:


df_can.loc['Japan']


# In[20]:


print(df_can.columns)


# In[21]:


df_can.loc['Japan', '2001']


# In[22]:


df_can.loc['Kenya', '2009']


# In[23]:


df_can[['Continent']]


# In[24]:


print(dtypes(df_can.columns))


# In[ ]:


print(df_can.dtypes)


# In[ ]:


#check type of column heads

[print (type(x)) for x in df_can.columns.values]


# In[25]:


#let's declare a variable that will allow us to easily call upon the full range of years:

years = list(map(str, range(1980, 2014)))
years


# In[26]:


#Let's filter the dataframe to show the data on Asian countries (AreaName = Asia).

condition = df_can[['Continent']] == 'Asia'
print(condition)


# In[27]:


# 2. pass this condition into the dataFrame

df_can[condition]


# In[28]:


# we can pass multiple criteria in the same line.
# let's filter for AreaNAme = Asia and RegName = Southern Asia

df_can[(df_can['Continent']=='Asia') & (df_can['Region']=='Southern Asia')]


# In[29]:


#Eastern Africa

df_can[(df_can['Continent']=='Africa')&(df_can['Region']=='Eastern Africa')]


# In[30]:


#let's review the changes we have made to our dataframe.

print('Dataframe dimension:', df_can.shape)
print(df_can.columns)
df_can.head(2)


# In[31]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib as mpl
import matplotlib.pyplot as plt


# In[32]:


print(plt.style.available)
mpl.style.use(['ggplot'])


# In[33]:


years = list(map(str, range(1980, 2014)))
years


# In[34]:


#Question: Plot a line graph of immigration from Haiti using df.plot()


# In[35]:


haiti = df_can.loc['Haiti', years]
haiti.head()


# In[36]:


#plot line plot

haiti.plot()


# In[37]:


haiti.index = haiti.index.map(int) 
haiti.plot(kind='line')

plt.title('Immigration from Haiti')
plt.xlabel('Years')
plt.ylabel('Number of immigrants')
plt.show()


# In[38]:


haiti.plot(kind = 'line')

plt.title("Immigration from Haiti")
plt.xlabel('No of Immigrants')
plt.ylabel('Years')

#annonate the 2010 earthquake.
#syntax plt.text(x, y, label)

plt.text(2001, 6100, '2010 Earthquake')

plt.show()


# In[39]:


#Immigration from Ethiopia

Pakistan = df_can.loc['Pakistan', years]
Pakistan.head()


# In[40]:


Pakistan.plot()


# In[41]:


Pakistan.plot(kind='line')

plt.title('Immigration from Pakistan')
plt.xlabel('No of Immigrants')
plt.ylabel('Years')


plt.show()


# In[42]:


#Let's compare the number of immigrants from India and China from 1980 to 2010

df_CI = df_can.loc[['China', 'India'], years]
df_CI.head()


# In[43]:


df_CI= df_CI.transpose()
df_CI.head()


# In[44]:


df_CI.plot(kind='line')

plt.title('Immigrants from China and India')
plt.xlabel('No of Immigrants')
plt.ylabel('Country')

plt.show()


# In[45]:


#Compare the trend of top 5 countries that contributed the most to immigration to Canada.


# In[46]:


df_can.sort_values(by = 'Total', ascending=False, inplace=True)
df_top5 = df_can.head(5)
df_top5


# In[47]:


df_top5 = df_top5[years].transpose()
df_top5


# In[48]:


df_top5.plot(kind='line', figsize=(14,8))

plt.title('Immigration trend of top 5 countries')
plt.xlabel('No of Immigrants')
plt.ylabel('Years')


# In[49]:


#Area Plot

df_top5.plot(kind='area', stacked=False,figsize=(20,10))

plt.title('Immigration trend of Top 5 Countries')
plt.ylabel('Countries')
plt.xlabel('No of Immigrants')

plt.show()


# In[50]:


#Area Plot

df_top5.plot(kind='area',alpha=0.25, stacked=False,figsize=(20,10))

plt.title('Immigration trend of Top 5 Countries')
plt.ylabel('Countries')
plt.xlabel('No of Immigrants')

plt.show()


# In[51]:


df_bottom5 = df_can.tail(5)
df_bottom5


# In[52]:


df_bottom5=df_bottom5[years].transpose()
df_bottom5.head()


# In[53]:


#Area stacked plot

df_bottom5.plot(kind='area', alpha=0.45, stacked=False,figsize=(20,10))

plt.title('Immigration Trend for Bottom 5 Countries')
plt.ylabel('No of Immigrants')
plt.xlabel('Years')
plt.show()


# In[54]:


# Transparency value = 0.55

ax=df_bottom5.plot(kind='area', alpha=0.55, stacked=False, figsize=(20,10))
ax.set_title('Immigration Trend for Bottom 5 Countries')
ax.set_xlabel('Years')
ax.set_ylabel('No of Immigrants')


# In[55]:


df_can[['2013']].head()


# In[56]:


# np.histogram returns 2 values

count, bin_edges = np.histogram(df_can['2013'])

print(count) #frequency count
print(bin_edges) # bin ranges, default = 10 bins


# In[57]:


df_can['2013'].plot(kind='hist', figsize=(8,5))

plt.title('Histogram of Immigration from 195 Countries in 2013')
plt.xlabel('No of Immigrants')
plt.ylabel('No of Countries')

plt.show()


# In[58]:


count, bin_edges = np.histogram(df_can['2013'])

df_can['2013'].plot(kind='hist', figsize=(8,5), xticks=bin_edges)

plt.title('Histogram of Immigration from 195 Countries in 2013')
plt.xlabel('No of Immigrants')
plt.ylabel('No of Countries')

plt.show()


# In[59]:


#What is the immigration distribution for Denmark, Norway, and Sweden for years 1980 - 2013?


# In[60]:


df_can.loc[['Denmark', 'Norway', 'Sweden'], years]


# In[61]:


df_t = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()

df_t.plot(kind='hist', figsize=(10, 6))

plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.xlabel('Number of Immigrants')
plt.ylabel('Number of Years')

plt.show()


# In[62]:


#Let's make a few modifications to our graph

#increase the bin size to 15 by passing in bins parameter;
         #set transparency to 60% by passing in alpha parameter;
         #label the x-axis by passing in x-label parameter;
         #change the colors of the plots by passing in color parameter.


# In[63]:


#let's make xticks

count, bin_edges = np.histogram(df_t, 15)

print(count)
print(bin_edges)


# In[64]:


#Let's plot the graph

df_t.plot(kind='hist', alpha=0.60, bins = 15,xticks=bin_edges, figsize=(10,6), color=
          ['coral', 'darkslateblue', 'mediumseagreen'])

plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.xlabel('No of immigrants')
plt.ylabel('No of years')

plt.show()


# In[65]:


##Let's also adjust the min and max x-axis labels to remove the extra gap on the edges of the plot. 
#We can pass a tuple (min,max) using the xlim paramater, as shown below.


# In[66]:


count, bin_edges = np.histogram(df_t, 15)
xmax= bin_edges[0]-10     #  first bin value is 31.0, adding buffer of 10 for aesthetic purposes 
xmin=bin_edges[-1]+10     #  last bin value is 308.0, adding buffer of 10 for aesthetic purposes

df_t.plot(kind='hist', alpha=0.60, bins=15,
         xticks=bin_edges, figsize=(10,6),stacked=True,
         color=['coral', 'darkslateblue', 'mediumseagreen'],
         xlim=(xmin,xmax))

plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.xlabel('No of immigrants')
plt.ylabel('No of years')

plt.show()


# In[67]:


#Use the scripting layer to display the immigration distribution for Greece, Albania, and Bulgaria for years 1980 - 2013? 
#Use an overlapping plot with 15 bins and a transparency value of 0.35.

df_G = df_can.loc[['Greece','Albania','Bulgaria'], years].transpose()
df_G


# In[68]:


count, bins_edge = np.histogram(df_G, 15)
xmin=bins_edge[0] - 10
xmax=bins_edge[-1] + 10

df_G.plot(kind='hist',
          alpha=0.35,
         bins=15,
         xticks=bins_edge,
         stacked=True,
         xlim=(xmin,xmax),
         color=['darkorchid','darkred','darkviolet'])

plt.title('Histogram of Immigration from Greece, Albania, and Bulgaria from 1980 - 2013')
plt.xlabel=('No of Years')
plt.ylabel=('No of Immigrants')

plt.show()


# In[69]:


#The 2008 - 2011 Icelandic Financial Crisis was a major economic and political event in Iceland. 
#Relative to the size of its economy, Iceland's systemic banking collapse was the largest experienced by any country 
#in economic history. 
#The crisis led to a severe economic depression in 2008 - 2011 and significant political unrest.

#Let's compare the number of Icelandic immigrants (country = 'Iceland') to Canada from year 1980 to 2013.


# In[70]:


iceland = df_can.loc[['Iceland'], years].transpose()
iceland.head(3)


# In[71]:


#vertical bar chart

iceland.plot(kind='bar', figsize=(10,6))

plt.title('Iceland Immigration Trend')
plt.xlabel=('Year')
plt.ylabel=('No of Immigrants')

plt.show()


# In[72]:


iceland.index = iceland.index.map(int)

iceland.plot(kind='bar', figsize=(10,6), rot=90)

plt.title('Iceland Immigration Trend')
plt.ylabel=('No of immigrants')
plt.xlabel=('Year')

#annotate arrow
plt.annotate('',
            xy=(32,70),
            xytext=(28,20),
            xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3',color='blue',lw=2)
            )


#annotate text
plt.annotate('2008 - 2011 Financial Crisis', # text to display
            xy=(28, 30),                     # start the text at at point (year 2008 , pop 30)
            rotation=72.5,                   # based on trial and error to match the arrow
            va='bottom',                     # want the text to be vertically 'bottom' aligned
            ha='left')                       # want the text to be horizontally 'left' algned.


plt.show()


# In[73]:


#horizontal Bar Plot


# In[74]:


df_can.sort_values(by='Total', ascending=True, inplace=True)
df_top15 = df_can['Total'].tail(15).transpose()
df_top15


# In[75]:


df_top15.plot(kind='barh', figsize=(12, 12), color='steelblue')

plt.title('Top 15 Conuntries Contributing to the Immigration to Canada between 1980 - 2013')
plt.xlabel=('Countries')
plt.ylabel=('No of Immigrants')

#annotate value labels to each country
for index, value in enumerate(df_top15):
    label = format(int(value), ',')
    
# place text at the end of bar (subtracting 47000 from x, and 0.1 from y to make it fit within the bar)
plt.annotate(label, xy=(value - 47000, index - 0.1), color='violet')

plt.show()


# In[76]:


# Pie Chart

#Let's explore the proportion (percentage) of new immigrants grouped by continents for the entire time 
#period from 1980 to 2013.


# In[77]:


df_continents = df_can.groupby('Continent', axis=0).sum()


# In[78]:


print(type(df_can.groupby('Continent', axis=0)))


# In[79]:


df_continents.head(7)


# In[80]:


# Plot the pie chart
color_list=['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink']
explode_list=[0.1, 0, 0, 0, 0.1, 0.1]  # ratio for each continent with which to offset each wedge.

df_continents['Total'].plot(kind='pie',
                           figsize=(15, 6),
                           autopct='%1.1f%%',    # add in percentages
                           startangle=90,        # start angle - 90 Africa
                           shadow=True,
                           labels=None,          # turn off labels on pie chart
                           pctdistance=1.12,     #the ratio between the center of each pie slice and the start of the text generated by autopct
                           colors=color_list,
                           explode=explode_list)          # add shadow
plt.title('Immigration to Canada by Continent')
plt.axis('equal')

plt.legend(labels=df_continents.index, loc='upper left')

plt.show()


# In[81]:


#New immigrants in 2013

df_2013 = df_continents['2013']
df_2013.head(5)


# In[82]:


color_list = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink']
explode_list = [0.1, 0, 0, 0, 0.1, 0.1]

df_2013.plot(kind='pie',
             figsize=(15, 6),
             autopct='%1.f%%',
             startangle=90,
             shadow=True,
             labels=None,
             pctdistance=1.12,
             colors=color_list,
             explode=explode_list)

plt.title('Percentage of Immigrants in Canada in 2013')
plt.axis('equal')

plt.legend(labels=df_2013.index, loc='upper left')

plt.show()


# In[83]:


#Box Plots

#Let's plot the box plot for the Japanese immigrants between 1980 - 2013.


# In[84]:


df_japan = df_can.loc[['Japan'], years].transpose()
df_japan.head(4)


# In[85]:


#Plot the boxplot

df_japan.plot(kind='box',
             figsize=(8, 6))

plt.title('Box Plot of Japanese Immigrants in Canada')
plt.ylabel=('No of Immigrants')

plt.show()


# In[86]:


df_japan.describe()


# In[87]:


df_CI = df_can.loc[['China', 'India'], years].transpose()
df_CI.head(5)


# In[88]:


df_CI.plot(kind='box', figsize=(10, 6))

plt.title('Box Plot For China and India Immigration Trend')
plt.ylabel=('No of Immigrants')

plt.show()


# In[89]:


#Vertical Box plot

df_CI.plot(kind='box', figsize=(10, 6), vert=False)

plt.title('Box Plot For China and India Immigration Trend')
plt.ylabel=('No of Immigrants')

plt.show()


# In[90]:


#Subplots

fig = plt.figure()   #to create the figure

ax0 = fig.add_subplot(1,2,1)   # add subplot 1 (1 row, 2 columns, first plot)
ax1 = fig.add_subplot(1,2,2)   # add subplot 2 (1 row, 2 columns, second plot)

#subplot1 : Box Plot
df_CI.plot(kind='box', figsize=(20, 6), vert=False, color='blue', ax=ax0)
ax0.set_title('Box Plots of Immigrants from China and India (1980 - 2013)')
ax0.set_xlabel=('Number of Immigrants')
ax0.set_ylabel=('Countries')

#subplot2: Line Plot
df_CI.plot(kind='line', figsize=(20, 6), color='violet', ax=ax1)
ax1.set_title('Line Plots of Immigrants from China and India (1980 - 2013)')
ax1.set_xlabel=('Number of Immigrants')
ax1.set_ylabel=('Years')

plt.show


# In[91]:


#Create a box plot to visualize the distribution of the top 15 countries (based on total immigration) grouped by the decades 
#1980s, 1990s, and 2000s.


# In[92]:


#Step 1

df_top15=df_can.sort_values(['Total'], ascending=False, axis=0).head(15)
df_top15 


# In[93]:


#Step 2

years_80s = list(map(str, range(1980, 1990)))
years_90s = list(map(str, range(1990, 2000)))
years_00s = list(map(str, range(2000, 2010)))

# slice the original dataframe df_can to create a series for each decade
df_80s = df_top15.loc[:, years_80s].sum(axis=1)
df_90s = df_top15.loc[:, years_90s].sum(axis=1)
df_00s = df_top15.loc[:, years_00s].sum(axis=1)

# merge the three series into a new data frame
new_df = pd.DataFrame({'1980s':df_80s, '1990s':df_90s, '2000s':df_00s})

#display new DataFrame
new_df


# In[94]:


#To learn more about our new dataframe

new_df.describe()


# In[95]:


#Box Plot


new_df.plot(kind='box', figsize=(10, 6), vert=False)

plt.title('Immigration from top 15 countries for decades 80s, 90s and 2000s')

plt.show()


# In[96]:


# let's check how many entries fall above the outlier threshold in 2000s

# Outlier = any value > 1.5 * IQR

new_df = new_df.reset_index()
new_df[new_df['2000s'] > 209611.5]


# In[97]:


new_df = new_df.reset_index()
new_df[new_df['1980s'] > 42485]


# In[98]:


new_df = new_df.reset_index()
new_df[new_df['1990s'] > 65192.5 ]


# In[99]:


#Scatter Plots

#Using a scatter plot, let's visualize the trend of total immigrantion to Canada (all countries combined) for the years 
#1980 - 2013.


# In[100]:


#Get the dataset

# we can use the sum()method to get the total population per year
df_tot = pd.DataFrame(df_can[years].sum(axis=0))

# change the years to type int (useful for regression later on)
df_tot.index=map(int, df_tot.index)

## reset the index to put in back in as a column in the df_tot dataframe
df_tot.reset_index(inplace=True)

#rename columns
df_tot.columns=['Years','Total']

#view the final dataframe
df_tot.head(3)


# In[1]:


# step 2. Plot the data
df_tot.plot(kind='scatter',x = 'Years', y='Total',figsize=(10, 6), color='darkblue')

plt.title('Total Immigration to Canada from 1980 - 2013')
plt.xlabel=('Year')
plt.ylabel=('Number of Immigrants')

plt.show()


# In[102]:


#We can clearly observe an upward trend in the data: as the years go by, the total number of immigrants increases.
#We can mathematically analyze this upward trend using a regression line (line of best fit).

#So let's try to plot a linear line of best fit, and use it to predict the number of immigrants in 2015.

#Step 1: Get the equation of line of best fit. We will use Numpy's polyfit() method by passing in the following:

#        x: x-coordinates of the data.
#        y: y-coordinates of the data.
#       deg: Degree of fitting polynomial. 1 = linear, 2 = quadratic, and so on.


# In[103]:


x = df_tot['Years']
y = df_tot['Total']
fit = np.polyfit(x,y, deg=1)

fit


# In[104]:


#The output is an array with the polynomial coefficients, highest powers first. 
#Since we are plotting a linear regression y= a * x + b, our output has 2 elements [5.56709228e+03, -1.09261952e+07] with 
#the the slope in position 0 and intercept in position 1.

#Step 2: Plot the regression line on the scatter plot.


# In[105]:


#Step 2: Plot the regression line on the scatter plot.

df_tot.plot(kind='scatter', x = 'Years', y = 'Total', figsize=(10, 6), color = 'darkblue')

plt.title('Total Immigration to Canada from 1980-2013')
plt.xlabel=('Years')
plt.ylabel=('No of Immigrants')

#plot line of best fit
plt.plot(x, fit[0] * x + fit[1], color='red')
plt.annotate('y ={0:.0f} x + {1:.0f}'. format(fit[0], fit[1]), xy=(2000, 150000))


plt.show()

# print out the line of best fit
'No. Immigrants = {0:.0f} * Year + {1:.0f}'.format(fit[0], fit[1]) 


# In[106]:


#Create a scatter plot of the total immigration from Denmark, Norway, and Sweden to Canada from 1980 to 2013?


# In[107]:


#Step 1. Get the data
df_countries =df_can.loc[['Denmark', 'Norway','Sweden'], years].transpose()
df_total = pd.DataFrame(df_countries.sum(axis=1))
df_total.reset_index(inplace=True)
df_total.columns=['Years', 'Total']
 
# change column year from string to int to create scatter plot
df_total['Years'] = df_total['Years'].astype(int)

df_total.head()


# In[108]:


# Step 2. Plot the data
df_total.plot(kind='scatter', x = 'Years', y = 'Total', figsize=(10, 6), color='darkblue')

plt.title('Immigration Trend for Denmark, Norway and Sweden')
plt.xlabel=('Years')
plt.ylabel=('No of Immigrants')

plt.show()


# In[109]:


#Bubble Plots

#Let's start by analyzing the effect of Argentina's great depression.

#Argentina suffered a great depression from 1998 to 2002, which caused widespread unemployment, riots, the fall of the 
#government, and a default on the country's foreign debt. 
#In terms of income, over 50% of Argentines were poor, and seven out of ten Argentine children were poor at the depth of 
#the crisis in 2002.

#Let's analyze the effect of this crisis, and compare Argentina's immigration to that of it's neighbour Brazil. 
#Let's do that using a bubble plot of immigration from Brazil and Argentina for the years 1980 - 2013. 
#We will set the weights for the bubble as the normalized value of the population for each year.


# In[110]:


#Step 1: Get the data

#transposed dataframe
df_can_t = df_can[years].transpose()

# cast the Years (the index) to type int
df_can_t.index=map(int, df_can_t.index)

# let's label the index. This will automatically be the column name when we reset the index
df_can_t.index.name = ('Year')

#reset index to bring the year in as a column
df_can_t.reset_index(inplace=True)

#view the changes
df_can_t.head()


# In[111]:


#Step 2: Create normalized weights

#normalize Brazil data
norm_Brazil = (df_can_t['Brazil'] - df_can_t['Brazil'].min())/(df_can_t['Brazil'].max() - df_can_t['Brazil'].min())

#normalize Argentina data
norm_Argentina = (df_can_t['Argentina'] - df_can_t['Brazil'].min())/(df_can_t['Argentina'].max() - df_can_t['Argentina'].min())


# In[112]:


#To plot two different scatter plots in one plot, we can include the axes one plot into the other by passing it via the ax 
#parameter.
#We will also pass in the weights using the s parameter. 
#Given that the normalized weights are between 0-1, they won't be visible on the plot. 
#Therefore, we will:
      #multiply weights by 2000 to scale it up on the graph, and,
      #add 10 to compensate for the min value (which has a 0 weight and therefore scale with x 2000)


# In[113]:


#Step 3: Plot the data

#Brazil
ax0 = df_can_t.plot(kind='scatter',
                   x = 'Year',
                   y = 'Brazil',
                   figsize=(14, 8),
                   alpha = 0.5,
                   color = 'green',
                   s = norm_Brazil * 200 + 10,
                   xlim = (1975, 2015))

#Argentina
ax1 = df_can_t.plot(kind='scatter',
                   x = 'Year',
                   y = 'Argentina',
                   figsize=(14, 8),
                   alpha = 0.5,
                   color = 'blue',
                   s = norm_Argentina * 200 + 10,
                   ax = ax0)

ax0.set_ylabel=('Number of Immigrants')
ax0.set_title('Immigration from Brazil and Argentina from 1985 - 2013')
ax0.legend(['Brazil', 'Argentina'], loc='upper left', fontsize='x-large')


# In[114]:


#The size of the bubble corresponds to the magnitude of immigrating population for that year, compared to the 1980 - 2013 data. 
#The larger the bubble is, the more immigrants are in that year.

#From the plot above, we can see a corresponding increase in immigration from Argentina during the 1998 - 2002 great depression.
#We can also observe a similar spike around 1985 to 1993. In fact, Argentina had suffered a great depression from 1974 to 1990, 
#just before the onset of 1998 - 2002 great depression.

#On a similar note, Brazil suffered the Samba Effect where the Brazilian real (currency) dropped nearly 35% in 1999. 
#There was a fear of a South American financial crisis as many South American countries were heavily dependent on industrial 
#exports from Brazil. 
#The Brazilian government subsequently adopted an austerity program, and the economy slowly recovered over the years, 
#culminating in a surge in 2010. 
#The immigration data reflect these events.


# In[115]:


#Bubble Plots for China and India


# In[116]:


#Step 1: Normalize the data

#Normalize China data
norm_China = (df_can_t['China'] - df_can_t['China'].min())/(df_can_t['China'].max()-df_can_t['China'].min())

#Normalize India data
norm_India = (df_can_t['India'] - df_can_t['India'].min())/(df_can_t['India'].max()-df_can_t['India'].min())


# In[117]:


#Step 2: Plot the data

ax0 = df_can_t.plot(kind='scatter',
              x='Year',
              y='China',
              figsize=(14, 8),
              alpha=0.5,
              color = 'violet',
              s = norm_China * 200 + 10,
              xlim=(1975, 2015))

ax1 = df_can_t.plot(kind='scatter',
              x = 'Year',
              y = 'India',
              figsize=(14, 8),
              alpha = 0.5,
              color = 'darkblue',
              s = norm_India * 200 + 10,
              ax = ax0)

ax0.set_ylabel('No of Immigrants')
ax0.set_title('Immigration from China India from 1985 - 2013')
ax0.legend(['China', 'India'], loc = 'upper left', fontsize='x-large')


# In[118]:


#Waffle Charts

#Let's create a new dataframe
df_dsn = df_can.loc[['Denmark', 'Sweden', 'Norway'], :]
df_dsn


# In[119]:


#Step 1: Determine the proportion of each category with respect to the total.

# compute the proportion of each category with respect to the total
total_values = df_dsn['Total'].sum()
category_proportions = df_dsn['Total']/total_values

#print out proportions
pd.DataFrame({'Category Proportion' : category_proportions})


# In[120]:


#Step 2: Detemine overall size of the waffle chart

width = 40
height = 10

total_num_tiles = width * height
print(f'Total number of tiles is {total_num_tiles}')


# In[121]:


#Step 3: The third step is using the proportion of each category to determe it respective number of tiles

#compute the number of tiles for each category
tiles_per_category = (category_proportions * total_num_tiles).round().astype(int)

#print out number of tiles per category
pd.DataFrame({'Number of tiles' : tiles_per_category})


# In[122]:


#Step 4: The fourth step is creating a matrix that resembles the waffle chart and populating it.

#initialize the waffle chart as an empty matrix
waffle_chart = np.zeros((height, width), dtype=np.uint)

#define indices to loop through waffle chart
category_index = 0
tile_index = 0

#populate the waffle chart
for col in range(width):
    for row in range(height):
        tile_index += 1
        
         # if the number of tiles populated for the current category is equal to its corresponding allocated tiles...
        if tile_index > sum(tiles_per_category[0: category_index]):
            # ...proceed to the next category
            category_index += 1
            
            
        # set the class value to an integer, which increases with class    
        waffle_chart[row, col] = category_index
        
        
print('Waffle Chart Populated')


# In[123]:


#Let's take a peek at how the matrix looks like

waffle_chart


# In[124]:


#Step 5: Map the waffle chart into a visual

#instantiate a new figure object
fig = plt.figure()

#use matshow to display the waffle chart
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap = colormap)
plt.colorbar()
plt.show()


# In[125]:


#Step 6: Prettify the chart

#instantiate a new figure object
fig = plt.figure()

#use matshow to display the waffle chart
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap=colormap)
plt.colorbar()

#get the axis
ax = plt.gca()

#set minor ticks
ax.set_xticks(np.arange(-5, (width), 1), minor=True)
ax.set_yticks(np.arange(-5, (height), 1), minor=True)

#add gridlines based on minor tricks
ax.grid(which = 'minor', color='w', linestyle='-', linewidth=2)

plt.xticks([])
plt.yticks([])
plt.show()


# In[126]:


import matplotlib.patches as mpatches


# In[127]:


# Step 7: Create a legend and add it to chart

# instantiate a new figure object
fig = plt.figure()

# use matshow to display the waffle chart
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap=colormap)
plt.colorbar()

# get the axis
ax = plt.gca()

# set minor ticks
ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
ax.set_yticks(np.arange(-.5, (height), 1), minor=True)
    
# add gridlines based on minor ticks
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

plt.xticks([])
plt.yticks([])

#compute cummulative sum of individual categories to match color schemes between chart and legend
values_cumsum = np.cumsum(df_dsn['Total'])
total_values = values_cumsum[len(values_cumsum) -1 ]

#create legend
legend_handles = []
for i, category in enumerate(df_dsn.index.values):
    label_str = category + '(' + str(df_dsn['Total'][i]) + ')'
    color_val = colormap(float(values_cumsum[i])/total_values)
    legend_handles.append(mpatches.Patch(color=color_val, label=label_str))
                                     
#add legend to chart
plt.legend(handles = legend_handles,
          loc = 'lower central',
          ncol = len(df_dsn.index.values),
          bbox_to_anchor=(0., -0.2, 0.95, .1)) 
                                     

plt.show()                                     


# In[ ]:


def create_waffle_chart(categories, values, height, width, colormap, value_sign=''):

    # compute the proportion of each category with respect to the total
    total_values = sum(values)
    category_proportions = [(float(value) / total_values) for value in values]

    # compute the total number of tiles
    total_num_tiles = width * height # total number of tiles
    print ('Total number of tiles is', total_num_tiles)
    
    # compute the number of tiles for each catagory
    tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]

    # print out number of tiles per category
    for i, tiles in enumerate(tiles_per_category):
        print (df_dsn.index.values[i] + ': ' + str(tiles))
    
    # initialize the waffle chart as an empty matrix
    waffle_chart = np.zeros((height, width))

    # define indices to loop through waffle chart
    category_index = 0
    tile_index = 0

    # populate the waffle chart
    for col in range(width):
        for row in range(height):
            tile_index += 1

            # if the number of tiles populated for the current category 
            # is equal to its corresponding allocated tiles...
            if tile_index > sum(tiles_per_category[0:category_index]):
                # ...proceed to the next category
                category_index += 1       
            
            # set the class value to an integer, which increases with class
            waffle_chart[row, col] = category_index
    
    # instantiate a new figure object
    fig = plt.figure()

    # use matshow to display the waffle chart
    colormap = plt.cm.coolwarm
    plt.matshow(waffle_chart, cmap=colormap)
    plt.colorbar()

    # get the axis
    ax = plt.gca()

    # set minor ticks
    ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
    ax.set_yticks(np.arange(-.5, (height), 1), minor=True)
    
    # add dridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    plt.xticks([])
    plt.yticks([])

    # compute cumulative sum of individual categories to match color schemes between chart and legend
    values_cumsum = np.cumsum(values)
    total_values = values_cumsum[len(values_cumsum) - 1]

    # create legend
    legend_handles = []
    for i, category in enumerate(categories):
        if value_sign == '%':
            label_str = category + ' (' + str(values[i]) + value_sign + ')'
        else:
            label_str = category + ' (' + value_sign + str(values[i]) + ')'
            
        color_val = colormap(float(values_cumsum[i])/total_values)
        legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

    # add legend to chart
    plt.legend(
        handles=legend_handles,
        loc='lower center', 
        ncol=len(categories),
        bbox_to_anchor=(0., -0.2, 0.95, .1)
    )
    plt.show()


# In[ ]:


width = 40 # width of chart
height = 10 # height of chart

categories = df_dsn.index.values # categories
values = df_dsn['Total'] # correponding values of categories

colormap = plt.cm.coolwarm # color map class


# In[ ]:


create_waffle_chart(categories, values, height, width, colormap)


# In[ ]:


df_can.head()


# In[128]:


get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud, STOPWORDS


# In[129]:


# total immigration from 1980 -2013

total_immigration = df_can['Total'].sum()
total_immigration


# In[130]:


#Using countries with single-word names, let's duplicate each country's name based on how much they 
#contribute to the total immigration.


# In[131]:


max_words = 90
word_string = ''
for Country in df_can.index.values:
    if Country.count(" ") == 0:
        repeat_num_times = int(df_can.loc[Country, 'Total']/ total_immigration * max_words)
        word_string = word_string + ((Country + '') * repeat_num_times)
        
        
word_string


# In[132]:


#We are not dealing with any stopwords here, so there is no need to pass them when creating the word cloud.

#create the word cloud
wordcloud = WordCloud(background_color = 'pink').generate(word_string)

print('Word Cloud created!')


# In[133]:


# display the cloud
plt.figure(figsize=(14, 18))

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[134]:


# Regression Plots

get_ipython().system('pip install seaborn')

import seaborn as sns
print('Seaborn Installed and imported')


# In[135]:


#Create a new dataframe that stores that total number of landed immigrants to Canada per year from 1980 to 2013.


# we can use the sum() method to get the total population per year
df_tot = pd.DataFrame(df_can[years].sum(axis=0))

# change the years to type float (useful for regression later on)
df_tot.index = map(float, df_tot.index)

# reset the index to put in back in as a column in the df_tot dataframe
df_tot.reset_index(inplace=True)

# rename columns
df_tot.columns=['Years', 'Total']

# view the final dataframe
df_tot.head(4)


# In[136]:


#generate a regression plot

sns.regplot(x ='Years', y='Total', data = df_tot)


# In[ ]:


#And let's increase the size of markers so they match the new size of the figure, and add a title and x- and y-labels.


# In[149]:


plt.figure(figsize=(15, 10))

sns.set(font_scale=1.5)
sns.set_style('whitegrid')
ax = sns.regplot(x='Years', y='Total', data=df_tot, marker = '*', color='green', scatter_kws={'s': 200})

ax.set_title('Total Immigration to Canada from 1980 - 2013')
ax.set(xlabel='Years', ylabel='Total Immigrants')
plt.show()


# In[ ]:


#Regression line to visualize the total immigration from Denmark, Sweden, and Norway to Canada from 1980 to 2013.


# In[156]:


df_countries = df_can.loc[['Denmark', 'Sweden','Norway'], years].transpose()
df_total = pd.DataFrame(df_countries.sum(axis=1))
df_total.reset_index(inplace=True)
df_total.columns=['Years', 'Total']
df_total['Years'] = df_total['Years'].astype(int)

#plot the regression line
plt.figure(figsize=(15, 10))

sns.set(font_scale=1.5)
sns.set_style('whitegrid')

ax = sns.regplot(x='Years', y='Total',data=df_total, color='blue', marker='+', scatter_kws={'s':200})
ax.set(xlabel='Years', ylabel='Total Immigration')
ax.set_title('Total Immigration from Denmark, Sweden, Norway to Canada from 1980-2013')


# In[158]:


#Folium

get_ipython().system('pip install folium')
import folium

print('Follium is installed and imported')


# In[160]:


#define the world map
world_map = folium.Map()

#display world map
world_map


# 

# In[163]:


# define the world map centered around Canada with a low zoom level
world_map = folium.Map(location=[56.130, -106.35], zoom_start=4)

#display world map
world_map


# In[164]:


#Create a map of Mexico with a zoom level of 4.
world_map = folium.Map(location=[23.6345, -102.5528], zoom_start=4)

world_map


# In[172]:


#Chloropeth Maps

# download countries geojson file
import pandas as pd
import requests
import json

URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/world_countries.json'
response = requests.get(URL)
data = response.json()

world_geo = data

print('GeoJSON file loaded!')


# In[173]:


#Now that we have the GeoJSON file, let's create a world map, centered around [0, 0] latitude and longitude values, 
#with an initisal zoom level of 2.


# In[175]:


#create a plain world map

world_map = folium.Map(location=[0, 0], zoom_start=2)


# In[187]:


df_can.reset_index(drop = False, inplace=True)
df_can.columns


# In[192]:


# generate choropleth map using the total immigration of each country to Canada from 1980 to 2013

folium.Choropleth(geo_data = world_geo,
                     data=df_can,
                     columns=['Country', 'Total'],
                     key_on ='feature.properties.name',
                     fill_color='YlOrRd',
                     fill_opcaity=0.7,
                     fill_line=0.2,
                     legend_name='Immigration to Canada').add_to(world_map)

#display map
world_map


# In[193]:


#As per our Choropleth map legend, the darker the color of a country and the closer the color to red, the higher the number
#of immigrants from that country. 
#Accordingly, the highest immigration over the course of 33 years (from 1980 to 2013) was from China, India, and the 
#Philippines, followed by Poland, Pakistan, and interestingly, the US.


# In[194]:


#Let's fix the legend


# In[196]:


# create a numpy array of length 6 and has linear spacing from the minimum total immigration to the maximum total immigration
threshold_scale = np.linspace(df_can['Total'].min(),
                              df_can['Total'].max(),
                              6, dtype=int)

threshold_scale = threshold.scale.tolist()
thresholdscale[-1] = threshold[-1] + 1

world_map=folium.Map(location=[0, 0], zoom_start=2)
folium.Choropleth(geo_data = world_geo,
                     data=df_can,
                     columns=['Country', 'Total'],
                     key_on ='feature.properties.name',
                     fill_color='YlOrRd',
                     fill_opcaity=0.7,
                     fill_line=0.2,
                     legend_name='Immigration to Canada').add_to(world_map)

#display map
world_map


# In[ ]:




