#!/usr/bin/env python
# coding: utf-8

# In[7]:


# Import libraries
import pandas as pd
import numpy as np

import matplotlib as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import requests
import lxml
from geopy.geocoders import Nominatim # module to convert an address into latitude and longitude values
import folium # plotting library
from sklearn.cluster import KMeans

from opencage.geocoder import OpenCageGeocode
from pprint import pprint

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[6]:


from rightmove_webscraper import RightmoveData

url = "https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E87490&sortType=10&propertyTypes=bungalow%2Cdetached%2Cflat%2Cpark-home%2Csemi-detached%2Cterraced&maxDaysSinceAdded=1&includeSSTC=false&mustHave=&dontShow=&furnishTypes=&keywords="
rm = RightmoveData(url)


# In[8]:


rm.average_price


# In[16]:


rm.get_results


# In[17]:


rm.results_count


# In[18]:


rm.summary()


# In[19]:


rm.summary(by="postcode")


# In[21]:


df_rm = pd.DataFrame(rm.get_results)
df_rm.to_csv(path_or_buf = 'RightMoveAPIResults.csv')


# In[22]:


# Get lxlm
get_ipython().system('conda install -c anaconda lxml -y')

# Install requests library
get_ipython().system('conda install -c anaconda requests -y')

# Install geopy library
get_ipython().system('conda install -c conda-forge geopy --yes ')

# Install folium library
get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')

# Install rightmove-webscraper (https://github.com/toby-p/rightmove_webscraper.py)
get_ipython().system('pip install -U rightmove-webscraper')

# Install opencage
get_ipython().system('pip install opencage')


# In[23]:



# Additional imports for analysis:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


print(df_rm.shape)
df_rm.head(3)


# In[25]:


get_ipython().system('pip install --upgrade numpy')
get_ipython().system('pip install --upgrade seaborn')


# In[26]:


import seaborn as sns
def plot_by_type(rm: RightmoveData):
    """Bar chart of count of results by number of bedrooms."""
    df = rm.summary()
    labels = [f"{i}-b" if i != 0 else "Studio" for i in df["number_bedrooms"]]
    x = df.index
    y = df["count"]
    sns.set_style("dark")
    sns.set_palette(sns.color_palette("pastel"))
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.title("London Sales: Number of Listings by Apartment Type", size = 18)
    plt.xlabel("Apartment Type", size = 14)
    plt.ylabel("Number of Listings", size = 14)
    plt.xticks(size = 14)
    plt.yticks(size = 12)
    plt.bar(x, y, tick_label=labels)
    plt.savefig('London Rentals_Number of Listings by Apartment Type.png', dpi=600, bbox_inches='tight')
    plt.show()

plot_by_type(rm)


# In[27]:


def plot_by_postcode(rm: RightmoveData, number_to_plot: int = 25):
    """Plot count of results by postcode."""
    df = rm.summary("postcode")
    df.sort_values(by="count", ascending=False, inplace=True)
    df = df.reset_index(drop=True)[:number_to_plot]
    x, y = df["postcode"], df["count"]
    ymax = ((df["count"].max() // 5) + 1) * 5
    sns.set_palette(sns.color_palette("pastel"))
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.bar(x.index, height=y)
    ax.set_title(f"London Sales: {number_to_plot} Most Active Postcode Areas", size=18)
    ax.set_xlabel("Postcode Area", size=14)
    ax.set_ylabel("Number of Listings", size=14)
    ax.set_xticks(x.index)
    ax.set_xlim(-1, x.index[-1]+1)
    ax.set_xticklabels(x.values, rotation=45)
    ax.set_yticks(range(0, ymax, 5))
    plt.savefig('Most Active Postcode Areas.png', dpi=300, bbox_inches='tight')
    return fig

f = plot_by_postcode(rm, number_to_plot=25)


# In[28]:


df_districts = pd.read_csv("London Districts.csv")


# In[29]:


df_rm.rename(columns = {"postcode":"PostCode"}, inplace=True)


# In[30]:


df_rm.shape


# In[31]:


postal_codes = df_districts["PostCode"]


# In[32]:


import json

latitudes = [] # Initializing the latitude array
longitudes = [] # Initializing the longitude array

for postal_code in postal_codes : 
    place_name = postal_code + " London" # Formats the place name
    url = 'https://api.opencagedata.com/geocode/v1/json?q={}&key={}'.format(place_name, "63eaae5de3224e53856a8130eace70c3") # Gets the proper url to make the API call
    obj = json.loads(requests.get(url).text) # Loads the JSON file in the form of a python dictionary
    
    results = obj['results'] # Extracts the results information out of the JSON file
    lat = results[0]['geometry']['lat'] # Extracts the latitude value
    lng = results[0]['geometry']['lng'] # Extracts the longitude value
    
    latitudes.append(lat) # Appending to the list of latitudes
    longitudes.append(lng) # Appending to the list of longitudes


# In[33]:


df_districts['Latitude'] = latitudes
df_districts['Longitude'] = longitudes
df_districts.head()


# In[35]:


df_rm_lat_long = df_rm
df_rm_address = df_rm_lat_long["address"]


# In[36]:


# Set the OpenCageGeocode requirements:
key = '63eaae5de3224e53856a8130eace70c3'
geocoder = OpenCageGeocode(key)


# In[37]:



# Create a new empty dataframe with the columns that are important for our purpose
df_info = pd.DataFrame(columns = ["i", "address", "Latitude_a", "Longitude_a", "county", "Postcode_complete", "state_district", "suburb"])


# In[38]:


# Define some empty list
lat_list = []
lng_list = []
county_list = []
postcode_complete_list = []
state_district_list = []
suburb_list = []
address_list = []

# Create a loop to find and store the data for each address.
for i in range(0, len(df_rm["address"])):
    address = df_rm["address"][i]
    query = u'{}'.format(address)
    results = geocoder.geocode(query)

    # In order to avoid errors, define an if-else statement to insert 'np.nan' when the 'results' variable is blank or some of the results are missing
    if not results:
        address_ = df_rm["address"][i]
        lat_ = np.nan
        lng_ = np.nan
        county_ = np.nan
        postcode_complete = np.nan
        state_district = np.nan
        suburb = np.nan
    
    else:
        address_ = df_rm["address"][i]

        if 'lat' in results[0]['geometry']:
            lat_ = results[0]['geometry']['lat']
        else:
            lat_ = np.nan

        if 'lng' in results[0]['geometry']:
            lng_ = results[0]['geometry']['lng']
        else:
            lng_ = np.nan

        if 'county' in results[0]['components']:
            county_ = results[0]['components']['county']
        else:
            county_ = np.nan

        if 'postcode' in results[0]['components']:
            postcode_complete = results[0]['components']['postcode']
        else:
            postcode_complete = np.nan

        if 'state_district' in results[0]['components']:
            state_district = results[0]['components']['state_district']
        else:
            state_district = np.nan

        if 'suburb' in results[0]['components']:
            suburb = results[0]['components']['suburb']
        else:
            suburb = np.nan

        lat_list.append(lat_)
        lng_list.append(lng_)
        county_list.append(county_)
        postcode_complete_list.append(postcode_complete)
        state_district_list.append(state_district)
        suburb_list.append(suburb)
        address_list.append(address_)
        
        #(columns = ["i", "address", "Latitude_a", "Longitude_a", "county", "Postcode_complete", "state_district", "suburb"])
        df_info = df_info.append(pd.Series([i, address_, lat_, lng_, county_, postcode_complete, state_district, suburb], index=df_info.columns), ignore_index = True)


# In[39]:


print(df_info.shape)
df_info.head(3)


# In[40]:


df_info = df_info.set_index('i')


# In[41]:


df_merged = df_rm.merge(df_districts, how='left', on="PostCode")


# In[42]:


print(df_merged.shape)
df_merged.head(3)


# In[43]:


df_merged = df_merged.reset_index()
df_merged.rename(columns = {"index" : "i"}, inplace = True)
df_info = df_info.reset_index()


# In[44]:


df_merged_copy = df_merged
df_info_copy = df_info


# In[45]:


df_location = df_merged_copy.merge(df_info_copy, how='outer', on=["i"])


# In[46]:


print(df_location.shape)
df_location.tail(3)


# In[47]:


df_clustering = df_location


# In[48]:



df_clustering.T.apply(lambda x: x.nunique(), axis=1)


# In[49]:


df_clustering.dropna(subset=['Latitude_a', 'Longitude_a'], inplace=True)
print(df_clustering.shape)
df_clustering.rename(columns={"level_0" : "i-match"}, inplace=True)
df_clustering.reset_index()
df_clustering.head(3)


# In[50]:


CLIENT_ID = '0WCXVMCPDCZ1GOT00KYGFK500CMISIXTQR21ZZMMNGQBOK2U' # your Foursquare ID
CLIENT_SECRET = 'FDDWJJRI24RCXI5MGMDTGDVWWSQTMNBDMSQOGUKTWK0HUZVK' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[51]:


def getNearbyVenues(names, latitudes, longitudes, i, price, types, radius = 500, LIMIT = 100):
    
    venues_list=[]
    myhitcount = 0
    for name, lat, lng, i, price, types in zip(names, latitudes, longitudes, i, price, types):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
        
        if (lat != np.nan) and (lng != np.nan) and myhitcount<950:
            
            # make the GET request
            results = requests.get(url).json()["response"]['groups'][0]['items']
            myhitcount=myhitcount+1
            print(myhitcount)
            # return only relevant information for each nearby venue
            venues_list.append([(
                name, 
                lat, 
                lng,
                i,
                price,
                types,
                v['venue']['name'], 
                v['venue']['location']['lat'], 
                v['venue']['location']['lng'],  
                v['venue']['categories'][0]['name']) for v in results])

            nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
            nearby_venues.columns = ['address_x', 
                                      'address Latitude', 
                                      'address Longitude',
                                      'i',
                                      'price',
                                      'type',
                                      'Venue', 
                                      'Venue Latitude', 
                                      'Venue Longitude', 
                                      'Venue Category']
        else:
            continue
    
    return(nearby_venues)


# In[52]:


df_address_venues = getNearbyVenues(names = df_clustering['address_x'],
                                   latitudes = df_clustering['Latitude_a'],
                                   longitudes = df_clustering['Longitude_a'],
                                   i = df_clustering['i'],
                                   price = df_clustering['price'],
                                   types = df_clustering['type']
                                  )


# In[53]:


df_clustering.loc[22]


# In[54]:


df_address_venues.groupby(['address_x', 'i', 'price', 'type']).count().head(3)


# In[55]:


print("There are " , len(df_address_venues["Venue Category"].unique()), " unique categories in the dataset")


# In[56]:


print("There are " , len(df_address_venues["Venue Category"]), " total categories in the dataset")


# In[57]:


# one hot encoding
df_venues_dummies = pd.get_dummies(df_address_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighbourhood column back to dataframe
df_venues_dummies['address'] = df_address_venues['address_x']
df_venues_dummies['i'] = df_address_venues['i']
df_venues_dummies['price'] = df_address_venues['price']
df_venues_dummies['type'] = df_address_venues['type']

# move address column to the first column
temp_address = df_venues_dummies['address']
df_venues_dummies.drop(labels=['address'], axis=1,inplace = True)
df_venues_dummies.insert(0, 'address', temp_address)

# move i column to the second column
temp_i = df_venues_dummies['i']
df_venues_dummies.drop(labels=['i'], axis=1,inplace = True)
df_venues_dummies.insert(1, 'i', temp_i)

# move price column to the third column
temp_price = df_venues_dummies['price']
df_venues_dummies.drop(labels=['price'], axis=1,inplace = True)
df_venues_dummies.insert(2, 'price', temp_price)

# move type column to the fourth column
temp_type = df_venues_dummies['type']
df_venues_dummies.drop(labels=['type'], axis=1,inplace = True)
df_venues_dummies.insert(3, 'type', temp_type)


# In[58]:


df_venues_dummies.shape


# In[59]:


df_dcategories = df_venues_dummies.groupby(['address', 'i', 'price', 'type']).mean().reset_index()


# In[64]:


df_dcategories.shape


# In[63]:


df_dcategories.to_csv(path_or_buf = 'df_dcategories.csv')


# In[65]:


# Create a dataframe with the top 20 venues

def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[66]:



num_top_venues = 20

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['address', 'i', 'price', 'type']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
address_venues_sorted = pd.DataFrame(columns=columns)
address_venues_sorted['address'] = df_dcategories['address']
address_venues_sorted['i'] = df_dcategories['i']
address_venues_sorted['price'] = df_dcategories['price']
address_venues_sorted['type'] = df_dcategories['type']

for ind in np.arange(df_dcategories.shape[0]):
    address_venues_sorted.iloc[ind, 4:] = return_most_common_venues(df_dcategories.iloc[ind, 3:], num_top_venues)

address_venues_sorted.head(3)


# In[67]:


address_venues_sorted.to_csv(path_or_buf = '20_Venue_address_venues_sorted.csv')


# In[68]:


address_venues_sorted.shape


# In[69]:



i_values = address_venues_sorted['i']


# In[70]:


from sklearn.cluster import KMeans
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
import pickle

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[71]:


df_dcategories_nn = df_dcategories.drop(columns={'address', 'i', 'price', 'type'})
df_dcategories_nn.head(3)


# In[72]:


df_dcategories_nn.to_csv(path_or_buf='df_dcategries_nn.csv')


# In[76]:


k_clusters = 13

k_clusters_fit = KMeans(k_clusters, random_state = 4).fit(df_dcategories_nn)


# In[74]:


get_ipython().system('pip install yellowbrick')


# In[75]:


# Elbow Method for K means
# Import ElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(2,30), timings= True)
visualizer.fit(df_dcategories_nn)        # Fit data to visualizer
visualizer.show() 


# In[77]:


k_clusters_fit.labels_


# In[78]:


# add clustering labels
# df.insert(loc, column_name, values)
address_venues_sorted.insert(0, "Cluster Label", k_clusters_fit.labels_)


# In[79]:


# Create a copy of the DataFrame
df_final = df_clustering

df_final = df_final[df_final['i'].isin(i_values)]
df_final.shape


# In[80]:



# Merge the DataFrames
df_final = df_final.merge(address_venues_sorted, how='outer', on=["i"])

# Convert Cluster Label in integers
df_final["Cluster Label"] = df_final["Cluster Label"].astype(int)


# In[81]:


df_final.shape


# In[82]:


df_final.to_csv(path_or_buf='DM_FinalOutput.csv')


# In[83]:


df_final_clean = df_final.drop(columns= ['i', 'PostCode', 'search_date', 'address_y',  'state_district'])


# In[84]:


df_final_clean.to_csv(path_or_buf='DM_FinalOutput_Clean.csv')


# In[85]:


df_plot1 = df_final_clean.groupby(['Cluster Label']).count()

df_plot1.reset_index().plot(x="Cluster Label", y="price_x", kind="bar", figsize=(15,8))
plt.xlabel("Cluster Label", size = 12)
plt.ylabel("Number of offers", size = 12)
plt.title("Number of offers by Cluster")

plt.show()


# In[86]:


pickle.dump(k_clusters_fit, open("k_clusters_fit.pkl", "wb"))


# In[87]:


# London latitude and longitude
latitude = 51.509865
longitude = -0.118092

# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=12)

# set color scheme for the clusters
x = np.arange(k_clusters)
ys = [i + x + (i*x)**2 for i in range(k_clusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(df_final_clean['Latitude_a'], df_final_clean['Longitude_a'], df_final_clean['address_x'], df_final_clean['Cluster Label']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[88]:


def generateBaseMap(default_location=[latitude, longitude], default_zoom_start=8):
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
    return base_map


# In[89]:


from folium.plugins import HeatMap

base_map = generateBaseMap(default_zoom_start = 13)
HeatMap(data=df_final_clean[['Latitude_a', 'Longitude_a', 'price_x']].groupby(['Latitude_a', 'Longitude_a']).mean().reset_index().values.tolist(), radius=12, max_zoom=12).add_to(base_map)


# In[90]:


base_map


# In[91]:


df_final_clean.to_csv(path_or_buf = 'Final_clusters_labeled.csv')


# In[92]:


print(df_final_clean.groupby(['District Name']).head().shape)
df_final_grouped = df_final_clean.groupby(['District Name', 'type_x']).mean()
df_final_grouped.head(3)


# In[93]:


df_final_grouped.to_csv(path_or_buf = 'Final_District_Grouped_clusters_labeled.csv')


# In[94]:


df_2bed = df_final_grouped.loc[df_final_grouped['number_bedrooms'] == 2].sort_values(by=['price_x'])
df_2bed.head()


# In[95]:


df_2bed.to_csv(path_or_buf = 'Final_df_2bed_District_Grouped_clusters_labeled.csv')


# In[96]:


clusters_grouped = df_final_clean.groupby(['Cluster Label', 'District Name', 'type_x']).mean()
clusters_grouped.loc[clusters_grouped['number_bedrooms'] == 2].sort_values(by=['price_x']).head(3)


# In[97]:


clusters_grouped.to_csv(path_or_buf = 'Final_df_All_Grouped_clusters_labeled.csv')


# In[98]:


df_3bd = clusters_grouped.reset_index()


# In[99]:


df_3bd = df_3bd.loc[df_3bd['number_bedrooms'] == 3]


# In[100]:


df_3bd = df_3bd.groupby(['District Name']).mean()


# In[101]:


df_3bd = df_3bd.reset_index()
df_3bd.sort_values(by='price_x', inplace = True)


# In[102]:


import matplotlib.pyplot as plt
df_3bd.plot('District Name', 'price_x', kind='bar', figsize=(25,8))

plt.xlabel('District Name', size = 12)
plt.ylabel("Average Price", size = 12)
plt.title("Price for a 3 bed flat in London", size = 14)
plt.xticks(size = 12, rotation = 45)
plt.yticks(size = 12)

plt.show()


# In[103]:


df_3bd.to_csv(path_or_buf = 'Final_df_3bed_District_Grouped_clusters_labeled.csv')


# In[104]:


df_1bed = clusters_grouped.reset_index()


# In[105]:


df_1bed = df_1bed.loc[df_1bed['number_bedrooms'] == 1]


# In[106]:


df_1bed = df_1bed.groupby(['District Name']).mean()


# In[107]:



df_1bed = df_1bed.reset_index()
df_1bed.sort_values(by='price_x', inplace = True)


# In[108]:


import matplotlib.pyplot as plt
df_1bed.plot('District Name', 'price_x', kind='bar', figsize=(25,8))

plt.xlabel('District Name')
plt.ylabel("Average month")
plt.title("Price for a 1 bedroom flat in London")
plt.xticks(size = 12)
plt.yticks(size = 12)

plt.show()


# In[109]:


df_1bed.to_csv(path_or_buf = 'Final_df_1bed_District_Grouped_clusters_labeled.csv')


# In[110]:


df_2bed = df_2bed.loc[df_2bed['number_bedrooms'] == 2]


# In[111]:



df_2bed = df_2bed.groupby(['District Name']).mean()


# In[112]:



df_2bed = df_2bed.reset_index()
df_2bed.sort_values(by='price_x', inplace = True)


# In[113]:


import matplotlib.pyplot as plt
df_2bed.plot('District Name', 'price_x', kind='bar', figsize=(25,8))

plt.xlabel('District Name')
plt.ylabel("Average Price")
plt.title("Price for a 2 bedroom flat in London")
plt.xticks(size = 12)
plt.yticks(size = 12)
plt.show()


# In[114]:


df_2bed.to_csv(path_or_buf = 'Final_df_2bedroom_District_Grouped_clusters_labeled.csv')


# In[115]:


df_plot1 = df_final_clean.groupby(['Cluster Label']).count()

df_plot1.reset_index().plot(x="Cluster Label", y="price_x", kind="bar", figsize=(15,8))
plt.xticks(size = 12)
plt.yticks(size = 12)

plt.show()


# In[116]:


df_plot1.to_csv(path_or_buf = 'Final_Plot_Grouped_clusters_labeled.csv')


# In[117]:


from matplotlib import pyplot

corr = df_final_clean.corr()

plt.figure(figsize=(15,15))

ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    #cmap=sns.diverging_palette(20, 220, n=200),
    cmap="YlGnBu",
    square=True
    
)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# In[118]:


k_clusters_fit = pickle.load(open("k_clusters_fit.pkl", "rb"))
df_dcategories_nn = pd.read_csv('df_dcategries_nn.csv')


# In[119]:


address = input('Insert your address')
bedroom = input('How many badrooms are you looking for?')
price = input("What's your budget?")


# In[120]:



key = '63eaae5de3224e53856a8130eace70c3'
geocoder = OpenCageGeocode(key)

query = u'{}'.format(address)
results = geocoder.geocode(query)

latitude = results[0]['geometry']['lat']
longitude = results[0]['geometry']['lng']

print(u'%f;%f;%s;%s' % (results[0]['geometry']['lat'], 
                        results[0]['geometry']['lng'],
                        results[0]['components']['country_code'],
                        results[0]['annotations']['timezone']['name']))


# In[121]:



df_your_location = pd.DataFrame(columns = ['address', 'latitude', 'longitude', 'number bedrooms'])
df_your_location = df_your_location.append(pd.Series([address, latitude, longitude, int(bedroom)], index=df_your_location.columns), ignore_index = True)


# In[122]:


columns_list = df_dcategories_nn.columns
columns_list = columns_list.tolist()


# In[123]:



CLIENT_ID = '0WCXVMCPDCZ1GOT00KYGFK500CMISIXTQR21ZZMMNGQBOK2U' # your Foursquare ID
CLIENT_SECRET = 'FDDWJJRI24RCXI5MGMDTGDVWWSQTMNBDMSQOGUKTWK0HUZVK' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[124]:


def getNearbyVenuesExp(names, latitudes, longitudes, bedrooms, radius = 500, LIMIT = 100):
    
    venues_list=[]
    for name, lat, lng, bedroom in zip(names, latitudes, longitudes, bedrooms):
        print(name)
        myhitcount=0
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
        
        if (lat != np.nan) and (lng != np.nan) and myhitcount<1:
            
            # make the GET request
            results = requests.get(url).json()["response"]['groups'][0]['items']

            # return only relevant information for each nearby venue
            venues_list.append([(
                name, 
                lat, 
                lng,
                bedroom,
                v['venue']['name'], 
                v['venue']['location']['lat'], 
                v['venue']['location']['lng'],  
                v['venue']['categories'][0]['name']) for v in results])

            nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
            nearby_venues.columns = ['address', 
                                      'latitude', 
                                      'longitude',
                                      'bedroom',
                                      'Venue', 
                                      'Venue Latitude', 
                                      'Venue Longitude', 
                                      'Venue Category']
        else:
            continue
    
    return(nearby_venues)


# In[125]:


df_address_venues = getNearbyVenuesExp(names = df_your_location['address'],
                                   latitudes = df_your_location['latitude'],
                                   longitudes = df_your_location['longitude'],
                                   bedrooms = df_your_location['number bedrooms']
                                  )


# In[126]:


df_address_venues.groupby(['address', 'bedroom']).count()


# In[127]:


# one hot encoding
df_venues_dummies = pd.get_dummies(df_address_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighbourhood column back to dataframe
df_venues_dummies['address'] = df_address_venues['address']
df_venues_dummies['bedroom'] = df_address_venues['bedroom']

# move address column to the first column
temp_address = df_venues_dummies['address']
df_venues_dummies.drop(labels=['address'], axis=1,inplace = True)
df_venues_dummies.insert(0, 'address', temp_address)

# move bedroom column to the second column
temp_i = df_venues_dummies['bedroom']
df_venues_dummies.drop(labels=['bedroom'], axis=1,inplace = True)
df_venues_dummies.insert(1, 'bedroom', temp_i)


df_dcategories = df_venues_dummies.groupby(['address', 'bedroom']).mean().reset_index()
df_dcategories.head()


# In[128]:


# Create a dataframe with the top 20 venues

def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[129]:



num_top_venues = 20

if num_top_venues <= (len(df_dcategories.columns) - 2):
    num_top_venues = num_top_venues
else:
    num_top_venues = (len(df_dcategories.columns) - 2)

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['address', 'bedroom']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
address_venues_sorted = pd.DataFrame(columns=columns)
address_venues_sorted['address'] = df_dcategories['address']
address_venues_sorted['bedroom'] = df_dcategories['bedroom']

for ind in np.arange(df_dcategories.shape[0]):
    address_venues_sorted.iloc[ind, 2:] = return_most_common_venues(df_dcategories.iloc[ind, 1:], num_top_venues)


# In[130]:


address_venues_sorted.head()


# In[131]:


df_prediction = df_dcategories.drop(columns = ['address', 'bedroom'])


# In[132]:


df_col = {}

for col in columns_list[1:]:
    df_col[col] = 0
    
df_columns_df = pd.DataFrame.from_dict(df_col, orient = 'index')
df_columns_df = df_columns_df.transpose()

df_columns_df.update(df_prediction)
df_columns_df


# In[133]:



cluster_target = k_clusters_fit.predict(df_columns_df)[0]


# In[134]:


cluster_target


# In[135]:


df_final_clean.head(2)


# In[136]:


df_final_clean[
    (df_final_clean.number_bedrooms == float(bedroom))
    & (df_final_clean['Cluster Label'] == cluster_target)
    & (df_final_clean.price_x <= float(price))
].sort_values(by='price_x', ascending = True).drop(columns=["District Name", "Latitude", "Longitude", "address", "price_y", "type_y"])


# In[137]:


k_clusters = 20

k_clusters_fit = KMeans(k_clusters, random_state = 4).fit(df_dcategories_nn)


# In[138]:


k_clusters_fit.labels_


# In[139]:



# add clustering labels
# df.insert(loc, column_name, values)
address_venues_sorted.insert(0, "Cluster Label", k_clusters_fit.labels_)


# In[141]:


address_venues_sorted


# In[142]:


address_venues_sorted = pd.read_csv("20_Venue_address_venues_sorted.csv")


# In[143]:


address_venues_sorted


# In[144]:


address_venues_sorted.insert(0, "Cluster Label", k_clusters_fit.labels_)


# In[145]:


# Create a copy of the DataFrame
df_final = df_clustering

df_final = df_final[df_final['i'].isin(i_values)]
df_final.shape


# In[146]:


# Merge the DataFrames
df_final = df_final.merge(address_venues_sorted, how='outer', on=["i"])

# Convert Cluster Label in integers
df_final["Cluster Label"] = df_final["Cluster Label"].astype(int)


# In[147]:


df_final.shape


# In[148]:


df_final.to_csv(path_or_buf='DM_FinalOutput.csv')


# In[149]:


df_final_clean = df_final.drop(columns= ['i', 'PostCode', 'search_date', 'address_y',  'state_district'])


# In[150]:


df_final_clean.to_csv(path_or_buf='DM_FinalOutput_Clean.csv')


# In[151]:


df_plot1 = df_final_clean.groupby(['Cluster Label']).count()

df_plot1.reset_index().plot(x="Cluster Label", y="price_x", kind="bar", figsize=(15,8))
plt.xlabel("Cluster Label", size = 12)
plt.ylabel("Number of offers", size = 12)
plt.title("Number of offers by Cluster")

plt.show()


# In[152]:


pickle.dump(k_clusters_fit, open("k_clusters_fit.pkl", "wb"))


# In[153]:


# London latitude and longitude
latitude = 51.509865
longitude = -0.118092

# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=12)

# set color scheme for the clusters
x = np.arange(k_clusters)
ys = [i + x + (i*x)**2 for i in range(k_clusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(df_final_clean['Latitude_a'], df_final_clean['Longitude_a'], df_final_clean['address_x'], df_final_clean['Cluster Label']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[154]:


def generateBaseMap(default_location=[latitude, longitude], default_zoom_start=8):
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
    return base_map


# In[155]:


from folium.plugins import HeatMap

base_map = generateBaseMap(default_zoom_start = 13)
HeatMap(data=df_final_clean[['Latitude_a', 'Longitude_a', 'price_x']].groupby(['Latitude_a', 'Longitude_a']).mean().reset_index().values.tolist(), radius=12, max_zoom=12).add_to(base_map)


# In[156]:


base_map


# In[157]:


df_final_clean.to_csv(path_or_buf = 'Final_clusters_labeled.csv')


# In[158]:


print(df_final_clean.groupby(['District Name']).head().shape)
df_final_grouped = df_final_clean.groupby(['District Name', 'type_x']).mean()
df_final_grouped.head(3)


# In[159]:


df_final_grouped.to_csv(path_or_buf = 'Final_District_Grouped_clusters_labeled.csv')


# In[160]:


df_2bed = df_final_grouped.loc[df_final_grouped['number_bedrooms'] == 2].sort_values(by=['price_x'])
df_2bed.head()


# In[161]:


df_2bed.to_csv(path_or_buf = 'Final_df_2bed_District_Grouped_clusters_labeled.csv')


# In[162]:


clusters_grouped = df_final_clean.groupby(['Cluster Label', 'District Name', 'type_x']).mean()
clusters_grouped.loc[clusters_grouped['number_bedrooms'] == 2].sort_values(by=['price_x']).head(3)


# In[163]:


clusters_grouped.to_csv(path_or_buf = 'Final_df_All_Grouped_clusters_labeled.csv')


# In[164]:


df_3bd = clusters_grouped.reset_index()


# In[165]:


df_3bd = df_3bd.loc[df_3bd['number_bedrooms'] == 3]


# In[166]:


df_3bd = df_3bd.groupby(['District Name']).mean()


# In[167]:


df_3bd = df_3bd.reset_index()
df_3bd.sort_values(by='price_x', inplace = True)


# In[168]:


import matplotlib.pyplot as plt
df_3bd.plot('District Name', 'price_x', kind='bar', figsize=(25,8))

plt.xlabel('District Name', size = 12)
plt.ylabel("Average Price", size = 12)
plt.title("Price for a 3 bed flat in London", size = 14)
plt.xticks(size = 12, rotation = 45)
plt.yticks(size = 12)

plt.show()


# In[169]:


df_3bd.to_csv(path_or_buf = 'Final_df_3bed_District_Grouped_clusters_labeled.csv')


# In[170]:


df_1bed = clusters_grouped.reset_index()


# In[171]:


df_1bed = df_1bed.loc[df_1bed['number_bedrooms'] == 1]


# In[172]:


df_1bed = df_1bed.groupby(['District Name']).mean()


# In[173]:



df_1bed = df_1bed.reset_index()
df_1bed.sort_values(by='price_x', inplace = True)


# In[174]:


import matplotlib.pyplot as plt
df_1bed.plot('District Name', 'price_x', kind='bar', figsize=(25,8))

plt.xlabel('District Name')
plt.ylabel("Average month")
plt.title("Price for a 1 bedroom flat in London")
plt.xticks(size = 12)
plt.yticks(size = 12)

plt.show()


# In[175]:


df_1bed.to_csv(path_or_buf = 'Final_df_1bed_District_Grouped_clusters_labeled.csv')


# In[176]:


df_2bed = clusters_grouped.reset_index()


# In[177]:



df_2bed = df_2bed.loc[df_2bed['number_bedrooms'] == 2]


# In[178]:


df_2bed = df_2bed.groupby(['District Name']).mean()


# In[179]:



df_2bed = df_2bed.reset_index()
df_2bed.sort_values(by='price_x', inplace = True)


# In[180]:


import matplotlib.pyplot as plt
df_2bed.plot('District Name', 'price_x', kind='bar', figsize=(25,8))

plt.xlabel('District Name')
plt.ylabel("Average Price")
plt.title("Price for a 2 bedroom flat in London")
plt.xticks(size = 12)
plt.yticks(size = 12)
plt.show()


# In[181]:


df_2bed.to_csv(path_or_buf = 'Final_df_2bedroom_District_Grouped_clusters_labeled.csv')


# In[182]:


df_plot1 = df_final_clean.groupby(['Cluster Label']).count()

df_plot1.reset_index().plot(x="Cluster Label", y="price_x", kind="bar", figsize=(15,8))
plt.xticks(size = 12)
plt.yticks(size = 12)

plt.show()


# In[183]:


df_plot1.to_csv(path_or_buf = 'Final_Plot_Grouped_clusters_labeled.csv')


# In[184]:


from matplotlib import pyplot

corr = df_final_clean.corr()

plt.figure(figsize=(15,15))

ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    #cmap=sns.diverging_palette(20, 220, n=200),
    cmap="YlGnBu",
    square=True
    
)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# In[185]:


k_clusters_fit = pickle.load(open("k_clusters_fit.pkl", "rb"))
df_dcategories_nn = pd.read_csv('df_dcategries_nn.csv')


# In[186]:


address = input('Insert your address')
bedroom = input('How many badrooms are you looking for?')
price = input("What's your budget?")


# In[187]:


key = '63eaae5de3224e53856a8130eace70c3'
geocoder = OpenCageGeocode(key)

query = u'{}'.format(address)
results = geocoder.geocode(query)

latitude = results[0]['geometry']['lat']
longitude = results[0]['geometry']['lng']

print(u'%f;%f;%s;%s' % (results[0]['geometry']['lat'], 
                        results[0]['geometry']['lng'],
                        results[0]['components']['country_code'],
                        results[0]['annotations']['timezone']['name']))


# In[188]:



df_your_location = pd.DataFrame(columns = ['address', 'latitude', 'longitude', 'number bedrooms'])
df_your_location = df_your_location.append(pd.Series([address, latitude, longitude, int(bedroom)], index=df_your_location.columns), ignore_index = True)


# In[189]:


columns_list = df_dcategories_nn.columns
columns_list = columns_list.tolist()


# In[190]:



CLIENT_ID = '0WCXVMCPDCZ1GOT00KYGFK500CMISIXTQR21ZZMMNGQBOK2U' # your Foursquare ID
CLIENT_SECRET = 'FDDWJJRI24RCXI5MGMDTGDVWWSQTMNBDMSQOGUKTWK0HUZVK' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[191]:


def getNearbyVenuesExp(names, latitudes, longitudes, bedrooms, radius = 500, LIMIT = 100):
    
    venues_list=[]
    for name, lat, lng, bedroom in zip(names, latitudes, longitudes, bedrooms):
        print(name)
        myhitcount=0
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
        
        if (lat != np.nan) and (lng != np.nan) and myhitcount<1:
            
            # make the GET request
            results = requests.get(url).json()["response"]['groups'][0]['items']

            # return only relevant information for each nearby venue
            venues_list.append([(
                name, 
                lat, 
                lng,
                bedroom,
                v['venue']['name'], 
                v['venue']['location']['lat'], 
                v['venue']['location']['lng'],  
                v['venue']['categories'][0]['name']) for v in results])

            nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
            nearby_venues.columns = ['address', 
                                      'latitude', 
                                      'longitude',
                                      'bedroom',
                                      'Venue', 
                                      'Venue Latitude', 
                                      'Venue Longitude', 
                                      'Venue Category']
        else:
            continue
    
    return(nearby_venues)


# In[192]:


df_address_venues = getNearbyVenuesExp(names = df_your_location['address'],
                                   latitudes = df_your_location['latitude'],
                                   longitudes = df_your_location['longitude'],
                                   bedrooms = df_your_location['number bedrooms']
                                  )


# In[193]:


df_address_venues.groupby(['address', 'bedroom']).count()


# In[194]:


# one hot encoding
df_venues_dummies = pd.get_dummies(df_address_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighbourhood column back to dataframe
df_venues_dummies['address'] = df_address_venues['address']
df_venues_dummies['bedroom'] = df_address_venues['bedroom']

# move address column to the first column
temp_address = df_venues_dummies['address']
df_venues_dummies.drop(labels=['address'], axis=1,inplace = True)
df_venues_dummies.insert(0, 'address', temp_address)

# move bedroom column to the second column
temp_i = df_venues_dummies['bedroom']
df_venues_dummies.drop(labels=['bedroom'], axis=1,inplace = True)
df_venues_dummies.insert(1, 'bedroom', temp_i)


df_dcategories = df_venues_dummies.groupby(['address', 'bedroom']).mean().reset_index()
df_dcategories.head()


# In[195]:


# Create a dataframe with the top 20 venues

def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[196]:



num_top_venues = 20

if num_top_venues <= (len(df_dcategories.columns) - 2):
    num_top_venues = num_top_venues
else:
    num_top_venues = (len(df_dcategories.columns) - 2)

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['address', 'bedroom']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
address_venues_sorted = pd.DataFrame(columns=columns)
address_venues_sorted['address'] = df_dcategories['address']
address_venues_sorted['bedroom'] = df_dcategories['bedroom']

for ind in np.arange(df_dcategories.shape[0]):
    address_venues_sorted.iloc[ind, 2:] = return_most_common_venues(df_dcategories.iloc[ind, 1:], num_top_venues)


# In[197]:


address_venues_sorted.head()


# In[198]:


df_col = {}

for col in columns_list[1:]:
    df_col[col] = 0
    
df_columns_df = pd.DataFrame.from_dict(df_col, orient = 'index')
df_columns_df = df_columns_df.transpose()

df_columns_df.update(df_prediction)
df_columns_df


# In[201]:



cluster_target = k_clusters_fit.predict(df_columns_df)[0]


# In[200]:


df_columns_df['Afhgan Restaurant'] = 0


# In[202]:


cluster_target


# In[203]:


df_final_clean.head(2)


# In[204]:


df_final_clean[
    (df_final_clean.number_bedrooms == float(bedroom))
    & (df_final_clean['Cluster Label'] == cluster_target)
    & (df_final_clean.price_x <= float(price))
].sort_values(by='price_x', ascending = True).drop(columns=["District Name", "Latitude", "Longitude", "address", "price_y", "type_y"])


# In[205]:


df_final_clean


# In[208]:


df_dcategories_nn


# In[268]:


###Importing Libraries
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn import metrics
from sklearn import datasets


# In[269]:


KNN = knn(n_neighbors=20)


# In[275]:


Y = df_dcategories_nn.iloc[:,0]


# In[257]:


Y


# In[276]:


X = df_dcategories_nn.iloc[:,1:-1]


# In[261]:


X


# In[278]:


MyKnn=KNN.fit(X, Y)


# In[280]:


Y = KNN.predict(X)


# In[281]:


Y


# In[301]:


address_venues_sorted = pd.read_csv("20_Venue_address_venues_sorted.csv")


# In[302]:


address_venues_sorted.insert(0, "Knn Cluster Label", Y)


# In[303]:


df_knnfinal = df_clustering

df_knnfinal = df_knnfinal[df_knnfinal['i'].isin(i_values)]
df_knnfinal.shape


# In[304]:



# Merge the DataFrames
df_knnfinal = df_knnfinal.merge(address_venues_sorted, how='outer', on=["i"])

# Convert Cluster Label in integers
df_knnfinal["Knn Cluster Label"] = df_knnfinal["Knn Cluster Label"].astype(int)


# In[305]:


df_knnfinal.shape


# In[306]:


df_knnfinal.to_csv(path_or_buf='DM_KnnFinalOutput.csv')


# In[307]:


df_knnfinal_Clean = df_knnfinal.drop(columns= ['i', 'PostCode', 'search_date', 'address_y',  'state_district'])


# In[308]:


df_knnfinal_Clean.to_csv(path_or_buf='DM_KnnFinalOutput_Clean.csv')


# In[309]:


df_knnplot1 = df_knnfinal_Clean.groupby(['Knn Cluster Label']).count()

df_knnplot1.reset_index().plot(x="Knn Cluster Label", y="price_x", kind="bar", figsize=(15,8))
plt.xlabel("Knn Cluster Label", size = 12)
plt.ylabel("Number of offers", size = 12)
plt.title("Number of offers by Cluster")

plt.show()


# In[310]:


pickle.dump(MyKnn, open("MyKnn.pkl", "wb"))


# In[316]:


# London latitude and longitude
latitude = 51.509865
longitude = -0.118092

# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=12)

# set color scheme for the clusters
x = np.arange(Y.all())
ys = [i + x + (i*x)**2 for i in Y]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(df_knnfinal_Clean['Latitude_a'], df_knnfinal_Clean['Longitude_a'], df_knnfinal_Clean['address_x'], df_knnfinal_Clean['Knn Cluster Label']):
    label = folium.Popup(str(poi) + ' Knn Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[317]:


def generateBaseMap(default_location=[latitude, longitude], default_zoom_start=8):
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
    return base_map


# In[318]:


from folium.plugins import HeatMap

base_map = generateBaseMap(default_zoom_start = 13)
HeatMap(data=df_knnfinal_Clean[['Latitude_a', 'Longitude_a', 'price_x']].groupby(['Latitude_a', 'Longitude_a']).mean().reset_index().values.tolist(), radius=12, max_zoom=12).add_to(base_map)


# In[319]:


base_map


# In[320]:


df_knnfinal_Clean.to_csv(path_or_buf = 'KnnFinal_clusters_labeled.csv')


# In[321]:


print(df_knnfinal_Clean.groupby(['District Name']).head().shape)
df_knnfinal_grouped = df_knnfinal_Clean.groupby(['District Name', 'type_x']).mean()
df_knnfinal_grouped.head(3)


# In[322]:


df_knnfinal_grouped.to_csv(path_or_buf = 'knnFinal_District_Grouped_clusters_labeled.csv')


# In[323]:



df_knn2bed = df_knnfinal_grouped.loc[df_final_grouped['number_bedrooms'] == 2].sort_values(by=['price_x'])
df_knn2bed.head()


# In[324]:


df_knn2bed.to_csv(path_or_buf = 'knnFinal_df_2bed_District_Grouped_clusters_labeled.csv')


# In[326]:


knnclusters_grouped = df_knnfinal_Clean.groupby(['Knn Cluster Label', 'District Name', 'type_x']).mean()
knnclusters_grouped.loc[knnclusters_grouped['number_bedrooms'] == 2].sort_values(by=['price_x']).head(3)


# In[327]:


knnclusters_grouped.to_csv(path_or_buf = 'knnFinal_df_All_Grouped_clusters_labeled.csv')


# In[328]:


df_knn3bd = knnclusters_grouped.reset_index()


# In[329]:


df_knn3bd = df_knn3bd.loc[df_knn3bd['number_bedrooms'] == 3]


# In[330]:


df_knn3bd = df_knn3bd.groupby(['District Name']).mean()


# In[331]:


df_knn3bd = df_knn3bd.reset_index()
df_knn3bd.sort_values(by='price_x', inplace = True)


# In[332]:


import matplotlib.pyplot as plt
df_knn3bd.plot('District Name', 'price_x', kind='bar', figsize=(25,8))

plt.xlabel('District Name', size = 12)
plt.ylabel("Average Price", size = 12)
plt.title("Price for a 3 bed flat in London", size = 14)
plt.xticks(size = 12, rotation = 45)
plt.yticks(size = 12)

plt.show()


# In[333]:


df_knn2bed = knnclusters_grouped.reset_index()


# In[335]:



df_knn2bed = df_knn2bed.loc[df_knn2bed['number_bedrooms'] == 2]


# In[336]:


df_knn2bed = df_knn2bed.groupby(['District Name']).mean()


# In[337]:


df_knn2bed = df_knn2bed.reset_index()
df_knn2bed.sort_values(by='price_x', inplace = True)


# In[338]:


import matplotlib.pyplot as plt
df_knn2bed.plot('District Name', 'price_x', kind='bar', figsize=(25,8))

plt.xlabel('District Name')
plt.ylabel("Average Price")
plt.title("Price for a 2 bedroom flat in London")
plt.xticks(size = 12)
plt.yticks(size = 12)
plt.show()


# In[339]:


df_knn2bed.to_csv(path_or_buf = 'knnFinal_df_2bedroom_District_Grouped_clusters_labeled.csv')


# In[345]:


df_knnplot1 = df_knnfinal_Clean.groupby(['Knn Cluster Label']).count()

df_knnplot1.reset_index().plot(x="Knn Cluster Label", y="price_x", kind="bar", figsize=(15,8))
plt.xticks(size = 12)
plt.yticks(size = 12)

plt.show()


# In[346]:


df_knnplot1.to_csv(path_or_buf = 'knnFinal_Plot_Grouped_clusters_labeled.csv')


# In[347]:


from matplotlib import pyplot

corr = df_knnfinal_Clean.corr()

plt.figure(figsize=(15,15))

ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    #cmap=sns.diverging_palette(20, 220, n=200),
    cmap="YlGnBu",
    square=True
    
)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# In[348]:


MyKnn = pickle.load(open("MyKnn.pkl", "rb"))
df_dcategories_nn = pd.read_csv('df_dcategries_nn.csv')


# In[406]:


address = input('Insert your address')
bedroom = input('How many badrooms are you looking for?')
price = input("What's your budget?")


# In[407]:


key = '63eaae5de3224e53856a8130eace70c3'
geocoder = OpenCageGeocode(key)

query = u'{}'.format(address)
results = geocoder.geocode(query)

latitude = results[0]['geometry']['lat']
longitude = results[0]['geometry']['lng']

print(u'%f;%f;%s;%s' % (results[0]['geometry']['lat'], 
                        results[0]['geometry']['lng'],
                        results[0]['components']['country_code'],
                        results[0]['annotations']['timezone']['name']))


# In[408]:



df_your_location = pd.DataFrame(columns = ['address', 'latitude', 'longitude', 'number bedrooms'])
df_your_location = df_your_location.append(pd.Series([address, latitude, longitude, int(bedroom)], index=df_your_location.columns), ignore_index = True)


# In[409]:


columns_list = df_dcategories_nn.columns
columns_list = columns_list.tolist()


# In[410]:



CLIENT_ID = '0WCXVMCPDCZ1GOT00KYGFK500CMISIXTQR21ZZMMNGQBOK2U' # your Foursquare ID
CLIENT_SECRET = 'FDDWJJRI24RCXI5MGMDTGDVWWSQTMNBDMSQOGUKTWK0HUZVK' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[411]:


def getNearbyVenuesExp(names, latitudes, longitudes, bedrooms, radius = 500, LIMIT = 100):
    
    venues_list=[]
    for name, lat, lng, bedroom in zip(names, latitudes, longitudes, bedrooms):
        print(name)
        myhitcount=0
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
        
        if (lat != np.nan) and (lng != np.nan) and myhitcount<1:
            
            # make the GET request
            results = requests.get(url).json()["response"]['groups'][0]['items']

            # return only relevant information for each nearby venue
            venues_list.append([(
                name, 
                lat, 
                lng,
                bedroom,
                v['venue']['name'], 
                v['venue']['location']['lat'], 
                v['venue']['location']['lng'],  
                v['venue']['categories'][0]['name']) for v in results])

            nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
            nearby_venues.columns = ['address', 
                                      'latitude', 
                                      'longitude',
                                      'bedroom',
                                      'Venue', 
                                      'Venue Latitude', 
                                      'Venue Longitude', 
                                      'Venue Category']
        else:
            continue
    
    return(nearby_venues)


# In[412]:


df_address_venues = getNearbyVenuesExp(names = df_your_location['address'],
                                   latitudes = df_your_location['latitude'],
                                   longitudes = df_your_location['longitude'],
                                   bedrooms = df_your_location['number bedrooms']
                                  )


# In[413]:


df_address_venues.groupby(['address', 'bedroom']).count()


# In[414]:


# one hot encoding
df_venues_dummies = pd.get_dummies(df_address_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighbourhood column back to dataframe
df_venues_dummies['address'] = df_address_venues['address']
df_venues_dummies['bedroom'] = df_address_venues['bedroom']

# move address column to the first column
temp_address = df_venues_dummies['address']
df_venues_dummies.drop(labels=['address'], axis=1,inplace = True)
df_venues_dummies.insert(0, 'address', temp_address)

# move bedroom column to the second column
temp_i = df_venues_dummies['bedroom']
df_venues_dummies.drop(labels=['bedroom'], axis=1,inplace = True)
df_venues_dummies.insert(1, 'bedroom', temp_i)


df_dcategories = df_venues_dummies.groupby(['address', 'bedroom']).mean().reset_index()
df_dcategories.head()


# In[415]:


# Create a dataframe with the top 20 venues

def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[416]:



num_top_venues = 20

if num_top_venues <= (len(df_dcategories.columns) - 2):
    num_top_venues = num_top_venues
else:
    num_top_venues = (len(df_dcategories.columns) - 2)

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['address', 'bedroom']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
address_venues_sorted = pd.DataFrame(columns=columns)
address_venues_sorted['address'] = df_dcategories['address']
address_venues_sorted['bedroom'] = df_dcategories['bedroom']

for ind in np.arange(df_dcategories.shape[0]):
    address_venues_sorted.iloc[ind, 2:] = return_most_common_venues(df_dcategories.iloc[ind, 1:], num_top_venues)


# In[417]:


num_top_venues = 20

if num_top_venues <= (len(df_dcategories.columns) - 2):
    num_top_venues = num_top_venues
else:
    num_top_venues = (len(df_dcategories.columns) - 2)

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['address', 'bedroom']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
address_venues_sorted = pd.DataFrame(columns=columns)
address_venues_sorted['address'] = df_dcategories['address']
address_venues_sorted['bedroom'] = df_dcategories['bedroom']

for ind in np.arange(df_dcategories.shape[0]):
    address_venues_sorted.iloc[ind, 2:] = return_most_common_venues(df_dcategories.iloc[ind, 1:], num_top_venues)

address_venues_sorted.head()


# In[418]:


df_knnprediction = df_dcategories.drop(columns = ['address', 'bedroom'])


# In[419]:


df_knncol = {}

for col in columns_list[1:]:
    df_knncol[col] = 0
    
df_knncolumns_df = pd.DataFrame.from_dict(df_knncol, orient = 'index')
df_knncolumns_df = df_knncolumns_df.transpose()

df_knncolumns_df.update(df_knnprediction)
df_knncolumns_df


# In[420]:


df_knncolumns_df.shape


# In[369]:


X


# In[421]:


df_knncolumns_df


# In[422]:


df_knncolumns_df = df_knncolumns_df.drop('Adult Boutique', 1)


# In[430]:


knncluster_target = KNN.predict(df_knncolumns_df)


# In[431]:


dfX = pd.DataFrame(data=X)


# In[432]:


dfX.shape


# In[433]:


df_knncolumns_df.shape


# In[434]:


knncluster_target


# In[435]:


df_knnfinal_Clean.head(2)


# In[438]:


df_knnfinal_Clean[
    (df_knnfinal_Clean.number_bedrooms == float(bedroom))
    & (df_knnfinal_Clean['Knn Cluster Label'] == 42)
    & (df_knnfinal_Clean.price_x <= float(price))
].sort_values(by='price_x', ascending = True).drop(columns=["District Name", "Latitude", "Longitude", "address", "price_y", "type_y"])


# In[439]:


dfPrice = pd.read_csv('PricePrediction.csv')


# In[441]:


dfPrice.head(3)


# In[454]:


#separate the other attributes from the predicting attribute
Xlm = dfPrice.iloc[:,[2,4,5]].values
ylm = dfPrice.iloc[:,0].values
#separte the predicting attribute into Y for model training 


# In[455]:


Xlm


# In[448]:


ylm


# In[456]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')


# In[457]:


imputer = imputer.fit(Xlm[:,[0,1,2]])
Xlm[:,[0,1,2]] = imputer.transform(Xlm[:,[0,1,2]])


# In[458]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(Xlm,ylm,test_size = 0.2, random_state= 0)#feature scaling


# In[459]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[460]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)#predicting the test set results
y_pred = regressor.predict(X_test)


# In[466]:


df1 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df1.plot(kind='bar')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='white')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[471]:


import statsmodels.api as sm


# In[472]:


X_opt = Xlm[:,[0,1,2]]
regressor_OLS = sm.OLS(endog = ylm, exog = X_opt).fit()
regressor_OLS.summary()


# In[473]:


# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[474]:


get_ipython().system('pip install xgboost')


# In[510]:


# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[525]:


# split data into train and test sets
seed = 0
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(Xlm, ylm, test_size=test_size, random_state=seed)


# In[526]:


xgbr = xgb.XGBRegressor(verbosity=0) 
print(xgbr)


# In[527]:


# fit model no training data
model = XGBRegressor(verbosity=0)
model.fit(X_train, y_train)


# In[528]:


# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# In[523]:


score = XGBRegressor.score(X_train,y_train)  


# In[529]:


# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[514]:


Xlm


# In[515]:


ylm


# In[530]:



import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# In[531]:


xtrain, xtest, ytrain, ytest=train_test_split(Xlm, ylm, test_size=0.33)


# In[532]:


xgbr = xgb.XGBRegressor(verbosity=0) 
print(xgbr)


# In[533]:


xgbr.fit(xtrain, ytrain)


# In[534]:


score = xgbr.score(xtrain, ytrain)  
print("Training score: ", score)


# In[535]:


scores = cross_val_score(xgbr, xtrain, ytrain,cv=10)
print("Mean cross-validation score: %.2f" % scores.mean())


# In[536]:


ypred = xgbr.predict(xtest)
mse = mean_squared_error(ytest, ypred)
print("MSE: %.2f" % mse)


# In[537]:


print("RMSE: %.2f" % (mse**(1/2.0)))


# In[539]:


x_ax = range(len(ytest))
plt.plot(x_ax, ytest, label="original")
plt.plot(x_ax, ypred, label="predicted")
plt.title(" London test and predicted data")
plt.legend()
plt.show()


# In[540]:


logylm = np.log(ylm) 


# In[541]:


xtrain, xtest, ytrain, ytest=train_test_split(Xlm, logylm, test_size=0.33)


# In[542]:


xgbr = xgb.XGBRegressor(verbosity=0) 
print(xgbr)


# In[543]:


xgbr.fit(xtrain, ytrain)


# In[544]:


score = xgbr.score(xtrain, ytrain)  
print("Training score: ", score)


# In[545]:


scores = cross_val_score(xgbr, xtrain, ytrain,cv=10)
print("Mean cross-validation score: %.2f" % scores.mean())


# In[546]:


ypred = xgbr.predict(xtest)
mse = mean_squared_error(ytest, ypred)
print("MSE: %.2f" % mse)


# In[547]:


print("RMSE: %.2f" % (mse**(1/2.0)))


# In[548]:


x_ax = range(len(ytest))
plt.plot(x_ax, ytest, label="original")
plt.plot(x_ax, ypred, label="predicted")
plt.title(" London test and predicted data")
plt.legend()
plt.show()


# In[549]:


import seaborn as sns
sns.distplot(ylm);


# In[550]:


sns.distplot(logylm);


# In[554]:


#scatter plot number of bedroom/saleprice
data = pd.concat(ylm, Xlm[0])
data.plot.scatter(x='Number of bedrooms', y='SalePrice', ylim=(0,8000000));


# In[555]:


dfPrice = pd.read_csv('PricePrediction.csv')


# In[569]:


dfPrice.plot.scatter(x='number_bedrooms', y='price_x',xlim=(0,20), ylim=(0,80000000));


# In[559]:


dfPrice.head


# In[ ]:




