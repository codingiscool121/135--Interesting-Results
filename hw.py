import csv
import plotly.express as pe
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
stars=[]
with open('star_with_gravity.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        stars.append(row)

filter=[]
with open('filtered_data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        filter.append(row)


headers= stars[0]
planetdata=stars[1:]

headersnew = filter[0]
headerdata = filter[1:]

headers[0]= 'Row_Number'

data = pd.read_csv('star_with_gravity.csv')

mass = data["Mass"].tolist()
radius = data["Radius"].tolist()
distance = data["Distance"].tolist()
gravity = data["Gravity"].tolist()

mass.pop(0)
radius.pop(0)
distance.pop(0)
gravity.pop(0)

data2 = pd.read_csv('filtered_data.csv')
starnamenew = data2['Star_name'].tolist()
massnew = data2['Mass'].tolist()
radiusnew = data2['Radius'].tolist()
distancenew = data2['Distance'].tolist()
gravitynew = data2['Gravity'].tolist()

kg=[]

for i in mass:
    kg = i*(1.98855e+30)
print("kilo",kg)

for i in radius:
    meters=i*(1.496e+11)
print("meters",meters)

starmass = []
starradius = []
starname = []
for i in planetdata:
    starmass.append(i[3])
    starradius.append(i[4])
    starname.append(i[1])

planetgravity = []
X = []

for index, name in enumerate(starname):
    #6371000= Earth radius in meters
    #5.972e24= Earth mass in kg
    planetgravity.append(6.67e-11*(float(starmass[index])*5.972e+24)/(float(starradius[index])**2)*6371000*6371000)

scatter = pe.scatter(x=starradius, y=starmass, size=planetgravity, title='Star Gravity by Name', hover_data=[starname])
scatter.show()

for index, planetmass in enumerate(starmass):
    templist=[starradius[index], planetmass]
    X.append(templist)
#Within cluster sum of squares
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))
plt.title('The Elbow Method')
sb.lineplot(range(1,11), wcss, color= 'blue', marker= 'o', markersize=5, linewidth=2)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

lgplanettypescatter = pe.scatter(x=starradius, y=starmass, title='Low Gravity Star Mass by Type', color=starname)
lgplanettypescatter.show()

#terrestrial and super earth from lower gravity planets
distance100= []
for i in distance:
    if i<=100:
        distance100.append(i)

gravityreal=[]
for i in gravity:
    if i>150 or i<350:
        gravityreal.append(i)

with open("filtered_data.csv", "a+") as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(headers)
    csvwriter.writerows(planetdata)

graph = pe.bar(x=starnamenew, y=gravitynew, title='Gravity by Star Name', color=starname)
graph.show()

graph2 = pe.bar(x=starnamenew, y=massnew, title='Mass by Star Name', color=starname)
graph2.show()

graph3 = pe.bar(x=starnamenew, y=radiusnew, title='Radius by Star Name', color=starname)
graph3.show()

graph4 = pe.bar(x=starnamenew, y=distancenew, title='Distance by Star Name', color=starname)
graph4.show()
