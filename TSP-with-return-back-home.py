# @author Victor GALISSON
# @date 2017/1
# @brief Project for Optimization Class, resolving the Traveling Salesperson Problem (TSP)

### -------------------------------------------------------------------------------------------- ###
### -------------- SETUP ----------------------------------------------------------------------- ###
### -------------------------------------------------------------------------------------------- ###
import matplotlib.pyplot as plt
import numpy as np
from math import factorial, sqrt, inf
from random import shuffle

from matplotlib import gridspec
import time

## ------ VARIABLES ----------------------------------------------------------------------------- ##

# -- files related
cities_travelTimes = []
cities_names = []
cities_locations = []

# -- others
xs = []
ys = []
nbCitiesDisplayed = 12
order = list(np.linspace(0, nbCitiesDisplayed-1, nbCitiesDisplayed, dtype = int))

bestDistance = inf
bestOrder = list(order)

# -- GA related
population = []
newPopulation = []
fitness = []
nb_population = 100
nbGenerations = 2000
mutationRate = 0.1                  # low mutation rate as we don't want to focus too much on exploration

## ------ OPENING FILES ------------------------------------------------------------------------- ##

fTravelTimes = open('uk12_dist.txt', 'r')
fNames = open('uk12_name.txt', 'r')
fLocations = open('uk12_xy.txt', 'r')
img = plt.imread("uk12_map.png")


## ------ FUNCTIONS DECLARATIONS ---------------------------------------------------------------- ##

def remove_newlineCharacter( str ) :
    if(str[-1:]=='\n'):
        return str[:-1]
    else:
        return str

# -- adapt data for plotting (basically instead of {[x1,y1], [x2,y2], ...} it is { [x1 x2 ...],[y1 y2 ...]}
def adapt(order):
    global xs, ys
    xs.clear()
    ys.clear()
    # done here and not during the parsing so swapping cities is easier
    i = 0
    
    for index in order:
        city = cities_locations[index]
        xs.append(city[0])                # adding each x value to x list
        ys.append(city[1])                # adding each y value to y list


def calcDistance(listPoints, order) :
    total = 0
    
    for i in range(0, len(order)-1):
        cityAIndex = order[i]
        cityA = listPoints[cityAIndex]
        cityBIndex = order[i+1]
        cityB = listPoints[cityBIndex]
        d =  sqrt((float(cityB[0]) - float(cityA[0]))**2 + ((float(cityB[1]) - float(cityA[1]))**2))
        total += d

    total +=  sqrt((float(listPoints[order[0]][0]) - float(listPoints[order[nbCitiesDisplayed-1]][0]))**2 + ((float(listPoints[order[0]][1]) - float(listPoints[order[nbCitiesDisplayed-1]][1]))**2))
    return total


 # ------------ BRUTE FORCE FUNCTION ---------------------
def nextOrder():
    # largest I such that P[i]<P[i+1].
    largestI = -1
    for i in range(0, nbCitiesDisplayed-1):
        if(order[i] < order[i+1]):
            largestI = i

    if (largestI == -1):
        return True

    # the largest J such that P[I]<P[j]
    largestJ = -1
    for j in range(0, nbCitiesDisplayed):
        if(order[largestI] < order[j]):
            largestJ = j

    # swap P[I] and P[J]
    order[largestI], order[largestJ] = order[largestJ], order[largestI]

    # reverse from P[I+1] to P[n]
    order[largestI+1:nbCitiesDisplayed] = order[largestI+1:nbCitiesDisplayed][::-1]

    return False


## ------- GENETIC ALGORITHM FUNCTIONS -------------------------

def calcFitness():
    global bestDistance, bestOrder
    
    for j in range(nb_population):
        d = calcDistance(cities_locations, population[j])
        if( d < bestDistance):
            bestDistance = d                        # record best distance
            bestOrder = population[j][:]            # record new best order

            # LIVE PLOTTING BEST SOLUTION
            plt.gcf().clear()
            adapt(bestOrder)
            xs.append(xs[0])
            ys.append(ys[0])
            
            plt.imshow(img)
            plt.plot(xs, ys, 'ro-')
            plt.pause(0.001) 
            
        fitness.append(1/(d+1))


# used to normalize fitness to be between 0 and 1
def normalizeFitness():
    total = 0
    
    for i in range(nb_population):
        total += fitness[i]
    for i in range(nb_population):
        fitness[i] = fitness[i]/total


def nextGeneration():
    global population, newPopulation
    newPopulation.clear()
    
    for i in range(nb_population):
        orderA = pick(population, fitness)
        orderB = pick(population, fitness)
        order = crossOver(orderA, orderB,)
        order = mutate(order, mutationRate)
        newPopulation.append(order)

    population = newPopulation[:]


# roulette wheel selection
def pick(p_list , prob):
    index = 0
    r = np.random.random_sample()
    
    while(r > 0):
        r = r - prob[index]
        index += 1
    index-= 1
    
    return p_list[index]
    

def mutate(p_order, mutationRate):
    if(np.random.random_sample() < mutationRate):
        indexA = np.random.randint(len(p_order))
        indexB = np.random.randint(len(p_order))
        p_order[indexA], p_order[indexB] = p_order[indexB], p_order[indexA]

    return p_order

# can't do a normal crossover because we can't duplicate item in a chromosome
def crossOver( orderA, orderB ):
    start = np.random.randint(nbCitiesDisplayed)
    end = np.random.randint(start, nbCitiesDisplayed)
    newOrder = orderA[start:end][:]

    for k in range(nbCitiesDisplayed):
        city = orderB[k]
        if (city not in newOrder):
            newOrder.append(city)
    
    return newOrder


## ------- PARSING DATA FROM FILES -------------------------------------------------------------- ##

# -- NAMES --
cities_names = fNames.readlines()[2:]           # read all the lines except first 2 lines
tmpCities = []                                  # temporary list loop usage 

for city in cities_names:
    tmpCities.append(remove_newlineCharacter(city))

cities_names = list(tmpCities[:nbCitiesDisplayed]) # make a real copy and not just give them same reference (like x = y would do)

tmpCities.clear()
#print(cities_names) # TEST

# -- CITY TO CITY TRAVEL TIMES --
cities_travelTimes = fTravelTimes.readlines()[2:]   # read all the lines except first 2 lines

for city in cities_travelTimes:
    tmpCities.append(remove_newlineCharacter(city))

cities_travelTimes.clear()

for distance_oneToEveryone in tmpCities:
    list_splitted = distance_oneToEveryone.split(' ')
    tmp = list(filter(None,list_splitted))      # separate every numbers and remove empty elements
    cities_travelTimes.append(tmp[:nbCitiesDisplayed])

cities_travelTimes = list(cities_travelTimes[:nbCitiesDisplayed])

tmpCities.clear()
# print(cities_travelTimes) # TEST

## -- LOCATIONS --
cities_locations = fLocations.readlines()[8:]

for location in cities_locations:
    tmp = remove_newlineCharacter(location).split('\t')
    tmp[0] = float(tmp[0])
    tmp[1] = float(tmp[1])
    tmpCities.append(tmp)

cities_locations = list(tmpCities[:nbCitiesDisplayed])
# print(cities_locations) # TEST


### -------------------------------------------------------------------------------------------- ###
### -------------- MAIN ------------------------------------------------------------------------ ###
### -------------------------------------------------------------------------------------------- ###

## ---- GENETIC ALGORITHM ----------------------------

# init population
for k in range(nb_population):
    population.append(order[:])
    shuffle(population[k])

n = 0
while n < nbGenerations:
    if(n%(int(nbGenerations*0.1)) == 0):  # every 10% using modulo %
        print(int(n/nbGenerations*100), "%")
    
    calcFitness()
    normalizeFitness()
    nextGeneration()
    n += 1


print("100 %")
print("Numbers of explored solutions :", nbGenerations*nb_population )
orderGA = bestOrder[:]
print("Order:", bestOrder)



####---- BRUTE FORCE - aka - ALL POSSIBILITIES --- UNCOMMENT TO TRY & COMMENT OUT LINES 337 339
##
##bestDistance = inf
##bestOrder = list(np.linspace(0, nbCitiesDisplayed-1, nbCitiesDisplayed, dtype = int))
##
##finished = False  
##advancement = 0                                         # increment to calculate progression
##totalPermutations = factorial(nbCitiesDisplayed)
##
##while (finished == False):
##    
##    if(advancement%(int(totalPermutations*0.1)) == 0):  # every 10% using modulo %
##        print(int(advancement/totalPermutations*100), "%")
##    advancement+=1
##
##    newDistance = calcDistance(cities_locations, order)
##    if( newDistance < bestDistance):
##        bestDistance = newDistance            # record best distance
##        bestOrder = list(order)               # record new best order
##
####        # LIVE PLOTTING BEST SOLUTION
####        plt.gcf().clear()
####        adapt(bestOrder)
####        plt.plot(xs, ys, 'bx-')
####        plt.pause(0.001)        
##    
##    finished = nextOrder()
##
##print("100 %")
##print("Numbers of explored solutions :", totalPermutations )
##print("Best Solution :",bestOrder)
##bestSolution = bestOrder
##bestDistanceSolution = bestDistance




## -------------- PLOTTING  -----------------------------------------
plt.ion()
fig = plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
##plt.gcf().clear() # clearing, useful if live plotting solutions
gs = gridspec.GridSpec(2, 2)


# -- plot GA solution -----
adapt(orderGA)
xs.append(xs[0])
ys.append(ys[0])

ax1 = fig.add_subplot(gs[0,0])
ax1.imshow(img)
ax1.plot(xs, ys, 'ro-')
ax1.set_title("Genetic Algorithm Solution")
for X, Y, k in zip(xs, ys, orderGA):
    ax1.annotate('{}'.format(k), xy=(X,Y), xytext=(-5, 5), ha='right',
                textcoords='offset points') # format(cities_names[k])


# -- plot best solution----
bestSolution = [0,6,4,5,2,8,9,10,11,7,1,3] # result obtained through brute force run after 3h
bestDistanceSolution = calcDistance(cities_locations, bestSolution)

adapt(bestSolution)
xs.append(xs[0])
ys.append(ys[0])

ax2 = fig.add_subplot(gs[0,1])
ax2.imshow(img)
ax2.plot(xs, ys, 'bx-')
ax2.set_title("Best Solution")
for X, Y, name in zip(xs, ys, bestSolution):
    ax2.annotate('{}'.format(name), xy=(X,Y), xytext=(-5, 5), ha='right',
                textcoords='offset points')


# -- bar graph ----------
distancePlot = fig.add_subplot(gs[1,0:])
y = [bestDistance, bestDistanceSolution]
v = "GA : " + str(int(100*(y[0]/y[1]*100-100))/100) +"% off"
x = [v ,"BS"]
width = 1/3
barlist = distancePlot.barh(x,y, width)
barlist[0].set_color('r')
barlist[1].set_color('b')
distancePlot.set_title("Total Distance by algorithm")

plt.show()

# ------- END -------------------------------------------------------------------------------------

fTravelTimes.close()
fNames.close()
fLocations.close()
