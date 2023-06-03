"""
Helper file for solve.py

contains

"""
import math
import random
from point import Point
from instance import Instance
from solution import Solution


"""
IDEA:
function that calcuates the probabilty 
of adding a tower at a specific point, has some sort
of alpha value as the temperature analog.
based on the function we add/move/delete a tower at 
that point. as temp lowers we move delete add less.

Problem: how to move towers effectively?
"""


def grid_then_delete(instance: Instance) -> Solution:
    grid_density = 5
    grid_length = instance.grid_side_length

    T_i = []
    T_f = []
    for x in range(0, grid_length):
        for y in range(0, grid_length):
            if (x % grid_density == 0 and y % grid_density == 0):
                T_i += [Point(x,y)]
    
    for tower in T_i:
        if not all([tower.distance_obj(city) > 3 for city in instance.cities]):
            T_f += [tower]

    #Ensure full coverage:
    for city in instance.cities:
        if not any([city.distance_obj(tower) <= 3 for tower in T_f]):
            T_f += [Point(int(city.x + r_s() * 2 * random.uniform(0, 1)), int(city.y + r_s() * 2 * random.uniform(0, 1)))]


    return Solution(
        instance=instance,
        towers=T_f,
    )

#Random sign helper function
def r_s():
    return 1 if random.random() < 0.5 else -1


"""
Inputs: 
    -solution(contains instance, and list of towers)
    -temperature
Outputs:
    -New list of towers

Notes:
new list guarenteed to be valid
currently a random update function - higher temp moves points from original pos more
"""

def move_towers (temperature, sol):
    #New set of towers we return:
    T_new = []
    inst = sol.instance

    
    #Desirability of various grid locations:
    """
    side = inst.grid_side_length
    arr = [[0 for i in range(side)] for j in range(side)]
    for x in range(0,side-1):
        for y in range(0,side-1):
            arr[x][y] = sum([city.distance_obj(Point(x,y)) <= inst.coverage_radius for city in inst.cities])
            arr[x][y] -= sum([tower.distance_obj(Point(x,y)) <= inst.penalty_radius for tower in sol.towers])
    """
    #Update the towers
    for tower in sol.towers:
        x = -1
        y = -1
        while x < 0 or x > inst.grid_side_length:
            x = int(tower.x + r_s() * 3*temperature)
        while y < 0 or y > inst.grid_side_length:
            y = int(tower.y + r_s() * 3*temperature) 
        T_new += [Point(x,y)] 


    # Remove towers inside each others service radius
    for tower in T_new:
        if any([tower is not tow and tower.distance_obj(tow) < inst.coverage_radius for tow in T_new]):
            T_new.remove(tower)    

    # Determine how many cities each tower covers
    numCover = []
    for tower in T_new:
        numCover += [sum([tower.distance_obj(city) <= inst.coverage_radius for city in inst.cities])]
    #print(numCover)
    
    # Remove towers that cover the same city
    # - remove the one that covers fewer cities
    for city in inst.cities:
        covered = []
    T_alt = T_new.copy()
    for i in range(0,len(T_new)):
        for city in inst.cities:
            if T_new[i].distance_obj(city) <= inst.coverage_radius:
                if city in covered and T_new[i] in T_alt:
                    T_alt.remove(T_new[i])
                    #
                    # numCover.pop(i)
                else:
                    covered += [city]
    T_new = T_alt.copy()

    #Check any useless towers
    for tower in T_new:
        if not any([tower.distance_obj(city) <= inst.coverage_radius for city in inst.cities]):
            T_new.remove(tower)

    # Check still covering - if not add a tower
    for city in inst.cities:
        if not any([city.distance_obj(tower) <= inst.coverage_radius for tower in T_new]):
            
            # Compute point furthest from all other towers that still covers city
            minPoint = Point(city.x,city.y)
            minSum = sum([minPoint.distance_obj(tower).value for tower in T_new])
            for x in range(city.x-inst.coverage_radius, city.x + inst.coverage_radius+1):
                for y in range(city.y-inst.coverage_radius, city.y + inst.coverage_radius+1):
                    testPoint = Point(x,y)
                    if (testPoint.distance_obj(city) <= inst.coverage_radius):
                        testSum = sum([testPoint.distance_obj(tower).value for tower in T_new])
                        if (testSum < minSum):
                            minPoint = testPoint
                            minSum = testSum
            
            T_new += [minPoint]
            # T_new += [Point(city.x, city.y)]
            # ^^ adding a tower on top of city
    return T_new


def move_towers_simple (temperature, sol):
    towers = sol.towers
    inst = sol.instance
    # New set of towers we return:
    new_towers = []

    # Decide whether to Move Add or Remove
    if len(towers) < 10:
        action = 85 # always add when under 5 towers(patchy fix to weird index bug when 0 towers)
    else:
        action = random.uniform(0,100)

    # Move a tower 1 unit in a random direction
    if (action < 50):
        index = math.floor(random.uniform(0,len(sol.towers)+1))
        moving_tower = towers[math.floor(random.uniform(0,len(sol.towers)))]
        towers.remove(moving_tower)
        direction = math.floor(random.uniform(0,4))
        if (direction == 0):
            moving_tower = Point(moving_tower.x + 1, moving_tower.y) 
        elif (direction == 1):
            moving_tower = Point(moving_tower.x, moving_tower.y + 1)
        elif (direction == 2):
            moving_tower = Point(moving_tower.x - 1, moving_tower.y)
        else:
            moving_tower = Point(moving_tower.x, moving_tower.y - 1)
    # ^^ Note: changed back to +1 in single direction from 3*temperature 
    # (also must cast to int to use float number when making a point)

        new_towers = towers.copy()
        new_towers += [moving_tower]

    # Add a tower
    if (50 < action and action < 90):
        add_tower = inst.cities[math.floor(random.uniform(0,len(inst.cities)))]
        new_towers = towers.copy()
        new_towers += [add_tower]

    # Remove a tower
    if (90 < action):
        remove_tower = towers[math.floor(random.uniform(0,len(sol.towers)))]
        new_towers = towers.copy()
        new_towers.remove(remove_tower)

    return new_towers    