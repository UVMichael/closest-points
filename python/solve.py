"""Solves an instance.

Modify this file to implement your own solvers.

For usage, run `python3 solve.py --help`.
"""

import argparse
import math
from pathlib import Path
import random
import sys
from typing import Callable, Dict

from instance import Instance
from file_wrappers import StdoutFileWrapper
from solution import Solution
from file_wrappers import StdinFileWrapper
from point import Point
from array import *
import solve_move_towers


def solve_naive(instance: Instance) -> Solution:
    return Solution(
        instance=instance,
        towers=instance.cities,
    )

#class Instance:
#    grid_side_length: int
#    coverage_radius: int
#    penalty_radius: int
#    cities: List[Point]

def fully_covered(towers, instance):
    for city in instance.cities:
        if not any([city.distance_obj(tower) <= 3 for tower in towers]):
            return False
    return True

def num_coverage(tower, curr_towers, inst):
    covered_cities = []
    for city in uncovered_cities:
        if tower.distance_obj(city) <= inst.coverage_radius:
            covered_cities.append(city)
    return covered_cities


def tower_options(curr_towers, blacklist, inst):
    options = {}
    for x in range(inst.grid_side_length):
        for y in range(inst.grid_side_length):
            tower = Point(x, y)
            if tower not in curr_towers and tower not in blacklist:
                covered = num_coverage(tower, curr_towers, inst)
                num_covered = len(covered)
                if (num_covered not in options):
                    options[num_covered] = [(tower, covered)]
                else:
                    options[num_covered].extend([(tower, covered)])
    return options[max(options)]

def blacklistedTowers(inst):
    blacklist = set()
    for x in range(inst.grid_side_length):
        for y in range(inst.grid_side_length):
            tower = Point(x, y)
            if not any([tower.distance_obj(city) <= inst.coverage_radius for city in inst.cities]):
                blacklist.add(tower)
    return blacklist

def minDist(currTowers, t):
    minDist = 1000000000000
    for tower in currTowers:
        d = tower.distance_obj(t)
        if d < minDist:
            minDist = d
    return minDist

def greedy(instance: Instance) -> Solution:
    global uncovered_cities
    sol = Solution(instance=instance, towers=[])
    blacklist = blacklistedTowers(instance)
    uncovered_cities = instance.cities.copy()
    while uncovered_cities:
        options = tower_options(set(sol.towers), blacklist, instance)
        opt_covered = []
        tower = min(options, key=lambda x: minDist(sol.towers, x[0]))
        tower_sol = Solution(instance=instance, towers=sol.towers + [tower[0]])
        opt_covered = tower[1]
        sol = tower_sol
        for city in opt_covered:
            uncovered_cities.remove(city)

    tower_iterator = sol.towers.copy()
    towers_new = sol.towers.copy()
    for tower in tower_iterator:
        dupe_towers = towers_new.copy()
        dupe_towers.remove(tower)
        dupe_sol = Solution(instance=instance, towers=dupe_towers)
        if dupe_sol.valid():
            towers_new.remove(tower)   
 
    sol.towers = towers_new
    return sol

def greedy_slow(instance: Instance) -> Solution:
    global uncovered_cities
    sol = Solution(instance=instance, towers=[])
    blacklist = blacklistedTowers(instance)
    uncovered_cities = instance.cities.copy()
    while uncovered_cities:
        options = tower_options(set(sol.towers), blacklist, instance)
        opt_tower = options[0][0]
        opt_covered = options[0][1]
        opt_sol = Solution(instance=instance, towers=sol.towers + [opt_tower])
        opt_cost = opt_sol.trainingPenalty()
        for tower, covered in options:
            new_sol = Solution(instance=instance, towers=sol.towers + [tower])
            penal = new_sol.trainingPenalty()
            if penal <= opt_cost:
                opt_sol = new_sol
                opt_covered = covered
                opt_tower = tower
                opt_cost = penal
        sol = opt_sol
        for city in opt_covered:
            uncovered_cities.remove(city)

    tower_iterator = sol.towers.copy()
    towers_new = sol.towers.copy()
    for tower in tower_iterator:
        dupe_towers = towers_new.copy()
        dupe_towers.remove(tower)
        dupe_sol = Solution(instance=instance, towers=dupe_towers)
        if dupe_sol.valid():
            towers_new.remove(tower)   
 
    sol.towers = towers_new
    return sol

def move_towers_MW (temperature, sol):
    #New set of towers we return:
    inst = sol.instance
    dupe_towers = sol.towers.copy()
    dupe_sol = Solution(instance=inst, towers=dupe_towers)
    
    chooser = random.choices([0,1], weights=[0.5,0.5])[0]
    if (chooser == 0): # add tower
        coord = get(random.choices(actors, weights=w)[0])
        chosen_tower = Point(coord[0], coord[1])
        #need to make sure this point is not a duplicate
        
        while chosen_tower in sol.towers:
            coord = get(random.choices(actors, weights=w)[0])
            chosen_tower = Point(coord[0], coord[1])
        dupe_towers.append(chosen_tower)
        succ = sol.penalty() > dupe_sol.penalty()
        updateWeight(chosen_tower,succ)

    if (chooser == 1) : #remove tower
        
        #create a subset
        subset = []
        for tower in sol.towers:
            res= w[(tower.x*grid_length) + tower.y]
            subset.append(res)
        coord = get(random.choices(sol.towers, weights=subset)[0])
        chosen_tower = Point(coord[0], coord[1])
        dupe = sol.towers.copy()
        dupe.remove(chosen_tower)
        if not fully_covered(dupe, inst):
            #update weight
            updateWeight(chosen_tower, False)
            return dupe_sol.towers
        dupe_sol.towers.remove(chosen_tower)
        succ = sol.penalty() > dupe_sol.penalty()
        updateWeight(chosen_tower,succ)
        

    """
        coord = get(random.choices(actors, weights=w)[0])
        
        chosen_tower = Point(coord[0], coord[1])
        dupe = sol.towers.copy()
        while chosen_tower not in dupe:
            coord = get(random.choices(actors, weights=w)[0])
            chosen_tower = Point(coord[0], coord[1])
        dupe.remove(chosen_tower)
        if not fully_covered(dupe, inst):
            #update weight
            return dupe_sol.towers
        dupe_sol.towers.remove(chosen_tower)
        succ = sol.penalty() > dupe_sol.penalty()
        updateWeight(chosen_tower,succ)
        """""
    return dupe_towers   
    

"""    #Check any useless towers
    for tower in T_new:
        if not any([tower.distance_obj(city) <= inst.coverage_radius for city in inst.cities]):
            T_new.remove(tower)

    # Check still covering - issues with rounding back to an int inside T_new += ...
    for city in inst.cities:
        if not any([city.distance_obj(tower) <= inst.coverage_radius for tower in T_new]):
            T_new += [Point(city.x, city.y)]
            """       

#--------------------------------------

def get(i):
    row_index = i // grid_length
    col_index = i % grid_length
    return row_index, col_index


# multiplicative weight calculator
# let each x and y combination be a unque tower combination ex T_x_y.A+
# each x,y combination is an individual 
def init_MW(instance):
    #innitialize an array of all 1 for the start of the multiplicative weights. 
    global w
    global actors
    w = []
    actors = range(0, grid_length**2)
    for c in range(0, grid_length): #assume row major order. 
        for r in range(0, grid_length):
            tower = Point(r, c)
            if not any([city.distance_obj(tower) <= 3 for city in instance.cities]):
                w.append(0)
            else: 
                w.append(1)
    
       
# updates the weights of each possible tower by decreasing the weight of all towers
# in the worse iteration that are not in the better iteration 
def updateWeight(tower, success):
    if (success):
        return
    else:
        w[(tower.x*grid_length) + tower.y] = (1-epsilon)*w[(tower.x*grid_length) + tower.y]


def anneal_1(instance: Instance) -> Solution:
    ''' Why cluttering up the main function with move_towers specific stuff?? '''
                # global epsilon
                # global grid_length
                # grid_length = instance.grid_side_length
                # epsilon = (math.log2(grid_length**2)/(temp/cooling))**(1/2)
    # Parameter Variables:
    temp = 10
    cooling = .001

    
    #Function Variables:
    T = [Point(0,0)]
    T_new = []
    sol = Solution(instance=instance, towers=T)
    P_delta = 0

    # Get intial solution
    T_new = solve_move_towers.grid_then_delete(instance).towers
    sol = Solution(instance=instance, towers=T_new)
    true_optimal = sol
    optimal_cost = true_optimal.trainingPenalty()

    while temp > 0:
        # UPDATE FUNCTION:
        T_new = solve_move_towers.move_towers(temp, sol)
        sol_new = Solution(instance=instance, towers=T_new)
        #sol_new.deduplicate()
        P_delta = sol_new.trainingPenalty() - sol.trainingPenalty()
        
        #print('Absolute T-Penalty:', sol_new.trainingPenalty())
        #print('Delta:', P_delta)
    
        #Accept the Update?
        if(P_delta <= 0):
            sol = sol_new
            if (sol_new.trainingPenalty() - optimal_cost <= 0):
                true_optimal = sol_new
                optimal_cost = true_optimal.trainingPenalty()
        else : # if new solution is not better, accept it with a probability of e^(-cost/temp)
            try:
                res = math.exp(-P_delta/temp)
            except OverflowError:
                res = 0
            if random.uniform(0,1) <= res:
                sol = Solution(instance=instance, towers=T_new)
        temp -= cooling
            
    sol.deduplicate()

    # Remove Useless towers at end
    # - ones that dont cover any towers
    # - ones that redudantly cover
    tower_iterator = true_optimal.towers.copy()
    towers_new = true_optimal.towers.copy()
    for tower in tower_iterator:
        dupe_towers = towers_new.copy()
        dupe_towers.remove(tower)
        dupe_sol = Solution(instance=instance, towers=dupe_towers)
        if dupe_sol.valid():
            towers_new.remove(tower)
    
    true_optimal = Solution(instance=instance, towers=towers_new)
    return true_optimal

def anneal_greed(instance: Instance) -> Solution:
    ''' Why cluttering up the main function with move_towers specific stuff?? '''
                # global epsilon
                # global grid_length
                # grid_length = instance.grid_side_length
                # epsilon = (math.log2(grid_length**2)/(temp/cooling))**(1/2)
    # Parameter Variables:
    temp = 1.5
    cooling = .001

    
    #Function Variables:
    T = [Point(0,0)]
    T_new = []
    sol = Solution(instance=instance, towers=T)
    P_delta = 0

    # Get intial solution
    T_new = greedy(instance).towers
    sol = Solution(instance=instance, towers=T_new)
    true_optimal = sol
    sol_cost = true_optimal.trainingPenalty()
    optimal_cost = sol_cost

    while temp > 0:
        # UPDATE FUNCTION:
        T_new = solve_move_towers.move_towers(temp, sol)
        sol_new = Solution(instance=instance, towers=T_new)
        new_cost = sol_new.trainingPenalty()
        sol_new.deduplicate()
        P_delta = new_cost - sol_cost
        '''
        if (logic == 0): #move
            P_delta = sol_new.penalty_local(new) - sol.penalty_local(old)

        if (logic == 1): #add
            P_delta = sol_new.penalty_local(new) - sol.penalty_local(old)
        
        if (logic == 2): #remove
            P_delta = sol_new.penalty_local(new) - sol.penalty_local(old)
        '''
        #Accept the Update?
       
        if(P_delta <= 0):
            sol = sol_new
            if (new_cost - optimal_cost <= 0):
                true_optimal = sol_new
                optimal_cost = new_cost
        else : # if new solution is not better, accept it with a probability of e^(-cost/temp)
            try:
                res = math.exp(-P_delta/temp)
            except OverflowError:
                res = 0
            if random.uniform(0,1) <= res:
                sol = Solution(instance=instance, towers=T_new)
        temp -= cooling
            
    sol.deduplicate()

    # Remove Useless towers at end
    # - ones that dont cover any towers
    # - ones that redudantly cover
    tower_iterator = true_optimal.towers.copy()
    towers_new = true_optimal.towers.copy()
    for tower in tower_iterator:
        dupe_towers = towers_new.copy()
        dupe_towers.remove(tower)
        dupe_sol = Solution(instance=instance, towers=dupe_towers)
        if dupe_sol.valid():
            towers_new.remove(tower)
    
    true_optimal = Solution(instance=instance, towers=towers_new)
    print(true_optimal.penalty())
    return true_optimal
#--------------------------------------
#---------------------------------------

#--------------------------------------
#---------------------------------------

#--------------------------------------
#---------------------------------------






























































def ann_2(instance: Instance) -> Solution:
    initial = 90 #initial temp of the system
    final = .1 # final temp of the system
    delta = .1 #change in tempurature per itteration
    epsilon = 0.1 # update rule for MWU - how fast probabilities converge to 0
    current = initial
    solution = current
    weights = []
    init_MW(weights, instance)
    tower_choices = range(0, grid_length**2) #initialize an array with value the same length as the weights array and the indices as its values
    curr_towers = generate_new_towers(weights, tower_choices, instance)
    while current > final: # this amount of itterations

        ### REPLACE WITH GENERATE NEIGHBOR which will use MW. NEED TO CONSIDER SWAP AND REMOVE
        #add_posx = random.choice(range(0, instance.grid_side_length))
        #add_posy = random.choice(range(0, instance.grid_side_length))
        #temp_point = [add_posx, add_posy] #new tower to consider adding
        ###

        tower_choices = range(0, grid_length**2) #initialize an array with value the same length as the weights array and the indices as its values
        new_towers = generate_new_towers(weights, tower_choices, instance) # generate an a new set of towers based on weights

        sol = Solution(instance=instance, towers=curr_towers)
        sol_new = Solution(instance=instance, towers=new_towers)
        print(current)
        cost_diff = sol.penalty() - sol_new.penalty() #diffrence between adding and not adding that tower
        #^ might need to edit so that if it fails it has a weight of inf instead of the penalty only
        # we might need to consider the cost diff of replacing a random tower with the newly calcualted on
        #ex we are purely choosing randomly but we need to implement a way to do it based on multiplicative weights 
        # this is where I would try and implement multiplicative weights function TODO
        
        if cost_diff > 0: # if new solution is better, keep that set of towers for now
            curr_towers = sol_new.towers
            updateWeights(weights, new_towers, curr_towers, epsilon, instance) # decrease weight of the old towers that differ from new towers
        else:
            updateWeights(weights, curr_towers, new_towers, epsilon, instance) # decrease weight of the new towers that differ from old towers

        
#        if cost_diff > 0:
#            curr_towers = curr_towers.append(temp_point)
#        else : # if new solution is not better, accept it with a probability of e^(-cost/temp)
#            if random.uniform(0,1) < math.exp(-cost_diff/current):
#                curr_towers = curr_towers.append(temp_point)
        current -= delta
    final = Solution(instance=instance, towers=curr_towers)
    return final

# generates new towers based on weights
def generate_new_towers(weights, choices, instance):
    towers = set()
    dist = [w / sum(weights) for w in weights] # probability distribution derived from weights
    initial_index = random.choices(choices, weights=dist, k=1)[0]
    initial_tower = Point(initial_index // grid_length, initial_index % grid_length) # choose an initial tower
    towers.add(initial_tower)
    while not fully_covered(towers, instance): # while we don't have full coverage
        chosen_index = random.choices(choices, weights=dist, k=1)[0]
        tower = Point(chosen_index // grid_length, chosen_index % grid_length)
        towers.add(tower)
    return towers


#could maybe implement with tabu search 
#NEED TO IMPLEMENT!
#the idea I was thinking of is to add a point, remove a point.
#https://stackoverflow.com/questions/45449762/how-to-find-neighboring-solutions-in-simulated-annealing
def Gen_neighbor(weights):
    random.choice(w) # need to return the idex not the elem.
    return weights    

SOLVERS: Dict[str, Callable[[Instance], Solution]] = {
    "naive": solve_naive,
    "grid": solve_move_towers.grid_then_delete,
    "anneal_1": anneal_1,
    "anneal_2": ann_2,
    "anneal_greed": anneal_greed,
    "greedy": greedy
}

# You shouldn't need to modify anything below this line.
def infile(args):
    if args.input == "-":
        return StdinFileWrapper()

    return Path(args.input).open("r")

def outfile(args):
    if args.output == "-":
        return StdoutFileWrapper()

    return Path(args.output).open("w")

def main(args):
    with infile(args) as f:
        instance = Instance.parse(f.readlines())
        solver = SOLVERS[args.solver]
        solution = solver(instance)
        #assert solution.valid() FIXME UNCOMMENT!!
        with outfile(args) as g:
            print("# Penalty: ", solution.penalty(), file=g)
            solution.serialize(g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve a problem instance.")
    parser.add_argument("input", type=str, help="The input instance file to "
                        "read an instance from. Use - for stdin.")
    parser.add_argument("--solver", required=True, type=str,
                        help="The solver type.", choices=SOLVERS.keys())
    parser.add_argument("output", type=str, 
                        help="The output file. Use - for stdout.", 
                        default="-")
    main(parser.parse_args())
