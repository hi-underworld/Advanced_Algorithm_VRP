from typing import List, Tuple
import random
import math
from utils import Vehicle, DropPoint, calculate_distance
from orders_processing import check_path

#using the GA to find the shortest path for the vehicle to deliver the orders
def find_shortest_path_GA(vehicle:Vehicle, depot: Tuple[float, float]) -> Vehicle:
    #initialize the population
    route_population = []

    initial_population_num = 100
    generation_num = 100
    # generate the initial population
    for i in range(initial_population_num):
        route = vehicle.route.copy()
        random.shuffle(route)
        if route not in route_population:
            route_population.append(route)


    if len(route_population) <= initial_population_num/2:
        generation_num = 1


    for generation in range(generation_num):
        print(f"Generation: {generation}")

        #select the route which satisfy the i time window constraints
        selected_routes = []
        for i in range(len(route_population)):
            if check_path(vehicle.orders, depot, vehicle, start_time=0, route=route_population[i]):
                selected_routes.append(route_population[i])
        

        #calculate the fitness of the selected routes
        selected_fitness_distances = []
        for route in selected_routes:
            fitness_distance = 0
            for i in range(len(route)):
                if i == 0:
                    fitness_distance += calculate_distance(depot, [route[i].x, route[i].y])
                else:
                    fitness_distance += calculate_distance([route[i-1].x,route[i-1].y], [route[i].x, route[i].y])
            fitness_distance += calculate_distance([route[-1].x,route[-1].y], depot)
            selected_fitness_distances.append(fitness_distance)
        

        #select the shortest route as the optimal route
        optimal_route = selected_routes[0]
        print(f"Optimal route: {optimal_route}")


        #sort the selected routes(List[DropPoint]) by the fitness distance and select half of the routes to generate the next generation
        zip_selected_routes = zip(selected_routes, selected_fitness_distances)
        for route, fitness_distance in sorted(zip_selected_routes, key=lambda x: x[1]):
            selected_routes.append(route)
        selected_routes = selected_routes[: math.ceil(len(selected_routes)/2)]
        
        next_generation = selected_routes

        for i in range(len(selected_routes)):
            mutation_rate = 0.3
            #geneerate a probability between 0 and 1, if the probability is less than the mutation rate, the route will be mutated
            if random.random() < mutation_rate:
                mutated_route = mutation(selected_routes[i])
                if mutated_route not in next_generation:
                    next_generation.append(mutated_route)


        route_population = next_generation
        if len(route_population) == 1:
            break
        
    vehicle.route = optimal_route
    return vehicle

def mutation(selected_route:List[DropPoint]) -> List[DropPoint]:
    num_mutated_points = random.randint(1, len(selected_route))
    # shuffle the mutated points in the selected route
    for i in range(num_mutated_points):
        k = random.randint(0, len(selected_route)-1)
        j = random.randint(0, len(selected_route)-1)
        if k != j:
            selected_route[k], selected_route[j] = selected_route[j], selected_route[k]
    return selected_route