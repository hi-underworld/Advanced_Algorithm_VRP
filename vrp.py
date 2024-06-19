import random
import math
from typing import List, Tuple



class Vehicle:
    def __init__(self, vehicle_id: int, capacity: int, speed: int ):
        self.vehicle_id = vehicle_id
        self.capacity = capacity
        self.route = []
        self.load = 0
        self.speed = speed  # 1 unit distance per minute
        self.orders = []
        

class DropPoint:
    def __init__(self, x: float, y: float,id:int):
        self.x = x
        self.y = y
        self.id = id
        self.depot = None

class Order:
    def __init__(self, order_id: int,  destination: DropPoint, demand: int, time_window: Tuple[int, int], priority: int):
        self.order_id = order_id
        self.destination = destination
        self.demand = demand
        self.time_window = time_window
        self.priority = priority
        

class DePot:
    def __init__(self, x: float, y: float, id:int):
        self.x = x
        self.y = y
        self.id =id
        


def initialize_vehicles(num_vehicles: int, capacity: int, speed: int) -> List[Vehicle]:
    vehicles = []
    for i in range(num_vehicles):
        vehicles.append(Vehicle(i, capacity, speed))
    return vehicles

def initialize_drop_points(num_drop_points: int,max_distance:float) -> List[DropPoint]:
    drop_points = []
    thetas = [random.uniform(0, 2 * math.pi) for _ in range(num_drop_points)]
    distances = [random.uniform(0, max_distance) for _ in range(num_drop_points)]

    drop_point = [(distances[i]* math.cos(thetas[i]), distances[i] * math.sin(thetas[i])) for i in range(len(thetas))]
    for k in range(num_drop_points):
        x = drop_point[k][0]
        y = drop_point[k][1]
        drop_points.append(DropPoint(x, y, k + 1))
    return drop_points

def initialize_depots(num_depots, radium: float) -> List[DePot]:
    thetas = [random.uniform(0, 2 * math.pi) for _ in range(num_depots)]
    depots = []
    depot = [(radium* math.cos(thetas[i]), radium * math.sin(thetas[i])) for i in range(len(thetas))]
    for k in range(num_depots):
        x = depot[k][0]
        y = depot[k][1]
        depots.append(DePot(x, y, k+1))
    return depots

#using the KNN algorithm to find the depot that is closest to the drop point
def find_closest_depot(drop_points: List[DropPoint], depots: List[DePot], therehold: float):
    for drop_point in drop_points:
        # 统计当前drop point半径therehold范围内的drop point的对应的不同depot的数量
        depot_count = {}
        for drop_point_ in drop_points:
            if drop_point_ != drop_point:
                distance = calculate_distance([drop_point.x, drop_point.y], [drop_point_.x, drop_point_.y])
                if distance <= therehold and drop_point_.depot is not None:
                    if drop_point_.depot.id in depot_count:
                        depot_count[drop_point_.depot.id] += 1
                    else:
                        depot_count[drop_point_.depot.id] = 1

        # 找到当前drop point半径therehold范围内的drop point的对应的不同depot的数量最多的depot,并将当前drop point的depot设置为这个depot
        if len(depot_count) != 0:
            max_count = max(depot_count.values())
            for depot_id, count in depot_count.items():
                if count == max_count:
                    drop_point.depot = depots[depot_id-1]
                    print(f"Drop point {drop_point.id} is closest to depot {depot_id}.")
                    break
        else:
            # 如果当前drop point半径therehold范围内没有drop point对应的depot,则将当前drop point的depot设置为距离当前drop point最近的depot
            min_distance = float('inf')
            for depot in depots:
                distance = calculate_distance([drop_point.x, drop_point.y], [depot.x, depot.y])
                if distance < min_distance:
                    min_distance = distance
                    drop_point.depot = depot
            print(f"Drop point {drop_point.id} is closest to depot {drop_point.depot.id}.")

# classify the orders by its destination belonging to the sam depot
def classify_orders_by_depot(orders: List[Order]) -> List[Order]:
    orders_by_depot = {}
    for order in orders:
        if order.destination.depot.id in orders_by_depot:
            orders_by_depot[order.destination.depot.id].append(order)
        else:
            orders_by_depot[order.destination.depot.id] = [order]
    return orders_by_depot


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def generate_orders(current_orders_num: int, drop_points: List[DropPoint], current_time: int, max_orders: int) -> List[Order]:
    orders = []
    order_id = current_orders_num
    for i in range(len(drop_points)):
        for j in range(random.randint(0, max_orders)):
            order_id += 1
            destination = drop_points[i]
            demand = 1  # Assuming each order has a demand of 1
            priority = random.choice([1, 2, 3])
            if priority == 1:
                time_window = (current_time, current_time + 180)  # 3 hours
            elif priority == 2:
                time_window = (current_time, current_time + 90)   # 1.5 hours
            else:
                time_window = (current_time, current_time + 30)   # 0.5 hour
            #print(order_id, destination,destination.id, demand, time_window, priority)
            orders.append(Order(order_id,destination , demand, time_window, priority))
    return orders

def sort_orders_by_window_end(orders: List[Order]) -> List[Order]:
    return sorted(orders, key=lambda order: order.time_window[1])

def check_orders_due(orders: List[Order], vehicle_speed: int, dealing_window:Tuple[float, float] ) -> List[Order]:
    due_orders = []
    for order in orders:
        depot = order.destination.depot
        depot = [depot.x, depot.y]
        transport_time = calculate_distance(depot, [order.destination.x, order.destination.y]) / vehicle_speed
        if order.time_window[1] -  transport_time <= dealing_window[1]: 
            # print(f"order.time_window:{order.time_window})")
            # print(f"dealing_window:{dealing_window})")
            # print(f"Transport time: {transport_time}")
            due_orders.append(order)
    if len(due_orders) == 0:
        return None
    else:
        print(len(due_orders))
        for due_order in due_orders:
            print(f"Due orders at drop point {due_order.destination.id}:")
            print(f"Order ID: {due_order.order_id}, destination.id: {due_order.destination.id},depot_id:{due_order.destination.depot.id} Demand: {due_order.demand}, Time Window: {due_order.time_window}, Priority: {due_order.priority}")
        return due_orders

# remove the orders that are due from the orders_to_be_delivered list
def remove_due_orders(orders_to_be_delivered: List[Order], due_orders:List[Order]) -> List[Order]:
    for due_order in due_orders:
        orders_to_be_delivered.remove(due_order)
    return orders_to_be_delivered

# allocate the orders which are due to the same destination to the same vehicle
def initial_orders_to_vehicle(orders:List[Order], vehicle_id: int) -> List[Vehicle]:
    allocated_vehicles = []
    vehicle_capacity = 20
    speed = 1
    # calculate the number of vehicles needed to deliver the orders
    num_vehicles_needed = math.ceil(sum([order.demand for order in orders]) / vehicle_capacity)
    print(f"")
    for i in range(num_vehicles_needed):
        allocated_vehicles = initialize_vehicles(num_vehicles_needed, vehicle_capacity, 1)
    
    # allocate the orders to the vehicles
    id = vehicle_id
    i = 0
    while len(orders) != 0:
        while allocated_vehicles[i].load < vehicle_capacity:
            allocated_vehicles[i].vehicle_id = id + i  + 1
            allocated_vehicles[i].orders.append(orders.pop(0))
            allocated_vehicles[i].load += 1
            if allocated_vehicles[i].load == vehicle_capacity:
                i += 1
            if len(orders) == 0:
                # print("All orders are allocated to vehicles.")
                break
    return allocated_vehicles

# initialize the vehicle plan for the orders that are due in the current dealing window, orders's destination are the same can be delivered by the same vehicle
def initial_delivery_plan(due_orders:List[Order] , all_vehicles:List[Vehicle], depot: Tuple[float, float], current_time:float) -> List[Vehicle]:
    # classify the orders by its destination
    orders_by_destination = {}
    for order in due_orders:
        if order.destination.id in orders_by_destination:
            orders_by_destination[order.destination.id].append(order)
        else:
            orders_by_destination[order.destination.id] = [order]
    
    allocated_vehicles = []
    vehicle_id = len(all_vehicles)
    # initialize the route for each vehicle, each vechicle deals with the orders that are due at the same drop point
    for destination_id, orders in orders_by_destination.items():
        if len(orders) != 0:
            allocated_vehicles_i = initial_orders_to_vehicle(orders, vehicle_id = vehicle_id)
            vehicle_id += len(allocated_vehicles_i)
            # print(f"Number of vehicles allocated: {len(allocated_vehicles_i)}")
            allocated_vehicles.extend(allocated_vehicles_i)

    # generate the route for each vehicle
    for vehicle in allocated_vehicles:
        check_path(vehicle.orders, depot, vehicle, current_time, route=None)

    return allocated_vehicles

# calculate the distance the vehicle needs to travel to deliver the orders
def calculate_route_distance(vehicle: Vehicle, depot: Tuple[float, float]) -> float:
    route_distance = 0
    for i in range(len(vehicle.orders)):
        if i == 0:
            route_distance += calculate_distance(depot, [vehicle.orders[i].destination.x, vehicle.orders[i].destination.y])
        else:
            route_distance += calculate_distance([vehicle.orders[i-1].destination.x, vehicle.orders[i-1].destination.y], [vehicle.orders[i].destination.x, vehicle.orders[i].destination.y])
    route_distance += calculate_distance([vehicle.orders[-1].destination.x, vehicle.orders[-1].destination.y], depot)
    return route_distance

 

# aggregate vechicles until there is no more vehicle can be aggregated
def vehicles_aggregate(vehicles: List[Vehicle], depot: Tuple[float, float],current_time:float) -> List[Vehicle]:
    start_id = vehicles[0].vehicle_id 
    aggregatation_flag = True
    while aggregatation_flag:
        aggregatation_flag = False
        for i in range(len(vehicles)):
            for j in range(i+1, len(vehicles)):
                #print(f"i: {i}, j: {j}")
                if check_aggregate_capacity(vehicles[i], vehicles[j]) and check_aggregate_time(vehicles[i], vehicles[j], depot, current_time):
                    vehicles[i].orders.extend(vehicles[j].orders)
                    vehicles[i].load += vehicles[j].load
                    vehicles.pop(j)
                    aggregatation_flag = True
                    break
            if aggregatation_flag:
                break
    
    for vehicle in vehicles:
        vehicle.vehicle_id = start_id
        start_id += 1

    return vehicles
                
# check the capacity of the aggregated vehicle
def check_aggregate_capacity(vehicle1: Vehicle, vehicle2: Vehicle) -> bool:
    capacity = vehicle1.capacity
    if vehicle1.load + vehicle2.load <= capacity:
        #print("The aggregated vehicle has enough capacity.")
        return True
    else:
        #print("The aggregated vehicle does not have enough capacity.")
        return False

# check the time is enough for the aggregated vehicle to deliver the orders
def check_aggregate_time(vehicle1: Vehicle, vehicle2: Vehicle, depot: Tuple[float, float], start_time:float) -> bool:
    orders = []
    orders.extend(vehicle1.orders)
    orders.extend(vehicle2.orders)
    if check_path(orders, depot, vehicle1,start_time, route=None):
        #print("The aggregated vehicle can deliver the orders in time.")
        return True
    else:
        #print("The aggregated vehicle cannot deliver the orders in time.")
        return False


def generate_path_orders_from_orders(orders: List[Order], speed: float,route:List[DropPoint] ) -> List[Order]:
    path_orders = []
    speed = speed
        
    #classify the orders by its destination
    orders_by_destination = {}
    for order in orders:
        if order.destination.id in orders_by_destination:
            orders_by_destination[order.destination.id].append(order)
        else:
            orders_by_destination[order.destination.id] = [order]
    
    # find the orders that are due the earliest at each drop point
    for destination_id, orders in orders_by_destination.items():
        orders = sort_orders_by_window_end(orders)
        path_orders.append(orders[0])
    
    sorted_path_orders = []
    if route is not None:
        for i in range(len(route)):
            destination_id = route[i].id
            sorted_path_order = [order for order in path_orders if order.destination.id == destination_id]
            sorted_path_orders.extend(sorted_path_order)
    else:
        sorted_path_orders = path_orders
    
    return sorted_path_orders


#check the path is available for the vehicle to deliver the orders
def check_path(vehicle_orders: List[Order], depot: Tuple[float, float], vehicle:Vehicle, start_time:float, route:List[DropPoint]) -> bool:
    vehicle_orders = sort_orders_by_window_end(vehicle_orders)
    speed = vehicle.speed
    path_orders = generate_path_orders_from_orders(vehicle_orders, speed, route)
    real_window_ends = []
    # calculate the real window end for each order
    current_time = start_time
    for i in range(len(path_orders)):
        if i == 0:
            transport_time = calculate_distance(depot, [path_orders[i].destination.x, path_orders[i].destination.y]) / speed
            real_window_end = current_time + transport_time
            real_window_ends.append(real_window_end)
            current_time = real_window_end
        else:
            transport_time = calculate_distance([path_orders[i-1].destination.x, path_orders[i-1].destination.y], [path_orders[i].destination.x, path_orders[i].destination.y]) / speed
            real_window_end = current_time + transport_time
            real_window_ends.append(real_window_end)
            current_time = real_window_end

    for i in range(len(path_orders)):
        if real_window_ends[i] > path_orders[i].time_window[1]:
            return False
        
    vehicle_route = []
    for path_order in path_orders:
        vehicle_route.append(path_order.destination)
    vehicle.route = vehicle_route
    return True


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
        

#simulate the whole process
def main():
    num_drop_points = 20
    num_depots = 5
    vehicle_speed = 1  # 1 unit distance per minute
    max_distance = 5
    depot = (0, 0)
    
    time_interval = 12  # minutes
    time_sensitivity = 1 # minutes
    simulation_duration = 8 * 60  # 8 hours
    max_orders = 5  # Max orders generated per interval per drop point

    #generate drop points which distances from depot limited to max_distance randomly
    drop_points = initialize_drop_points(num_drop_points,max_distance)
    
    #generate depots which distances from the origin limited to max_distance randomly
    depots = initialize_depots(num_depots, max_distance)

    current_time = 0

    orders_to_be_delivered = []

    all_vehicles = []
    current_orders_num = 0
    dealing_window = [0,0]

    #find the closest depot for each drop point
    find_closest_depot(drop_points, depots, therehold=0.5)


    while current_time < simulation_duration:
        if current_time % time_interval == 0:
            new_orders = generate_orders(current_orders_num,drop_points, current_time, max_orders)
            current_orders_num += len(new_orders)
            orders_to_be_delivered.extend(new_orders)
            orders_to_be_delivered = sort_orders_by_window_end(orders_to_be_delivered)
            dealing_window = [current_time, current_time + time_interval]
           
            #check orders due in the current dealing  window
            due_orders = check_orders_due(orders_to_be_delivered, vehicle_speed=vehicle_speed, dealing_window=dealing_window)

            if not due_orders:
                print("No orders due in the current dealing window.")
               
            else:
                due_orders_by_depots = classify_orders_by_depot(due_orders)
                for depot_id, due_orders_by_depot in due_orders_by_depots.items():
                    print(f'dealing the orders starting from depot {depot_id}...')
                    depot = depots[depot_id-1]
                    depot = [depot.x, depot.y]
                    #initialize the vehicle plan for the orders that are due in the current dealing window, orders's destination are the same can be delivered by the same vehicle
                    allocated_vehicles = initial_delivery_plan(due_orders_by_depot, all_vehicles, depot, current_time)

                    #aggregate vechicles until there is no more vehicle can be aggregated
                    aggregated_vehicles = vehicles_aggregate(allocated_vehicles, depot, current_time)

                    for aggregated_vehicle in aggregated_vehicles:
                        #using the GA to find the shortest path for the vehicle to deliver the orders
                        print(f"searching for the shortest path for vehicle {aggregated_vehicle.vehicle_id}")
                        aggregated_vehicle = find_shortest_path_GA(aggregated_vehicle, depot)


                    #print the aggregated vehicles' orders and routes
                    for aggregated_vehicle in aggregated_vehicles:
                        print(f"Aggregated vehicle {aggregated_vehicle.vehicle_id} has orders:")
                        for order in aggregated_vehicle.orders:
                            print(f"Order ID: {order.order_id}, destination.id: {order.destination.id}, Demand: {order.demand}, Time Window: {order.time_window}, Priority: {order.priority}")
                        print(f"Aggregated vehicle {aggregated_vehicle.vehicle_id} has route:")
                        for drop_point in aggregated_vehicle.route:
                            print(f"Drop point ID: {drop_point.id}")

                    all_vehicles.extend(aggregated_vehicles)

                    #remove the orders that are due from the orders_to_be_delivered list
                    orders_to_be_delivered = remove_due_orders(orders_to_be_delivered, due_orders_by_depot)
                    print(len(orders_to_be_delivered))
            

        current_time += time_sensitivity

if __name__ == "__main__":
    main()
