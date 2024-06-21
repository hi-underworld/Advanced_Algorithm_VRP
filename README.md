# Advanced_Algorithm_VRP(高级算法大作业-无人机配送路径规划问题)

## 1、问题描述

无人机可以快速解决最后10 公里的配送，本作业要求设计一个算法，实现如下图所示区域
的无人机配送的路径规划。在此区域中，共有j 个配送中心，任意一个配送中心有用户所需
要的商品，其数量无限，同时任一配送中心的无人机数量无限。该区域同时有 k 个卸货点
（无人机只需要将货物放到相应的卸货点即可），假设每个卸货点会随机生成订单，一个订
单只有一个商品，但这些订单有优先级别，分为三个优先级别（用户下订单时，会选择优先
级别，优先级别高的付费高）：

- 一般：3 小时内配送到即可； 
- 较紧急：1.5 小时内配送到； 
- 紧急：0.5 小时内配送到。 

将时间离散化，也就是每隔t 分钟，所有的卸货点会生成订单（0-m 个订单），同时每
隔t 分钟，系统要做成决策，包括

- 哪些配送中心出动多少无人机完成哪些订单； 
- 每个无人机的路径规划，即先完成那个订单，再完成哪个订单，...，最后返回原来的配送
  中心；

（ps：系统做决策时，可以不对当前的某些订单进行配送，因为当前某些订单可能紧急程度
不高，可以累积后和后面的订单一起配送。）



## 2、目标

一段时间内（如一天），所有无人机的总配送路径最短



## 3、约束条件

满足订单的优先级别要求



## 4、基本假设

- 无人机一次最多只能携带n 个物品；
- 无人机一次飞行最远路程为20 公里（无人机送完货后需要返回卸货点）； 
- 无人机的速度为60 公里/小时； 
- 配送中心的无人机数量无限； 
- 任意一个配送中心都能满足用户的订货需求；



## 5、VRP&TWVRP简介

车辆路径规划问题（VRP） 是一个NP-hard问题。这意味着在一般情况下，没有已知的高效算法可以在多项式时间内解决它。具体来说，VRP问题属于组合优化问题的类别，其复杂性源于需要考虑多个约束条件，如车辆容量、路径长度、时间窗口等。虽然存在一些启发式算法和近似算法来解决VRP问题，但在一般情况下，找到确切的最优解仍然是一个挑战。TWVRP 是指时间窗口车辆路径规划问题（Time-Window Vehicle Routing Problem，TWVRP）。在这个问题中，除了考虑车辆容量、路径长度等因素外，还需要满足客户的时间窗口限制。具体来说，每个客户点都有一个指定的时间窗口，在这个时间窗口内必须被服务。TWVRP是VRP问题的一个变体，增加了时间维度的约束，使得路径规划更具挑战性。



## 6、解决方案-带时间窗的订单调度及路径规划模型

### 6.1模型概要

整个模型主要分为三个子模型：

1. **卸货点分类模型：**将所有卸货点依据配送中心进行分类。每个$drop\_point_i$有对应的$depot_j$。 所有卸货点为$drop\_point_i$的订单，在后续的订单调度配送的过程中，均由配送中心$depot_j$的无人机进行配送。
2. **订单调度模型：**将处理订单的最小时间间隔设置为t（即订单生成的时间间隔），每个处理订单的时间间隔内，只处理当前时间间隔内必须送出的订单，进而保证所有订单都能在规定的时间窗口内完成。而对于所有必须在第$i$个时间间隔内必须完成的所有订单，采用订单调度模型对这些订单的配送进行调度，并考虑分配哪些订单到一台无人机，使得一个时间窗口内无人机的路径尽可能短进而逼近整个时间周期内无人机的路径最短
3. **车辆最短路径模型**：所有订单完成调度后，对所有分配了订单的车辆进行路径规划。对于类似TSP问题的订单最短配送路径规划，使用遗传算法搜索使得每台无人的路径尽可能短进而逼近一个时间窗口内无人机的路径最短。



### 6.2、卸货点分类模型

卸货点分类模型旨在简化整个问题。由于n个配送中心到m个卸货点的vrp问题相对复杂，因此考虑使用卸货点分类模型，将问题简化为n个（1个配送中心到k个卸货点的vrp问题）。相当于在订单调度开始前，根据配送中心以及卸货点的相对位置确定每个卸货点的订单由固定的配送中心处理。



#### 6.2.1、主要步骤 

1. **初始化**：随机生成的m个卸货点和n个配送中心，即卸货点列表（`drop_points`）和配送中心列表（`depots`），并设置一个阈值（`therehold`）。 
2.  **统计阈值范围内的配送中心**：对于每个卸货点（`drop_point_i`），计算其阈值范围（半径为`therehold`）内其他卸货点的配送中心数量。 如果某个卸货点`drop_point_j`和`drop_point_i`的距离在阈值范围内且已分配了对应的配送中心，则统计该配送中心的数量。 
3. **分配最接近的配送中心**： 如果以`drop_point_i`为圆心，阈值为半径的范围内有卸货点分配了对应的配送中心，则选择数量最多的配送中心分配给当前卸货点`drop_point_i`。如果没有，则选择距离当前卸货点最近的配送中心。



下面是一个例子，表示如何为卸货点分配配送中心。灰色的点表示`drop_point_i`，阈值为半径的范围内有5个卸货点，相同的颜色表示被分配到同一个配送中心，不同颜色表示被分配到不同的配送中心。因为4个点有2个红色即两个卸货点被分配到了配送中心`depot_j`，数量是最多的，因此，将`drop_point_i`分配到`depot_j`。

 <img src="https://github.com/hi-underworld/Advanced_Algorithm_VRP/blob/main/images/knn.png" alt="initial_plan.drawio" style="zoom: 100%;" />


#### 6.2.2、代码实现

```python
#using the algorithm to find the depot that is closest to the drop point
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

```





### 6.3、订单调度模型

订单调度模型主要是将处理订单的最小时间间隔设置为t（即订单生成的时间间隔），每个处理订单的时间间隔内，只处理当前时间间隔内必须送出的订单，进而保证所有订单都能在规定的时间窗口内完成。而对于所有必须在第$i$​个时间间隔内必须完成的所有订单，在进行订单分配。主要为以下几个步骤：



#### 6.3.1、主要步骤

1. **统计最小订单处理时间间隔内应处理的订单**：对于当前待处理的所有订单，计算从其对应的配送中心（`depot`）直接到订单目的地（`drop_point`）的运输时间。 如果订单的截止时间减去运输时间（即订单最晚开始处理时间）首次小于当前处理窗口的结束时间，则认为订单应当在当前订单处理窗口处理。统计所有应当在当前订单处理时间窗口的订单`due_orders`。以下是一个统计订单的例子，订单的颜色表示应当处理这些订单的时间间隔。**后续的订单均在一个处理订单的最小时间间隔内进行分配调度**。


   <img src="https://github.com/hi-underworld/Advanced_Algorithm_VRP/blob/main/images/due_order.drawio.svg" alt="initial_plan.drawio" style="zoom: 100%;" />


2. **订单分配初始化**：在统计完当前处理订单的最小时间间隔内的订单`due_orders`后，初始化一个订单配送的方案：将所有`drop_point`即卸货点相同的订单`due_orders_drop_point_i`用`num_vehicles_needed`台无人机运送，`num_vehicles_needed`等于`due_orders_drop_point_i`的重量除以一台无人机的承重`capacity`的值向上取整。最终总是会有小于等于k台无人机没有满载，k为这批订单中不同的卸货点的数量。以下是一个订单分配初始化的一个例子：

   <img src="https://github.com/hi-underworld/Advanced_Algorithm_VRP/blob/main/images/initial_plan.drawio.svg" alt="initial_plan.drawio" style="zoom: 100%;" />

   由于步骤1中统计最小订单处理时间间隔内应处理的订单的规则，这样分配的订单一定能够在订单的截止时间前完成。即这种订单分配策略一定合法。

3.  **针对未满载的无人机进行订单再分配**：对于已经满载的无人机，无需在考虑订单再分配，直接运送至该批订单相同的卸货点即可。而对于未满载的无人机，则需要考虑订单的再分配，即不断将两台无人机的订单进行合并由一台无人机完成配送，直到不存在能够合并订单的两台无人机（可能的原因包括无人机满载，无人机运输单次里程达到上限，无人机无法满足订单的截止时间）。这可以降低这批订单无人机的里程，可以通过简单的数学来证明。

   假设现在有两台无人未满载。由于订单初始化的规则，一定不会存在两台无人机，他们装载了卸货点相同的订单。现在无人机`vehicle_i`和`vehicle_j`分别装载了i，j种不同的订单（订单按照不同的卸货点分类）。对于两台无人机的任意一条运输路径`path_i`,` path_j`，定义：
   $$
   path_i = depot - drop\_point_{i1} - ...drop\_point_{ii}-depot\\
   path_j = depot-drop\_point_{j1} - ...drop\_point_{jj}-depot
   $$
   对于`path_i`,` path_j`的合并运输路径`path_{i+j}`,定义：
   $$
   path_{i+j} = depot - drop\_point_{i1} - ...drop\_point_{ii}-drop\_point_{jj} - ...drop\_point_{j1}-depot
   $$
   因为
   $$
   dist（drop\_point_{ii}, depot）+ dist（drop\_point_{jj}, depot）> dist（drop\_point_{ii}, drop\_point_{jj}）
   $$
   所以
   $$
   dist（path_i）+ dist(path_j) > dist(path_{i + j})
   $$
   
   <img src="https://github.com/hi-underworld/Advanced_Algorithm_VRP/blob/main/images/proof.drawio.svg" alt="proof.drawio" style="zoom: 100%;" />

   因此，可以得出结论，在满足约束条件的情况下，不断地合并无人机的订单，一定可以降低一个订单处理时间间隔内无人机的总的运输里程。



#### 6.3.2、代码实现

1. **统计最小订单处理时间间隔内应处理的订单**

   ```python
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
   ```

   

2. **订单分配初始化**

   ```python
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
   ```

   

3. **针对未满载的无人机进行订单再分配**

   ```python
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
   ```

这些代码中还有一些检查无人机订单分配的约束满足条件以及订单时间约束满足条件的函数。这里没有附上，后面会给全部的源码。



### 6.4、车辆最短路径模型

在完成了订单的分配之后，每台无人机的订单确定。则需要在订单确定的情况下进行路径规划，以最小化无人机的运输里程。这个问题类似TSP问题，不过带有时间约束，即满足该台无人机的订单的时间需求的约束，求最短路径。考虑使用遗传算法求解，

#### 6.4.1、主要步骤 

1. **初始化种群**：初始化种群列表`route_population`。 生成初始种群，数量为`initial_population_num`。每条路线为所有订单的卸货点的随机排列。

2.  **遗传算法迭代**：遗传算法的迭代次数为`generation_num`。 在每一代中执行以下操作：选择满足时间窗口约束以及无人机里程上限约束的路线`selected_routes`。 计算选择路线的适应度，即路线总距离`selected_fitness_distances`。 选择适应度最小的路线作为最优路线`optimal_route`。对选择的路线进行排序，并选择一半的路线生成下一代。对选中的路线进行变异，生成新的路线。

   由于路径需要合法，且每个卸货点只经过一次，因此，变异策略不能使用传统的变异策略，变异只能发生再一个个体上。变异策略考虑交换随机次数的随机两个卸货点的位置。

   

3.  **更新车辆最短路径**：每代选出最短的路径，经过一定的遗传代数后，选出整个过程中最短的路径作为该台无人机的最短路径。

对一个订单处理时间间隔内，对已经完成订单分配的所有无人机使用遗传算法搜索满足约束的最短路径。



#### 6.4.2、代码实现

```python
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
```





## 7、附件

### 7.1、完整代码(vrp.py)

```python
import random
import math
from typing import List, Tuple
import json

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
    
    #write all the vehicles' orders and routes to a json file
    vehicles = []
    for vehicle in all_vehicles:
        vehicle_dict = {}
        vehicle_dict['vehicle_id'] = vehicle.vehicle_id
        vehicle_dict['orders'] = []
        for order in vehicle.orders:
            order_dict = {}
            order_dict['order_id'] = order.order_id
            order_dict['destination_id'] = order.destination.id
            order_dict['demand'] = order.demand
            order_dict['time_window'] = order.time_window
            order_dict['priority'] = order.priority
            vehicle_dict['orders'].append(order_dict)
        vehicle_dict['route'] = [drop_point.id for drop_point in vehicle.route]
        vehicles.append(vehicle_dict)
    
    with open('vehicles.json', 'w') as f:
        json.dump(vehicles, f, indent=4)

if __name__ == "__main__":
    main()

```



### 7.2、代码运行说明及结果

执行以下命令即可模拟订单生成并执行上述模型的操作

```shell
python3 vrp.py
```

运行结果保存在vehicles.json文件中,例子如下：

```json
[
    {
        "vehicle_id": 1,
        "orders": [
            {
                "order_id": 1,
                "destination_id": 2,
                "demand": 1,
                "time_window": [
                    0,
                    30
                ],
                "priority": 3
            },
            {
                "order_id": 10,
                "destination_id": 6,
                "demand": 1,
                "time_window": [
                    0,
                    30
                ],
                "priority": 3
            },
            {
                "order_id": 12,
                "destination_id": 6,
                "demand": 1,
                "time_window": [
                    0,
                    30
                ],
                "priority": 3
            },
            {
                "order_id": 13,
                "destination_id": 6,
                "demand": 1,
                "time_window": [
                    0,
                    30
                ],
                "priority": 3
            }
        ],
        "route": [
            6,
            2
        ]
    },
```

