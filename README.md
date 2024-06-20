# Advanced_Algorithm_VRP(高级算法大作业-无人机配送路径规划问题)

## 问题描述

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



## 目标

一段时间内（如一天），所有无人机的总配送路径最短



## 约束条件

满足订单的优先级别要求



## 基本假设

- 无人机一次最多只能携带n 个物品；
- 无人机一次飞行最远路程为20 公里（无人机送完货后需要返回配送点）； 
- 无人机的速度为60 公里/小时； 
- 配送中心的无人机数量无限； 
- 任意一个配送中心都能满足用户的订货需求；

## VRP&TWVRP简介
车辆路径规划问题（VRP） 是一个NP-hard问题。这意味着在一般情况下，没有已知的高效算法可以在多项式时间内解决它。具体来说，VRP问题属于组合优化问题的类别，其复杂性源于需要考虑多个约束条件，如车辆容量、路径长度、时间窗口等。虽然存在一些启发式算法和近似算法来解决VRP问题，但在一般情况下，找到确切的最优解仍然是一个挑战。TWVRP 是指时间窗口车辆路径规划问题（Time-Window Vehicle Routing Problem，TWVRP）。在这个问题中，除了考虑车辆容量、路径长度等因素外，还需要满足客户的时间窗口限制。具体来说，每个客户点都有一个指定的时间窗口，在这个时间窗口内必须被服务。TWVRP是VRP问题的一个变体，增加了时间维度的约束，使得路径规划更具挑战性。

## 解决方案-带时间窗的订单调度模型

### 模型概要
整个模型主要分为5个模块：

卸货点分类模型：将所有卸货点依据配送中心进行分类。每隔$drop_point_i$有对应的$depot_i$。 所有卸货点为drop_point_i的订单，在后续的订单调度配送的过程中，均由配送中心depot_i的无人机进行配送。

车辆规划与路径优化：

对每个配送中心的到期订单，初始化车辆计划，并尝试合并车辆以减少运输成本。
使用遗传算法（GA）寻找车辆的最短配送路径。
打印每辆车的订单和路线信息。
订单处理：

从订单列表中移除已完成的订单，继续处理剩余订单。


# Advanced_Algorithm_VRP(高级算法大作业-无人机配送路径规划问题)

[TOC]







## 问题描述

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



## 目标

一段时间内（如一天），所有无人机的总配送路径最短



## 约束条件

满足订单的优先级别要求



## 基本假设

- 无人机一次最多只能携带n 个物品；
- 无人机一次飞行最远路程为20 公里（无人机送完货后需要返回卸货点）； 
- 无人机的速度为60 公里/小时； 
- 配送中心的无人机数量无限； 
- 任意一个配送中心都能满足用户的订货需求；



## VRP&TWVRP简介

车辆路径规划问题（VRP） 是一个NP-hard问题。这意味着在一般情况下，没有已知的高效算法可以在多项式时间内解决它。具体来说，VRP问题属于组合优化问题的类别，其复杂性源于需要考虑多个约束条件，如车辆容量、路径长度、时间窗口等。虽然存在一些启发式算法和近似算法来解决VRP问题，但在一般情况下，找到确切的最优解仍然是一个挑战。TWVRP 是指时间窗口车辆路径规划问题（Time-Window Vehicle Routing Problem，TWVRP）。在这个问题中，除了考虑车辆容量、路径长度等因素外，还需要满足客户的时间窗口限制。具体来说，每个客户点都有一个指定的时间窗口，在这个时间窗口内必须被服务。TWVRP是VRP问题的一个变体，增加了时间维度的约束，使得路径规划更具挑战性。



## 解决方案-带时间窗的订单调度模型

### 模型概要

整个模型主要分为三个子模型：

卸货点分类模型：将所有卸货点依据配送中心进行分类。每个$drop\_point_i$有对应的$depot_j$。 所有卸货点为$drop\_point_i$的订单，在后续的订单调度配送的过程中，均由配送中心$depot_j$的无人机进行配送。

订单调度模型：将处理订单的最小时间间隔设置为t（即订单生成的时间间隔），每个处理订单的时间间隔内，只处理当前时间间隔内必须送出的订单，进而保证所有订单都能在规定的时间窗口内完成。而对于所有必须在第$i$个时间间隔内必须完成的所有订单，采用订单调度模型对这些订单的配送进行调度，并考虑分配哪些订单到一台无人机。

车辆最短路径模型：所有订单完成调度后，对所有分配了订单的车辆进行路径规划。对于类似TSP问题的订单最短配送路径规划，使用遗传算法搜索近似的最短路径。



### 卸货点分类模型

卸货点分类模型旨在简化整个问题。由于n个配送中心到m个卸货点的vrp问题相对复杂，因此考虑使用卸货点分类模型，将问题简化为n个（1个配送中心到k个卸货点的vrp问题）。相当于在订单调度开始前，根据配送中心以及卸货点的相对位置确定每个卸货点的订单由固定的配送中心处理。



#### 主要步骤 

1. **初始化**：随机生成的m个卸货点和n个配送中心，即卸货点列表（`drop_points`）和配送中心列表（`depots`），并设置一个阈值（`therehold`）。 
2. . **统计阈值范围内的配送中心**：对于每个卸货点（`drop_point_i`），计算其阈值范围（半径为`therehold`）内其他卸货点的配送中心数量。 如果某个卸货点`drop_point_j`和`drop_point_i`的距离在阈值范围内且已分配了对应的配送中心，则统计该配送中心的数量。 
3. **分配最接近的配送中心**： 如果以`drop_point_i`为圆心，阈值为半径的范围内有卸货点分配了对应的配送中心，则选择数量最多的配送中心分配给当前卸货点`drop_point_i`。如果没有，则选择距离当前卸货点最近的配送中心。



下面是一个例子，表示如何为卸货点分配配送中心。灰色的点表示`drop_point_i`，阈值为半径的范围内有5个卸货点，相同的颜色表示被分配到同一个配送中心，不同颜色表示被分配到不同的配送中心。因为4个点有2个红色即两个卸货点被分配到了配送中心`depot_j`，数量是最多的，因此，将`drop_point_i`分配到`depot_j`。

<img src="/Users/underworld/Desktop/Advanced_Algorithm_VRP/images/knn.png" alt="knn" style="zoom:60%;" />



#### 代码实现

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





### 订单调度模型

