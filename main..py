from vrp_simulator import simulator
import json
import matplotlib.pyplot as plt

def main():
    #run the simulator with different k values and save the results into a json file
    #set the m as the reconducting times
    m = 10
    results = {}
    for i in range (1, 10):
        results[str(i)] = {}
        for j in range(m):
            print(f"Simulating with k = {i} for the {j+1}th time...")
            results[str(i)][str(j)] = simulator(k=i)
    
    with open('results.json', 'w') as f:
        json.dump(results, f)
    
    average_results = {}
    for i in range(1, 10):
        average_results[str(i)] = sum(results[str(i)].values())/m

    #plot the results and save the plot into a pdf file
    plt.plot(average_results.keys(), average_results.values())
    plt.xlabel('k')
    plt.ylabel('Whole vehicles distance')
    plt.title('Whole vehicles distance vs. k')
    plt.savefig('whole_vehicles_distance_vs_k.pdf')


if __name__ == "__main__":
    main()