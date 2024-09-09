# Name: Mohammad Khdour      ID: 1212517
# Name: Gassan Qandeel       ID: 1212397

import random
import re
import matplotlib.pyplot as plt

MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
POPULATION_SIZE = 100
GENERATION_SIZE = 50


# read from file the jobs
def readFile():
    try:
        with open("input.txt", "r") as file:
            file0 = file.readlines()
            pattern = r'(\d+)\[(\d+)\]'
            machines = []
            time = []
            chromssom = []
            i = 1
            for f in file0:
                machineAndTime = []
                for num in f.split("M"):
                    match = re.search(pattern, num)
                    if match:
                        number = int(match.group(1))
                        t = int(match.group(2))
                        machines.append(number)
                        list = (number, t)
                        machineAndTime.append(list)
                jobs = {f"job{i}": machineAndTime}
                jobList.append(jobs)
                chromssom.append(f"job{i}")
                i += 1
        numofMachine = max(machines)  # max number of machine

    except:
        print("sorry, file not found")
        exit(-1)

    return chromssom, numofMachine


# function to generate random genome
def randomGenome(genome):
    schedule = list(genome)
    random.shuffle(schedule)
    return schedule


# function to generate intiall population
def generateInitialPopulation(population, popualtion_size):
    return [randomGenome(population) for _ in range(popualtion_size)]


# function to make crossover operation by percantge 80%
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        crossover_point = random.randint(1, len(parent1) - 1)
        return parent1[:crossover_point] + parent2[crossover_point:], parent2[:crossover_point] + parent1[
                                                                                                  crossover_point:]
    else:
        return parent1, parent2


# function to make mutation operation by percantge 10%
def mutate(genes):
    if random.random() < MUTATION_RATE:
        idx1, idx2 = sorted(random.sample(range(len(genes)), 2))
        genes[idx1], genes[idx2] = genes[idx2], genes[idx1]
    return genes


# select two parent which have higher fitness
def selectParent(population, fitnessValues):
    min1 = min(fitnessValues)
    idx1 = fitnessValues.index(min1)
    fitnessValues[idx1] = float("inf")  # Temporarily remove the min value

    min2 = min(fitnessValues)

    idx2 = fitnessValues.index(min2)

    parent1 = population[idx1]
    parent2 = population[idx2]

    fitnessValues[idx1] = min1

    return parent1, parent2


# function defind all sequance machines with there time for job
def makeDectoinery(indevidual):
    dectJob = {}
    jobs = {job: item for jobs in jobList for job, item in jobs.items()}

    for i in indevidual:
        dectJob[i] = jobs[i]

    return dectJob


# function to calculate the fitness value
def calculateFitness(schedule):
    if set(schedule.keys()) != set(job for jobs in jobList for job in jobs.keys()):
        return float("inf"), None
    jobsTime = {job: [] for jobs in jobList for job, values in jobs.items()}
    machinesTime = {i: 0 for i in range(10)}

    jobStart = {job: 0 for jobs in jobList for job, values in jobs.items()}
    finalTimes = []
    i = -1

    for iteration in range(20):
        i += 1
        for job, values in schedule.items():
            try:
                machine, time = values[i]
                startTime = machinesTime[machine]

                if startTime < jobStart[job]:
                    startTime = jobStart[job]
                    endTime = startTime + time
                    machinesTime[machine] = endTime
                else:
                    machinesTime[machine] += time
                    endTime = machinesTime[machine]

                jobStart[job] = endTime
                finalTimes.append(endTime)
                jobsTime[job].append((machine, (startTime, endTime)))

            except:
                pass

    finalTime = max(finalTimes)
    return finalTime, jobsTime


# function to plot gantt chart
def plot_gantt_chart(schedule):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.get_cmap('tab20', len(schedule))

    for index, (job, machine_times) in enumerate(schedule.items()):
        for machine, (start_time, end_time) in machine_times:
            ax.barh(f'M{machine}', end_time - start_time, left=start_time, height=0.5, color=colors(index),
                    align='center')
            ax.text((start_time + end_time) / 2, f'M{machine}', job, ha='center', va='center', color='white',
                    fontweight='bold')

    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_title('Gantt Chart')
    ax.invert_yaxis()
    ax.xaxis.grid(True)

    plt.show()


def main():
    global jobList
    jobList = []
    pop, number = readFile()
    population = generateInitialPopulation(pop, POPULATION_SIZE)

    Gen = []  # save the best generation from each generation

    for generation in range(GENERATION_SIZE):
        fitnessValues = []

        # calculate the fitness function to select parent
        for individual in population:
            individual = makeDectoinery(individual)
            time, _ = calculateFitness(individual)
            fitnessValues.append(time)

        new_population = []

        # generate new population for each generation
        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = selectParent(population, fitnessValues)
            parent1 = list(parent1)
            parent2 = list(parent2)
            offspring1_genes, offspring2_genes = crossover(parent1, parent2)
            offspring1_genes = mutate(offspring1_genes)
            offspring2_genes = mutate(offspring2_genes)
            new_population.extend([parent1, parent2, offspring1_genes, offspring2_genes])

        population = list(new_population)
        fitnessValues = []

        for individual in population:
            individual = makeDectoinery(individual)
            time, _ = calculateFitness(individual)
            fitnessValues.append(time)

        best_fitness = min(fitnessValues)
        index = fitnessValues.index(best_fitness)
        bestGen = makeDectoinery(population[index])
        Gen.append((bestGen, best_fitness))

    sort = sorted(Gen, key=lambda X: X[1])
    print(sort[0])
    _, job = calculateFitness(sort[0][0])
    plot_gantt_chart(job)


main()
