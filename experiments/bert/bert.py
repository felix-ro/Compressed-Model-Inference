import time
from transformers import pipeline

pipe = pipeline('fill-mask', model='distilbert-base-uncased')

latencies = []
iterations = 50
for _ in range(iterations):
    start_time = time.time()
    res = pipe("The White man worked as a [MASK].")
    end_time = time.time()

    latency = (end_time - start_time) * 1000
    latencies.append(latency)

# Calculate average latency
average_latency = sum(latencies) / iterations
print("The average distilbert latency is: " + str(average_latency) + "ms")

pipe = pipeline('fill-mask', model='bert-base-uncased')

latencies = []
iterations = 50
for _ in range(iterations):
    start_time = time.time()
    res = pipe("The White man worked as a [MASK].")
    end_time = time.time()

    latency = (end_time - start_time) * 1000
    latencies.append(latency)

# Calculate average latency
average_latency = sum(latencies) / iterations
print("The average bert latency is: " + str(average_latency) + "ms")

