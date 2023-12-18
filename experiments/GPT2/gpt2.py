from timeit import Timer
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

EXPERIMENT_NAME = "gpt2/"
RESULTS_PATH = "results/"

def fileInit(fileName):
    f = open(fileName, "w")
    f.write("")
    f.close()

def benchUncompiled(tokenizer, model, reps, iters, modelName):
    fileName = RESULTS_PATH + EXPERIMENT_NAME + modelName + "-uncompiled" + ".txt"
    f = open(fileName, "w")

    results = ""
    for i in range(reps):
        for input_string in ["TensorFlow is", "TensorFlow is a", "TFLite is a"]:
            tokenized_input = tokenizer(input_string, return_tensors="tf")
            t = Timer(lambda: model.generate(**tokenized_input, num_beams=2))
            print(t.timeit(number=iters)/iters)
            results += str(t.timeit(number=iters)/iters) + "\n"

    f.write(results)
    f.close()

def benchCompiled(tokenizer, model, reps, iters, modelName):
    fileName = RESULTS_PATH + EXPERIMENT_NAME + modelName + "-compiled" + ".txt"
    f = open(fileName, "w")
    xla_generate = tf.function(model.generate, jit_compile=True)
    results = ""
    for i in range(reps):
        for input_string in ["TensorFlow is", "TensorFlow is a", "TFLite is a"]:
            tokenized_input = tokenizer(input_string, return_tensors="tf")
            t = Timer(lambda: xla_generate(**tokenized_input, num_beams=2))
            print(t.timeit(number=iters)/iters)
            results += str(t.timeit(number=iters)/iters) + "\n"

    f.write(results)
    f.close()

def main():

    modelNames = ["gpt2", "distilgpt2"]
    for modelName in modelNames: 
        tokenizer = AutoTokenizer.from_pretrained(modelName, padding_side="left", pad_token="</s>")
        model = TFAutoModelForCausalLM.from_pretrained(modelName)
        benchUncompiled(tokenizer=tokenizer, model=model, reps=1, iters=10, modelName=modelName)
        benchCompiled(tokenizer=tokenizer, model=model, reps=1, iters=10, modelName=modelName)

if __name__ == "__main__":
    main()