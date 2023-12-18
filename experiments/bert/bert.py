from timeit import Timer
from transformers import DistilBertTokenizer, TFDistilBertModel, BertTokenizer, TFBertModel

EXPERIMENT_NAME = "bert/"
RESULTS_PATH = "results/"
FILE_NAME_COMPRESSED_RESULTS = "results-compressed.txt"

def benchCompressed(tokenizer, model, reps, iters, modelName):
    fileName = RESULTS_PATH + EXPERIMENT_NAME + modelName + ".txt"
    f = open(fileName, "w")
    results = ""
    for i in range(reps):
        for input_string in ["TensorFlow is", "TensorFlow is a", "TFLite is a"]:
            tokenized_input = tokenizer(input_string, return_tensors='tf')
            t = Timer(lambda: model(tokenized_input))
            print(t.timeit(number=iters)/iters)
            results += str(t.timeit(number=iters)/iters) + "\n"

    f.write(results)
    f.close()

def benchUncompressed(tokenizer, model, reps, iters, modelName):
    fileName = RESULTS_PATH + EXPERIMENT_NAME + modelName + ".txt"
    f = open(fileName, "w")
    results = ""
    for i in range(reps):
        for input_string in ["TensorFlow is", "TensorFlow is a", "TFLite is a"]:
            tokenized_input = tokenizer(input_string, return_tensors='tf')
            t = Timer(lambda: model(tokenized_input))
            print(t.timeit(number=iters)/iters)
            results += str(t.timeit(number=iters)/iters) + "\n"

    f.write(results)
    f.close()

def main(): 
    modelNames = ["distilbert-base-uncased", "bert-base-uncased"]

    for modelName in modelNames:
        if modelName == "distilbert-base-uncased":
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
            benchCompressed(tokenizer=tokenizer, model=model, reps=1, iters=100, modelName=modelName)
        else: 
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = TFBertModel.from_pretrained("bert-base-uncased")
            benchUncompressed(tokenizer=tokenizer, model=model, reps=1, iters=100, modelName=modelName)


if __name__ == "__main__":
    main()