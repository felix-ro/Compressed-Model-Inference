import time
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

FILE_NAME_UNCOMPRESSED_RESULTS = "results-uncompressed.txt"
FILE_NAME_UNCOMPRESSED_COMPILED_RESULTS = "results-uncompressed-compiled.txt"
FILE_NAME_COMPRESSED_RESULTS = "results-compressed.txt"
FILE_NAME_COMPRESSED_COMPILED_RESULTS = "results-compressed-compiled.txt"

def fileInit(fileName):
    f = open(fileName, "w")
    f.write("")
    f.close()

def benchUncompressed(tokenizer, model):
    f = open(FILE_NAME_UNCOMPRESSED_RESULTS, "a")
    for input_string in ["TensorFlow is", "TensorFlow is a", "TFLite is a"]:
        tokenized_input = tokenizer(input_string, return_tensors="tf")

        start = time.time_ns()
        generated_tokens = model.generate(**tokenized_input, num_beams=2)
        end = time.time_ns()
        f.write(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
        print(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
        decoded_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        print(f"Generated -- {decoded_text}")
    f.close()

def benchUncompressedCompiled(tokenizer, model):
    f = open(FILE_NAME_UNCOMPRESSED_COMPILED_RESULTS, "a")
    xla_generate = tf.function(model.generate, jit_compile=True)
    for input_string in ["TensorFlow is", "TensorFlow is a", "TFLite is a", "TensorFlow is"]:
        tokenized_input = tokenizer(input_string, return_tensors="tf")

        tokenized_input = tokenizer(input_string, pad_to_multiple_of=8, padding=True, return_tensors="tf")
        start = time.time_ns()
        generated_tokens = xla_generate(**tokenized_input, num_beams=2)
        end = time.time_ns()
        print(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
        f.write(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
        decoded_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        print(f"Generated -- {decoded_text}")
    f.close()

def benchCompressed(tokenizer, model):
    f = open(FILE_NAME_COMPRESSED_RESULTS, "a")
    for input_string in ["TensorFlow is", "TensorFlow is a", "TFLite is a"]:
        tokenized_input = tokenizer(input_string, return_tensors="tf")

        start = time.time_ns()
        generated_tokens = model.generate(**tokenized_input, num_beams=2)
        end = time.time_ns()
        f.write(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
        print(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
        decoded_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        print(f"Generated -- {decoded_text}")
    f.close()

def benchCompressedCompiled(tokenizer, model):
    f = open(FILE_NAME_COMPRESSED_COMPILED_RESULTS, "a")
    xla_generate = tf.function(model.generate, jit_compile=True)
    for input_string in ["TensorFlow is", "TensorFlow is a", "TFLite is a", "TensorFlow is"]:
        tokenized_input = tokenizer(input_string, return_tensors="tf")

        tokenized_input = tokenizer(input_string, pad_to_multiple_of=8, padding=True, return_tensors="tf")
        start = time.time_ns()
        generated_tokens = xla_generate(**tokenized_input, num_beams=2)
        end = time.time_ns()
        print(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
        f.write(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
        decoded_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        print(f"Generated -- {decoded_text}")
    f.close()

def main():
    fileInit(FILE_NAME_UNCOMPRESSED_RESULTS)
    fileInit(FILE_NAME_UNCOMPRESSED_COMPILED_RESULTS)
    fileInit(FILE_NAME_COMPRESSED_RESULTS)
    fileInit(FILE_NAME_COMPRESSED_COMPILED_RESULTS)

    tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left", pad_token="</s>")
    model = TFAutoModelForCausalLM.from_pretrained("gpt2")
    tokenizerCompressed = AutoTokenizer.from_pretrained("distilgpt2", padding_side="left", pad_token="</s>")
    modelCompressed = TFAutoModelForCausalLM.from_pretrained("distilgpt2")

    benchUncompressed(tokenizer=tokenizer, model=model)
    benchUncompressedCompiled(tokenizer=tokenizer, model=model)
    benchCompressed(tokenizer=tokenizerCompressed, model=modelCompressed)
    benchCompressedCompiled(tokenizer=tokenizerCompressed, model=modelCompressed)

if __name__ == "__main__":
    main()