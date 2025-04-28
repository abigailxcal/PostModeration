import transformers
import inspect

print("Transformers version:", transformers.__version__)
print("TrainingArguments source:", inspect.getfile(transformers.TrainingArguments))

