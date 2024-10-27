import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer,DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import Dataset


# Load the dataset
df = pd.read_csv("Book_updated.csv")
print(df.head())  # Show the first few rows for verification


# Combine title, ingredients, and instructions into one text prompt for training
#df['recipe_text'] = df['Recipe'] + "\nIngredients:\n" + df['Ingredients']
df['recipe_text'] = df['Recipe'] + "\nIngredients:\n" + df['Ingredients'] + "\nInstructions:\n" + df['Instructions']


print("stage 1")
# Load a pre-trained GPT-2 model and tokenizer
#model_name = "gpt2-medium"  # You can switch to "gpt2" for a smaller model
model_name = "gpt2"  # You can switch to "gpt2" for a smaller model
#model_name = "gpt2lmhead"  # You can switch to "gpt2" for a smaller model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


print ("stage 2")


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("inside if : after stage 2")


def tokenize_function(examples):
   
    return tokenizer(examples['recipe_text'], return_tensors="pt", max_length=512, truncation=True, padding="max_length")


dataset = Dataset.from_pandas(df)
tokenized_dataset = dataset.map(tokenize_function, batched=True)


# Create labels - they should be identical to input_ids for language modeling tasks
tokenized_dataset = tokenized_dataset.map(lambda examples: {'labels': examples['input_ids']}, batched=True)
# Set the pad_token to eos_token if it's not already set


print("stage 3")


# Tokenize the recipes (with padding)
inputs = tokenizer(
    df['recipe_text'].tolist(),
    return_tensors="pt",
    max_length=512,
    truncation=True,
    padding="max_length"
)
#inputs['labels'] = inputs['input_ids'].clone()
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
print("stage 4")


# Prepare the dataset for the Trainer


#class RecipeDataset(torch.utils.data.Dataset):
 #   def __init__(self, encodings):
        #self.encodings = encodings


 #   def __getitem__(self, idx):
  #      item = {key: val[idx] for key, val in self.encodings.items()}
  #      return item


  #  def __len__(self):
  #      return len(self.encodings['input_ids'])


# Create dataset
#dataset = RecipeDataset(inputs)
#print("stage 4")


# Set up Trainer
training_args = TrainingArguments(
    output_dir="./recipe_generator3",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=100,
    save_total_limit=1,
    logging_dir="./logs",
    evaluation_strategy="no"
)






trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=inputs["input_ids"],
    data_collator=data_collator
)
print("stage 5")
#print (model)  
#print(dataset)
#print(training_args)
# Train the model
trainer.train()








# Function to match input ingredients with recipe dataset
def match_ingredients(input_ingredients, recipe_ingredients):
    input_ingredients_set = set(ingredient.strip().lower() for ingredient in input_ingredients.split(','))
    recipe_ingredients_set = set(ingredient.strip().lower() for ingredient in recipe_ingredients.split(','))


    # Find the common ingredients between input and recipe
    matched_ingredients = input_ingredients_set.intersection(recipe_ingredients_set)
   
    return len(matched_ingredients) >= 2  # Return True if 2 or more ingredients match


# Function to search the whole dataset and return all matching recipes
def generate_recipes(prompt):
    input_ingredients = prompt.lower()  # Process input ingredients
    matched_recipes = []


    # Check for matching recipes in the dataset
    for index, row in df.iterrows():
        if match_ingredients(input_ingredients, row['Ingredients']):
            matched_recipes.append(
                f"Recipe: {row['Recipe']}\n"
                f"Ingredients: {row['Ingredients']}\n"
                f"Instructions: {row['Instructions']}\n"
                f"Cuisine: {row['Cuisines']}\n"  
                
            )
            #matched_recipes.append(f"Recipe: {row['Recipe']}\nIngredients: {row['Ingredients']}\nInstructions: {row['Instructions']}\n")
            #matched_recipes.append(f"Recipe: {row['Recipe']}\nIngredients: {row['Ingredients']}\n")


    if matched_recipes:
        return "\n".join(matched_recipes)
    else:
        return "No matching recipe found in the dataset. Try generating a new recipe with GPT-2!"






while True:
    prompt=input("enter the ingredients separated with comma or enter exit:")

    if prompt == "exit":
        break
    else:
        print(generate_recipes(prompt))
        
   

