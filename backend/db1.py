import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
import torch

# Load the dataset
df = pd.read_csv("Book1_revised.csv")

print(df.head())

# Combine title, ingredients, and instructions into one text prompt for training
df['recipe_text'] = df['Recipe'] + "\nIngredients:\n" + df['Ingredients'] 

# Load a pre-trained GPT-2 model and tokenizer
model_name = "gpt2-medium"  # or use "gpt2" for a smaller model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# **Set the pad_token to eos_token**
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the recipes (with padding)
inputs = tokenizer(
    df['recipe_text'].tolist(), 
    return_tensors="pt", 
    max_length=512, 
    truncation=True, 
    padding="max_length"
)

# Set up Trainer
training_args = TrainingArguments(
    output_dir="./recipe_generator",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=inputs["input_ids"]
)

# Train the model
trainer.train()

# Function to match input ingredients with recipe dataset
def match_ingredients(input_ingredients, recipe_ingredients):
    # Split ingredients into individual words or items for matching
    input_ingredients = set([ingredient.strip().lower() for ingredient in input_ingredients.split(',')])
    recipe_ingredients = set([ingredient.strip().lower() for ingredient in recipe_ingredients.split(',')])

    # Find the common ingredients between input and recipe
    matched_ingredients = input_ingredients.intersection(recipe_ingredients)
    
    return len(matched_ingredients) >= 2  # Return True if 2 or more ingredients match

# Modified function to search the whole dataset and return all matching recipes
def generate_recipes(prompt):
    input_ingredients = prompt.lower().split(',')  # Split user input by commas
    matched_recipes = []

    # Check for matching recipes in the dataset
    for index, row in df.iterrows():
        recipe_ingredients = row['Ingredients']  # Ingredients from the dataset
        if match_ingredients(prompt, recipe_ingredients):
            # If 2 or more ingredients match, collect the recipe name and its ingredients
            matched_recipes.append(f"Recipe: {row['Recipe']}\nIngredients: {row['Ingredients']}\n")

    if matched_recipes:
        # If any matching recipes were found, return them
        return "\n".join(matched_recipes)
    else:
        # If no matching recipe is found, you can still generate a new recipe using GPT-2
        return "No matching recipe found in the dataset. Try generating a new recipe with GPT-2!"

# Example prompt with ingredients
prompt = "chicken, garlic, lemon"
print(generate_recipes(prompt))
