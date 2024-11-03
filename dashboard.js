// Function to search recipes by name
function searchRecipeByName() {
    const recipeName = document.getElementById("recipeSearchBar").value;
    if (recipeName) {
        // Replace this alert with logic to show results for recipeName search
        alert(`Searching for recipes with the name: ${recipeName}`);
    }
}

// Function to search recipes by ingredients
function searchByIngredients() {
    const ingredients = [];
    for (let i = 1; i <= 5; i++) {
        const ingredient = document.getElementById(`ingredient${i}`).value;
        if (ingredient) {
            ingredients.push(ingredient);
        }
    }
    if (ingredients.length > 0) {
        // Replace this alert with logic to show results for ingredients search
        alert(`Searching for recipes with ingredients: ${ingredients.join(", ")}`);
    } else {
        alert("Please enter at least one ingredient.");
    }
}
