// Sample recipe data
const recipes = [
    {
        title: "Veggie Breakfast Burrito",
        description: "A hearty vegetarian breakfast burrito packed with veggies.",
        course: "breakfast",
        diet: "veg",
        cuisine: "mexican"
    },
    {
        title: "Chicken Caesar Salad",
        description: "A classic chicken Caesar salad with homemade dressing.",
        course: "lunch",
        diet: "non-veg",
        cuisine: "italian"
    },
    {
        title: "Vegan Chocolate Cake",
        description: "A delicious vegan chocolate cake made with almond milk.",
        course: "dessert",
        diet: "vegan",
        cuisine: "american"
    },
    {
        title: "Pasta Primavera",
        description: "Italian pasta with fresh spring vegetables.",
        course: "lunch",
        diet: "veg",
        cuisine: "italian"
    },
    {
        title: "Chinese Vegetable Stir Fry",
        description: "A quick stir fry with mixed vegetables and tofu.",
        course: "dinner",
        diet: "veg",
        cuisine: "chinese"
    }
];

// Function to apply filters and update recipe display
function applyFilters() {
    const courseFilter = document.getElementById("courseFilter").value;
    const dietFilter = document.getElementById("dietFilter").value;
    const cuisineFilter = document.getElementById("cuisineFilter").value;

    const filteredRecipes = recipes.filter(recipe => {
        const courseMatch = courseFilter === "all" || recipe.course === courseFilter;
        const dietMatch = dietFilter === "all" || recipe.diet === dietFilter;
        const cuisineMatch = cuisineFilter === "all" || recipe.cuisine === cuisineFilter;
        return courseMatch && dietMatch && cuisineMatch;
    });

    displayRecipes(filteredRecipes);
}
