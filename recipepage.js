const urlParams = new URLSearchParams(window.location.search);
const cardId = urlParams.get('id');

// Fetch recipe data based on cardId (replace with your data source)
fetchRecipeData(cardId)
    .then(recipeData => {
        // Update the recipe details on the page
        document.getElementById('recipe-details').innerHTML = `
            <h1>${recipeData.name}</h1>
            <ul>
                ${recipeData.ingredients.map(ingredient => <li>${ingredient}</li>).join('')}
            </ul>
            <p>Course: ${recipeData.course}</p>
            <p>Dietary Preferences: ${recipeData.dietaryPreferences}</p>
            <p>Cuisine: ${recipeData.cuisine}</p>
            <p>${recipeData.instructions}</p>
        `;
    })
    .catch(error => {
        console.error('Error fetching recipe data:', error);
    });