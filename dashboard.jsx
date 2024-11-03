// src/App.jsx

import React from 'react';
import './App.css';

const App = () => {
    return (
        <div className="app">
            <header className="header">
                <div className="header-logo">
                    <h1>Taste The Best Food</h1>
                </div>
                <nav className="header-nav">
                    <a href="#all-recipes">All recipes</a>
                    <a href="#trending">Trending</a>
                    <a href="#blog">Blog</a>
                    <a href="#about">About</a>
                    <a href="#contact">Contact</a>
                </nav>
                <div className="header-actions">
                    <button className="icon-button">🔍</button>
                    <button className="icon-button">❤</button>
                    <button className="submit-button">Submit recipe</button>
                </div>
            </header>

            <section className="banner">
                <h2>Delicious Recipes</h2>
            </section>

            <main className="main-content">
                <section className="most-popular">
                    <h3>Most popular</h3>
                    <div className="recipes">
                        <RecipeCard
                            imageUrl="path/to/chicken-pizza.jpg"
                            title="Gourmet BBQ Chicken Pizza"
                            time="30 min"
                            servings="4 person"
                            rating="122"
                        />
                        <RecipeCard
                            imageUrl="path/to/carbonara.jpg"
                            title="Classic Spaghetti Carbonara"
                            time="20 min"
                            servings="2 person"
                            rating="71"
                        />
                        <RecipeCard
                            imageUrl="path/to/burger.jpg"
                            title="Gourmet Burger"
                            time="30 min"
                            servings="4 person"
                            rating="54"
                        />
                        <RecipeCard
                            imageUrl="path/to/salmon.jpg"
                            title="Succulent Salmon Fillets"
                            time="25 min"
                            servings="2 person"
                            rating="27"
                        />
                    </div>
                </section>

                <section className="trending">
                    <h3>Trending</h3>
                    <div className="recipes">
                        {/* Add trending recipe cards here */}
                    </div>
                </section>
            </main>
        </div>
    );
};

const RecipeCard = ({ imageUrl, title, time, servings, rating }) => (
    <div className="recipe-card">
        <img src={imageUrl} alt={title} className="recipe-image" />
        <div className="recipe-info">
            <h4>{title}</h4>
            <p><span>⏱ {time}</span> • <span>👥 {servings}</span></p>
            <p>⭐ {rating}</p>
            <button className="view-recipe">View recipe</button>
        </div>
    </div>
);

export default App;