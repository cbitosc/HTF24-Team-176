// script.js
function handleSubmit(event) {
    event.preventDefault(); // Prevent form submission

    const formData = {
        name: document.getElementById('name').value,
        email: document.getElementById('email').value,
        password: document.getElementById('password').value,
    };

    console.log('Form submitted:', formData);
    alert('Signup successful! Check the console for details.');
}

function signupWithGoogle() {
    alert("Redirecting to Google Sign-Up...");
    // Here you would typically handle Google Sign-In logic
}
