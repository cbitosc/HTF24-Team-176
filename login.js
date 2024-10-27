// script.js
function handleLogin(event) {
    event.preventDefault(); // Prevent form submission

    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    // Here you can add your login logic, like sending a request to your backend
    console.log('Login submitted:', { email, password });
    alert('Login successful! Check the console for details.');
}

function loginWithGoogle() {
    alert("Redirecting to Google Login...");
    // Here you would typically handle Google Sign-In logic
}

