// Firebase initialization (compat)
// Replace the config values with your Firebase project's keys from the Firebase console
// Project Settings → General → Your apps → Firebase SDK snippet → Config
(function () {
	if (!window.firebase) {
		console.error("Firebase SDK not loaded. Ensure firebase-app-compat.js is included before this file.");
		return;
	}

	var firebaseConfig = {
		apiKey: "AIzaSyDXYPzz8lbuRSJef6eQiHkAuJFaDfg3mMY",
		authDomain: "aleraa-b0e0b.firebaseapp.com",
		projectId: "aleraa-b0e0b",
		storageBucket: "aleraa-b0e0b.firebasestorage.app",
		messagingSenderId: "609644245663",
		appId: "1:609644245663:web:1427fa0f8d42ca2786a8aa",
		measurementId: "G-MMM9GYP8M5"
	};

	// Initialize Firebase (compat)
	if (!window.firebase.apps || window.firebase.apps.length === 0) {
		window.firebase.initializeApp(firebaseConfig);
	}

	// Expose a simple helper to access auth
	window.getFirebaseAuth = function () {
		return window.firebase.auth();
	};
})();


