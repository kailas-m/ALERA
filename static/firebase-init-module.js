// Firebase v12 modular initialization as a JavaScript module
// Uses your provided config and exposes minimal APIs on window.firebaseMod for non-module scripts
import { initializeApp } from 'https://www.gstatic.com/firebasejs/12.3.0/firebase-app.js';
import { getAnalytics } from 'https://www.gstatic.com/firebasejs/12.3.0/firebase-analytics.js';
import {
	getAuth,
	onAuthStateChanged,
	createUserWithEmailAndPassword,
	signInWithEmailAndPassword,
	signOut
} from 'https://www.gstatic.com/firebasejs/12.3.0/firebase-auth.js';

const firebaseConfig = {
	apiKey: 'AIzaSyDXYPzz8lbuRSJef6eQiHkAuJFaDfg3mMY',
	authDomain: 'aleraa-b0e0b.firebaseapp.com',
	projectId: 'aleraa-b0e0b',
	storageBucket: 'aleraa-b0e0b.firebasestorage.app',
	messagingSenderId: '609644245663',
	appId: '1:609644245663:web:1427fa0f8d42ca2786a8aa',
	measurementId: 'G-MMM9GYP8M5'
};

const app = initializeApp(firebaseConfig);
try { getAnalytics(app); } catch (_) { /* analytics may fail on http or non-browser */ }

const auth = getAuth(app);

// Expose limited modular API to window for non-module consumers
window.firebaseMod = {
	app,
	auth,
	onAuthStateChanged,
	createUserWithEmailAndPassword,
	signInWithEmailAndPassword,
	signOut
};


