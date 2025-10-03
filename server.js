const express = require('express');
const bodyParser = require('body-parser');
const nodemailer = require('nodemailer');
const path = require('path');
const multer = require('multer');
const fs = require('fs');
// TensorFlow is optional; if not installed the prediction endpoints will be disabled
let tf = null;
try {
    tf = require('@tensorflow/tfjs-node');
} catch (err) {
    console.warn('Optional dependency @tensorflow/tfjs-node not installed — prediction endpoints will be disabled.');
}
require('dotenv').config();

// Small startup debug: report whether email env vars are present (masked, no secrets)
const maskEmail = (em) => {
    if (!em || typeof em !== 'string') return 'not-set';
    const parts = em.split('@');
    if (parts.length !== 2) return 'set';
    return parts[0].charAt(0) + '***@' + parts[1];
};
try {
    console.log('EMAIL_USER:', process.env.EMAIL_USER ? maskEmail(process.env.EMAIL_USER) : 'not-set', 'EMAIL_PASS set:', !!process.env.EMAIL_PASS);
} catch (e) { /* ignore logging errors */ }

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware to parse JSON and URL-encoded form data
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Serve static files (CSS, JS, images)
app.use(express.static(path.join(__dirname)));

// Set up multer for file uploads
const upload = multer({ dest: 'uploads/' });

// Load the model (only if tf is available)
let model = null;
async function loadModel() {
    if (!tf) return;
    try {
        const modelPath = 'file://F:/PROJECT/Trial000/models/allergy_detector.h5';
        model = await tf.loadLayersModel(modelPath);
        console.log('Model loaded');
    } catch (err) {
        console.error('Failed to load model:', err && err.message ? err.message : err);
        model = null;
    }
}
if (tf) loadModel();

// Route to handle image upload and prediction
app.post('/predict', upload.single('file'), async (req, res) => {
    if (!tf || !model) {
        return res.status(503).json({ error: 'Prediction service unavailable (model or tfjs not loaded)' });
    }

    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
    }

    try {
        // Load and preprocess the image
        const imagePath = req.file.path;
        const imageBuffer = fs.readFileSync(imagePath);
        const image = await tf.node.decodeImage(imageBuffer, 3);
        const resizedImage = tf.image.resizeBilinear(image, [224, 224]); // Adjust size as needed
        const normalizedImage = resizedImage.div(tf.scalar(255));
        const batchedImage = normalizedImage.expandDims(0);

        // Make prediction
        const predictions = await model.predict(batchedImage);
        const results = predictions.arraySync()[0];

        // Send the results back to the client
        res.status(200).json({ results });
    } catch (error) {
        console.error('Error processing image:', error && error.message ? error.message : error);
        res.status(500).json({ error: 'Error processing image' });
    }
});

// Contact route: accepts JSON { name, email, message }
app.post('/contact', async (req, res) => {
    const { name, email, message } = req.body || {};

    // Validate the input
    if (!name || !email || !message) {
        return res.status(400).json({ error: 'All fields are required' });
    }

    try {
        // Configure transporter from environment variables. Preferred: SMTP host/port.
        const transporterConfig = {};

        if (process.env.EMAIL_HOST && process.env.EMAIL_PORT && process.env.EMAIL_USER && process.env.EMAIL_PASS) {
            transporterConfig.host = process.env.EMAIL_HOST;
            transporterConfig.port = parseInt(process.env.EMAIL_PORT, 10);
            transporterConfig.secure = process.env.EMAIL_SECURE === 'true' || transporterConfig.port === 465; // true for 465, false for other ports
            transporterConfig.auth = {
                user: process.env.EMAIL_USER,
                pass: process.env.EMAIL_PASS
            };
        } else if (process.env.EMAIL_USER && process.env.EMAIL_PASS) {
            // Fallback to Gmail using OAuth/app-password style credentials (not recommended for production)
            transporterConfig.service = 'gmail';
            transporterConfig.auth = {
                user: process.env.EMAIL_USER,
                pass: process.env.EMAIL_PASS
            };
        } else {
            console.warn('Email credentials not set in environment. Skipping sendMail.');
            return res.status(500).json({ error: 'Email server not configured' });
        }

        const transporter = nodemailer.createTransport(transporterConfig);

        // Log which configured sender will be used (masked)
        console.log('Configured mail sender:', maskEmail(process.env.EMAIL_FROM || process.env.EMAIL_USER));

        const mailOptions = {
            from: process.env.EMAIL_FROM || process.env.EMAIL_USER,
            to: process.env.EMAIL_TO || process.env.EMAIL_USER,
            subject: `New Contact Form Submission from ${name}`,
            text: `Name: ${name}\nEmail: ${email}\nMessage: ${message}`,
            replyTo: email
        };

        const info = await transporter.sendMail(mailOptions);
        console.log('Contact email sent:', info && info.response ? info.response : info);
        return res.status(200).json({ message: 'Message sent successfully' });
    } catch (err) {
        console.error('Error sending contact email:', err && err.message ? err.message : err);
        return res.status(500).json({ error: 'Error sending email' });
    }
});

app.post('/signup', (req, res) => {
    const { name, email, password } = req.body;

    // Validate the input
    if (!name || !email || !password) {
        return res.status(400).json({ error: 'All fields are required' });
    }

    // Here you would typically save the user to a database
    console.log('User signed up:', { name, email, password });

    return res.status(200).json({ message: 'Signup successful' });
});

app.post('/login', (req, res) => {
    const { email, password } = req.body;

    // Validate the input
    if (!email || !password) {
        return res.status(400).json({ error: 'All fields are required' });
    }

    // Here you would typically check the user's credentials against a database
    console.log('User logged in:', { email, password });

    return res.status(200).json({ message: 'Login successful' });
});

// Serve the HTML files
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.get('/about', (req, res) => {
    res.sendFile(path.join(__dirname, 'about.html'));
});

app.get('/contact', (req, res) => {
    res.sendFile(path.join(__dirname, 'contact.html'));
});

app.get('/allergy', (req, res) => {
    res.sendFile(path.join(__dirname, 'allergy.html'));
});

app.get('/upload', (req, res) => {
    res.sendFile(path.join(__dirname, 'uploadDNAimage.html'));
});

app.get('/signup', (req, res) => {
    res.sendFile(path.join(__dirname, 'signup.html'));
});

app.get('/login', (req, res) => {
    res.sendFile(path.join(__dirname, 'login.html'));
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});