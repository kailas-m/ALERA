# ShopBot - AI-Powered E-commerce Platform

A modern, commercial-grade e-commerce website built with React frontend and Flask backend, featuring an AI-powered shopping assistant powered by Groq API.

## 🚀 Features

### Frontend (React)
- **Modern UI/UX**: Professional commercial website design
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- **Product Catalog**: Beautiful product grid with filtering and search
- **Shopping Cart**: Real-time cart management with React state
- **AI Chat Interface**: Interactive chatbot for shopping assistance
- **Category Filtering**: Filter products by electronics, clothing, home, books
- **Product Ratings**: Star ratings and reviews display
- **Smooth Animations**: CSS animations and transitions

### Backend (Flask)
- **RESTful API**: Clean API endpoints for all operations
- **Groq AI Integration**: Intelligent shopping assistant
- **Product Management**: CRUD operations for products
- **Shopping Cart**: Session-based cart management
- **Search Functionality**: Advanced product search with filters
- **Category Management**: Dynamic category handling

### AI Features
- **Smart Recommendations**: AI-powered product suggestions
- **Natural Language Processing**: Understand shopping queries
- **Contextual Responses**: Maintains conversation context
- **Shopping Assistance**: Help with product comparisons and decisions

## 🛠️ Technology Stack

- **Frontend**: React 18, HTML5, CSS3, JavaScript ES6+
- **Backend**: Flask, Python 3.8+
- **AI**: Groq API (Llama models)
- **Styling**: Custom CSS with modern design patterns
- **Icons**: Font Awesome 6
- **Fonts**: Inter (Google Fonts)

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/shopbot-ecommerce.git
   cd shopbot-ecommerce
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   export GROQ_API_KEY="your_groq_api_key_here"
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:5000`

## 🔧 Configuration

### Environment Variables
- `GROQ_API_KEY`: Your Groq API key for AI functionality

### Product Database
The application includes a sample product database with:
- Electronics (phones, laptops, headphones)
- Clothing (shoes, jeans, t-shirts)
- Home & Garden (kitchen appliances, cleaning, lighting)
- Books (self-help, business)

## 📱 Usage

### For Customers
1. **Browse Products**: Use category filters or search bar
2. **Add to Cart**: Click "Add to Cart" on any product
3. **AI Assistant**: Click the chat icon for shopping help
4. **Get Recommendations**: Ask the AI for product suggestions
5. **Checkout**: Complete your purchase (demo mode)

### For Developers
1. **API Endpoints**: Use the REST API for integrations
2. **Customize Products**: Modify the PRODUCTS_DB in app.py
3. **Styling**: Update static/styles.css for design changes
4. **AI Prompts**: Customize AI responses in the chat function

## 🔌 API Endpoints

- `GET /api/products` - Get all products
- `GET /api/products?category=electronics` - Filter by category
- `GET /api/search?q=iphone` - Search products
- `POST /api/add-to-cart` - Add product to cart
- `POST /api/chat` - Chat with AI assistant
- `GET /api/categories` - Get all categories

## 🎨 Customization

### Adding New Products
```python
# In app.py, add to PRODUCTS_DB
"electronics": [
    {
        "id": 13,
        "name": "New Product",
        "price": 299,
        "rating": 4.5,
        "category": "electronics",
        "description": "Product description"
    }
]
```

### Styling Changes
- Modify `static/styles.css` for design updates
- Use CSS custom properties for consistent theming
- Responsive breakpoints are already configured

### AI Behavior
- Update the system prompt in `query_groq()` function
- Modify response handling in the chat endpoint
- Add new shopping-related keywords in `is_shopping_related()`

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
1. Set up a production WSGI server (Gunicorn)
2. Configure reverse proxy (Nginx)
3. Set up SSL certificates
4. Configure environment variables
5. Set up monitoring and logging

## 📊 Performance

- **Frontend**: Optimized React components with minimal re-renders
- **Backend**: Efficient Flask routes with proper error handling
- **AI**: Fast Groq API responses with caching
- **Database**: In-memory product storage (easily replaceable with SQL)

## 🔒 Security

- Input validation on all API endpoints
- XSS protection with proper escaping
- CSRF protection for forms
- Secure session management
- API rate limiting (recommended for production)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Groq for providing the AI API
- React team for the amazing frontend framework
- Flask team for the lightweight backend framework
- Font Awesome for the beautiful icons

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Contact: support@shopbot.com
- Documentation: https://docs.shopbot.com

---

**ShopBot** - Making online shopping intelligent and delightful! 🛒✨


