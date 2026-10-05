💳 AI-Powered Credit Card Expense Analyzer

Intelligent expense management system using Google Gemini AI and FastAPI


<img width="1338" height="630" alt="Screenshot (484)" src="https://github.com/user-attachments/assets/cf8587d2-a188-4b21-89d4-67eee993a1ea" />

<img width="1345" height="629" alt="Screenshot (488)" src="https://github.com/user-attachments/assets/a5e3447c-676e-426c-99af-dc66a7494329" />

<img width="1342" height="625" alt="Screenshot (485)" src="https://github.com/user-attachments/assets/c7b045c0-c6dc-4ec6-9054-509a33192a5e" />

Features
AI-Powered Analysis

Intelligent PDF Parsing: Extracts transactions from credit card statements using Google Gemini AI
Smart Categorization: Automatically classifies transactions into 10+ spending categories
Natural Language Insights: Generates personalized recommendations and spending analysis
Anomaly Detection: Identifies unusual transactions and spending patterns
----

 Financial Intelligence

Multi-Currency Support: INR, USD, EUR with proper localization
Spending Trends: Visual analytics with interactive charts
Budget Analysis: AI-powered budget recommendations
Category Breakdown: Detailed spending distribution
---

Technical Excellence

Production-Ready API: Built with FastAPI for high performance
Dual Parsing Method: LLM + regex fallback for reliability
Batch Processing: Optimized for cost and speed
Real-Time Dashboard: Interactive web interface with Chart.js


Demo
Try it out with demo data or upload your own credit card statement!
Live Demo: [Coming Soon]
Sample Output

Spending Overview
Total Spent: ₹17,881.90
Transactions: 90
Average: ₹198.69

 AI Insights:
✓ Highest spending: Shopping & Retail (₹17,008.80)
✓ Recommendation: Set a budget cap for discretionary spending
⚠️ Monitor: Large transactions detected

🛠️ Tech Stack
Backend:

Python 3.11+
FastAPI
Google Gemini AI API
Pandas
Pdfplumber / PyPDF2
Pydantic

Frontend:

HTML5 / CSS3 / JavaScript
Chart.js for visualizations
Responsive design

Deployment:


Uvicorn ASGI server



🚀 Quick Start
Prerequisites

Python 3.8 or higher
Google Gemini API key (Get it here)
Credit card statement PDFs

Installation

Clone the repository

```
git clone https://github.com/yourusername/credit-card-analyzer.git
cd credit-card-analyzer
```

Create virtual environment


```
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
````

Install dependencies

````
pip install -r requirements.txt
````

Set up environment variables

Create a .env file in the project root:
````
envGEMINI_API_KEY=your_gemini_api_key_here
````

Run the API server

````
uvicorn fastapi_app:app --reload
````

Open the frontend

Open frontend.html in your browser or visit:
http://localhost:8000/docs  # API Documentation

📖 Usage
API Endpoints
1. Analyze Statement
````
POST /api/analyze
Content-Type: multipart/form-data

# Upload a PDF credit card statement
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@statement.pdf"
````
Response:
````
json{
  "analysis_id": "abc-123",
  "transactions": [...],
  "summary": {
    "total_spent": 17881.90,
    "transaction_count": 90,
    "by_category": {...}
  },
  "insights": {
    "observations": [...],
    "recommendations": [...]
  }
}
````
2. Get Demo Data
   
   ```
   POST /api/demo
   ```

# Returns sample transaction data

3. Get Analysis by ID
```
bash
GET /api/analysis/{analysis_id}

````

# Retrieve previously processed analysis
4. Get Categories
````
bash
GET /api/categories
````

# Returns list of spending categories
Web Interface

Open frontend.html in your browser
Click "Choose PDF File" or "Try Demo Data"
View interactive dashboard with:

Spending metrics
Category breakdown (pie chart)
Top merchants (bar chart)
AI-generated insights




🏗️ Project Structure
```
credit-card-analyzer/
│
├── fastapi_app.py           # Main FastAPI backend
├── frontend.html            # Web interface
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (create this)
├── .gitignore              # Git ignore file
│
├── screenshots/            # Demo images
│   └── dashboard.png
│
└── README.md              # This file
```

 Acknowledgments

Google Gemini AI - LLM API
FastAPI - Web framework
Chart.js - Visualizations
Pdfplumber - PDF parsing


📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

Author
Reshma Thomas


 Show Your Support
Give a ⭐️ if this project helped you!

🐳 Run with Docker
```
bash
cp .env.example .env          # then put your own GEMINI_API_KEY in .env
docker build -t cc-analyzer .
docker run --env-file .env -p 8000:8000 cc-analyzer
```
Open http://localhost:8000/docs. The API key is passed at runtime and is never baked into the image.

Troubleshooting: if /api/analyze returns "No transactions found", open /api/stats — the last_gemini_error field shows exactly why Gemini failed (bad key, retired model, quota). To switch models, set GEMINI_MODEL in .env.
