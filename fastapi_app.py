"""
Credit Card Analyzer - FastAPI Backend
Run: uvicorn fastapi_app:app --reload
API Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import google.generativeai as genai
from dotenv import load_dotenv
import os
import tempfile
import pdfplumber
import PyPDF2
import pandas as pd
from datetime import datetime
import json
import re
import uuid

# Load environment
load_dotenv()

# Configuration
CURRENCY_SYMBOL = "₹"  # Change to "$" for USD, "€" for EUR, etc.
CURRENCY_CODE = "INR"  # Change to "USD", "EUR", etc.

# Initialize FastAPI
app = FastAPI(
    title="Credit Card Analyzer API",
    description="AI-powered credit card statement analysis using Gemini",
    version="1.0.0"
)

# CORS middleware (allows frontend to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    gemini_model = None

# In-memory storage (use Redis/Database in production)
analysis_storage = {}


# ============ Pydantic Models ============

class Transaction(BaseModel):
    date: str
    merchant: str
    amount: float
    category: str
    description: Optional[str] = ""

class AnalysisSummary(BaseModel):
    total_spent: float
    transaction_count: int
    average_transaction: float
    by_category: Dict[str, float]
    top_merchant: str

class AIInsights(BaseModel):
    observations: List[str]
    recommendations: List[str]
    concern: str

class AnalysisResponse(BaseModel):
    analysis_id: str
    transactions: List[Transaction]
    summary: AnalysisSummary
    insights: AIInsights
    processed_at: str


# ============ Helper Functions ============

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF"""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        # Fallback to PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except:
            raise HTTPException(status_code=400, detail="Failed to extract text from PDF")
    
    return text


def parse_transactions_with_gemini(text: str) -> List[dict]:
    """Parse transactions using Gemini"""
    if not gemini_model:
        raise HTTPException(status_code=500, detail="Gemini API not configured")
    
    # Truncate if too long
    if len(text) > 8000:
        text = text[:8000]
    
    prompt = f"""
    Extract ALL transactions from this credit card statement.
    Return ONLY valid JSON (no markdown, no explanation):
    
    {{"transactions": [{{"date": "2024-11-15", "merchant": "Store", "amount": 123.45, "description": "details"}}]}}
    
    Statement text:
    {text}
    """
    
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config={"temperature": 0.1, "max_output_tokens": 4096},
            request_options={"timeout": 60}
        )
        
        json_text = response.text.strip()
        
        # Clean markdown
        if '```' in json_text:
            parts = json_text.split('```')
            for part in parts:
                if 'json' in part or ('{' in part and '}' in part):
                    json_text = part.replace('json', '').strip()
        
        # Extract JSON
        start_idx = json_text.find('{')
        end_idx = json_text.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_text = json_text[start_idx:end_idx]
        
        data = json.loads(json_text)
        return data.get('transactions', [])
        
    except Exception as e:
        # Fallback to regex parser
        return fallback_parse(text)


def fallback_parse(text: str) -> List[dict]:
    """Regex-based fallback parser"""
    transactions = []
    patterns = [
        r'(\d{1,2}/\d{1,2}/\d{2,4})\s+(.+?)\s+[\$]?([\d,]+\.\d{2})',
        r'(\d{4}-\d{2}-\d{2})\s+(.+?)\s+[\$]?([\d,]+\.\d{2})',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.MULTILINE)
        for match in matches:
            try:
                date_str = match.group(1)
                merchant = match.group(2).strip()[:50]
                amount = float(match.group(3).replace(',', ''))
                
                if amount > 50000 or amount < 0.01:
                    continue
                
                # Parse date
                if '/' in date_str:
                    parts = date_str.split('/')
                    if len(parts[2]) == 2:
                        date = datetime.strptime(date_str, '%m/%d/%y')
                    else:
                        date = datetime.strptime(date_str, '%m/%d/%Y')
                else:
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                
                transactions.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'merchant': merchant,
                    'amount': amount,
                    'description': merchant
                })
            except:
                continue
    
    # Remove duplicates
    seen = set()
    unique = []
    for t in transactions:
        key = (t['date'], t['merchant'], t['amount'])
        if key not in seen:
            seen.add(key)
            unique.append(t)
    
    return unique


def categorize_transactions(transactions: List[dict]) -> List[dict]:
    """Categorize transactions with Gemini"""
    if not gemini_model:
        # Fallback categorization
        for txn in transactions:
            txn['category'] = simple_categorize(txn['merchant'])
        return transactions
    
    batch_size = 10
    for i in range(0, len(transactions), batch_size):
        batch = transactions[i:i+batch_size]
        
        txn_info = [{'merchant': t['merchant'], 'amount': t['amount']} for t in batch]
        
        prompt = f"""
        Categorize into: Groceries, Dining & Restaurants, Transportation, Entertainment, 
        Utilities & Bills, Healthcare, Shopping & Retail, Travel, Subscriptions, Other
        
        Transactions: {json.dumps(txn_info)}
        
        Return JSON array: ["Category1", "Category2", ...]
        """
        
        try:
            response = gemini_model.generate_content(prompt)
            json_text = response.text.strip()
            
            if '```' in json_text:
                json_text = json_text.split('```')[1].replace('json', '').strip()
            
            categories = json.loads(json_text)
            
            for j, cat in enumerate(categories):
                if i+j < len(transactions):
                    transactions[i+j]['category'] = cat
        except:
            # Fallback
            for j in range(len(batch)):
                if i+j < len(transactions):
                    transactions[i+j]['category'] = simple_categorize(batch[j]['merchant'])
    
    return transactions


def simple_categorize(merchant: str) -> str:
    """Simple keyword-based categorization"""
    merchant_lower = merchant.lower()
    
    if any(w in merchant_lower for w in ['grocery', 'market', 'food', 'walmart', 'target']):
        return 'Groceries'
    elif any(w in merchant_lower for w in ['restaurant', 'cafe', 'pizza', 'starbucks']):
        return 'Dining & Restaurants'
    elif any(w in merchant_lower for w in ['gas', 'fuel', 'uber', 'lyft']):
        return 'Transportation'
    elif any(w in merchant_lower for w in ['netflix', 'spotify', 'movie']):
        return 'Entertainment'
    elif any(w in merchant_lower for w in ['electric', 'water', 'internet', 'phone']):
        return 'Utilities & Bills'
    elif any(w in merchant_lower for w in ['amazon', 'store', 'shop']):
        return 'Shopping & Retail'
    elif any(w in merchant_lower for w in ['airline', 'hotel', 'travel']):
        return 'Travel'
    else:
        return 'Other'


def generate_summary(transactions: List[dict]) -> dict:
    """Generate spending summary"""
    df = pd.DataFrame(transactions)
    
    by_category = df.groupby('category')['amount'].sum().to_dict()
    
    return {
        'total_spent': float(df['amount'].sum()),
        'transaction_count': len(df),
        'average_transaction': float(df['amount'].mean()),
        'by_category': {k: float(v) for k, v in by_category.items()},
        'top_merchant': df.groupby('merchant')['amount'].sum().idxmax()
    }


def generate_insights(summary: dict) -> dict:
    """Generate AI insights"""
    if not gemini_model:
        return {
            'observations': ['API not configured'],
            'recommendations': ['Set up Gemini API key'],
            'concern': 'Unable to generate insights'
        }
    
    prompt = f"""
    Analyze spending: {json.dumps(summary)}
    
    Return JSON:
    {{
        "observations": ["obs1", "obs2", "obs3"],
        "recommendations": ["rec1", "rec2"],
        "concern": "concern text"
    }}
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        json_text = response.text.strip()
        
        if '```' in json_text:
            json_text = json_text.split('```')[1].replace('json', '').strip()
        
        return json.loads(json_text)
    except:
        top_cat = max(summary['by_category'].items(), key=lambda x: x[1])
        return {
            'observations': [
                f"Total spending: {summary['total_spent']:.2f}",
                f"Top category: {top_cat[0]} ({top_cat[1]:.2f})",
                f"Average transaction: {summary['average_transaction']:.2f}"
            ],
            'recommendations': [
                f"Consider budget for {top_cat[0]}",
                "Review recurring subscriptions"
            ],
            'concern': 'Monitor large transactions'
        }


# ============ API Endpoints ============

@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "online",
        "service": "Credit Card Analyzer API",
        "version": "1.0.0",
        "gemini_configured": gemini_model is not None,
        "currency": CURRENCY_CODE,
        "currency_symbol": CURRENCY_SYMBOL
    }


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_statement(file: UploadFile = File(...)):
    """
    Analyze credit card statement PDF
    
    - Upload PDF file
    - Returns transactions, summary, and AI insights
    """
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    
    try:
        # Extract text
        text = extract_text_from_pdf(tmp_path)
        
        if len(text) < 50:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Parse transactions
        transactions = parse_transactions_with_gemini(text)
        
        if not transactions:
            raise HTTPException(status_code=400, detail="No transactions found in PDF")
        
        # Categorize
        transactions = categorize_transactions(transactions)
        
        # Generate summary
        summary = generate_summary(transactions)
        
        # Generate insights
        insights = generate_insights(summary)
        
        # Store result
        analysis_id = str(uuid.uuid4())
        result = {
            'analysis_id': analysis_id,
            'transactions': transactions,
            'summary': summary,
            'insights': insights,
            'processed_at': datetime.now().isoformat()
        }
        
        analysis_storage[analysis_id] = result
        
        return result
        
    finally:
        # Clean up
        os.unlink(tmp_path)


@app.get("/api/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Retrieve previously processed analysis"""
    if analysis_id not in analysis_storage:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_storage[analysis_id]


@app.post("/api/demo")
async def get_demo_data():
    """Generate demo transaction data for testing"""
    import random
    
    dates = pd.date_range(start='2024-11-01', end='2024-11-30', freq='D')
    merchants = ['Amazon', 'Starbucks', 'Whole Foods', 'Shell', 'Netflix', 
                'Uber', 'Target', 'Walmart', 'Apple', 'Spotify']
    
    transactions = []
    for date in dates:
        for _ in range(random.randint(1, 3)):
            merchant = random.choice(merchants)
            transactions.append({
                'date': date.strftime('%Y-%m-%d'),
                'merchant': merchant,
                'amount': round(random.uniform(5, 150), 2),
                'category': simple_categorize(merchant),
                'description': f'Purchase at {merchant}'
            })
    
    summary = generate_summary(transactions)
    insights = generate_insights(summary)
    
    return {
        'analysis_id': 'demo',
        'transactions': transactions,
        'summary': summary,
        'insights': insights,
        'processed_at': datetime.now().isoformat()
    }


@app.get("/api/categories")
async def get_categories():
    """Get list of available categories"""
    return {
        'categories': [
            'Groceries',
            'Dining & Restaurants',
            'Transportation',
            'Entertainment',
            'Utilities & Bills',
            'Healthcare',
            'Shopping & Retail',
            'Travel',
            'Subscriptions',
            'Other'
        ]
    }


@app.get("/api/stats")
async def get_stats():
    """Get API usage statistics"""
    return {
        'total_analyses': len(analysis_storage),
        'gemini_configured': gemini_model is not None,
        'api_status': 'healthy'
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)