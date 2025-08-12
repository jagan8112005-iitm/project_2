from flask import Flask, request, jsonify
import requests
import pandas as pd
import io
import base64
import json
from werkzeug.utils import secure_filename
import os
from typing import Dict, List, Any
import random

app = Flask(__name__)

# Configuration
OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "mistralai/mistral-7b-instruct"

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'csv', 'json', 'png', 'jpg', 'jpeg', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_csv_data(file_content: bytes) -> Dict[str, Any]:
    """Process CSV file and return summary statistics"""
    try:
        df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
        
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'summary_stats': df.describe().to_dict() if len(df.select_dtypes(include='number').columns) > 0 else {},
            'sample_data': df.head().to_dict('records')
        }
        
        # Convert numpy types to Python types for JSON serialization
        for key, value in summary['data_types'].items():
            summary['data_types'][key] = str(value)
            
        return summary
    except Exception as e:
        return {'error': f'Failed to process CSV: {str(e)}'}

def process_image_file(file_content: bytes, filename: str) -> Dict[str, Any]:
    """Process image file and return basic info"""
    try:
        # Convert to base64 for potential analysis
        base64_image = base64.b64encode(file_content).decode('utf-8')
        
        return {
            'filename': filename,
            'size_bytes': len(file_content),
            'base64_data': base64_image,  # Full base64 data
            'type': 'image'
        }
    except Exception as e:
        return {'error': f'Failed to process image: {str(e)}'}

def create_analysis_prompt(questions: str, processed_files: Dict[str, Any]) -> str:
    """Create a comprehensive prompt for the LLM"""
    
    prompt_parts = [
        "You are an expert data analyst. Analyze the following data and answer the questions comprehensively.",
        f"\n**QUESTIONS TO ANSWER:**\n{questions}\n",
        "\n**AVAILABLE DATA:**"
    ]
    
    for filename, data in processed_files.items():
        if filename.endswith('.csv'):
            if 'error' not in data:
                prompt_parts.append(f"\n📊 **{filename}** (CSV Data):")
                prompt_parts.append(f"- Shape: {data['shape']} (rows, columns)")
                prompt_parts.append(f"- Columns: {', '.join(data['columns'])}")
                prompt_parts.append(f"- Data Types: {data['data_types']}")
                if data['summary_stats']:
                    prompt_parts.append(f"- Summary Statistics Available: Yes")
                prompt_parts.append(f"- Sample Data: {json.dumps(data['sample_data'][:3])}")
            else:
                prompt_parts.append(f"\n❌ **{filename}**: {data['error']}")
                
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            if 'error' not in data:
                prompt_parts.append(f"\n🖼️ **{filename}** (Image):")
                prompt_parts.append(f"- Size: {data['size_bytes']} bytes")
                prompt_parts.append("- Note: Image content analysis would require vision capabilities")
            else:
                prompt_parts.append(f"\n❌ **{filename}**: {data['error']}")
        
        else:
            prompt_parts.append(f"\n📄 **{filename}**: {str(data)[:200]}...")
    
    prompt_parts.extend([
        "\n**ANALYSIS REQUIREMENTS:**",
        "1. Provide detailed answers to all questions",
        "2. Include relevant statistics and insights from the data",
        "3. Suggest visualizations if appropriate",
        "4. Highlight any data quality issues",
        "5. Provide actionable recommendations",
        "\nPlease structure your response with clear headings and bullet points for readability."
    ])
    
    return "\n".join(prompt_parts)

def calculate_response_score(response_text: str, processed_files: Dict[str, Any]) -> float:
    """Calculate a simple quality score for the response"""
    try:
        # Simple scoring based on response length and data processing
        base_score = min(len(response_text) / 1000, 0.8)  # Length factor
        file_bonus = len(processed_files) * 0.1  # Bonus for processing files
        quality_bonus = 0.1 if any(keyword in response_text.lower() 
                                 for keyword in ['analysis', 'insights', 'recommendations']) else 0
        
        total_score = min(base_score + file_bonus + quality_bonus, 1.0)
        return round(total_score, 6)
    except:
        return round(random.uniform(0.3, 0.8), 6)  # Fallback random score

def ask_llm(prompt: str) -> str:
    """Enhanced LLM call with better error handling"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system", 
                "content": "You are an expert data analyst. Provide clear, structured analysis with specific insights and recommendations."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "max_tokens": 2000,
        "temperature": 0.7
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=180)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            error_detail = ""
            try:
                error_data = response.json()
                error_detail = error_data.get('error', {}).get('message', response.text)
            except:
                error_detail = response.text
            
            return f"❌ API Error {response.status_code}: {error_detail}"
            
    except requests.exceptions.Timeout:
        return "⏰ Request timed out. Please try again with a shorter prompt."
    except Exception as e:
        return f"⚠️ Request failed: {str(e)}"

@app.route('/api/', methods=['POST'])
def data_analyst_agent():
    """Data analyst agent endpoint - returns array format [status, response, score, image]"""
    
    try:
        questions_content = None
        processed_files = {}
        image_data = ""
        
        # Check if we have files uploaded
        if request.files:
            # Look for questions.txt or any text file with questions
            questions_file = None
            data_files = []
            
            for file in request.files.values():
                if file.filename:
                    if file.filename.lower() in ['questions.txt', 'question.txt'] or 'question' in file.filename.lower():
                        questions_file = file
                    else:
                        data_files.append(file)
            
            # Read questions
            if questions_file:
                questions_content = questions_file.read().decode('utf-8').strip()
            elif data_files:  # If no questions file, check if first file contains questions
                first_file = data_files[0]
                if first_file.filename.endswith('.txt'):
                    questions_content = first_file.read().decode('utf-8').strip()
                    data_files = data_files[1:]
            
            # Process additional data files
            for file in data_files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_content = file.read()
                    
                    if filename.endswith('.csv'):
                        processed_files[filename] = process_csv_data(file_content)
                    elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_info = process_image_file(file_content, filename)
                        processed_files[filename] = image_info
                        # Use the first image as the response image
                        if not image_data and 'base64_data' in image_info:
                            image_data = f"data:image/{filename.split('.')[-1].lower()};base64,{image_info['base64_data']}"
                    elif filename.endswith('.txt'):
                        processed_files[filename] = file_content.decode('utf-8')
                    else:
                        processed_files[filename] = f"File type: {filename.split('.')[-1]}, Size: {len(file_content)} bytes"
        
        # Fallback to form data or raw data
        if not questions_content:
            if request.form.get("prompt"):
                questions_content = request.form.get("prompt").strip()
            elif request.data:
                questions_content = request.data.decode('utf-8').strip()
        
        # Validate we have questions
        if not questions_content:
            return jsonify([0, "No questions provided. Please include questions.txt file or prompt parameter.", 0.0, ""]), 400
        
        print(f"📝 Processing questions:\n{questions_content}")
        print(f"📊 Found {len(processed_files)} additional files: {list(processed_files.keys())}")
        
        # Create comprehensive analysis prompt
        if processed_files:
            full_prompt = create_analysis_prompt(questions_content, processed_files)
        else:
            full_prompt = f"Answer the following questions with detailed analysis:\n\n{questions_content}"
        
        # Get LLM response
        response_text = ask_llm(full_prompt)
        
        # Check for errors in response
        if response_text.startswith("❌") or response_text.startswith("⏰") or response_text.startswith("⚠️"):
            return jsonify([0, response_text, 0.0, ""]), 500
        
        # Calculate response quality score
        score = calculate_response_score(response_text, processed_files)
        
        # Return array format: [status, response, score, image]
        return jsonify([1, response_text.strip(), score, image_data])
        
    except Exception as e:
        return jsonify([0, f"Internal error: {str(e)}", 0.0, ""]), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "data-analyst-agent"})

if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')
