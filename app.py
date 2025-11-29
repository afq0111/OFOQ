# IMPORTS
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import google.generativeai as genai
import json
import os

# ============================================================
# CONFIGURATION
# ============================================================

# Get API key from environment variable (set this in Render dashboard)
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# ============================================================
# LOAD/TRAIN ML MODELS
# ============================================================

print("Loading ML models...")

# Load your dataset
df = pd.read_csv('children_dataset.csv')

# Feature engineering
df['score_variance'] = df[['matching_score', 'sequencing_score', 'memory_score', 'auditory_score']].var(axis=1)
df['visual_auditory_gap'] = df['matching_score'] - df['auditory_score']
df['cognitive_avg'] = (df['attention_score'] + df['memory_field_score']) / 2
df['academic_avg'] = (df['language_score'] + df['math_score']) / 2
df['best_score'] = df[['matching_score', 'sequencing_score', 'memory_score', 'auditory_score']].max(axis=1)
df['worst_score'] = df[['matching_score', 'sequencing_score', 'memory_score', 'auditory_score']].min(axis=1)
df['score_range'] = df['best_score'] - df['worst_score']

feature_columns = [
    'age', 'total_sessions', 'avg_progress', 'avg_score', 'success_rate',
    'matching_score', 'sequencing_score', 'memory_score', 'auditory_score',
    'attention_score', 'memory_field_score', 'language_score', 'math_score',
    'score_variance', 'visual_auditory_gap', 'cognitive_avg', 'academic_avg', 'score_range'
]

X = df[feature_columns].copy()
le_diagnosis = LabelEncoder()
df['diagnosis_encoded'] = le_diagnosis.fit_transform(df['medical_diagnosis'])
X['diagnosis'] = df['diagnosis_encoded']

# Train models
rf_strength = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_strength.fit(X, df['dominant_strength'])

rf_weakness = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_weakness.fit(X, df['primary_weakness'])

rf_style = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
rf_style.fit(X, df['learning_style'])

rf_trend = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_trend.fit(X, df['progress_trend'])

print("✅ ML models loaded")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def predict_child_profile(child_data):
    """Make ML predictions"""
    features = {
        'age': child_data['child_basic_info']['age'],
        'total_sessions': child_data['overall_progress']['total_sessions'],
        'avg_progress': child_data['overall_progress']['avg_progress'],
        'avg_score': child_data['overall_progress']['avg_score'],
        'success_rate': child_data['overall_progress']['success_rate'],
        'matching_score': child_data['progress_by_game_type']['matching'],
        'sequencing_score': child_data['progress_by_game_type']['sequencing'],
        'memory_score': child_data['progress_by_game_type']['memory'],
        'auditory_score': child_data['progress_by_game_type']['auditory'],
        'attention_score': child_data['progress_by_learning_field']['attention'],
        'memory_field_score': child_data['progress_by_learning_field']['memory'],
        'language_score': child_data['progress_by_learning_field']['language'],
        'math_score': child_data['progress_by_learning_field']['math']
    }

    game_scores = [features['matching_score'], features['sequencing_score'],
                   features['memory_score'], features['auditory_score']]
    features['score_variance'] = np.var(game_scores)
    features['visual_auditory_gap'] = features['matching_score'] - features['auditory_score']
    features['cognitive_avg'] = (features['attention_score'] + features['memory_field_score']) / 2
    features['academic_avg'] = (features['language_score'] + features['math_score']) / 2
    features['best_score'] = max(game_scores)
    features['worst_score'] = min(game_scores)
    features['score_range'] = features['best_score'] - features['worst_score']

    diagnosis = child_data['child_basic_info'].get('medical_diagnosis', 'unknown')
    try:
        features['diagnosis'] = le_diagnosis.transform([diagnosis])[0]
    except:
        features['diagnosis'] = 0

    feature_vector = pd.DataFrame([features])[X.columns]

    return {
        'dominant_strength': rf_strength.predict(feature_vector)[0],
        'primary_weakness': rf_weakness.predict(feature_vector)[0],
        'learning_style': rf_style.predict(feature_vector)[0],
        'progress_trend': rf_trend.predict(feature_vector)[0]
    }

def generate_gemini_prompt(child_data, ml_predictions, teacher_notes=""):
    """Generate short prompt"""
    age = child_data['child_basic_info']['age']
    diagnosis = child_data['child_basic_info'].get('medical_diagnosis', 'غير محدد')
    sessions = child_data['overall_progress']['total_sessions']
    avg_score = child_data['overall_progress']['avg_score']

    matching = child_data['progress_by_game_type']['matching']
    memory = child_data['progress_by_game_type']['memory']
    auditory = child_data['progress_by_game_type']['auditory']

    attention = child_data['progress_by_learning_field']['attention']
    language = child_data['progress_by_learning_field']['language']
    math_score = child_data['progress_by_learning_field']['math']

    strength = ml_predictions['dominant_strength']
    weakness = ml_predictions['primary_weakness']

    notes = teacher_notes[:200] if teacher_notes else "لا توجد"

    return f"""أنت خبير تربية خاصة. حلل وأعط توصيات بالعربية فقط.

طفل {age} سنوات | {diagnosis} | {sessions} جلسة | معدل {avg_score:.0f}%
الأداء: مطابقة {matching:.0f}% | ذاكرة {memory:.0f}% | سمعي {auditory:.0f}%
المهارات: انتباه {attention:.0f}% | لغة {language:.0f}% | رياضيات {math_score:.0f}%
AI: قوة={strength} | ضعف={weakness}
معلم: {notes}

مهم: الرد بالعربية 100%. استخدم هذا الهيكل:
{{"highlighted_strengths":["نقطة 1","نقطة 2","نقطة 3"],"areas_for_improvement":["مجال 1","مجال 2"],"suggested_activities":["نشاط 1","نشاط 2","نشاط 3"],"near_term_goals":["هدف 1","هدف 2"]}}

JSON فقط بدون markdown."""

def call_gemini(prompt):
    """Call Gemini API"""
    model = genai.GenerativeModel(
        'gemini-2.0-flash',
        system_instruction="رد بالعربية فقط. استخدم JSON المحدد."
    )

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.3,
            max_output_tokens=1000,
        )
    )

    response_text = response.text.strip()

    # Clean markdown
    if '```json' in response_text:
        response_text = response_text.split('```json')[1].split('```')[0]
    elif '```' in response_text:
        response_text = response_text.split('```')[1].split('```')[0]

    return json.loads(response_text.strip())

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'Child Assessment API is active',
        'endpoints': {
            '/analyze': 'POST - Analyze child performance',
            '/health': 'GET - Check API status'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'healthy', 'ml_models': 'loaded', 'gemini': 'connected'})

@app.route('/analyze', methods=['POST'])
def analyze_child():
    """
    Main endpoint to analyze child performance
    """
    try:
        # Get request data
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Validate required fields
        required_fields = ['child_basic_info', 'overall_progress',
                          'progress_by_game_type', 'progress_by_learning_field']

        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Extract teacher notes
        teacher_notes = data.get('teacher_notes', '')

        # Step 1: ML Predictions
        ml_predictions = predict_child_profile(data)

        # Step 2: Generate Gemini prompt
        prompt = generate_gemini_prompt(data, ml_predictions, teacher_notes)

        # Step 3: Call Gemini
        recommendations = call_gemini(prompt)

        # Step 4: Return response
        return jsonify({
            'success': True,
            'child_id': data['child_basic_info'].get('child_id'),
            'ml_predictions': ml_predictions,
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }), 500

# ============================================================
# RUN FOR PRODUCTION
# ============================================================

if __name__ == '__main__':
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
