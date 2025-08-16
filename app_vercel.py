from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import whisper
import json
import os
import tempfile
from rapidfuzz import fuzz
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for caching
model = None
quran_db = None
preprocessed_cache = {}

def initialize_app():
    """Initialize the application with model and database loading"""
    global model, quran_db
    
    if model is None:
        logger.info("Loading Whisper model...")
        model = whisper.load_model("base")
        logger.info("Whisper model loaded successfully")
    
    if quran_db is None:
        quran_db = load_quran_database()
        logger.info(f"Loaded {len(quran_db)} surahs")

def load_quran_database():
    """Load all Quran surahs from JSON files"""
    quran_data = {}
    surah_dir = os.path.join(os.path.dirname(__file__), 'quran-json', 'surah')
    
    for i in range(1, 115):  # 114 surahs
        file_path = os.path.join(surah_dir, f'{i}.json')
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                surah_key = str(i)
                if surah_key in data:
                    surah_info = data[surah_key]
                    quran_data[i] = {
                        'number': int(surah_info['number']),
                        'name': surah_info['name'],
                        'name_arabic': surah_info['name'],
                        'name_latin': surah_info['name_latin'],
                        'number_of_ayah': int(surah_info['number_of_ayah']),
                        'verses': []
                    }
                    
                    # Extract verses
                    for verse_num in range(1, int(surah_info['number_of_ayah']) + 1):
                        verse_key = str(verse_num)
                        if verse_key in surah_info['text'] and verse_key in surah_info['translations']['id']['text']:
                            quran_data[i]['verses'].append({
                                'number': verse_num,
                                'text': surah_info['text'][verse_key],
                                'translation': surah_info['translations']['id']['text'][verse_key]
                            })
    
    return quran_data

def clean_verse_text(text):
    """Clean Arabic verse text for better matching"""
    if not text:
        return ""
    
    # Remove diacritics and normalize
    text = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', text)  # Remove diacritics and tatweel
    text = re.sub(r'[\u06D6-\u06ED]', '', text)  # Remove Quranic annotation marks
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()
    
    return text

def clean_translation_text(text):
    """Clean Indonesian translation text for better matching"""
    if not text:
        return ""
    
    # Remove punctuation and normalize
    text = re.sub(r'[.,;:!?()\[\]{}"\'-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    
    return text

def preprocess_arabic_text(text):
    """Preprocess Arabic text for fuzzy matching"""
    if not text:
        return ""
    
    # Cache preprocessing results
    if text in preprocessed_cache:
        return preprocessed_cache[text]
    
    # Clean and normalize
    processed = clean_verse_text(text)
    
    # Store in cache
    preprocessed_cache[text] = processed
    
    return processed

def find_matching_verses(transcribed_text, min_score=50.0, max_results=10):
    """Find matching verses using fuzzy string matching"""
    if not transcribed_text or not quran_db:
        return []
    
    # Clean transcribed text
    processed_transcription = clean_translation_text(transcribed_text)
    transcription_length = len(processed_transcription.split())
    
    if transcription_length < 3:
        min_score = max(min_score, 70.0)
    
    # Find individual verse matches
    individual_matches = find_individual_verse_matches(processed_transcription, transcription_length, min_score)
    
    # Find consecutive verse matches
    consecutive_matches = find_consecutive_verse_matches(processed_transcription, transcription_length, min_score)
    
    # Combine and sort all matches
    all_matches = individual_matches + consecutive_matches
    all_matches.sort(key=lambda x: x['score'], reverse=True)
    
    # Remove duplicates and limit results
    seen = set()
    unique_matches = []
    
    for match in all_matches:
        match_key = (match['surah'], tuple(match['verses']))
        if match_key not in seen:
            seen.add(match_key)
            unique_matches.append(match)
            if len(unique_matches) >= max_results:
                break
    
    return unique_matches

def find_individual_verse_matches(processed_transcription, transcription_length, min_score):
    """Find individual verse matches"""
    matches = []
    
    for surah_num, surah_data in quran_db.items():
        for verse in surah_data['verses']:
            # Check translation match
            cleaned_translation = clean_translation_text(verse['translation'])
            if cleaned_translation:
                score = fuzz.ratio(processed_transcription, cleaned_translation)
                
                if score >= min_score:
                    matches.append({
                        'surah': surah_num,
                        'surah_name': surah_data['name_latin'],
                        'surah_arabic': surah_data['name_arabic'],
                        'verses': [verse['number']],
                        'score': score,
                        'match_type': 'individual',
                        'text': verse['text'],
                        'translation': verse['translation']
                    })
    
    return matches

def find_consecutive_verse_matches(processed_transcription, transcription_length, min_score):
    """Find consecutive verse matches"""
    matches = []
    max_consecutive = min(5, max(2, transcription_length // 10))
    
    for surah_num, surah_data in quran_db.items():
        verses = surah_data['verses']
        
        for start_idx in range(len(verses)):
            for length in range(2, min(max_consecutive + 1, len(verses) - start_idx + 1)):
                end_idx = start_idx + length
                verse_group = verses[start_idx:end_idx]
                
                # Combine translations
                combined_translation = ' '.join([v['translation'] for v in verse_group])
                cleaned_combined = clean_translation_text(combined_translation)
                
                if cleaned_combined:
                    score = fuzz.ratio(processed_transcription, cleaned_combined)
                    
                    if score >= min_score:
                        matches.append({
                            'surah': surah_num,
                            'surah_name': surah_data['name_latin'],
                            'surah_arabic': surah_data['name_arabic'],
                            'verses': [v['number'] for v in verse_group],
                            'score': score,
                            'match_type': 'consecutive',
                            'text': ' '.join([v['text'] for v in verse_group]),
                            'translation': combined_translation
                        })
    
    return matches

@app.route('/')
def index():
    """Serve the main page"""
    initialize_app()
    return render_template('index.html')

@app.route('/api/identify-verse', methods=['POST'])
def identify_verse():
    """API endpoint to identify Quranic verses from audio"""
    try:
        initialize_app()
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            temp_filename = temp_file.name
        
        try:
            # Transcribe audio using Whisper
            result = model.transcribe(temp_filename, language='id')
            transcribed_text = result['text'].strip()
            
            if not transcribed_text:
                return jsonify({
                    'success': False,
                    'error': 'No speech detected in audio',
                    'transcription': '',
                    'matches': []
                })
            
            # Find matching verses
            matches = find_matching_verses(transcribed_text, min_score=50.0, max_results=10)
            
            # Filter high-confidence matches
            high_confidence_matches = [m for m in matches if m['score'] >= 80.0]
            
            return jsonify({
                'success': True,
                'transcription': transcribed_text,
                'matches': high_confidence_matches if high_confidence_matches else matches[:5],
                'total_matches': len(matches)
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
                
    except Exception as e:
        logger.error(f"Error in identify_verse: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'transcription': '',
            'matches': []
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    initialize_app()
    return jsonify({'status': 'healthy', 'surahs_loaded': len(quran_db) if quran_db else 0})

# Vercel handler function
def handler(request):
    """Main handler function for Vercel"""
    return app(request.environ, lambda status, headers: None)

# For local development
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)