from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import whisper
import json
import os
import tempfile
from rapidfuzz import fuzz
import re
import logging
import requests

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Whisper model
logger.info("Loading Whisper model...")
model = whisper.load_model("base")
logger.info("Whisper model loaded successfully")

# Load Quran database
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
                            verse_data = {
                                'verse_number': verse_num,
                                'arabic_text': surah_info['text'][verse_key],
                                'indonesian_translation': surah_info['translations']['id']['text'][verse_key]
                            }
                            quran_data[i]['verses'].append(verse_data)
    
    logger.info(f"Loaded {len(quran_data)} surahs from database")
    return quran_data

# Load the database
quran_db = load_quran_database()

# Cache for preprocessed verse texts to improve performance
preprocessed_cache = {}

def clean_verse_text(text):
    """Clean verse text from repeated numbering and formatting issues"""
    if not text:
        return ""
    
    # Remove all existing verse numbers first
    text = re.sub(r'\(\d+\)\s*', '', text)
    # Remove any remaining parentheses with numbers
    text = re.sub(r'\(\d+\)', '', text)
    # Clean up extra spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def clean_translation_text(text):
    """Clean translation text from repeated numbering and formatting issues"""
    if not text:
        return ""
    
    # Remove all existing verse numbers first
    text = re.sub(r'\(\d+\)\s*', '', text)
    # Remove any remaining parentheses with numbers
    text = re.sub(r'\(\d+\)', '', text)
    # Clean up extra spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def preprocess_arabic_text(text):
    """Preprocess Arabic text for better matching"""
    # Remove diacritics and normalize
    text = re.sub(r'[ًٌٍَُِّْٰٱ]', '', text)
    text = re.sub(r'[۝۞ۖۗۘۙۚۛۜ]', '', text)
    text = re.sub(r'[ࣖࣗࣘࣙࣚࣛࣜࣝࣞࣟ]', '', text)
    
    # Normalize common Whisper transcription variations
    # Handle common ending variations
    text = re.sub(r'ويلو\b', 'ويل', text)  # ويلو -> ويل
    text = re.sub(r'و\s+يوم', 'ويوم', text)  # و يوم -> ويوم
    
    # Handle word segmentation issues
    text = re.sub(r'يوم\s+ايذ', 'يومايذ', text)  # يوم ايذ -> يومايذ
    text = re.sub(r'يومايذ', 'يومىٕذ', text)  # يومايذ -> يومىٕذ
    
    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def find_best_matching_verse(transcribed_text, min_score=60.0):
    """Find the single best matching verse that most closely matches the entire transcribed text"""
    best_match = None
    best_score = 0
    
    # Preprocess the transcribed text
    processed_transcription = preprocess_arabic_text(transcribed_text)
    
    for surah_num, surah_data in quran_db.items():
        for verse in surah_data['verses']:
            # Use cached preprocessed text or create new one
            cache_key = f"{surah_num}_{verse['verse_number']}"
            if cache_key in preprocessed_cache:
                processed_verse = preprocessed_cache[cache_key]
            else:
                processed_verse = preprocess_arabic_text(verse['arabic_text'])
                preprocessed_cache[cache_key] = processed_verse
            
            # Calculate similarity scores using different methods
            # Focus on ratio and token_sort_ratio for full text matching
            ratio_score = fuzz.ratio(processed_transcription, processed_verse)
            token_sort_score = fuzz.token_sort_ratio(processed_transcription, processed_verse)
            
            # Weight ratio score higher for full text matching
            weighted_score = (ratio_score * 0.7) + (token_sort_score * 0.3)
            
            if weighted_score > best_score and weighted_score >= min_score:
                best_score = weighted_score
                best_match = {
                    'surah_number': surah_num,
                    'surah_name': surah_data['name'],
                    'surah_name_arabic': surah_data['name'],
                    'surah_name_latin': surah_data['name_latin'],
                    'verse_number': verse['verse_number'],
                    'verse_text': verse['arabic_text'],
                    'translation': verse['indonesian_translation'],
                    'similarity_score': round(weighted_score, 2)
                }
    
    return best_match

def classify_verse_type(matches):
    """Classify the type of verse based on matches"""
    if not matches:
        return "unknown"
    
    best_match = matches[0]
    
    # Check if it's consecutive verses (beruntun dalam surat yang sama)
    if best_match.get('verse_count', 1) > 1 and best_match.get('end_verse_number'):
        return "consecutive"
    
    # Check for identical verses (ayat tunggal kembar identik)
    # Look for verses with identical text in different locations
    best_verse_text = best_match['verse_text']
    identical_locations = []
    
    # Preprocess the best match text for comparison
    processed_best_text = preprocess_arabic_text(best_verse_text)
    
    for match in matches[:10]:  # Check top 10 matches
        processed_match_text = preprocess_arabic_text(match['verse_text'])
        
        # Check if texts are identical after preprocessing
        if (processed_match_text == processed_best_text and 
            match['similarity_score'] >= 85 and
            (match['surah_number'] != best_match['surah_number'] or 
             match['verse_number'] != best_match['verse_number'])):
            identical_locations.append(match)
    
    # If we found identical verses, check if they are actually from different locations
    # (not just consecutive verses from the same surah)
    if len(identical_locations) >= 1:
        # Check if we have verses from different surahs or non-consecutive verses
        different_surahs = any(match['surah_number'] != best_match['surah_number'] for match in identical_locations)
        
        if different_surahs:
            return "identical"
        
        # If all from same surah, check if they are non-consecutive
        same_surah_verses = [match for match in identical_locations if match['surah_number'] == best_match['surah_number']]
        if same_surah_verses:
            verse_numbers = [best_match['verse_number']] + [match['verse_number'] for match in same_surah_verses]
            verse_numbers.sort()
            
            # Check if verses are consecutive
            is_consecutive = all(verse_numbers[i] == verse_numbers[i-1] + 1 for i in range(1, len(verse_numbers)))
            
            if not is_consecutive:
                return "identical"  # Same text but not consecutive = identical verses
    
    # Default to single verse (ayat tunggal biasa)
    return "single"

def find_matching_verses(transcribed_text, min_score=50.0, max_results=10):
    """Find all matching verses from the Quran database with similarity above threshold"""
    matches = []
    
    # Preprocess the transcribed text
    processed_transcription = preprocess_arabic_text(transcribed_text)
    transcription_words = processed_transcription.split()
    transcription_length = len(transcription_words)
    
    # First, try to find matches with individual verses
    individual_matches = find_individual_verse_matches(processed_transcription, transcription_length, min_score)
    matches.extend(individual_matches)
    
    # Check if we have a very high individual match - skip consecutive if so for better performance
    has_very_high_individual = individual_matches and max(match['similarity_score'] for match in individual_matches) >= 85.0
    
    # Only try consecutive verse combinations if no very high individual match found
    if not has_very_high_individual:
        consecutive_matches = find_consecutive_verse_matches(processed_transcription, transcription_length, min_score)
        matches.extend(consecutive_matches)
    
    # Check if we have a high-scoring individual match (likely a single long verse)
    has_high_individual_match = (individual_matches and 
                                any(match['similarity_score'] >= 72.5 for match in individual_matches))
    
    # Remove duplicates and sort by score
    seen = set()
    unique_matches = []
    for match in matches:
        key = (match['surah_number'], match['verse_number'], match.get('end_verse_number', match['verse_number']))
        if key not in seen:
            seen.add(key)
            unique_matches.append(match)

    unique_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # Always group consecutive verses to compare with individual matches
    grouped_matches = group_consecutive_verses(unique_matches)
    
    # If we have a high-scoring individual match, prefer individual results
    # but still allow consecutive matches if they score significantly higher
    if has_high_individual_match:
        # Find the best individual match score
        best_individual_score = max(match['similarity_score'] for match in individual_matches)
        
        # Check if any grouped match significantly outperforms the best individual match
        best_grouped_score = max((match['similarity_score'] for match in grouped_matches), default=0)
        
        # If grouped match is significantly better (>5% difference), use grouped results
        if best_grouped_score > best_individual_score + 5:
            pass  # Use grouped_matches
        else:
            # Use individual matches without grouping for high-scoring individual matches
            grouped_matches = unique_matches

    # Return all matches if max_results is None, otherwise limit the results
    if max_results is None:
        return grouped_matches
    else:
        return grouped_matches[:max_results]

def find_individual_verse_matches(processed_transcription, transcription_length, min_score):
    """Find matches with individual verses"""
    matches = []
    best_score = 0
    
    for surah_num, surah_data in quran_db.items():
        for verse in surah_data['verses']:
            # Use cached preprocessed text or create new one
            cache_key = f"{surah_num}_{verse['verse_number']}"
            if cache_key in preprocessed_cache:
                processed_verse = preprocessed_cache[cache_key]
            else:
                processed_verse = preprocess_arabic_text(verse['arabic_text'])
                preprocessed_cache[cache_key] = processed_verse
            verse_words = processed_verse.split()
            verse_length = len(verse_words)
            
            # Calculate similarity scores using different methods
            ratio_score = fuzz.ratio(processed_transcription, processed_verse)
            partial_score = fuzz.partial_ratio(processed_transcription, processed_verse)
            token_sort_score = fuzz.token_sort_ratio(processed_transcription, processed_verse)
            token_set_score = fuzz.token_set_ratio(processed_transcription, processed_verse)
            
            # Calculate length ratio for penalty
            length_ratio = min(transcription_length, verse_length) / max(transcription_length, verse_length)
            
            # Improved scoring algorithm for better accuracy
            if transcription_length <= 5:  # Short transcriptions (like Bismillah)
                # For short transcriptions, prioritize exact matches
                if ratio_score >= 95:
                    # Very high ratio score - likely exact match
                    final_score = ratio_score
                elif length_ratio >= 0.7 and token_sort_score >= 85:
                    # Similar length with good token sort - likely good match
                    final_score = token_sort_score
                else:
                    # Apply penalty for length mismatch and use weighted average
                    weighted_score = (ratio_score * 0.5 + token_sort_score * 0.3 + partial_score * 0.2)
                    final_score = weighted_score * length_ratio
            else:
                # For longer transcriptions, use the original approach with slight modification
                # Use weighted average instead of max to avoid false positives
                weighted_score = (ratio_score * 0.4 + partial_score * 0.3 + 
                                token_sort_score * 0.2 + token_set_score * 0.1)
                final_score = weighted_score
            
            # Only include if score meets minimum threshold
            if final_score >= min_score:
                match = {
                    'surah_number': surah_num,
                    'surah_name': surah_data['name'],
                    'surah_name_arabic': surah_data['name'],
                    'surah_name_latin': surah_data['name_latin'],
                    'verse_number': verse['verse_number'],
                    'verse_text': verse['arabic_text'],
                    'translation': verse['indonesian_translation'],
                    'similarity_score': round(final_score, 2),
                    'debug_scores': {
                        'ratio': ratio_score,
                        'partial': partial_score,
                        'token_sort': token_sort_score,
                        'token_set': token_set_score,
                        'length_ratio': round(length_ratio, 2)
                    }
                }
                matches.append(match)
                
                # Update best score and implement early termination for very high scores
                if final_score > best_score:
                    best_score = final_score
                    # Early termination if we find a near-perfect match
                    if final_score >= 95.0:
                        logger.info(f"Early termination: Found near-perfect match with score {final_score:.2f}%")
                        return matches
    
    return matches

def find_consecutive_verse_matches(processed_transcription, transcription_length, min_score):
    """Find matches with consecutive verses (for cases like complete surahs or multiple verses)"""
    matches = []
    
    for surah_num, surah_data in quran_db.items():
        verses = surah_data['verses']
        
        # Try different combinations of consecutive verses (2-5 verses for better performance)
        for start_idx in range(len(verses)):
            for end_idx in range(start_idx + 1, min(start_idx + 6, len(verses) + 1)):
                # Combine consecutive verses
                combined_text = ' '.join([verse['arabic_text'] for verse in verses[start_idx:end_idx]])
                processed_combined = preprocess_arabic_text(combined_text)
                
                # Calculate similarity scores
                ratio_score = fuzz.ratio(processed_transcription, processed_combined)
                partial_score = fuzz.partial_ratio(processed_transcription, processed_combined)
                token_sort_score = fuzz.token_sort_ratio(processed_transcription, processed_combined)
                token_set_score = fuzz.token_set_ratio(processed_transcription, processed_combined)
                
                # Use weighted average for consecutive verses
                weighted_score = (ratio_score * 0.4 + partial_score * 0.3 + 
                                token_sort_score * 0.2 + token_set_score * 0.1)
                
                if weighted_score >= min_score:
                    # Create individual verse matches for grouping later
                    for i, verse in enumerate(verses[start_idx:end_idx]):
                        verse_match = {
                            'surah_number': surah_num,
                            'surah_name': surah_data['name'],
                            'surah_name_arabic': surah_data['name'],
                            'surah_name_latin': surah_data['name_latin'],
                            'verse_number': verse['verse_number'],
                            'verse_text': verse['arabic_text'],
                            'translation': verse['indonesian_translation'],
                            'similarity_score': round(weighted_score, 2),
                            'debug_scores': {
                                'ratio': ratio_score,
                                'partial': partial_score,
                                'token_sort': token_sort_score,
                                'token_set': token_set_score
                            }
                        }
                        matches.append(verse_match)
    
    return matches

def group_consecutive_verses(matches):
    """Group verses from the same surah into single results with verse ranges"""
    if not matches:
        return matches
    
    # Group by surah number
    surah_groups = {}
    for match in matches:
        surah_num = match['surah_number']
        if surah_num not in surah_groups:
            surah_groups[surah_num] = []
        surah_groups[surah_num].append(match)
    
    grouped = []
    for surah_num, surah_matches in surah_groups.items():
        # Sort verses by verse number
        surah_matches.sort(key=lambda x: x['verse_number'])
        
        # Check if we have multiple verses from the same surah
        if len(surah_matches) > 1:
            # Check if verses are actually consecutive
            consecutive_groups = []
            current_group = [surah_matches[0]]
            
            for i in range(1, len(surah_matches)):
                current_verse = surah_matches[i]['verse_number']
                prev_verse = surah_matches[i-1]['verse_number']
                
                # If verses are consecutive (difference of 1), add to current group
                if current_verse == prev_verse + 1:
                    current_group.append(surah_matches[i])
                else:
                    # Not consecutive, start a new group
                    consecutive_groups.append(current_group)
                    current_group = [surah_matches[i]]
            
            # Add the last group
            consecutive_groups.append(current_group)
            
            # Process each consecutive group
            for group in consecutive_groups:
                if len(group) > 1:
                    # Multiple consecutive verses - group them
                    grouped.append(create_grouped_result(group))
                else:
                    # Single verse - keep as individual
                    grouped.extend(group)
        else:
            # Single verse from this surah
            grouped.extend(surah_matches)
    
    # Sort by similarity score
    grouped.sort(key=lambda x: x['similarity_score'], reverse=True)
    return grouped

def create_grouped_result(verse_group):
    """Create a single result from a group of consecutive verses"""
    if len(verse_group) == 1:
        # Single verse, return as is but clean the text
        verse = verse_group[0].copy()
        # Clean verse text from any existing numbering
        verse['verse_text'] = clean_verse_text(verse['verse_text'])
        verse['translation'] = clean_translation_text(verse['translation'])
        return verse
    
    # Multiple verses, create combined result
    first_verse = verse_group[0]
    last_verse = verse_group[-1]
    
    # Untuk ayat berurutan, tampilkan setiap ayat dengan nomor yang jelas
    arabic_parts = []
    translation_parts = []
    
    for verse in verse_group:
        verse_num = verse['verse_number']
        clean_arabic = clean_verse_text(verse['verse_text'])
        clean_translation = clean_translation_text(verse['translation'])
        
        # Format dengan nomor ayat yang jelas
        arabic_parts.append(f"({verse_num}) {clean_arabic}")
        translation_parts.append(f"({verse_num}) {clean_translation}")
    
    # Gabungkan dengan pemisah baris baru untuk kejelasan
    combined_arabic = "\n\n".join(arabic_parts)
    combined_translation = "\n\n".join(translation_parts)
    
    # Calculate average similarity score
    avg_score = sum(verse['similarity_score'] for verse in verse_group) / len(verse_group)
    
    # Create verse range info untuk identifikasi
    start_verse = first_verse['verse_number']
    end_verse = last_verse['verse_number']
    
    # Buat informasi range ayat untuk identifikasi
    if start_verse == end_verse:
        verse_info = f"Ayat {start_verse}"
        verse_reference = f"{first_verse['surah_name_latin']} {start_verse}"
    else:
        verse_info = f"Ayat {start_verse}-{end_verse}"
        verse_reference = f"{first_verse['surah_name_latin']} {start_verse}-{end_verse}"
    
    return {
        'surah_number': first_verse['surah_number'],
        'surah_name': first_verse['surah_name'],
        'surah_name_arabic': first_verse['surah_name_arabic'],
        'surah_name_latin': first_verse['surah_name_latin'],
        'verse_number': start_verse,
        'end_verse_number': end_verse,
        'verse_reference': verse_reference,
        'verse_info': verse_info,
        'verse_count': len(verse_group),
        'verse_text': combined_arabic,
        'translation': combined_translation,
        'similarity_score': round(avg_score, 2),
        'debug_scores': {
            'individual_scores': [verse['similarity_score'] for verse in verse_group],
            'average_score': round(avg_score, 2)
        }
    }

def find_best_match(transcribed_text, threshold=70):
    """Find the single best matching verse (for backward compatibility)"""
    matches = find_matching_verses(transcribed_text, min_score=0.0, max_results=1)
    if matches and matches[0]['similarity_score'] >= threshold:
        return matches[0]
    return None

def calculate_absolute_ayah_number(surah, verse):
    """Calculate absolute ayah number (1-6236) from surah and verse"""
    # Complete mapping of verse counts for all 114 surahs
    surah_verse_counts = {
        1: 7, 2: 286, 3: 200, 4: 176, 5: 120, 6: 165, 7: 206, 8: 75, 9: 129, 10: 109,
        11: 123, 12: 111, 13: 43, 14: 52, 15: 99, 16: 128, 17: 111, 18: 110, 19: 98, 20: 135,
        21: 112, 22: 78, 23: 118, 24: 64, 25: 77, 26: 227, 27: 93, 28: 88, 29: 69, 30: 60,
        31: 34, 32: 30, 33: 73, 34: 54, 35: 45, 36: 83, 37: 182, 38: 88, 39: 75, 40: 85,
        41: 54, 42: 53, 43: 89, 44: 59, 45: 37, 46: 35, 47: 38, 48: 29, 49: 18, 50: 45,
        51: 60, 52: 49, 53: 62, 54: 55, 55: 78, 56: 96, 57: 29, 58: 22, 59: 24, 60: 13,
        61: 14, 62: 11, 63: 11, 64: 18, 65: 12, 66: 12, 67: 30, 68: 52, 69: 52, 70: 44,
        71: 28, 72: 28, 73: 20, 74: 56, 75: 40, 76: 31, 77: 50, 78: 40, 79: 46, 80: 42,
        81: 29, 82: 19, 83: 36, 84: 25, 85: 22, 86: 17, 87: 19, 88: 26, 89: 30, 90: 20,
        91: 15, 92: 21, 93: 11, 94: 8, 95: 8, 96: 19, 97: 5, 98: 8, 99: 8, 100: 11,
        101: 11, 102: 8, 103: 3, 104: 9, 105: 5, 106: 4, 107: 7, 108: 3, 109: 6, 110: 3,
        111: 5, 112: 4, 113: 5, 114: 6
    }
    
    absolute_number = 0
    for s in range(1, surah):
        if s in surah_verse_counts:
            absolute_number += surah_verse_counts[s]
    
    absolute_number += verse
    return absolute_number

# Qari audio functionality removed as requested

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/identify-verse', methods=['POST'])
def identify_verse():
    """API endpoint to identify Quranic verse from audio"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Check file extension
        allowed_extensions = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.webm'}
        file_ext = os.path.splitext(audio_file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'Unsupported file format. Allowed: {", ".join(allowed_extensions)}'}), 400
        
        # Check file size (50MB limit)
        audio_file.seek(0, 2)  # Seek to end
        file_size = audio_file.tell()
        audio_file.seek(0)  # Reset to beginning
        
        if file_size > 50 * 1024 * 1024:  # 50MB
            return jsonify({'error': 'File size exceeds 50MB limit'}), 400
        
        # Create temporary file with simple path (no spaces or special chars)
        import uuid
        import tempfile
        file_extension = file_ext.lstrip('.')  # Remove the dot from extension
        
        # Use system temp directory but with simple filename
        temp_dir = tempfile.gettempdir()
        temp_filename = f"audio_{uuid.uuid4().hex[:8]}.{file_extension}"
        temp_filepath = os.path.join(temp_dir, temp_filename)
        logger.info(f"Creating temporary file: {temp_filepath}")
        
        try:
            # Save the uploaded file
            audio_file.save(temp_filepath)
            logger.info(f"Temporary file saved: {temp_filepath}")
            
            # Verify file exists and has content
            if not os.path.exists(temp_filepath):
                raise FileNotFoundError(f"Temporary file not found: {temp_filepath}")
            
            file_size = os.path.getsize(temp_filepath)
            logger.info(f"Temporary file size: {file_size} bytes")
            
            if file_size == 0:
                raise ValueError("Uploaded file is empty")
            
            # Test file accessibility
            try:
                with open(temp_filepath, 'rb') as test_file:
                    test_file.read(1)  # Try to read first byte
                logger.info(f"File accessibility confirmed: {temp_filepath}")
            except Exception as access_error:
                logger.error(f"File access test failed: {str(access_error)}")
                raise access_error
        
            # Try multiple approaches for audio transcription
            transcribed_text = None
            
            # Method 1: Try direct librosa + Whisper (most reliable)
            logger.info("Method 1: Trying librosa + Whisper audio array approach...")
            try:
                import librosa
                import numpy as np
                
                # Load audio as numpy array with librosa
                audio_array, sr = librosa.load(temp_filepath, sr=16000)
                logger.info(f"Audio loaded with librosa: shape={audio_array.shape}, sr={sr}")
                
                # Ensure audio is not empty and has reasonable length
                if len(audio_array) > 0:
                    # Transcribe using audio array
                    result = model.transcribe(audio_array, language='ar', fp16=False)
                    transcribed_text = result['text'].strip()
                    logger.info(f"Method 1 successful: {transcribed_text}")
                else:
                    raise ValueError("Audio array is empty")
                    
            except Exception as method1_error:
                logger.warning(f"Method 1 failed: {str(method1_error)}")
                
                # Method 2: Try with pydub preprocessing
                logger.info("Method 2: Trying pydub preprocessing + Whisper...")
                try:
                    from pydub import AudioSegment
                    
                    # Load and preprocess audio with pydub
                    audio = AudioSegment.from_file(temp_filepath)
                    
                    # Convert to standard format
                    processed_filename = temp_filepath.replace(f'.{file_extension}', '_processed.wav')
                    audio = audio.set_frame_rate(16000).set_channels(1)
                    audio.export(processed_filename, format="wav")
                    logger.info(f"Audio preprocessed with pydub: {processed_filename}")
                    
                    # Try transcription with processed file
                    abs_path = os.path.abspath(processed_filename)
                    result = model.transcribe(abs_path, language='ar', fp16=False)
                    transcribed_text = result['text'].strip()
                    logger.info(f"Method 2 successful: {transcribed_text}")
                    
                    # Clean up processed file
                    if os.path.exists(processed_filename):
                        os.unlink(processed_filename)
                        
                except Exception as method2_error:
                    logger.warning(f"Method 2 failed: {str(method2_error)}")
                    
                    # Method 3: Try direct file transcription
                    logger.info("Method 3: Trying direct file transcription...")
                    try:
                        abs_path = os.path.abspath(temp_filepath)
                        result = model.transcribe(abs_path, language='ar', fp16=False)
                        transcribed_text = result['text'].strip()
                        logger.info(f"Method 3 successful: {transcribed_text}")
                        
                    except Exception as method3_error:
                        logger.error(f"Method 3 failed: {str(method3_error)}")
                        
                        # Method 4: Final fallback with no language specification
                        logger.info("Method 4: Final fallback without language specification...")
                        try:
                            # Load with librosa again but transcribe without language
                            audio_array, sr = librosa.load(temp_filepath, sr=16000)
                            result = model.transcribe(audio_array, fp16=False)
                            transcribed_text = result['text'].strip()
                            logger.info(f"Method 4 successful: {transcribed_text}")
                            
                        except Exception as method4_error:
                            logger.error(f"All transcription methods failed. Last error: {str(method4_error)}")
                            raise Exception("Audio transcription failed with all attempted methods")
            
            # Validate transcription result
            if not transcribed_text or len(transcribed_text.strip()) == 0:
                raise ValueError("Transcription resulted in empty text")
            
            # Clean up temporary file immediately after transcription
            if temp_filepath and os.path.exists(temp_filepath):
                try:
                    os.unlink(temp_filepath)
                    logger.info(f"Temporary file cleaned up: {temp_filepath}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temporary file: {cleanup_error}")
            
            # Find all matching verses with improved algorithm
            logger.info("Finding matching verses...")
            matches = find_matching_verses(transcribed_text, min_score=65.0, max_results=None)
            
            if matches:
                # Classify verse type using new function
                verse_type = classify_verse_type(matches)
                logger.info(f"Classified verse type: {verse_type}")
                
                # Prepare response based on verse type
                if verse_type == "single":
                    # Ayat tunggal biasa: tampilkan teks Arab dan terjemahan tanpa nomor ayat
                    best_match = matches[0]
                    
                    # Get clean text without verse numbers
                    clean_arabic = clean_verse_text(best_match['verse_text'])
                    clean_translation = clean_translation_text(best_match['translation'])
                    
                    # Prepare all matches for "Surat dan Ayat yang Teridentifikasi"
                    matches_data = []
                    for match in matches:
                        match_data = {
                            'surah_number': match['surah_number'],
                            'surah_name': match['surah_name'],
                            'surah_name_arabic': match['surah_name_arabic'],
                            'surah_name_latin': match['surah_name_latin'],
                            'verse_number': match['verse_number'],
                            'similarity_score': match['similarity_score']
                        }
                        matches_data.append(match_data)
                    
                    response_data = {
                        'success': True,
                        'transcription': transcribed_text,
                        'verse_type': 'single',  # Display as single verse type for UI consistency
                        'display_text': {
                            'arabic': clean_arabic,
                            'translation': clean_translation
                        },
                        'matches_count': len(matches_data),
                        'all_matches': matches_data
                    }
                    
                elif verse_type == "identical":
                    # Ayat tunggal kembar identik: tampilkan satu ayat saja tanpa nomor
                    best_match = matches[0]
                    
                    # Find the best matching individual verse from database
                    processed_transcription = preprocess_arabic_text(transcribed_text)
                    best_verse = None
                    best_score = 0
                    
                    # Search through all individual verses to find the best match
                    for surah_num, surah_data in quran_db.items():
                        for verse_data in surah_data['verses']:
                            verse_clean = preprocess_arabic_text(verse_data['arabic_text'])
                            score = fuzz.ratio(processed_transcription, verse_clean)
                            if score > best_score:
                                best_score = score
                                best_verse = {
                                    'surah_number': surah_num,
                                    'surah_name': surah_data['name'],
                                    'surah_name_latin': surah_data['name_latin'],
                                    'verse_number': verse_data['verse_number'],
                                    'arabic_text': verse_data['arabic_text'],
                                    'translation': verse_data['indonesian_translation']
                                }
                    
                    if best_verse:
                        # Get clean text without verse numbers from the best matching verse
                        clean_arabic = clean_verse_text(best_verse['arabic_text'])
                        clean_translation = clean_translation_text(best_verse['translation'])
                        
                        # Find all identical verses in database
                        matches_data = []
                        target_verse_clean = preprocess_arabic_text(best_verse['arabic_text'])
                        
                        # Search through all verses to find identical ones
                        for surah_num, surah_data in quran_db.items():
                            for verse_data in surah_data['verses']:
                                verse_clean = preprocess_arabic_text(verse_data['arabic_text'])
                                if verse_clean == target_verse_clean:
                                    matches_data.append({
                                        'surah_number': surah_num,
                                        'surah_name': surah_data['name'],
                                        'surah_name_arabic': surah_data.get('name_arabic', surah_data['name']),
                                        'surah_name_latin': surah_data['name_latin'],
                                        'verse_number': verse_data['verse_number'],
                                        'similarity_score': 100.0  # Identical verses have 100% similarity
                                    })
                        
                        # Sort by surah number and verse number for consistent ordering
                        matches_data.sort(key=lambda x: (x['surah_number'], x['verse_number']))
                    else:
                        # Fallback to original logic if target verse not found
                        clean_arabic = clean_verse_text(best_match['verse_text'])
                        clean_translation = clean_translation_text(best_match['translation'])
                        matches_data = [{
                            'surah_number': best_match['surah_number'],
                            'surah_name': best_match['surah_name'],
                            'surah_name_arabic': best_match.get('surah_name_arabic', best_match['surah_name']),
                            'surah_name_latin': best_match['surah_name_latin'],
                            'verse_number': best_match['verse_number'],
                            'similarity_score': best_match['similarity_score']
                        }]
                    
                    response_data = {
                        'success': True,
                        'transcription': transcribed_text,
                        'verse_type': verse_type,
                        'display_text': {
                            'arabic': clean_arabic,
                            'translation': clean_translation
                        },
                        'matches_count': len(matches_data),
                        'all_matches': matches_data
                    }
                    
                elif verse_type == "consecutive":
                    # Ayat beruntun: tampilkan semua ayat dengan indikator nomor ayat
                    best_match = matches[0]
                    
                    # Build display text with verse indicators
                    display_arabic = ""
                    display_translation = ""
                    
                    if 'grouped_verses' in best_match:
                        # Handle grouped consecutive verses
                        for i, verse_data in enumerate(best_match['grouped_verses']):
                            verse_num = verse_data['verse_number']
                            clean_arabic = clean_verse_text(verse_data['verse_text'])
                            clean_trans = clean_translation_text(verse_data['translation'])
                            
                            if i > 0:
                                display_arabic += " "
                                display_translation += " "
                            
                            display_arabic += f"({verse_num}) {clean_arabic}"
                            display_translation += f"({verse_num}) {clean_trans}"
                    else:
                        # Single consecutive match
                        start_verse = best_match['verse_number']
                        end_verse = best_match.get('end_verse_number', start_verse)
                        
                        # Get individual verses from database
                        surah_data = quran_db.get(best_match['surah_number'])
                        if surah_data:
                            for verse_num in range(start_verse, end_verse + 1):
                                verse_data = next((v for v in surah_data['verses'] if v['verse_number'] == verse_num), None)
                                if verse_data:
                                    clean_arabic = clean_verse_text(verse_data['arabic_text'])
                                    clean_trans = clean_translation_text(verse_data['indonesian_translation'])
                                    
                                    if verse_num > start_verse:
                                        display_arabic += " "
                                        display_translation += " "
                                    
                                    display_arabic += f"({verse_num}) {clean_arabic}"
                                    display_translation += f"({verse_num}) {clean_trans}"
                    
                    # For consecutive verses, only show surah name in matches
                    matches_data = [{
                        'surah_number': best_match['surah_number'],
                        'surah_name': best_match['surah_name'],
                        'surah_name_arabic': best_match['surah_name_arabic'],
                        'surah_name_latin': best_match['surah_name_latin'],
                        'verse_range': f"{best_match['verse_number']}-{best_match.get('end_verse_number', best_match['verse_number'])}",
                        'similarity_score': best_match['similarity_score']
                    }]
                    
                    response_data = {
                        'success': True,
                        'transcription': transcribed_text,
                        'verse_type': verse_type,
                        'display_text': {
                            'arabic': display_arabic,
                            'translation': display_translation
                        },
                        'matches_count': len(matches_data),
                        'all_matches': matches_data
                    }
                
                else:
                    # Fallback to original format
                    matches_data = []
                    for match in matches:
                        match_data = {
                            'surah_number': match['surah_number'],
                            'surah_name': match['surah_name'],
                            'surah_name_arabic': match['surah_name_arabic'],
                            'surah_name_latin': match['surah_name_latin'],
                            'verse_number': match['verse_number'],
                            'end_verse_number': match.get('end_verse_number'),
                            'verse_reference': match.get('verse_reference', match['surah_name_latin']),
                            'verse_info': match.get('verse_info', ''),
                            'verse_count': match.get('verse_count', 1),
                            'verse_text': match['verse_text'],
                            'translation': match['translation'],
                            'similarity_score': match['similarity_score']
                        }
                        matches_data.append(match_data)
                    
                    response_data = {
                        'success': True,
                        'transcription': transcribed_text,
                        'verse_type': verse_type,
                        'matches_count': len(matches_data),
                        'best_match': matches_data[0] if matches_data else None,
                        'all_matches': matches_data
                    }
                
                logger.info(f"Found {len(response_data.get('all_matches', []))} matches. Verse type: {verse_type}")
                return jsonify(response_data)
            else:
                logger.info("No matches found above threshold")
                return jsonify({
                    'success': False,
                    'transcription': transcribed_text,
                    'message': 'Maaf, ayat tidak dapat diidentifikasi. Coba rekam dengan lebih jelas.'
                })
        
        except Exception as processing_error:
            # Clean up temporary file in case of error
            if 'temp_filepath' in locals() and temp_filepath and os.path.exists(temp_filepath):
                try:
                    os.unlink(temp_filepath)
                    logger.info(f"Temporary file cleaned up after error: {temp_filepath}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temporary file after error: {cleanup_error}")
            raise processing_error
    
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500



# Qari audio proxy endpoint removed as requested

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'surahs_loaded': len(quran_db)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)