import gradio as gr
import whisper
import json
import os
import tempfile
from rapidfuzz import fuzz
import re
import logging
import numpy as np

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
                            quran_data[i]['verses'].append({
                                'number': verse_num,
                                'text': surah_info['text'][verse_key],
                                'translation': surah_info['translations']['id']['text'][verse_key]
                            })
    
    logger.info(f"Loaded {len(quran_data)} surahs from database")
    return quran_data

quran_db = load_quran_database()

# Cache for preprocessed text
preprocessed_cache = {}

def clean_verse_text(text):
    """Clean Arabic verse text by removing diacritics and extra characters"""
    if not text:
        return ""
    
    # Remove common diacritics
    diacritics = ['ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ', 'ٰ', 'ٱ']
    cleaned = text
    for diacritic in diacritics:
        cleaned = cleaned.replace(diacritic, '')
    
    # Remove extra whitespace and normalize
    cleaned = ' '.join(cleaned.split())
    return cleaned.strip()

def clean_translation_text(text):
    """Clean translation text"""
    if not text:
        return ""
    
    # Remove HTML tags if any
    cleaned = re.sub(r'<[^>]+>', '', text)
    
    # Remove extra whitespace
    cleaned = ' '.join(cleaned.split())
    return cleaned.strip()

def preprocess_arabic_text(text):
    """Preprocess Arabic text for better matching"""
    if not text:
        return ""
    
    # Clean the text first
    cleaned = clean_verse_text(text)
    
    # Remove common prefixes and suffixes that might vary
    # Remove و (waw) at the beginning
    if cleaned.startswith('و '):
        cleaned = cleaned[2:]
    
    # Remove common particles
    particles = ['ال', 'في', 'من', 'إلى', 'على', 'عن', 'مع', 'بعد', 'قبل']
    words = cleaned.split()
    filtered_words = []
    for word in words:
        # Keep the word as is, just clean it
        filtered_words.append(word.strip())
    
    return ' '.join(filtered_words).strip()

def find_individual_verse_matches(processed_transcription, transcription_length, min_score):
    """Find individual verse matches"""
    matches = []
    best_score = 0
    
    for surah_num, surah_data in quran_db.items():
        for verse in surah_data['verses']:
            verse_number = verse['number']
            
            # Use cache for preprocessed text
            cache_key = f"{surah_num}_{verse_number}"
            if cache_key in preprocessed_cache:
                processed_verse = preprocessed_cache[cache_key]
            else:
                processed_verse = preprocess_arabic_text(verse['text'])
                preprocessed_cache[cache_key] = processed_verse
            
            if not processed_verse:
                continue
            
            # Calculate various similarity scores
            ratio_score = fuzz.ratio(processed_transcription, processed_verse)
            partial_score = fuzz.partial_ratio(processed_transcription, processed_verse)
            token_sort_score = fuzz.token_sort_ratio(processed_transcription, processed_verse)
            token_set_score = fuzz.token_set_ratio(processed_transcription, processed_verse)
            
            # Calculate length ratio for penalty
            verse_length = len(processed_verse.split())
            length_ratio = min(transcription_length, verse_length) / max(transcription_length, verse_length)
            
            # Enhanced scoring algorithm based on transcription length
            if transcription_length <= 5:  # Short transcription
                base_score = max(partial_score, token_set_score)
                length_bonus = length_ratio * 10
            elif transcription_length <= 15:  # Medium transcription
                base_score = (ratio_score * 0.3 + partial_score * 0.3 + 
                            token_sort_score * 0.2 + token_set_score * 0.2)
                length_bonus = length_ratio * 15
            else:  # Long transcription
                base_score = (ratio_score * 0.4 + partial_score * 0.2 + 
                            token_sort_score * 0.2 + token_set_score * 0.2)
                length_bonus = length_ratio * 20
            
            final_score = base_score + length_bonus
            
            if final_score >= min_score:
                matches.append({
                    'surah': surah_num,
                    'verse': verse_number,
                    'score': final_score,
                    'arabic_text': verse['text'],
                    'translation': verse['translation'],
                    'surah_name': surah_data['name_latin'],
                    'surah_arabic': surah_data['name_arabic'],
                    'match_type': 'individual',
                    'scores': {
                        'ratio': ratio_score,
                        'partial': partial_score,
                        'token_sort': token_sort_score,
                        'token_set': token_set_score,
                        'length_ratio': length_ratio,
                        'final': final_score
                    }
                })
                
                # Update best score and check for early termination
                if final_score > best_score:
                    best_score = final_score
                    
                # Early termination for very high scores
                if final_score >= 95.0:
                    return matches
    
    return matches

def find_consecutive_verse_matches(processed_transcription, transcription_length, min_score):
    """Find consecutive verse matches"""
    matches = []
    
    for surah_num, surah_data in quran_db.items():
        verses = surah_data['verses']
        
        # Try different consecutive verse combinations (2-5 verses)
        for start_idx in range(len(verses)):
            for num_verses in range(2, min(6, len(verses) - start_idx + 1)):
                end_idx = start_idx + num_verses
                consecutive_verses = verses[start_idx:end_idx]
                
                # Combine Arabic text
                combined_arabic = ' '.join([v['text'] for v in consecutive_verses])
                processed_combined = preprocess_arabic_text(combined_arabic)
                
                if not processed_combined:
                    continue
                
                # Calculate similarity scores
                ratio_score = fuzz.ratio(processed_transcription, processed_combined)
                partial_score = fuzz.partial_ratio(processed_transcription, processed_combined)
                token_sort_score = fuzz.token_sort_ratio(processed_transcription, processed_combined)
                token_set_score = fuzz.token_set_ratio(processed_transcription, processed_combined)
                
                # Weighted average for consecutive verses
                final_score = (ratio_score * 0.3 + partial_score * 0.3 + 
                             token_sort_score * 0.2 + token_set_score * 0.2)
                
                if final_score >= min_score:
                    # Combine translations
                    combined_translation = ' '.join([v['translation'] for v in consecutive_verses])
                    
                    matches.append({
                        'surah': surah_num,
                        'verse_start': consecutive_verses[0]['number'],
                        'verse_end': consecutive_verses[-1]['number'],
                        'score': final_score,
                        'arabic_text': combined_arabic,
                        'translation': combined_translation,
                        'surah_name': surah_data['name_latin'],
                        'surah_arabic': surah_data['name_arabic'],
                        'match_type': 'consecutive',
                        'verse_count': num_verses,
                        'scores': {
                            'ratio': ratio_score,
                            'partial': partial_score,
                            'token_sort': token_sort_score,
                            'token_set': token_set_score,
                            'final': final_score
                        }
                    })
    
    return matches

def find_matching_verses(transcribed_text, min_score=50.0, max_results=10):
    """Find matching verses from transcribed text"""
    if not transcribed_text or not transcribed_text.strip():
        return []
    
    processed_transcription = preprocess_arabic_text(transcribed_text)
    transcription_length = len(processed_transcription.split())
    
    if not processed_transcription:
        return []
    
    # Find individual verse matches
    individual_matches = find_individual_verse_matches(processed_transcription, transcription_length, min_score)
    
    # Skip consecutive matches if we have high-scoring individual matches
    consecutive_matches = []
    if not individual_matches or max([m['score'] for m in individual_matches]) < 85.0:
        consecutive_matches = find_consecutive_verse_matches(processed_transcription, transcription_length, min_score)
    
    # Combine and sort all matches
    all_matches = individual_matches + consecutive_matches
    all_matches.sort(key=lambda x: x['score'], reverse=True)
    
    return all_matches[:max_results]

def process_audio(audio_file):
    """Process audio file and return verse identification results"""
    try:
        if audio_file is None:
            return "❌ Tidak ada file audio yang diunggah.", "", "", ""
        
        # Transcribe audio using Whisper
        result = model.transcribe(audio_file, language='ar')
        transcribed_text = result['text'].strip()
        
        if not transcribed_text:
            return "❌ Tidak dapat mendeteksi teks dari audio.", "", "", ""
        
        # Find matching verses
        matches = find_matching_verses(transcribed_text, min_score=60.0, max_results=5)
        
        if not matches:
            return f"❌ Tidak ditemukan ayat yang cocok.\n\n📝 **Teks yang terdeteksi:**\n{transcribed_text}", "", "", ""
        
        # Format results
        best_match = matches[0]
        
        # Create result text
        if best_match['match_type'] == 'individual':
            verse_info = f"QS. {best_match['surah_name']} ({best_match['surah_arabic']}) Ayat {best_match['verse']}"
        else:
            verse_info = f"QS. {best_match['surah_name']} ({best_match['surah_arabic']}) Ayat {best_match['verse_start']}-{best_match['verse_end']}"
        
        result_text = f"✅ **Ayat berhasil diidentifikasi!**\n\n📍 **{verse_info}**\n\n🎯 **Tingkat Kemiripan:** {best_match['score']:.1f}%"
        
        arabic_text = best_match['arabic_text']
        translation = best_match['translation']
        transcription = f"📝 **Teks yang terdeteksi:**\n{transcribed_text}"
        
        return result_text, arabic_text, translation, transcription
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return f"❌ Terjadi kesalahan saat memproses audio: {str(e)}", "", "", ""

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="AyatFinder - Identifikasi Ayat Al-Quran", theme=gr.themes.Soft()) as demo:
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #2E86AB; margin-bottom: 10px;">🕌 AyatFinder</h1>
            <p style="font-size: 18px; color: #666; margin-bottom: 20px;">Identifikasi Ayat Al-Quran dari Audio dengan Teknologi AI</p>
            <p style="color: #888;">Unggah file audio bacaan ayat Al-Quran dan sistem akan mengidentifikasi ayat yang dibacakan</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3>📤 Upload Audio</h3>")
                audio_input = gr.Audio(
                    label="File Audio (MP3, WAV, M4A)",
                    type="filepath",
                    sources=["upload", "microphone"]
                )
                
                process_btn = gr.Button(
                    "🔍 Identifikasi Ayat", 
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                gr.HTML("<h3>📋 Hasil Identifikasi</h3>")
                result_output = gr.Markdown(label="Status")
                
                with gr.Row():
                    with gr.Column():
                        arabic_output = gr.Textbox(
                            label="📖 Teks Arab",
                            lines=3,
                            rtl=True,
                            interactive=False
                        )
                    
                    with gr.Column():
                        translation_output = gr.Textbox(
                            label="🇮🇩 Terjemahan Indonesia",
                            lines=3,
                            interactive=False
                        )
                
                transcription_output = gr.Textbox(
                    label="🎤 Transkripsi Audio",
                    lines=2,
                    interactive=False
                )
        
        # Event handlers
        process_btn.click(
            fn=process_audio,
            inputs=[audio_input],
            outputs=[result_output, arabic_output, translation_output, transcription_output]
        )
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 30px; border-top: 1px solid #eee;">
            <p style="color: #888; font-size: 14px;">Dikembangkan dengan ❤️ menggunakan Whisper AI dan RapidFuzz</p>
            <p style="color: #888; font-size: 12px;">Hanya menampilkan hasil dengan tingkat kemiripan ≥ 60%</p>
        </div>
        """)
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()