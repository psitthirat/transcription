from datetime import timedelta
from pydub import AudioSegment
from pydub.silence import split_on_silence
import numpy as np
import os
import speech_recognition as sr
from docx import Document

# Function to import audio file
def audio_import(file_name, import_folder):
    file_path = f"{import_folder}/{file_name}.wav"
    audio = AudioSegment.from_file(file_path)
    
    print("Audio file imported successfully.")
    audio_length(audio)
    return audio

# Function to convert milliseconds to MM:SS format
def convert_ms_to_timestamp(ms):
    return str(timedelta(milliseconds=ms))[:-7]

# Function to print audio length
def audio_length(audio):
    audio_length = convert_ms_to_timestamp(len(audio))
    return print(f"Audio length = {audio_length} min")

# Function to slice the audio file to the chunks
def audio_slice(audio, chunk_folder, file_name):
    
    print("Start slicing --")
    
    # Define parameters for silence detection
    min_silence_len = 1000  # in milliseconds

    # Segment length for dBFS analysis (e.g., 100 ms)
    segment_length = 100

    # Analyze dBFS levels for each segment
    dbfs_levels = [audio[i:i+segment_length].dBFS for i in range(0, len(audio), segment_length)]

    # Calculate the 20th percentile of dBFS levels
    silence_thresh = np.percentile([level for level in dbfs_levels if level != float('-inf')], 20)

    # Split audio at points of silence
    raw_chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=500  # Optional: keep some silence around the chunks
    )

    # Merge short chunks with adjacent chunks
    chunks = []
    min_chunk_length = 15 * 1000  # 10 seconds in milliseconds
    temp_chunk = None
    
    for chunk in raw_chunks:
        if temp_chunk is None:
            temp_chunk = chunk
        else:
            temp_chunk += chunk
        if len(temp_chunk) >= min_chunk_length:
            chunks.append(temp_chunk)
            temp_chunk = None

    # Check for any remaining short chunk at the end
    if temp_chunk and len(temp_chunk) < min_chunk_length and chunks:
        chunks[-1] += temp_chunk
    elif temp_chunk:
        chunks.append(temp_chunk)
    
    # Initialize start time
    start_time = 0
    
    # Create a directory for chunks
    chunk_dir = f"{chunk_folder}/{file_name}-chunks"
    os.makedirs(chunk_dir, exist_ok=True)
    
    # Export the slices and print their details
    for i, chunk in enumerate(chunks):

        # Format index as a three-digit number
        formatted_index = str(i+1).zfill(3)  # +1 if you want to start numbering from 001 instead of 000

        # Export the chunk to a file
        filename = f"{chunk_dir}/chunk_{formatted_index}.wav"
        chunk.export(filename, format="wav")
    
    return chunks

def save_transcription_to_docx(transcriptions, file_name, export_folder, milestone, last_file_path):
    # Define the new file path
    new_file_path = f"{export_folder}/transcription-{file_name}_(milestone: {milestone}).docx"
    
    # Check and delete the last milestone file
    if last_file_path and os.path.exists(last_file_path):
        os.remove(last_file_path)
        print(f"Deleted old milestone file: {last_file_path}")

    # Save the new milestone file
    doc = Document()
    for line in transcriptions:
        doc.add_paragraph(line)
    if not os.path.exists(f"{export_folder}"):
        os.makedirs(f"{export_folder}")
    doc.save(new_file_path)
    print(f"Transcription saved at milestone {milestone}.")
    return new_file_path

def transcribe(file_name, import_folder, export_folder, milestone=10):
    chunk_folder = os.path.join(import_folder, f"{file_name}-chunks")

    # Check if the chunk folder exists
    if not os.path.exists(chunk_folder):
        print(f"Chunk folder not found: {chunk_folder}")
        return

    print("Start transcribing --")

    # Initialize SpeechRecognition recognizer
    recognizer = sr.Recognizer()

    # Initialize variables for transcription
    start_time = 0
    transcriptions = []
    last_file_path = None

    # List all chunk files in the directory
    chunk_files = sorted([f for f in os.listdir(chunk_folder) if f.endswith('.wav')])
    print(f"Found {len(chunk_files)} chunks to transcribe.")

    # Loop over each chunk file and transcribe
    for i, chunk_file in enumerate(chunk_files):
        chunk_path = os.path.join(chunk_folder, chunk_file)
        print(f"Transcribing chunk {i+1}/{len(chunk_files)}: {chunk_path}")

        # Transcribe the chunk
        try:
            with sr.AudioFile(chunk_path) as source:
                audio_data = recognizer.record(source)
                transcription_text = recognizer.recognize_google(audio_data, language='th-TH')
        except sr.UnknownValueError:
            transcription_text = "ไม่สามารถอ่านไฟล์เสียงได้"
        except sr.RequestError as e:
            transcription_text = f"เกิดข้อผิดพลาดในการส่งข้อมูล: {e}"

        # Get timestamp in MM:SS format
        timestamp = convert_ms_to_timestamp(start_time)

        # Add timestamp and transcription to the list
        transcriptions.append(f"[#{i+1} {timestamp}] {transcription_text}")

        # Save milestone
        if (i + 1) % milestone == 0 or i == len(chunk_files) - 1:
            last_file_path = save_transcription_to_docx(transcriptions, file_name, export_folder, i + 1, last_file_path)

        # Update start time for the next chunk
        # Here you might want to use AudioSegment to determine the duration of each chunk
        chunk_audio = AudioSegment.from_file(chunk_path)
        chunk_duration = len(chunk_audio)
        start_time += chunk_duration

    return transcriptions, last_file_path

# Function to transcribe the audio file
def transcribe_full(file_name, import_folder, export_folder, milestone=10):
    
    # Import the audio
    audio = audio_import(file_name, import_folder)
    
    # Slice the audio
    chunks = audio_slice(audio, import_folder, file_name)
    print(f"Audio file sliced into {len(chunks)} chunks.")

    transcriptions, last_file_path = transcribe(file_name, import_folder, export_folder, milestone)    

    # Save the final transcription
    save_transcription_to_docx(transcriptions, file_name, export_folder, "complete", last_file_path)
    
    return '\n'.join(transcriptions)
            