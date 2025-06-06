import streamlit as st
import json
import os
from datetime import timedelta
from PIL import Image
from typing import List, Dict, Optional, Any, Tuple
import re
import pandas as pd
import numpy as np # For numerical operations
import jiwer # For WER calculation
from unidecode import unidecode
import gc
from pathlib import Path

# File upload and path resolution utilities
import zipfile
import tempfile
import shutil

def extract_uploaded_dataset(uploaded_file) -> Optional[str]:
    """Extract uploaded zip file and return the extraction directory."""
    if uploaded_file is None:
        return None
    
    try:
        # Create a temporary directory for this session
        if 'temp_extract_dir' not in st.session_state:
            st.session_state.temp_extract_dir = tempfile.mkdtemp(prefix="streamlit_dataset_")
        
        extract_dir = st.session_state.temp_extract_dir
        
        # Clear previous extraction
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        os.makedirs(extract_dir, exist_ok=True)
        
        # Extract the zip file
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        return extract_dir
    except Exception as e:
        st.error(f"Failed to extract uploaded file: {e}")
        return None

def find_json_file(base_dir: str) -> Optional[str]:
    """Find the main JSON dataset file in the directory."""
    if not base_dir or not os.path.exists(base_dir):
        return None
    
    # Look for JSON files
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                try:
                    # Validate it's a dataset JSON
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                            sample = data[0]
                            if any(key in sample for key in ['task_segment_id', 'participant_identifier', 'task_label']):
                                return json_path
                except:
                    continue
    return None

def resolve_media_paths(dataset: List[Dict], json_dir: str) -> List[Dict]:
    """Update file paths in dataset to point to gesture_grids/ and videos/ folders alongside the JSON."""
    if not json_dir or not os.path.exists(json_dir):
        return dataset
    
    # Define the expected folders
    gesture_grids_dir = os.path.join(json_dir, "gesture_grids")
    videos_dir = os.path.join(json_dir, "videos")
    
    # Create a mapping of gesture grid filenames to their full paths (including participant subfolders)
    gesture_grid_map = {}
    if os.path.exists(gesture_grids_dir):
        for participant_folder in os.listdir(gesture_grids_dir):
            participant_path = os.path.join(gesture_grids_dir, participant_folder)
            if os.path.isdir(participant_path):
                for filename in os.listdir(participant_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        full_path = os.path.join(participant_path, filename)
                        gesture_grid_map[filename] = full_path
    
    # Update paths in dataset
    updated_dataset = []
    for entry in dataset:
        updated_entry = entry.copy()
        
        # Update video paths to point to videos/ folder (flat structure)
        for video_field in ['video_filepath', 'video_filepath_orig', 'video_filepath_analysis']:
            if video_field in updated_entry and updated_entry[video_field]:
                original_path = updated_entry[video_field]
                filename = os.path.basename(original_path)
                new_path = os.path.join(videos_dir, filename)
                updated_entry[video_field] = new_path
        
        # Update gesture grid paths to point to gesture_grids/participant/ folders (nested structure)
        for grid_field in ['gesture_grid_filepaths', 'gesture_motion_sequence_grid_image_paths']:
            if grid_field in updated_entry and updated_entry[grid_field]:
                if isinstance(updated_entry[grid_field], list):
                    updated_paths = []
                    for path in updated_entry[grid_field]:
                        filename = os.path.basename(path)
                        # Use the mapping to find the correct participant subfolder
                        if filename in gesture_grid_map:
                            updated_paths.append(gesture_grid_map[filename])
                        else:
                            # Fallback: try to construct path using participant identifier
                            participant_id = updated_entry.get('participant_identifier', 'unknown')
                            fallback_path = os.path.join(gesture_grids_dir, str(participant_id), filename)
                            updated_paths.append(fallback_path)
                    updated_entry[grid_field] = updated_paths
        
        updated_dataset.append(updated_entry)
    
    return updated_dataset

# --- Configuration & Styling ---
def load_css(file_path: str):
    """Loads a CSS file and injects it into the Streamlit app."""
    try:
        with open(file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found at {file_path}. Using default Streamlit styling.")

def setup_page():
    """Configure page settings and load custom CSS."""
    st.set_page_config(
        layout="wide",
        page_title="Multimodal Aphasia Analysis Viewer",
        page_icon="🗣️"
    )
    if not os.path.exists("style.css"):
        st.warning("style.css not found. Creating a dummy style.css with basic styles. Please customize it for a better look.")
        try:
            with open("style.css", "w") as f:
                f.write("""
body { font-family: sans-serif; }
.page-header { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
.page-title { color: #333; text-align: center; }
.page-subtitle { color: #555; text-align: center; }
.sidebar-header { color: #007bff; margin-top: 0; }
.sidebar-subheader { color: #333; margin-top: 1rem; font-size: 1.1em; }
.section-header { color: #28a745; border-bottom: 2px solid #28a745; padding-bottom: 0.3em; margin-top: 1.5em; margin-bottom: 1em; }
.content-subheader { color: #17a2b8; margin-top: 1em; margin-bottom: 0.5em; }
.content-subheader-small { font-size: 0.9em; color: #6c757d; margin-top: 0.5em; }
.metadata-card { background-color: #e9ecef; padding: 10px; border-radius: 5px; margin-bottom: 15px; display: flex; flex-wrap: wrap; gap: 15px; }
.metadata-item { font-size: 0.9em; }
.transcript-text { background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 10px; border-radius: 5px; white-space: pre-wrap; word-wrap: break-word; font-family: monospace; }
.short-text { max-height: 200px; overflow-y: auto; }
.augmentation-marker { background-color: #fff3cd; color: #856404; padding: 0.1em 0.3em; border-radius: 0.2em; font-weight: bold; }
.main-content-placeholder { text-align: center; font-size: 1.2em; color: #6c757d; margin-top: 3rem; }
                """)
        except Exception as e:
            st.error(f"Could not create dummy style.css: {e}")
    load_css("style.css")


def display_app_header():
    """Displays the main title and subtitle of the application."""
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Multimodal Aphasia Analysis Viewer</h1>
        <p class="page-subtitle">Review Qwen-VL analysis of video segments with WER comparisons</p>
    </div>
    """, unsafe_allow_html=True)

# --- Utility Functions ---
def format_time(seconds: Optional[float]) -> str:
    """Format seconds into HH:MM:SS format."""
    if pd.isna(seconds) or not isinstance(seconds, (int, float)) or seconds < 0:
        return "0:00:00"
    try:
        td = timedelta(seconds=float(seconds))
        hours, remainder = divmod(td.total_seconds(), 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{int(hours):01}:{int(minutes):02}:{int(secs):02}"
    except (ValueError, TypeError):
        return "0:00:00"

def escape_html(text: Optional[Any]) -> str:
    """Safely escape HTML special characters in a string."""
    if pd.isna(text) or text is None:
        return ""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def get_entry_value(entry: Dict, base_key: str, default_val: Any = None) -> Any:
    """
    Safely retrieves a value from the entry dictionary, checking for common suffixes
    and direct key match. Updated for optimized pipeline field names.
    """
    if not entry or not isinstance(entry, dict):
        return default_val
    
    # Field mapping for optimized pipeline
    field_mappings = {
        'concatenated_filtered_cha_transcript': ['cha_transcript', 'concatenated_filtered_cha_transcript'],
        'speaker_labeled_asr_transcript': ['asr_full_text', 'speaker_labeled_asr_transcript'],
        'gesture_motion_sequence_grid_image_paths': ['gesture_grid_filepaths', 'gesture_motion_sequence_grid_image_paths'],
        'video_filepath_orig': ['video_filepath', 'video_filepath_orig'],
        'video_filepath_analysis': ['video_filepath', 'video_filepath_analysis'],
        'task_start_time_sec_orig': ['task_start_time_sec', 'task_start_time_sec_orig'],
        'task_start_time_sec_analysis': ['task_start_time_sec', 'task_start_time_sec_analysis'],
        'task_end_time_sec_orig': ['task_end_time_sec', 'task_end_time_sec_orig'],
        'task_end_time_sec_analysis': ['task_end_time_sec', 'task_end_time_sec_analysis']
    }
    
    # Check if we have a specific mapping for this key
    if base_key in field_mappings:
        for mapped_key in field_mappings[base_key]:
            if mapped_key in entry and pd.notna(entry[mapped_key]):
                return entry[mapped_key]
    
    # Original logic for other keys
    keys_to_try = [
        f"{base_key}_analysis_output",
        f"{base_key}",
        f"{base_key}_original_input"
    ]
    for key in keys_to_try:
        if key in entry and pd.notna(entry[key]):
            return entry[key]
    if base_key in entry and pd.notna(entry[base_key]):
        return entry[base_key]
    return default_val

def _display_task_description_for_entry(entry: Dict):
    """Displays the task description at the top of an entry's detailed view."""
    task_description_display = escape_html(get_entry_value(entry, 'task_description', "No description provided."))
    if task_description_display and task_description_display != "No description provided.":
        st.markdown(f"**Task Description:** {task_description_display}")
        st.markdown("---")

def display_metadata_card(entry: Dict):
    """Displays a card with metadata for the selected entry.
       This is now intended to be called under the video.
    """
    entry_id_display = escape_html(get_entry_value(entry, 'task_segment_id', "N/A"))
    task_label_display = escape_html(get_entry_value(entry, 'task_label', "Unknown Task"))
    participant_display = escape_html(get_entry_value(entry, 'participant_identifier', "N/A"))

    start_time_val = entry.get('task_start_time_sec_analysis', entry.get('task_start_time_sec_orig', 0.0))
    end_time_val = entry.get('task_end_time_sec_analysis', entry.get('task_end_time_sec_orig', 0.0))
    duration_val = get_entry_value(entry, 'clip_duration_sec')

    if duration_val is None or pd.isna(duration_val):
        if isinstance(start_time_val, (int, float)) and isinstance(end_time_val, (int, float)):
            duration_val = end_time_val - start_time_val if end_time_val >= start_time_val else 0.0
        else:
            duration_val = 0.0

    st.markdown(f"""
    <div class="metadata-card" style="margin-top: 15px;"> <span class="metadata-item"><b>Participant:</b> {participant_display}</span>
        <span class="metadata-item"><b>Task Segment ID:</b> {entry_id_display}</span>
        <span class="metadata-item"><b>Task Label:</b> {task_label_display}</span>
        <span class="metadata-item"><b>Segment Time (Analysis/Orig):</b> {format_time(start_time_val)} - {format_time(end_time_val)}</span>
        <span class="metadata-item"><b>Clip Duration:</b> {format_time(duration_val)}</span>
    </div>
    """, unsafe_allow_html=True)

    error_message = get_entry_value(entry, 'error_message')
    if error_message and str(error_message).strip() and "OK" not in str(error_message).upper() and "NONE" not in str(error_message).upper():
        st.error(f"**Processing Error for this entry:** {escape_html(error_message)}")

    # Removed pre-calculated WER display from here as its context section was removed.
    # If you have a different generic pre-calculated WER for the entry, you could display it.
    # precalculated_wer = get_entry_value(entry, 'some_other_precalculated_wer_key')
    # if precalculated_wer is not None and pd.notna(precalculated_wer):
    #     st.info(f"**Pre-calculated Metric:** {precalculated_wer:.4f}")


# --- Core Logic for WER and LLM Response Parsing ---
def preprocess_for_wer(text: Optional[str]) -> str:
    """
    Preprocesses text for WER calculation for a pure word-to-word comparison:
    - Transliterates Unicode text (including phonetic characters) to a basic ASCII representation.
    - Converts to string and lowercases.
    - Removes speaker labels like "Participant:".
    - Replaces hyphens and underscores with spaces to treat them as word separators.
    - Removes all characters that are not letters (a-z), numbers (0-9), or whitespace.
    - Normalizes whitespace to single spaces and removes leading/trailing spaces.
    """
    if pd.isna(text) or text is None:
        return ""

    text_as_str = str(text)
    try:
        text_str = unidecode(text_as_str)
    except Exception as e:
        # print(f"Unidecode failed for: {text_as_str[:50]}. Error: {e}") # Optional: for debugging
        text_str = text_as_str # Fallback to original string if unidecode fails
    
    text_str = text_str.lower()
    text_str = re.sub(r'^\s*\w+\s*:\s*', '', text_str, flags=re.IGNORECASE) # Speaker labels
    text_str = text_str.replace('-', ' ')
    text_str = text_str.replace('_', ' ')
    text_str = re.sub(r'[^a-z0-9\s]', '', text_str) # Keep only a-z, 0-9, space
    text_str = " ".join(text_str.split()) # Normalize whitespace
    return text_str

def calculate_wer(reference: Optional[str], hypothesis: Optional[str]) -> Optional[float]:
    """Calculates WER between reference and hypothesis strings."""
    if reference is None or hypothesis is None: return None
    try:
        ref_processed = preprocess_for_wer(reference)
        hyp_processed = preprocess_for_wer(hypothesis)
        if not ref_processed and not hyp_processed: return 0.0
        if not ref_processed: return 1.0 # All insertions if ref is empty
        if not hyp_processed: return 1.0 # All deletions if hyp is empty
        return jiwer.wer(ref_processed, hyp_processed)
    except Exception as e:
        print(f"Error calculating WER for ref='{str(reference)[:50]}...', hyp='{str(hypothesis)[:50]}...': {e}")
        return None

def calculate_participant_overall_wer(participant_tasks: List[Dict]) -> Optional[float]:
    """
    Calculates the overall WER for a participant by concatenating all their
    reference and hypothesis transcripts from their tasks.
    """
    if not participant_tasks:
        return None

    all_references_segments = []
    all_hypotheses_segments = []

    # Tasks should ideally be processed in a consistent order (e.g., chronological)
    # The participant_tasks from st.session_state.participant_data_map are already sorted by time.

    for task_entry in participant_tasks:
        ref_segment = get_entry_value(task_entry, 'concatenated_filtered_cha_transcript')
        hyp_segment = get_entry_value(task_entry, 'speaker_labeled_asr_transcript')

        # Only include segments where a reference transcript exists for a meaningful WER.
        if ref_segment is not None:
            all_references_segments.append(str(ref_segment))
            # If hypothesis is missing for an existing reference, use an empty string for jiwer to count deletions.
            all_hypotheses_segments.append(str(hyp_segment) if hyp_segment is not None else "")
        # If ref_segment is None, this task/segment is skipped for overall WER.

    if not all_references_segments: # No reference data found for this participant across all tasks
        return None

    # Concatenate all segments. A space is used as a separator.
    # preprocess_for_wer (called within calculate_wer) will handle final normalization.
    full_reference_text = " ".join(all_references_segments)
    full_hypothesis_text = " ".join(all_hypotheses_segments)

    return calculate_wer(full_reference_text, full_hypothesis_text)

def parse_model_response(raw_response_str: Optional[str]) -> Dict[str, str]:
    """Parses the raw LLM response string to extract augmented transcript and explanation."""
    augmented_transcript = "Not available in raw response."
    explanation = "Not available in raw response."

    if not raw_response_str or not isinstance(raw_response_str, str):
        return {"augmented_transcript": augmented_transcript, "explanation": explanation}

    try:
        start_aug_tag = "AUGMENTED_TRANSCRIPT_START"
        end_aug_tag = "AUGMENTED_TRANSCRIPT_END"
        aug_match = re.search(f"{re.escape(start_aug_tag)}(.*?){re.escape(end_aug_tag)}", raw_response_str, re.DOTALL)
        if aug_match:
            augmented_transcript = aug_match.group(1).strip()
    except Exception as e:
        print(f"Error parsing augmented transcript: {e}")

    try:
        start_exp_tag = "EXPLANATION_START"
        end_exp_tag = "EXPLANATION_END"
        exp_match = re.search(f"{re.escape(start_exp_tag)}(.*?){re.escape(end_exp_tag)}", raw_response_str, re.DOTALL)
        if exp_match:
            explanation = exp_match.group(1).strip()
    except Exception as e:
        print(f"Error parsing explanation: {e}")

    return {"augmented_transcript": augmented_transcript, "explanation": explanation}


# --- Data Loading & Preprocessing ---


def initialize_session_state_vars():
    """Initialize session state variables for navigation."""
    if 'participant_data_map' not in st.session_state:
        st.session_state.participant_data_map = {}
    if 'all_participants_ids' not in st.session_state:
        st.session_state.all_participants_ids = []
    if 'selected_participant_idx' not in st.session_state:
        st.session_state.selected_participant_idx = 0
    if 'selected_task_idx' not in st.session_state:
        st.session_state.selected_task_idx = 0
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'current_file_name' not in st.session_state:
        st.session_state.current_file_name = None




def prepare_participant_task_data(dataset: List[Dict]):
    """Populates session state with participant and task mappings, sorted by time."""
    if not dataset:
        st.session_state.participant_data_map = {}
        st.session_state.all_participants_ids = []
        st.session_state.selected_participant_idx = 0
        st.session_state.selected_task_idx = 0
        return

    participant_map = {}
    participant_ids_list = sorted(list(set(
        entry.get('participant_identifier') for entry in dataset if entry.get('participant_identifier')
    )))
    st.session_state.all_participants_ids = participant_ids_list

    for p_id in st.session_state.all_participants_ids:
        participant_tasks = sorted(
            [entry for entry in dataset if entry.get('participant_identifier') == p_id],
            key=lambda x: x.get('task_start_time_sec_analysis', x.get('task_start_time_sec_orig', 0))
        )
        participant_map[p_id] = participant_tasks
    st.session_state.participant_data_map = participant_map

    if st.session_state.selected_participant_idx >= len(st.session_state.all_participants_ids):
        st.session_state.selected_participant_idx = 0
        st.session_state.selected_task_idx = 0


# --- Sidebar Navigation for Detailed View ---
def display_detailed_view_sidebar_and_get_selection() -> Optional[Dict]:
    """Manages sidebar navigation for participants and tasks, returns selected entry."""
    st.sidebar.markdown('<h2 class="sidebar-header">Detailed Navigation</h2>', unsafe_allow_html=True)

    if not st.session_state.all_participants_ids:
        st.sidebar.warning("No participant data loaded or no participants found.")
        return None

    if st.session_state.selected_participant_idx >= len(st.session_state.all_participants_ids):
        st.session_state.selected_participant_idx = 0
        st.session_state.selected_task_idx = 0

    current_participant_id_by_idx = st.session_state.all_participants_ids[st.session_state.selected_participant_idx]
    selected_participant_id_from_dropdown = st.sidebar.selectbox(
        "Select Participant",
        options=st.session_state.all_participants_ids,
        index=st.session_state.selected_participant_idx,
        key=f"participant_select_{st.session_state.selected_participant_idx}"
    )

    if selected_participant_id_from_dropdown != current_participant_id_by_idx:
        st.session_state.selected_participant_idx = st.session_state.all_participants_ids.index(selected_participant_id_from_dropdown)
        st.session_state.selected_task_idx = 0
        # st.rerun() # Implicit rerun on selectbox change

    st.sidebar.markdown("Navigate Participants:")
    col_p_prev, col_p_next = st.sidebar.columns(2)
    if col_p_prev.button("⬅️ Prev. Participant", use_container_width=True, key="prev_p"):
        if st.session_state.selected_participant_idx > 0:
            st.session_state.selected_participant_idx -= 1
            st.session_state.selected_task_idx = 0
            st.rerun()
    if col_p_next.button("Next Participant ➡️", use_container_width=True, key="next_p"):
        if st.session_state.selected_participant_idx < len(st.session_state.all_participants_ids) - 1:
            st.session_state.selected_participant_idx += 1
            st.session_state.selected_task_idx = 0
            st.rerun()
    st.sidebar.markdown("---")

    actual_current_participant_id = st.session_state.all_participants_ids[st.session_state.selected_participant_idx]
    tasks_for_current_participant = st.session_state.participant_data_map.get(actual_current_participant_id, [])

    if not tasks_for_current_participant:
        st.sidebar.warning(f"No tasks found for participant {actual_current_participant_id}.")
        return None

    if st.session_state.selected_task_idx >= len(tasks_for_current_participant):
        st.session_state.selected_task_idx = 0

    task_options_display = [
        f"{idx+1}. {entry.get('task_label', 'Task')} (ID: {entry.get('task_segment_id', 'N/A')})"
        for idx, entry in enumerate(tasks_for_current_participant)
    ]
    current_task_display_name_by_idx = task_options_display[st.session_state.selected_task_idx]
    selected_task_display_name_from_dropdown = st.sidebar.selectbox(
        f"Select Task for {actual_current_participant_id}",
        options=task_options_display,
        index=st.session_state.selected_task_idx,
        key=f"task_select_{actual_current_participant_id}_{st.session_state.selected_task_idx}"
    )

    if selected_task_display_name_from_dropdown != current_task_display_name_by_idx:
        st.session_state.selected_task_idx = task_options_display.index(selected_task_display_name_from_dropdown)
        # st.rerun() # Implicit rerun

    st.sidebar.markdown(f"Navigate Tasks for {actual_current_participant_id}:")
    col_t_prev, col_t_next = st.sidebar.columns(2)
    if col_t_prev.button("⬅️ Prev. Task", use_container_width=True, key="prev_t"):
        if st.session_state.selected_task_idx > 0:
            st.session_state.selected_task_idx -= 1
            st.rerun()
    if col_t_next.button("Next Task ➡️", use_container_width=True, key="next_t"):
        if st.session_state.selected_task_idx < len(tasks_for_current_participant) - 1:
            st.session_state.selected_task_idx += 1
            st.rerun()

    if tasks_for_current_participant and st.session_state.selected_task_idx < len(tasks_for_current_participant):
        return tasks_for_current_participant[st.session_state.selected_task_idx]
    elif tasks_for_current_participant: # Fallback to first task if index became invalid
        st.session_state.selected_task_idx = 0
        return tasks_for_current_participant[0]
    return None


# --- Display Components (for Detailed View) ---
def display_media_and_frames(entry: Dict):
    """Displays video segment, then metadata card below it, and associated keyframes to the side."""
    st.markdown('<h2 class="section-header">Media & Keyframes</h2>', unsafe_allow_html=True)

    video_path = entry.get('video_filepath_orig', entry.get('video_filepath_analysis'))
    start_time = entry.get('task_start_time_sec_orig', entry.get('task_start_time_sec_analysis', 0))

    col_video_and_meta, col_keyframes_display = st.columns([2, 1])

    with col_video_and_meta:
        st.markdown('<h3 class="content-subheader">Video Segment</h3>', unsafe_allow_html=True)
        if video_path and isinstance(video_path, str) and os.path.exists(video_path):
            try:
                with open(video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                st.video(video_bytes, start_time=int(float(start_time)))
            except Exception as e:
                st.error(f"Error loading video: {escape_html(str(e))}")
        elif video_path:
            st.warning(f"Video file not found: {escape_html(video_path)}")
        else:
            st.info("Video path not available (checked 'video_filepath_orig' and 'video_filepath_analysis').")
        display_metadata_card(entry) # Call metadata card under the video

    with col_keyframes_display:
        st.markdown('<h3 class="content-subheader">Gesture Grid Images</h3>', unsafe_allow_html=True)
        
        # Try LLM selected frames first (if inference has been run)
        individual_frames_str = get_entry_value(entry, 'selected_llm_grid_paths', "[]")
        individual_frames = []
        try:
            loaded_frames = json.loads(individual_frames_str) if isinstance(individual_frames_str, str) else individual_frames_str
            if isinstance(loaded_frames, list): individual_frames = loaded_frames
        except json.JSONDecodeError:
            pass  # Fall back to gesture grid filepaths
        
        # Fall back to gesture grid filepaths from optimized pipeline
        frame_grid_list_str = get_entry_value(entry, 'gesture_motion_sequence_grid_image_paths', "[]")
        frame_grid_list = []
        try:
            loaded_grid_list = json.loads(frame_grid_list_str) if isinstance(frame_grid_list_str, str) else frame_grid_list_str
            if isinstance(loaded_grid_list, list): frame_grid_list = loaded_grid_list
        except json.JSONDecodeError:
            pass  # Continue with empty list

        def fix_image_path(fp):
            """Fix image path by prepending output_dataset if needed"""
            if isinstance(fp, str):
                # If path doesn't exist as-is, try prepending output_dataset/
                if not os.path.exists(fp) and not fp.startswith('output_dataset/'):
                    alt_path = os.path.join('output_dataset', fp)
                    if os.path.exists(alt_path):
                        return alt_path
                return fp
            return fp

        # Fix paths for LLM frames
        fixed_llm_frames = [fix_image_path(fp) for fp in individual_frames]
        valid_llm_frames = [fp for fp in fixed_llm_frames if isinstance(fp, str) and os.path.exists(fp)]
        
        if valid_llm_frames:
            # Display frames vertically with big red labels on top
            for i, frame_path in enumerate(valid_llm_frames):
                try:
                    st.markdown(f'<h4 style="color: red; font-size: 24px; font-weight: bold; margin-bottom: 5px;">LLM Frame {i+1}</h4>', unsafe_allow_html=True)
                    st.image(Image.open(frame_path), use_container_width=True)
                except Exception as e: 
                    st.warning(f"Error loading LLM Frame {i+1}") # Shorter error
        elif frame_grid_list:
            st.markdown('<h4 class="content-subheader-small">Gesture Sequence Grids (Input)</h4>', unsafe_allow_html=True)
            # Fix paths for gesture grid frames
            fixed_grid_frames = [fix_image_path(fp) for fp in frame_grid_list]
            valid_motion_frames = [fp for fp in fixed_grid_frames if isinstance(fp, str) and os.path.exists(fp)]
            
            if valid_motion_frames:
                # Display gesture grids vertically with big red labels on top
                for i, frame_path in enumerate(valid_motion_frames):
                    try: 
                        st.markdown(f'<h4 style="color: red; font-size: 24px; font-weight: bold; margin-bottom: 5px;">Motion Grid {i+1}</h4>', unsafe_allow_html=True)
                        st.image(Image.open(frame_path), use_container_width=True)
                    except Exception as e: 
                        st.warning(f"Error loading Motion Grid {i+1}") # Shorter error
            else: 
                st.info("Gesture motion grids provided, but files not found/empty.")
                # Debug info to help diagnose the issue
                if frame_grid_list:
                    st.write(f"Debug: Checked {len(frame_grid_list)} grid paths:")
                    for i, fp in enumerate(frame_grid_list[:3]):  # Show first 3 paths
                        fixed_fp = fix_image_path(fp)
                        exists = os.path.exists(fixed_fp) if fixed_fp else False
                        st.write(f"  {i+1}. Original: `{fp}` → Fixed: `{fixed_fp}` → Exists: {exists}")
        else: st.info("No keyframes found (checked 'selected_llm_grid_paths', 'gesture_motion_sequence_grid_image_paths').")


def display_llm_analysis(entry: Dict):
    """Displays analysis outputs in tabs - either LLM inference results or optimized pipeline augmented transcript."""
    st.markdown('<h2 class="section-header">Analysis Output</h2>', unsafe_allow_html=True)
    
    # Check if LLM inference has been run (model_augmented_transcript exists and is not default)
    llm_augmented_transcript = entry.get('model_augmented_transcript', "")
    llm_explanation = entry.get('model_explanation', "")
    
    # Check if this is actual LLM output or default values
    has_llm_results = (llm_augmented_transcript and 
                      "Not available" not in llm_augmented_transcript and 
                      llm_explanation and 
                      "Not available" not in llm_explanation)
    
    if has_llm_results:
        # Show LLM inference results
        tabs = st.tabs(["LLM Augmented Transcript", "LLM Explanation"])
        with tabs[0]:
            st.markdown('<h3 class="content-subheader">LLM Augmented Transcript (Full Analysis)</h3>', unsafe_allow_html=True)
            augmented_html = escape_html(llm_augmented_transcript)
            augmented_html = re.sub(r'(\[.*?\])', r'<span class="augmentation-marker">\1</span>', augmented_html)
            st.markdown(f'<div class="transcript-text">{augmented_html}</div>', unsafe_allow_html=True)
        with tabs[1]:
            st.markdown('<h3 class="content-subheader">LLM Comprehensive Explanation</h3>', unsafe_allow_html=True)
            st.markdown(f'<div class="transcript-text">{escape_html(llm_explanation)}</div>', unsafe_allow_html=True)
    else:
        # Show optimized pipeline results
        pipeline_augmented = entry.get('augmented_transcript', "")
        raw_asr = entry.get('asr_full_text', "")
        
        tabs = st.tabs(["Pipeline Augmented Transcript", "Raw ASR Transcript"])
        with tabs[0]:
            st.markdown('<h3 class="content-subheader">Pipeline Augmented Transcript (With Gesture Markers)</h3>', unsafe_allow_html=True)
            if pipeline_augmented:
                # Highlight gesture grid markers
                augmented_html = escape_html(pipeline_augmented)
                augmented_html = re.sub(r'(\(GESTURE_GRID [^)]+\))', r'<span class="augmentation-marker">\1</span>', augmented_html)
                st.markdown(f'<div class="transcript-text">{augmented_html}</div>', unsafe_allow_html=True)
            else:
                st.info("Augmented transcript not available. Run the optimized pipeline first.")
        with tabs[1]:
            st.markdown('<h3 class="content-subheader">Raw ASR Transcript</h3>', unsafe_allow_html=True)
            if raw_asr:
                st.markdown(f'<div class="transcript-text">{escape_html(raw_asr)}</div>', unsafe_allow_html=True)
            else:
                st.info("Raw ASR transcript not available.")
        
        if not has_llm_results:
            st.info("💡 **Tip:** Run the inference pipeline to get LLM analysis of the augmented transcript and gesture grids.")

def display_transcript_wer_comparisons(entry: Dict):
    """Displays WER comparisons between different transcripts."""
    st.markdown('<h2 class="section-header">Transcript WER Comparisons</h2>', unsafe_allow_html=True)
    st.markdown('<h3 class="content-subheader">CHA Transcript vs. ASR Transcript</h3>', unsafe_allow_html=True)
    
    # Get transcripts using the updated field mapping
    cha_tx = get_entry_value(entry, 'concatenated_filtered_cha_transcript')
    asr_tx = get_entry_value(entry, 'speaker_labeled_asr_transcript')

    if cha_tx is not None and asr_tx is not None:
        wer_cha_vs_asr = calculate_wer(cha_tx, asr_tx)
        if wer_cha_vs_asr is not None:
            st.metric(label="WER (CHA vs. ASR)", value=f"{wer_cha_vs_asr:.4f}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**CHA Transcript (Ground Truth):**")
            st.markdown(f"<div class='transcript-text short-text'>{escape_html(cha_tx)}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("**ASR Transcript:**")
            st.markdown(f"<div class='transcript-text short-text'>{escape_html(asr_tx)}</div>", unsafe_allow_html=True)
    else:
        st.info("CHA transcript or ASR transcript not available for comparison.")
        
        # Show what we do have
        if cha_tx is not None:
            st.markdown("**Available CHA Transcript:**")
            st.markdown(f"<div class='transcript-text short-text'>{escape_html(cha_tx)}</div>", unsafe_allow_html=True)
        if asr_tx is not None:
            st.markdown("**Available ASR Transcript:**")
            st.markdown(f"<div class='transcript-text short-text'>{escape_html(asr_tx)}</div>", unsafe_allow_html=True)

def display_selected_entry_details(selected_entry: Optional[Dict]):
    """Orchestrates the display of all details for the selected entry"""
    if not selected_entry:
        st.info("Select a participant and task from the sidebar to view details.")
        return
    _display_task_description_for_entry(selected_entry) # Task description at the top
    display_media_and_frames(selected_entry) # Handles video, then metadata card, then keyframes
    st.markdown("---")
    display_llm_analysis(selected_entry)
    st.markdown("---")
    display_transcript_wer_comparisons(selected_entry)

# --- Overview Tab Function ---
def display_overview_tab(dataset: List[Dict]):
    """Displays an overview of the dataset with aggregate metrics"""
    st.markdown('<h2 class="section-header">Dataset Overview & Metrics</h2>', unsafe_allow_html=True)
    if not dataset:
        st.warning("Dataset is empty.")
        return

    df_data = [] # This is for the per-entry metrics table, keep it as is.
    for entry in dataset:
        filtered_tx = get_entry_value(entry, 'concatenated_filtered_cha_transcript')
        parakeet_equivalent_tx = get_entry_value(entry, 'speaker_labeled_asr_transcript')
        augmented_llm_full = entry.get('model_augmented_transcript', "")
        # Try LLM selected keyframes first, fall back to gesture grid filepaths
        num_keyframes_str = get_entry_value(entry, 'selected_llm_grid_paths', "[]")
        num_keyframes_list = []
        try:
            loaded_kf = json.loads(num_keyframes_str) if isinstance(num_keyframes_str, str) else num_keyframes_str
            if isinstance(loaded_kf, list): num_keyframes_list = loaded_kf
        except: pass
        
        # If no LLM keyframes, use gesture grid filepaths
        if not num_keyframes_list:
            gesture_grids_str = get_entry_value(entry, 'gesture_motion_sequence_grid_image_paths', "[]")
            try:
                loaded_grids = json.loads(gesture_grids_str) if isinstance(gesture_grids_str, str) else gesture_grids_str
                if isinstance(loaded_grids, list): num_keyframes_list = loaded_grids
            except: pass
        
        num_keyframes = len(num_keyframes_list)

        explanation_text = entry.get('model_explanation', "")
        explanation_length = len(explanation_text) if "Not available" not in explanation_text else 0

        df_data.append({
            'task_segment_id': get_entry_value(entry, 'task_segment_id'),
            'participant_id': get_entry_value(entry, 'participant_identifier', "N/A"),
            'task_label': get_entry_value(entry, 'task_label', "Unknown"),
            'wer_cha_vs_speaker_asr': calculate_wer(filtered_tx, parakeet_equivalent_tx), # Segment WER
            'num_llm_keyframes': num_keyframes,
            'explanation_length': explanation_length
        })
    overview_df = pd.DataFrame(df_data)
    if overview_df.empty:
        st.info("No data to display in overview.")
        return
        
    total_entries = len(overview_df)

    # --- Combined WER and General Metrics (per segment/entry) ---
    st.markdown('<h3 class="content-subheader">Key Metrics (Per Segment)</h3>', unsafe_allow_html=True) # Clarified title
    wer_c_s_valid = overview_df['wer_cha_vs_speaker_asr'].dropna()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Segments", total_entries) # Changed label
    col2.metric("Mean Segment WER (CHA vs ASR)", f"{wer_c_s_valid.mean():.4f}" if not wer_c_s_valid.empty else "N/A")
    col3.metric("Median Segment WER (CHA vs ASR)", f"{wer_c_s_valid.median():.4f}" if not wer_c_s_valid.empty else "N/A")



    # --- LLM Analysis Metrics (per segment/entry) ---
    st.markdown('<h3 class="content-subheader">LLM Analysis Metrics (Per Segment)</h3>', unsafe_allow_html=True)
    avg_explanation_len = overview_df['explanation_length'].mean() if total_entries > 0 and overview_df['explanation_length'].notna().any() else 0

    col_llm1, col_llm2 = st.columns(2)
    col_llm1.metric("Avg. Explanation Length", f"{avg_explanation_len:.0f} chars")
    st.markdown("---")
    
    # --- Metrics by Task Label (per segment/entry) --- (Keep as is)
    st.markdown('<h3 class="content-subheader">Metrics by Task (Aggregated from Segments)</h3>', unsafe_allow_html=True) # Clarified title
    if not overview_df.empty and 'task_label' in overview_df.columns and overview_df['task_label'].notna().any():
        metrics_by_task = overview_df.groupby('task_label').agg(
            num_entries=('task_segment_id', 'count'),
            avg_wer=('wer_cha_vs_speaker_asr', 'mean'), # This is avg of segment WERs
            avg_keyframes=('num_llm_keyframes', 'mean'),
            avg_explanation_length=('explanation_length', 'mean')
        ).reset_index()
        metrics_by_task['avg_wer'] = metrics_by_task['avg_wer'].map(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
        metrics_by_task['avg_keyframes'] = metrics_by_task['avg_keyframes'].map(lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A')
        metrics_by_task['avg_explanation_length'] = metrics_by_task['avg_explanation_length'].map(lambda x: f'{x:.0f}' if pd.notna(x) else 'N/A')
        st.dataframe(metrics_by_task)
    else: st.info("No valid task labels found to group metrics.")
    st.markdown("---")


    # --- NEW SECTION: Overall WER by Participant ---
    st.markdown('<h3 class="content-subheader">Overall WER by Participant (CHA vs ASR)</h3>', unsafe_allow_html=True)
    participant_wer_data = []

    # Check if participant data is prepared (it should be if dataset was loaded)
    if 'participant_data_map' in st.session_state and st.session_state.participant_data_map:
        # Iterate through sorted participant IDs for consistent table order
        for p_id in st.session_state.all_participants_ids:
            participant_tasks = st.session_state.participant_data_map.get(p_id, [])
            
            if participant_tasks: # Ensure there are tasks for this participant
                overall_wer = calculate_participant_overall_wer(participant_tasks)
                participant_wer_data.append({
                    'Participant ID': p_id,
                    'Overall WER': overall_wer # Keep as float or None for now
                })
            else:
                participant_wer_data.append({
                    'Participant ID': p_id,
                    'Overall WER': 'No Tasks'
                })


    if participant_wer_data:
        participant_wer_df = pd.DataFrame(participant_wer_data)
        # Format WER for display after DataFrame creation
        participant_wer_df['Overall WER'] = participant_wer_df['Overall WER'].apply(
            lambda x: f'{x:.4f}' if isinstance(x, float) else ('N/A' if x is None else x)
        )
        st.dataframe(participant_wer_df.set_index('Participant ID')) # Set index for cleaner look
    else:
        st.info("Not enough data to calculate overall WER by participant. Ensure data is loaded and processed.")


# --- Main Application ---
def main():
    setup_page()
    initialize_session_state_vars()

    # JSON file upload interface
    st.sidebar.markdown("### 📄 Upload Analysis JSON")
    st.sidebar.markdown("**Setup Instructions:**")
    st.sidebar.markdown("1. Place your JSON file in a folder")
    st.sidebar.markdown("2. Create `videos/` and `gesture_grids/` subfolders")
    st.sidebar.markdown("3. Put videos in `videos/` and gesture grids in `gesture_grids/participant_id/`")
    st.sidebar.markdown("4. Upload the JSON file below")
    
    uploaded_json = st.sidebar.file_uploader(
        "Choose JSON analysis file", 
        type=['json'], 
        key="json_uploader",
        help="Upload the JSON file with analysis results. Videos and gesture grids should be in subfolders."
    )

    display_app_header()

    if not uploaded_json:
        st.markdown("""
        <div class='main-content-placeholder'>
        <h2>📄 JSON Upload Required</h2>
        <p>Please upload your analysis JSON file using the sidebar.</p>
        <p><strong>Required folder structure:</strong></p>
        <pre>
your_data_folder/
├── analysis.json          ← Upload this file
├── videos/
│   ├── participant1.mp4
│   ├── participant2.mp4
│   └── ...
└── gesture_grids/
    ├── participant1/
    │   ├── grid_001.png
    │   ├── grid_002.png
    │   └── ...
    ├── participant2/
    │   ├── grid_003.png
    │   ├── grid_004.png
    │   └── ...
    └── ...
        </pre>
        <p>The program will automatically look for videos in <code>videos/</code> and gesture grids in <code>gesture_grids/participant_id/</code>.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Process uploaded JSON file
    with st.spinner("📄 Processing uploaded JSON file..."):
        try:
            # Load JSON data
            json_data = json.loads(uploaded_json.getvalue().decode('utf-8'))
            
            if not isinstance(json_data, list) or len(json_data) == 0:
                st.error("❌ Invalid JSON format. Expected a list of analysis entries.")
                return
            
            # Validate it looks like an analysis dataset
            sample = json_data[0]
            if not isinstance(sample, dict) or not any(key in sample for key in ['task_segment_id', 'participant_identifier', 'task_label']):
                st.error("❌ JSON doesn't appear to be a valid analysis dataset.")
                return
            
            # Get the directory where the JSON conceptually lives
            json_dir = os.getcwd()  # Assume folders are in current working directory
            
            # Resolve media paths to point to local folders
            resolved_dataset = resolve_media_paths(json_data, json_dir)
            
            # Check if the expected folders exist
            gesture_grids_dir = os.path.join(json_dir, "gesture_grids")
            videos_dir = os.path.join(json_dir, "videos")
            
            folder_warnings = []
            if not os.path.exists(videos_dir):
                folder_warnings.append(f"⚠️ Videos folder not found: `{videos_dir}`")
            if not os.path.exists(gesture_grids_dir):
                folder_warnings.append(f"⚠️ Gesture grids folder not found: `{gesture_grids_dir}`")
            
            if folder_warnings:
                for warning in folder_warnings:
                    st.warning(warning)
                st.info("💡 Media files may not display correctly. Ensure you have `videos/` and `gesture_grids/` folders in your current directory.")
            
            # Process the dataset using existing logic
            processed_dataset = []
            for entry in resolved_dataset:
                # Parse LLM responses if present
                raw_response = entry.get('model_analysis_response_raw')
                if raw_response:
                    parsed_data = parse_model_response(raw_response)
                    entry['model_augmented_transcript'] = parsed_data['augmented_transcript']
                    entry['model_explanation'] = parsed_data['explanation']
                else:
                    entry.setdefault('model_augmented_transcript', "Not available.")
                    entry.setdefault('model_explanation', "Not available.")
                processed_dataset.append(entry)
            
            # Update session state
            st.session_state.dataset = processed_dataset
            st.session_state.current_file_name = uploaded_json.name
            
            # Prepare participant data
            prepare_participant_task_data(processed_dataset)
            
            st.sidebar.success(f"✅ Loaded {len(processed_dataset)} entries from {uploaded_json.name}")
            
            # Show folder status with participant breakdown
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📁 Expected Folders")
            
            # Videos folder status
            if os.path.exists(videos_dir):
                video_count = len([f for f in os.listdir(videos_dir) if f.lower().endswith(('.mp4', '.mov', '.avi'))])
                st.sidebar.success(f"📹 Videos: {video_count} files found")
            else:
                st.sidebar.error("📹 Videos folder missing")
                
            # Gesture grids folder status (with participant breakdown)
            if os.path.exists(gesture_grids_dir):
                participant_folders = [f for f in os.listdir(gesture_grids_dir) if os.path.isdir(os.path.join(gesture_grids_dir, f))]
                total_grids = 0
                for participant_folder in participant_folders:
                    participant_path = os.path.join(gesture_grids_dir, participant_folder)
                    grid_count = len([f for f in os.listdir(participant_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    total_grids += grid_count
                
                st.sidebar.success(f"🖼️ Gesture grids: {total_grids} files across {len(participant_folders)} participants")
                
                # Show participant breakdown in expander
                with st.sidebar.expander("👥 Participant breakdown"):
                    for participant_folder in sorted(participant_folders):
                        participant_path = os.path.join(gesture_grids_dir, participant_folder)
                        grid_count = len([f for f in os.listdir(participant_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                        st.write(f"• {participant_folder}: {grid_count} grids")
            else:
                st.sidebar.error("🖼️ Gesture grids folder missing")
            
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON file: {e}")
            return
        except Exception as e:
            st.error(f"❌ Failed to process JSON file: {e}")
            return

    if not st.session_state.get('dataset'):
        st.error("Dataset is not available. Please re-upload the JSON file.")
        return

    tab_detailed, tab_overview = st.tabs(["📄 Detailed Entry View", "📊 Overview"])

    with tab_detailed:
        current_entry_to_display = display_detailed_view_sidebar_and_get_selection()
        if current_entry_to_display:
            display_selected_entry_details(current_entry_to_display)
        elif st.session_state.all_participants_ids : # If participants exist but no task selected/error
            st.info("Select a participant and task from the sidebar to view details.")
        # If no participants, sidebar already shows a warning.

    with tab_overview:
        display_overview_tab(st.session_state.dataset)


if __name__ == "__main__":
    main()