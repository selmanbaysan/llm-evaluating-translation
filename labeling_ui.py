import argparse
import json
import os

import streamlit as st


def load_json_file(file_path: str) -> dict | None:
    """
    Loads a JSON file from the given path.

    This function attempts to open and load a JSON file. If the file is successfully loaded,
    the JSON data is returned as a dictionary. If a JSONDecodeError occurs, an error message is displayed
    using Streamlit and None is returned.

    Args:
        file_path: The path to the JSON file.

    Returns:
        A dictionary representing the JSON data or None if the file has an error.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            st.error(f"Invalid JSON in: {file_path}")
            return None


def save_decision(file_path: str, data: dict):
    """
    Saves the updated JSON data to the specified file.

    This function writes the given Python dictionary to the JSON file, formatting the JSON with
    indentation for better readability and ensuring ASCII characters are handled. A success message
    is displayed in streamlit to inform the user about the saved decision.

    Args:
        file_path: The path to the JSON file.
        data: The dictionary containing the JSON data to be saved.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    st.success(f"Decision saved for: {os.path.basename(file_path)}")


def main(data_folder: str):
    """
    Creates a Streamlit app to facilitate the LLM-as-a-Judge evaluation process.

    This function displays a user interface to allow manual evaluation of JSON files, each containing a question,
    AI-generated response, and explanation, along with AI's decision. Users can choose whether to pass or fail each instance,
    and these decisions are saved back to the JSON files. The UI also provides navigation buttons to move to the next/previous item

    Args:
        data_folder: The path to the directory containing JSON files to be evaluated.
    """
    st.title("LLM-as-a-Judge Evaluation")

    json_files = [f for f in os.listdir(data_folder) if f.lower().endswith(".json")]
    if not json_files:
        st.warning("No JSON files found in the specified directory.")
        return

    # Sort files numerically
    json_files.sort(key=lambda x: int(x.split(".")[0]))

    def get_first_undecided_index(files: list) -> int:
        """
            Gets the index of the first file without human decision.

            This function checks if the json has a decision already. If no decision is given,
            then this index is returned, else it continues with the other files.
            It returns 0 by default when all files are decided.

            Args:
                files: The list of json file names.
            Returns:
                The integer of the first undecided file
        """
        for i, file in enumerate(files):
            file_path = os.path.join(data_folder, file)
            data = load_json_file(file_path)
            if data and "human_decision" not in data:
                return i
        return 0  # Default to the first file if all have human_decision

    if "current_file_index" not in st.session_state:
        st.session_state.current_file_index = get_first_undecided_index(json_files)

    def save_and_next():
        """
            Saves the decision of current file and goes to the next file
        """
        if "current_data" in st.session_state and "file_path" in st.session_state:
            save_decision(st.session_state.file_path, st.session_state.current_data)
        if st.session_state.current_file_index < len(json_files) - 1:
            st.session_state.current_file_index += 1

    def save_and_next_undecided():
        """
            Saves the decision of current file and goes to the next undecided file
        """
        if "current_data" in st.session_state and "file_path" in st.session_state:
            save_decision(st.session_state.file_path, st.session_state.current_data)
        next_undecided_index = st.session_state.current_file_index + 1
        while next_undecided_index < len(json_files):
            file_path = os.path.join(data_folder, json_files[next_undecided_index])
            data = load_json_file(file_path)
            if data and ("human_decision" not in data or data["human_decision"] == "UNDECIDED" or data["human_decision"] is None):
                st.session_state.current_file_index = next_undecided_index
                return
            next_undecided_index += 1

    def save_and_prev():
        """
            Saves the decision of current file and goes to the previous file
        """
        if "current_data" in st.session_state and "file_path" in st.session_state:
            save_decision(st.session_state.file_path, st.session_state.current_data)
        if st.session_state.current_file_index > 0:
            st.session_state.current_file_index -= 1

    def save_and_prev_undecided():
        """
            Saves the decision of current file and goes to the previous undecided file
        """
        if "current_data" in st.session_state and "file_path" in st.session_state:
            save_decision(st.session_state.file_path, st.session_state.current_data)
        prev_undecided_index = st.session_state.current_file_index - 1
        while prev_undecided_index >= 0:
            file_path = os.path.join(data_folder, json_files[prev_undecided_index])
            data = load_json_file(file_path)
            if data and ("human_decision" not in data or data["human_decision"] == "UNDECIDED" or data["human_decision"] is None):
                st.session_state.current_file_index = prev_undecided_index
                return
            prev_undecided_index -= 1

    def save_and_jump_to_file():
        """
        Saves the decision of current file and jumps to the selected file
        """
        if "current_data" in st.session_state and "file_path" in st.session_state:
            save_decision(st.session_state.file_path, st.session_state.current_data)
        st.session_state.current_file_index = json_files.index(selected_file)

    selected_file = st.selectbox("Select JSON File", json_files, index=st.session_state.current_file_index, on_change = save_and_jump_to_file, key = "json_file_selector")

    file_path = os.path.join(data_folder, selected_file)
    current_data = load_json_file(file_path)
    if not current_data:
        return

    if 'human_decision' not in current_data or current_data['human_decision'] not in ['FAIL', 'UNDECIDED', 'PASS']:
        current_data['human_decision'] = 'UNDECIDED'

    st.session_state.file_path = file_path
    st.session_state.current_data = current_data

    col1, col2 = st.columns([1, 1])

    with col1:
        st.text_area("English Text", value=current_data.get("english_text", "N/A"), disabled=True, height=400)
        st.text_area("AI Decision", value=current_data.get("translation_is_valid", "N/A"), disabled=True, height=50)
    with col2:
        st.text_area("Turkish Text", value=current_data.get("turkish_text", "N/A"), disabled=True, height=400)
        human_decision = st.radio(
            "Human Decision",
            ["FAIL", "UNDECIDED", "PASS"],
            index=["FAIL", "UNDECIDED", "PASS"].index(current_data.get("human_decision", "UNDECIDED")),
            horizontal=True,
            on_change=None
        )
        current_data["human_decision"] = human_decision

    col3, _, col5 = st.columns([1, 1, 1])

    with col3:
        if st.button("Previous", on_click=save_and_prev):
            pass
        if st.button("Previous Undecided", on_click=save_and_prev_undecided):
            pass
    with col5:
        if st.button("Next", on_click=save_and_next):
            pass
        if st.button("Next Undecided", on_click=save_and_next_undecided):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate JSON files using a Streamlit UI")
    parser.add_argument("--data_folder_path", type=str, help="Path to the folder containing JSON files to evaluate")

    args = parser.parse_args()
    main(args.data_folder_path)
