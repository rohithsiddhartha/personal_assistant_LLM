import os
import json
from LLMManager import LLMManager

class ProfileProcessor:
    """
    A class to process user profiles by collecting text data, generating a profile summary,
    and writing the summary to a file.
    """
    def __init__(self, llm_manager, directory):
        """
        Initialize the ProfileProcessor with an instance of LLMManager and a directory to process.
        """
        self.llm_manager = llm_manager
        self.directory = directory

    def collect_texts(self):
        """
        Collect text from all .txt files in the specified directory and its subdirectories.
        """
        master_text = ""
        for root, _, files in os.walk(self.directory):
            # print(files)
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        text = f.read()
                    print(f"Processing {file_path}...")
                    master_text += f'\n {text} \n'
        # print("Master text", master_text)
        return master_text.strip()

    def process_profile(self):
        """
        Process the collected text to generate a profile summary using the LLMManager.
        """
        collected_text = self.collect_texts()
        profile_json = self.llm_manager.process_profile(collected_text)
        profile_dict = json.loads(profile_json)
        self.write_profile_to_file(profile_dict)

    def write_profile_to_file(self, profile_dict):
        """
        Write the generated profile summary to a text file.
        """
        output_file_path = os.path.join(self.directory, 'user_profile_summary.txt')
        with open(output_file_path, 'w') as file:
            file.write(self.format_profile(profile_dict))
        print(f"Profile summary written to {output_file_path}")

    def format_profile(self, profile_dict):
        """
        Format the profile dictionary into a readable string format.
        """
        formatted_profile = ""
        for section, attributes in profile_dict.items():
            formatted_profile += f"{section}\n"
            for key, value in attributes.items():
                formatted_profile += f"{key}: {value}\n"
            formatted_profile += "\n"
        return formatted_profile

# Example usage
if __name__ == "__main__":
    llm_manager = LLMManager()
    processor = ProfileProcessor(llm_manager, "/Users/rohithsiddharthareddy/Desktop/TakeHomeAssignments/personal_assistant_LLM/rohithprofile")
    processor.process_profile()
