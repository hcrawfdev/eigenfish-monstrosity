import base64
import sys
import pyperclip

def encode_js_to_base64(file_path):
    try:
        with open(file_path, 'rb') as file:
            js_content = file.read()
        base64_encoded = base64.b64encode(js_content).decode('utf-8')
        return base64_encoded
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_js_file>")
    else:
        js_file_path = sys.argv[1]
        result = encode_js_to_base64(js_file_path)
        print(result)
        
        try:
            pyperclip.copy(result)
            print("\nBase64 encoded string has been copied to your clipboard.")
        except pyperclip.PyperclipException:
            print("\nUnable to copy to clipboard. Please make sure you have the necessary dependencies installed.")
