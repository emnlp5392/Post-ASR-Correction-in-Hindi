# import pandas as pd
# from datasets import load_from_disk
# dataset = load_from_disk(f'Dataset/25/hf_format')
# train_dataset = dataset['train']
# print(train_dataset[0])





def get_instruction_from_file(keyword, filename="prompt.txt"):
    # Open and read the contents of the prompt file
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()

    # Split the content by keywords to organize sections
    sections = content.split('\n\n')  # Split by two newlines to separate sections

    # Search for the section that matches the keyword
    for section in sections:
        if section.startswith(keyword):
            # Remove the keyword part and return the rest of the section
            return section[len(keyword):].strip()  # Remove keyword and leading whitespace

    # If no section is found, return a default message
    return f"No section found for keyword: {keyword}"

# Example usage:
lis = ["WS", "CW", "EN", "EW", "HN", "AllError"]
# keyword = "All Error"  # Change this to any keyword you need
result = get_instruction_from_file(lis[5])

print(result)