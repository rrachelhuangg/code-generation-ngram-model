##  
# AUTHOR   : Rachel Huang, Jackson Taylor
# CREATED  : 20-2-2025
# EDITED   : 27-2-2025
# CONTAINS : Pre-proccessing methods to clean data and output cleaned methods to output file
##

import csv
import os
import re
import pandas as pd
from pygments.lexers.jvm import JavaLexer
from pygments.lexers import get_lexer_by_name
from pygments.token import Token

### Type 1 Clones ###
def remove_duplicates(data):
    """Remove duplicate methods based on method content.
      Almost Type-1 with the exception of comments
    """
    return data.drop_duplicates(subset="Method Java", keep="first")

def filter_ascii_methods(data):
    """Filter methods to include only those with ASCII characters."""
    data = data[data["Method Java"].apply(lambda x: all(ord(char) < 128 for char in x))]
    return data

# Three Approaches:
# 	1.	Data Distribution-Based Filtering: We eliminate outliers by analyzing the original data distribution, as demonstrated below.
# 	2.	Literature-Driven Filtering: We follow best practices outlined in research, such as removing methods exceeding 512 tokens in length.
# 	3.	Hybrid Approach: We combine elements from both the distribution-based and literature-driven methods.

def remove_outliers(data, lower_percentile=5, upper_percentile=95):
    """Remove outliers based on method length."""
    method_lengths = data["Method Java"].apply(len)
    lower_bound = method_lengths.quantile(lower_percentile / 100)
    upper_bound = method_lengths.quantile(upper_percentile / 100)
    return data[(method_lengths >= lower_bound) & (method_lengths <= upper_bound)]

def remove_boilerplate_methods(data):
    """Remove boilerplate methods like setters and getters."""
    boilerplate_patterns = [
        r"\bset[A-Z][a-zA-Z0-9_]*\(.*\)\s*{",  # Setter methods
        r"\bget[A-Z][a-zA-Z0-9_]*\(.*\)\s*{",  # Getter methods
    ]
    boilerplate_regex = re.compile("|".join(boilerplate_patterns))
    data = data[~data["Method Java"].apply(lambda x: bool(boilerplate_regex.search(x)))]
    return data

def remove_comments_from_dataframe(df: pd.DataFrame, method_column: str, language: str) -> pd.DataFrame:
    """
    Removes comments from Java methods in a DataFrame and adds a new column with cleaned methods.

    Args:
        df (pd.DataFrame): DataFrame containing the methods.
        method_column (str): Column name containing the raw Java methods.
        language (str): Programming language for the lexer (e.g., 'java').

    Returns:
        pd.DataFrame: Updated DataFrame with a new column 'Java Method No Comments'.
    """
    # Define a function to remove comments from a single method
    def remove_comments(code):
        lexer = get_lexer_by_name(language)
        tokens = lexer.get_tokens(code)
        # Filter out comments using a lambda function
        clean_code = ''.join(token[1] for token in tokens if not (lambda t: t[0] in Token.Comment)(token))

        return clean_code

    return df

def remove_multiline_comments(method_string:str) -> str:
    """Removes multiline comments from a method that is formatted into one long string"""
    pass

count = 0
total_methods = 0
for data_file in os.listdir("extracted"):
    methods = []
    with open("extracted/"+data_file, "r") as file:
        try:
            reader = csv.reader(file)
            header = next(reader)
            for row in reader:
                methods += [str(row[3])]
        except:
            continue

    data = pd.DataFrame({
        "Method Java": methods
    })

    if data.empty:
        continue
    else:
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)

        data = remove_duplicates(data)
        total_methods += len(data)
        data = filter_ascii_methods(data)
        total_methods += len(data)
        data = remove_outliers(data)
        total_methods += len(data)
        data = remove_boilerplate_methods(data)
        total_methods += len(data)
        data = remove_comments_from_dataframe(data, "Method Java", "Java")
        total_methods += len(data)

        lexer = JavaLexer()
        with open(f"tokens/tokens{count}.txt", "w") as file:
            for index, row in data.iterrows():
                row_string = str(row)
                method = row_string[row_string.find("Method Java")+11:row_string.find("Name:")].strip()
                tokens = [t[1] for t in lexer.get_tokens(method)]
                concat_tokens = ""
                token_count = 0
                for t in tokens:
                    if t not in ["\\", "n", "\n", "t", "\t"] and len(t) > 0:
                        if "private" in t and t[0]=="t":
                            concat_tokens += (t[1:]+" ")
                        elif "public" in t and t[0]=="t":
                            concat_tokens += (t[1:]+" ")
                        elif "protected" in t and t[0]=="t":
                            concat_tokens += (t[1:]+" ")
                        elif "out" in t and t[0]=="t":
                            concat_tokens += (t[1:]+" ")
                        elif "if" in t and t[0]=="t":
                            concat_tokens += (t[1:]+" ")
                        else:
                            concat_tokens += (t+" ")
                for regex in [r"\/\s*(\*)+.*(\*)+\s*\/",r"\/\/.*\\t+", r"\\t",r"\\n", r"\/\/"]:
                    concat_tokens = re.sub(regex, "", concat_tokens)
                concat_tokens = re.sub(r"\s{2,}"," ", concat_tokens)
                file.write(concat_tokens + "\n")
        count += 1

for tokens_file in os.listdir("tokens"):
    with open(f"all_tokens.txt", "w") as file:
        with open(f"tokens/{tokens_file}") as read_file:
            file.write(read_file.read())
