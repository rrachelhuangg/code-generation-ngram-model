AUTHOR  : Rachel Huang, Jackson Taylor\
CREATED : 17-2-2025

# N-gram probabilistic code token recommendation model

Model workflow: Download dataset of Java method code, clean and tokenize the data, train and test our n-gram model on the data for variable n, return sample code token generation results from optimal n-gram model

* The Java method dataset was curated by using Datahub with repo specs to make sure that we were collecting complex Java methods.
* Cleaning and tokenizing the data was done with a foundation of the Pydriller and Preprocessing notebooks. 
  * Pydriller was run in a tmux environment to ensure that it could complete downloading the large amount of data necessary to run the model, without disruption from closing our laptop.
  * After the Java methods were extracted and downloaded, the Preprocessing code in `tokenizer.py` was modified to iterate through the extracted files. The cleaned methods were then written to one large file by iterating through each dataframe. Each new line in the output file is a single method and tokens in each method are separated by a single space. 
  * Additional functionality had to be added to the proprocessing code to fix some discovered bugs.
      * Most notably, the pandas dataframe used to clean the data had a column width limitation, so code processing code was only applied to truncated versions of each method. This bug was fixed by removing the pandas dataframe column width.
      * The preprocessing code also did not remove multiline or single line comments when cleaning the data. This bug was fixed by applying regex to the data before writing to the output file.
      * In addition, the preprocessing code had trouble tokenizing space characters, such as "\t" and "\n". This bug was fixed by ensuring that each token was a non-space character before writing to the ouput file, as well as applying regex.

## Testing
#### Testing the data extraction and processing pipeline
*  `python tokenizer.py` cleans and tokenizes the extracted data that is located in the `extracted` directory in this repo

#### Testing the overall n-gram model
* `git clone https://github.com/rrachelhuangg/genai_project1.git`
* `cd genai_project1`
* `python model_controller.py tokens.txt`






