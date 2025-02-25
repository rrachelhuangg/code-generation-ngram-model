# N-gram probabilistic code token recommendation model

Model workflow: Download dataset of Java method code, clean and tokenize the data, train and test our n-gram model on the data for variable n, return sample code token generation results from optimal n-gram model

* The Java method dataset was curated by using Datahub with respository specifications to make sure that the data methods we collected were complex enough to support our model's functionality.
* Cleaning and tokenizing the data was done with a foundation of the Pydriller and Preprocessing notebooks given to us in class. 
  * To properly clean and tokenize the data, the Pydriller code file was run in a tmux environment, to ensure that it could complete downloading the large amount of data necessary to run the model, without exiting when our laptop was closed.
  * After the extracted methods were downloaded, the Preprocessing code was modified so that it would iterate through the extracted files and apply the code cleanup methods to the data. The cleaned methods were then written to one large file by iterating to each dataframe, where each new line in the file was a single method. Tokens in each method were separate by a single space. 
  * Additional functionality had to be added to the proprocessing code, as some bugs were discovered.
      * Most notably, the pandas dataframe used to clean the data had a column width limitation, so code processing code was only applied to truncated versions of each method. This bug was fixed by removing the pandas dataframe column width.
      * The preprocessing code also did not remove multiline or single line comments when cleaning the data. This bug was fixed by applying regex to the data before writing to the output file.
      * In addition, the preprocessing code had trouble tokenizing space characters, such as "\t" and "\n". This bug was fixed by ensuring that each token was a non-space character before writing to the ouput file, as well as applying regex.

## Testing
#### Testing the data extraction and processing pipeline

#### Testing the overall n-gram model
* `git clone https://github.com/rrachelhuangg/genai_project1.git`
* `cd genai_project1`
* `python model_controller.py tokens.txt`

Tools used: ![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54) 





