# HamOrSpam

This extension of a school project uses logistic regression to determine whether or not an email is Spam or not Spam (aka 'Ham').  Originally, you would pass the email as a String, however, I implemented it so you could pass files through the command line.  

SpamSingleEmail.py takes a single text file as input and outputs either Ham or Spam depening on the classification of the program. 

SpamFolder.py takes a directory of a folder full of text files and outputs the classification for each text file in that folder. 

Spam.ipynb was the original Jupyter Notebook for the project.

Data has the data used to train the model.

SpamBash.py has a version of the project that uses bash. 

In order to run the project, you must copy an email body text to a file, and save it. 
Then, clone the repo and run the following command from the command-line:

python SpamSingleEmail -d "/path/to/textfile.txt"
OR
python SpamFolder -d "/Folder/Full/Of/Text/Files/"
