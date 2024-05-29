## Sentiment Analysis Using DistilBERT

## Project Description:

* #### This project performs sentiment analysis on text input using a pre-trained DistilBERT model from Hugging Face.
  
* #### The model categorizes input text into three sentiment classes:

    * positive
           
    * neutral
           
    * negative.
    
* #### The text can be input from a file or directly by the user.

## Table of Contents

 * Installation
 * Usage
   
   Option 1: Analyze Text from a File
   
   Option 2: Analyze User Input Text
 * Preprocessing Steps
 * Model Details
 * License
## Installation
Clone the Repository:

      https://github.com/shehab-bs1553/ML_weekly_task-5_Sentiment-analysis.git

## Install Dependencies:

### Copy code

               pip install transformers
               pip install scipy
               pip install autocorrect
               pip install nltk

## Usage
#### Run the main function to start the sentiment analysis.

#### The script will prompt you to choose between analyzing text from a file or entering text directly.

      python sentiment_analysis/main.py

* #### Option 1: Analyze Text from a File

    Place your text file named input.txt in the ML_weekly_task-5_Sentiment-analysis/sentiment_analysis/ directory.
    
    Select option 1 when prompted. The code executes the text from the input.txt file.

* #### Option 2: Select option 2 when prompted.

    Enter your text in the console.

* #### Quitting

     Enter q to quit the program.


## Preprocessing Steps

  * #### Lowercasing: Converts all characters in the text to lowercase.
 
  * #### Punctuation Removal: Removes all punctuation from the text.
 
  * #### Spelling Correction: Correct spelling using the autocorrect library.
 
  * #### Stopword Removal: Removes common English stopwords using NLTK.

## Model Details
## The model used is 
      
      lxyuan/distilbert-base-multilingual-cased-sentiments-student
 a fine-tuned DistilBERT model for sentiment analysis.

## Labels:

* Positive
   
* Neutral
   
* Negative

## License

This project is licensed under the MIT License.
