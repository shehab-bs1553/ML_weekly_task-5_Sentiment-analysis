from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import string
from autocorrect import Speller
import nltk
from nltk.corpus import stopwords

# Download stop words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def option():
    option = input("\nEnter 1 for analysis from input.txt\nEnter 2 for analysis from user input\nEnter 'q' to quit\n\nYour option: ")
    if option == '1':
        tweet = open("ML_weekly_task-5_Sentiment-analysis/sentiment_analysis/input.txt").read()
        return tweet, option
    elif option == '2':
        tweet = input("\nEnter your tweet: ")
        return tweet, option
    elif option == 'q':
        return None, option
    else:
        return None, None

def preprocessing(tweet):
    tweet = tweet.lower()
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    
    # For spelling correction
    spell = Speller()
    tweet = ' '.join([spell(word) for word in tweet.split()])
    
    # Remove stop words
    tweet_words = [word for word in tweet.split() if word not in stop_words]
    tweet_processed = ' '.join(tweet_words)
    
    return tweet_processed

def model_train(tweet, tokenizer, model):
    encoded_tweet = tokenizer(tweet, return_tensors='pt')
    output = model(**encoded_tweet)
    scores = softmax(output[0][0].detach().numpy())
    return scores

def main():
    model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    while True:
        tweet, option_selected = option()
        if option_selected == 'q':
            print("Quitting...")
            break
        elif option_selected in ['1', '2']:
            tweet = preprocessing(tweet)
            labels = ["positive", "neutral", "negative"]
            scores = model_train(tweet, tokenizer, model)
            label_scores = list(zip(labels, scores))
            label_scores.sort(key=lambda x: x[1], reverse=True)

            for label, score in label_scores:
                print(f"{label}: {score:.4f}")
        else:
            print("Invalid option. Please try again.")
            break

if __name__ == "__main__":
    main()
