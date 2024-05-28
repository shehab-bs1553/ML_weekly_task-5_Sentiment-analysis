from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import string
from autocorrect import Speller

def option():
    option = input("\nEnter 1 for analysis from input.txt\nEnter 2 for analysis from user input\n\nYour option: ")
    if option == '1':
        tweet = open("sentiment_analysis\input.txt").read()
        return tweet
    elif option == '2':
        tweet = input("\nEnter your tweet: ")
        return tweet
    else:
        print("Invalid option")
        exit()

def preprocessing(tweet):
    tweet = tweet.lower()
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    spell = Speller()
    tweet = ' '.join([spell(word) for word in tweet.split()])
    return tweet

def model_train(tweet):
    
    tweet_words = []

    for word in tweet.split(' '):
        if word.startswith('@'):
            continue
        elif word.startswith('#'):
            continue
        elif word.startswith('http'):
            continue
        tweet_words.append(word)
    tweet_processed = ' '.join(tweet_words)

    model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    encoded_tweet = tokenizer(tweet_processed, return_tensors='pt')
    output = model(**encoded_tweet)
    # print(output[0][0])
    scores = softmax(output[0][0].detach().numpy())
    return scores


   
def main():
    
    tweet = preprocessing(option())
    labels = ["positive", "neutral", "negative"]
    scores = model_train(tweet)
    label_scores = list(zip(labels, scores))
    label_scores.sort(key=lambda x: x[1], reverse=True)

    for label, score in label_scores:
        print(f"{label}: {score:.4f}")

if __name__ == "__main__":
    main()
