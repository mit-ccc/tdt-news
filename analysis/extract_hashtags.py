#!/usr/bin/python3

"""
Reads a file of tweets (in the format of [user_id, {"result": [tweet list]}]
pairs, one pair per line), and prints the top hashtags referenced in the
body text of the tweets.

Example command line:

gunzip -c ~/Downloads/selected_social+interest_uids_2_recent-tweets.json | ./analysis/extract_hashtags.py --input -
"""

import argparse
import ujson as json
import sys

def extract_hashtags(s):
    """Given a string representing tweet text, return the set
    of hashtags mentioned."""
    return set(part[1:] for part in s.split() if part.startswith('#'))

def process_input_file(f, counts):
    """Given a file object f (of tweet lists) and a counts dictionary, read the
    tweets and add the hashtag counts for eligible tweets to the dictionary."""
    for line in f:
        user_id, api_response = json.loads(line)
        if "result" in api_response:
            for tweet_obj in api_response["result"]:
                process_tweet(tweet_obj, counts)

def process_tweet(tweet_obj, counts):
    """Given a tweet object, add its hashtag counts to a counts dictionary."""
    content = tweet_obj.get("full_text")
    for hashtag in extract_hashtags(content):
        counts[hashtag] = counts.get(hashtag, 0) + 1
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract top hashtags from tweets')
    parser.add_argument('--inputs', type=str,
                        help='Filenames of json files to read (comma-separated), or "-" to use stdin')
    args = parser.parse_args()

    # A dictionary mapping hashtag string to the number of times they occurred in the data
    hashtag_counts = {}

    for filename in args.inputs.split(","):
        if filename == "-":
            f = sys.stdin
        else:
            f = open(filename)
        with f:
            process_input_file(f, hashtag_counts)

    # Sort hashtags by count, highest first
    count_list = list(hashtag_counts.items())
    count_list.sort(key = lambda x : x[1], reverse=True)

    print ("Top hashtags:")
    print("\n".join("#%s: %d mentions" % (x) for x in count_list[:10]))
