# SUMBLR:Continuous Summarization of Evolving Tweet Streams 
- Implemented an online distributed algorithm for generating summaries of related tweets cluster created using k-means algorithm.  
- Outperformed existing algorithms in speed by **8 times** and accuracy by **10.8\%**.  
- Technologies: Python, scikit-learn, Tweepy, Flask, MongoDB, NLTK. 

## Overview
- Continuous Tweet Summarization.   
- Continuously monitor “Apple”-related tweets arriving from the stream and produce a continuous timeline which grows by time.  
- The range timeline during a period giving topic evolution in those weeks.  
- Drill down summary or Roll up summary  

## Methodology
### Tweet stream clustering module:
Tweet stream clustering algorithm, an online algorithm allowing for effective clustering of tweets with only one pass over the data.  

### High-level summarization module:
Generation of two kinds of summaries: online summaries and historical ones.  
Topic evolvement detection algorithm

## About Code and How to Run
### Code
3 files are used:  
	- dataStr.py   
	- twokenize.py  
	- summary.py  
 ### Dataset
 Datafile must be of the extension ".xls" only!!  
 Datafile must be a csv file with first column containing tweets and column 3 containing dates 
 
 ### Input
 Input file is named input.  
 Format of input file : tweet_id [space] summary_size
 
 ### Output
 Output is given in summay_uttarakhand  
 
 ### Run Command
 To run the code simply write "python summary.py " in the command line. 
