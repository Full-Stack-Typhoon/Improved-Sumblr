import re, math

######### function for getting cosine similarity ################
def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator
#################################################################


class TCV(object):
	"""This is the data structure class for Tweet Cluster Vector"""
	wordList = []
	clusterList = []
	numOfCLuster = 0
	forClusterId = 0
	def __init__(self,tweet):
		self.clusterId = TCV.forClusterId
		self.n = 1
		self.ts1 = tweet.ts
		self.ts2 = (tweet.ts)**2
		self.tweetsPresent = []
		self.tweetsPresent.append(tweet)
		self.sum_v = dict(tweet.tweetWordDict)
		self.wsum_v = dict(tweet.tweetWordDict)
		self.ft_set = []
		self.ft_set.append(tweet)
		tweet.clusterIdOfTweet = self.clusterId
		TCV.clusterList.append(self)
		TCV.numOfCLuster += 1
		TCV.forClusterId += 1

	def add_tweet(self,tweet):
		self.n += 1
		self.ts1 += tweet.ts
		self.ts2 += tweet.ts*tweet.ts
		tweet.clusterIdOfTweet = self.clusterId
		self.tweetsPresent.append(tweet)

		for k in tweet.tweetWordDict:
			flag = 0
			for key in self.sum_v:
				if key == k:
					self.sum_v[key] += float(tweet.tweetWordDict[k])
					flag = 1
					break
			if flag == 0:
				self.sum_v[k] = float(tweet.tweetWordDict[k])

			flag = 0
			for key in self.wsum_v:
				if key == k:
					self.wsum_v[key] += (float(tweet.w) * float(tweet.tweetWordDict[k]))
					flag = 1
					break
			if flag == 0:
				self.wsum_v[k] = float(tweet.w) * float(tweet.tweetWordDict[k])

		if len(self.ft_set) < 40:
			#print 'Simply adding tweet %s to cluster %s ft_set' %(tweet.id,self.clusterId)
			self.ft_set.append(tweet)
			#self.m += 1
		else:
			centroid = dict(self.wsum_v)
			for key in centroid:
				centroid[key] /= self.n
			cosine = {}
			cosine[get_cosine(centroid,tweet.tweetWordDict)] = []
			cosine[get_cosine(centroid,tweet.tweetWordDict)].append(tweet)
			for tweets in self.ft_set:
				newCos = get_cosine(centroid,tweets.tweetWordDict)
				if newCos in cosine:
					cosine[newCos].append(tweets)
				else:
					cosine[newCos] = []
					cosine[newCos].append(tweets)
			
			keyList = cosine.keys()
			keyList.sort(reverse= True)
			del keyList[-1]

			self.ft_set = []
			for key in keyList:
				#print 'cosine value is %s' % key
				for tups in cosine[key]:
					self.ft_set.append(tups)
					#print 'Complex adding tweet %s to cluster %s ft_set' %(tups.id,self.clusterId)

			#print "length of new ft_set is %s" % len(self.ft_set)		


class TWEET(object):
	tweetList = []
	def __init__(self,wordDict,tweet,ts,weight,ids):
		self.tweetWordDict = wordDict
		self.tweetStr = tweet
		self.ts = ts
		self.w = weight
		self.id = ids
		self.score = 0
		self.clusterIdOfTweet = -1

	def copy(tweet):
		return TWEET(tweet.tweetWordDict,tweet.ts,tweet.w,tweet.id)

