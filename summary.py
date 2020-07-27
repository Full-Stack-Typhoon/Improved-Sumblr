from xlrd import open_workbook 
import nltk
import twokenize
import dataStr
import re, math
from collections import Counter
import datetime
from nltk.corpus import stopwords
from scipy.stats import norm
import itertools
import time
import numpy
import time

stops = list(stopwords.words('english'))
stops.append('rt')
book=open_workbook('/home/vikas/Desktop/Social_com/processed_uttarakhand_flood_tweets.xls')
sheet=book.sheet_by_index(0)

################### Parameters ########################

NMAX = 150
MC = 0.7
BETA = 0.07
T1CANOPY = 0.7
l=5
#######################################################

####### Function for counting a value in a list ########
def count(word, list):
	count = 0
	for i in list:
		if i.rstrip().upper() == word.rstrip().upper():
			count = count+1
	return count
########################################################

################## Aggregate Cluster ###################

def aggregate(cluster1,cluster2):
	for key in cluster2.sum_v:
		if key in cluster1.sum_v:
			cluster1.sum_v[key] += cluster2.sum_v[key]
		else:
			cluster1.sum_v[key] = cluster2.sum_v[key]

	for key in cluster2.wsum_v:
		if key in cluster1.wsum_v:
			cluster1.wsum_v[key] += cluster2.wsum_v[key]
		else:
			cluster1.wsum_v[key] = cluster2.wsum_v[key]

	cluster1.n += cluster2.n
	cluster1.ts1 += cluster2.ts1
	cluster1.ts2 += cluster2.ts2

	newList = list(set(cluster1.ft_set) | set(cluster2.ft_set))
	centroid = dict(cluster1.wsum_v)
	for key in centroid:
		centroid[key] /= cluster1.n
	cosine = {}
	for tweets in newList:
		newCos = get_cosine(centroid,tweets.tweetWordDict)
		if newCos in cosine:
			cosine[newCos].append(tweets)
		else:
			cosine[newCos] = []
			cosine[newCos].append(tweets)
			
	keyList = cosine.keys()
	keyList.sort(reverse= True)

	focused_list = []
	
	for key in keyList:
		if len(focused_list) < 40:
			#print 'cosine value is %s' % key
			for tweet in cosine[key]:
				focused_list.append(tweet)
				#print 'Complex adding tweet %s to cluster %s ft_set' %(tweet.id,cluster1.clusterId)
		else:
			break

	cluster1.ft_set = focused_list

########################################################

################## Merging Clusters ####################

def merge_clusters():
	#print 'In Merging Clusters'
	cosineSorted = {}
	#print dataStr.TCV.numOfCLuster
	for i in range(dataStr.TCV.numOfCLuster):
		centroid1 = dict(dataStr.TCV.clusterList[i].wsum_v)
		for key in centroid1:
			centroid1[key] /= dataStr.TCV.clusterList[i].n
		for j in range(i+1,dataStr.TCV.numOfCLuster):	
			centroid2 = dict(dataStr.TCV.clusterList[j].wsum_v)
			for key in centroid2:
				centroid2[key] /= dataStr.TCV.clusterList[j].n

			cosine = get_cosine(centroid1,centroid2)
			tup = (dataStr.TCV.clusterList[i].clusterId,dataStr.TCV.clusterList[j].clusterId)
			#print 'for tuple %s and %s cosine is %s' %(str(dataStr.TCV.clusterList[i].clusterId),str(dataStr.TCV.clusterList[j].clusterId),str(cosine))
			if cosine in cosineSorted:
				cosineSorted[cosine].append(tup)
			else:
				cosineSorted[cosine] = []
				cosineSorted[cosine].append(tup)

	keyList = cosineSorted.keys()
	keyList.sort(reverse= True)

	#for key in keyList:
	#	print cosineSorted[key]

	temp = NMAX * (1 - MC)
	numMerged = 0
	CompositeCluster = {}
	for key in keyList:
		#print cosineSorted[key]
		if numMerged >= temp:
			#print 'Total Number of Clusters Merged = %s' % str(numMerged)
			break
		else:
			for everyTuple in cosineSorted[key]:
				#Checking whether both tuples are composite or not.
				comp1 = -1
				comp2 = -1
				for k in CompositeCluster:
					if everyTuple[0] in CompositeCluster[k]:
						comp1 = k
					if everyTuple[1] in CompositeCluster[k]:
						comp2 = k

				if comp1 == -1 and comp2 == -1:
					CompositeCluster[everyTuple[0]] = []
					CompositeCluster[everyTuple[0]].append(everyTuple[0])
					CompositeCluster[everyTuple[0]].append(everyTuple[1])
					cluster1 = None
					cluster2 = None
					for cluster in dataStr.TCV.clusterList:
						if everyTuple[0] == cluster.clusterId:
							cluster1 = cluster
						if everyTuple[1] == cluster.clusterId:
							cluster2 = cluster	
					aggregate(cluster1,cluster2)
					numMerged += 1

				elif comp1 == -1 and comp2 != -1:
					CompId = None
					for compClusterKey in CompositeCluster:
						if comp2 in CompositeCluster[compClusterKey]:
							CompositeCluster[compClusterKey].append(everyTuple[0])
							CompId = compClusterKey
							break
					for cluster in dataStr.TCV.clusterList:
						if everyTuple[0] == cluster.clusterId:
							cluster2 = cluster
						if CompId == cluster.clusterId:
							cluster1 = cluster	
					aggregate(cluster1,cluster2)
					numMerged += 1

				elif comp1 != -1 and comp2 == -1:
					CompId = None
					for compClusterKey in CompositeCluster:
						if comp1 in CompositeCluster[compClusterKey]:
							CompositeCluster[compClusterKey].append(everyTuple[1])
							CompId = compClusterKey
							break				
					for cluster in dataStr.TCV.clusterList:
						if everyTuple[1] == cluster.clusterId:
							cluster2 = cluster
						if CompId == cluster.clusterId:
							cluster1 = cluster
					aggregate(cluster1,cluster2)
					numMerged += 1

				elif comp1 != -1 and comp2 != -1:
					compId1 = None
					compId2 = None
					for compClusterKey in CompositeCluster:
						if comp1 in CompositeCluster[compClusterKey]:
							compId1 = compClusterKey
						if comp2 in CompositeCluster[compClusterKey]:
							compId2 = compClusterKey

					if compId1 != compId2:
						if compId1 in CompositeCluster:
							for vals in CompositeCluster[compId2]:
								CompositeCluster[compId1].append(vals)
						for cluster in dataStr.TCV.clusterList:
							if compId2 == cluster.clusterId:
								cluster2 = cluster
							if compId1 == cluster.clusterId:
								cluster1 = cluster
						aggregate(cluster1,cluster2)
						del CompositeCluster[compId2]
						numMerged += 1

	#print 'length of CompositeCluster is %s' % str(len(CompositeCluster))

	for key in CompositeCluster:
		for val in CompositeCluster[key]:
			if val != key:
				for cluster in dataStr.TCV.clusterList:
					if cluster.clusterId == val:
						#print 'Merging cluster with id %s to %s' % (val,key)
						dataStr.TCV.clusterList.remove(cluster)
						dataStr.TCV.numOfCLuster -= 1
						break

	#print 'New Length of ClusterList : %s' % str(len(dataStr.TCV.clusterList))
###############################################################


####### Storing initial 500 tweets in the DATA Structure #######
#for i in range(2826,3326):

start_time = time.time()
for i in range(1,201):
	tweet,number,date=sheet.row_values(i,0,3)
	#for line in itertools.islice(f,1,500):
	tweetWordList = twokenize.tokenize(tweet.encode('utf-8','replace'))
	tweetWordDict = {}
	for word in tweetWordList:
		word = word.upper()
		if word.lower() in stops:
			#print word.lower()
			pass
		else:
			if count(word, dataStr.TCV.wordList) == 0:
				dataStr.TCV.wordList.append(word)
			if word in tweetWordDict:
				tweetWordDict[word] += 1
			else:
				tweetWordDict[word] = 1

	total = 0
	for key in tweetWordDict:
		total += tweetWordDict[key]

	for key in tweetWordDict:
		tweetWordDict[key] = float(tweetWordDict[key]) /total

	x = date.split()
	x1 = x[0].split('-')
	x2 = x[1].split(':')
	#d = datetime()
	timestamp = time.mktime(datetime.datetime.strptime(date,"%Y-%m-%d %H:%M:%S").timetuple())
	#print timestamp
	newTweet = dataStr.TWEET(tweetWordDict,tweet,timestamp,1,i)
	#print "&&&&&&&"
	#print tweetWordDict
	dataStr.TWEET.tweetList.append(newTweet)
#################################################################

################## Delete Cluster UTIL ##########################
def remove_cluster(cluster):
	#print "##################################"
	#print "Deleting cluster "+str(cluster.clusterId)
	dataStr.TCV.clusterList.remove(cluster)

#################################################################

################## Delete Cluster Function ######################

def Delete_clusters(tweet):
	for cluster in dataStr.TCV.clusterList:
		ts1_sum=cluster.ts1
		ts2_sum=cluster.ts2
		mean=(ts1_sum)/(cluster.n)
		temp=(ts2_sum)/(cluster.n)
		variance=math.sqrt(temp-(mean**2))
		percentage=10
		q=(100- (percentage/2))/100
		Threshold=tweet.ts-3*3600*24
		#time=mean+variance*(norm.ppf(q))

		time = mean + variance *1.64485362175
		#print 'value of time is ' + str(time) + ' value of threshold is ' + str(Threshold)+"varaince ="+str(variance)
		if time<Threshold:
			remove_cluster(cluster)
		else:
			#print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
			print "Not deleting "+str(cluster.clusterId)

#################################################################

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

###################### CANOPY CLustering ########################
copy_dict = {}
for tweet in dataStr.TWEET.tweetList:
	copy_dict[tweet] = 1

while True:
	flag = 0
	for tweet in copy_dict:
		if copy_dict[tweet] == 1:
			copy_dict[tweet] = 0
			newCluster = dataStr.TCV(tweet)
			#print 'creating new cluster for tweet %s with id %s' % (tweet.id,newCluster.clusterId)
			for tweets in copy_dict:
				if copy_dict[tweets] == 1:
					cosine = get_cosine(tweet.tweetWordDict,tweets.tweetWordDict)
					if cosine >= T1CANOPY:
						#print 'adding tweet %s to cluster %s cosine is %s' % (tweets.id,newCluster.clusterId,cosine)
						newCluster.add_tweet(tweets)
					#if cosine >= T2CANOPY:
						copy_dict[tweets] = 0
			flag = 1
			break

	if dataStr.TCV.numOfCLuster > NMAX:
		merge_clusters()
	if flag == 0:
		break

#for cluster in dataStr.TCV.clusterList:
	#print cluster.clusterId
	#print cluster.n
	#print cluster.sum_v

####################################################

################  Getting MBS Function #############
def getMinBoundSim(cluster):

	#MBS = wsum_v.sum_v/(n* || wsum_v|| )
	intersection = set(cluster.wsum_v.keys()) & set(cluster.sum_v.keys())
	numerator = sum([cluster.wsum_v[x] * cluster.sum_v[x] for x in intersection])

	denominator = math.sqrt(sum([cluster.wsum_v[x]**2 for x in cluster.wsum_v.keys()]))

	MBS = BETA * numerator/(cluster.n * denominator)
	return MBS

####################################################


################ Lexrank of tweets #################
def power_method(m, epsilon ):
    n = len( m )
    p = [1.0 / n] * n
    while True:
        new_p = [0] * n
        for i in xrange( n ):
            for j in xrange( n ):
                new_p[i] += m[j][i] * p[j]
        total = 0
        for x in xrange( n ):
            total += ( new_p[i] - p[i] ) ** 2
        p = new_p
        if total < epsilon:
            break
    return p

def rank_documents(ft_set, threshold=0.1, tolerance=0.00001):
    n = len(ft_set)
    #Initialises the adjacency matrix
    adjacency_matrix = numpy.zeros([n, n])
    
    degree = numpy.zeros([n])
    scores = numpy.zeros([n])
    
    for i, tweeti in enumerate(ft_set):
        for j, tweetj in enumerate(ft_set):
            adjacency_matrix[i][j] = get_cosine(tweeti.tweetWordDict, tweetj.tweetWordDict)
            
            if adjacency_matrix[i][j] > threshold:
                adjacency_matrix[i][j] = 1.0
                degree[i] += 1
            else:
                adjacency_matrix[i][j] = 0
    
    for i in xrange(n):
        for j in xrange(n):
            if degree[i] == 0: degree[i] = 1.0 #at least similat to itself
            adjacency_matrix[i][j] = adjacency_matrix[i][j] / degree[i]

    scores = power_method(adjacency_matrix, tolerance)        
    
    for i in xrange( 0, n ):
        ft_set[i].score = scores[i]
    #return ft_set


####################################################
def make_dictionary(t):
	tweetWordList = twokenize.tokenize(tweet.encode('utf-8','replace'))
	tweetWordDict = {}
	for word in tweetWordList:
		word = word.upper()
		if word.lower() in stops:
			#print word.lower()
			pass
		else:
			if count(word, dataStr.TCV.wordList) == 0:
				dataStr.TCV.wordList.append(word)
			if word in tweetWordDict:
				tweetWordDict[word] += 1
			else:
				tweetWordDict[word] = 1

	total = 0
	for key in tweetWordDict:
		total += tweetWordDict[key]

	for key in tweetWordDict:
		tweetWordDict[key] = float(tweetWordDict[key]) /total
	return tweetWordDict	

def find_biggest_cluster(clusterList):
	maximum=0
	for cluster in clusterList:
		if cluster.n>maximum:
			maximum=cluster.n
	return maximum

def list_substraction(b,a,related_a):
	#print "len is"+str(len(related_a))+str(len(a))

	c=[]
	d=[]
	#print [i for i in a if not i in b or b.remove(i)]
	for i in a :
		if i not in b :
			c.append(i)
			#print "indxxxx is"+str(a.index(i))
			d.append(related_a[a.index(i)])
	return c,d

def avg_similarity(t,Summary):
	avg=0
	if len(Summary)==0:
		return 0
	dict1=make_dictionary(t)
	for i in Summary:
		dict2=make_dictionary(i)
		avg+=get_cosine(dict1,dict2)
	return float(avg)/len(Summary)

################# K-means Clustering ###############

i = 201
#i = 1000
#i = 5
#i = 3326
while i < 2826:
#while i < 4609:
#while i < 3826:
#while i < 1000:
#while i < 10:
#while i < 1500:
	#print 'here already %s' % i
	tweet,number,date=sheet.row_values(i,0,3)
	#print tweet
	tweetWordList = twokenize.tokenize(tweet.encode('utf-8','replace'))
	tweetWordDict = {}
	for word in tweetWordList:
		word = word.upper()
		if word.lower() in stops:
			#print word.lower()
			pass
		else:
			if count(word, dataStr.TCV.wordList) == 0:
				dataStr.TCV.wordList.append(word)
			if word in tweetWordDict:
				tweetWordDict[word] += 1
			else:
				tweetWordDict[word] = 1

	total = 0
	for key in tweetWordDict:
		total += tweetWordDict[key]

	for key in tweetWordDict:
		tweetWordDict[key] = float(tweetWordDict[key]) /total

	timestamp = time.mktime(datetime.datetime.strptime(date,"%Y-%m-%d %H:%M:%S").timetuple())
	#print timestamp
	newTweet = dataStr.TWEET(tweetWordDict,tweet,timestamp,1,i)
	dataStr.TWEET.tweetList.append(newTweet)

	#print 'Number of Clusters Before Deleteing : %s' % len(dataStr.TCV.clusterList)
	if i%10 == 0:
		Delete_clusters(newTweet)
	#print 'Number of Clusters After Deleting : %s' % len(dataStr.TCV.clusterList)

	maxCosine = -1
	maxCluster = None
	for cluster in dataStr.TCV.clusterList:
		centroid = dict(cluster.wsum_v)
		for key in centroid:
			centroid[key] /= cluster.n
		cosine = get_cosine(centroid,newTweet.tweetWordDict)
		#print 'found cosine is %s' % cosine
		if cosine > maxCosine:
			maxCosine = cosine
			maxCluster = cluster

	MBS = getMinBoundSim(maxCluster)

	#print 'For %s maxCosine is %s and MBS is %s' %(i,maxCosine,MBS)
	if(maxCosine > MBS):
		maxCluster.add_tweet(newTweet)
		#print 'adding tweet %s to %s' %(i,maxCluster.clusterId)
	else:
		dataStr.TCV(newTweet)
		#print 'Creating new Cluster'

	if dataStr.TCV.numOfCLuster > NMAX:
		merge_clusters()

	fts=[]
	list_tweets=[]
	for cluster in dataStr.TCV.clusterList:
		fts.extend(cluster.ft_set)
	for r in fts:
		list_tweets.append(r.tweetStr)
	rank_documents(fts)
	
	if i%50==0 or i==2825:
		#print "i is "+str(i)
		summary_file=open("summary_uttrakhand",'a')
		#print "hahahahaahah"
		#print len(fts)
		#print len(list_tweets)
		#print "heheheheheeheh"
		Summary=[]
		TWEET=[]
		TWEET_maxscores=[]
		cluster_size=[]
		sizes=[]
		maximum=find_biggest_cluster(dataStr.TCV.clusterList)
		for j in dataStr.TCV.clusterList:
			lexrank_scores=[]
			cluster_tweets=[]
			
			size=j.n
			
			for k in j.ft_set:
				#print k.tweetStr
				cluster_tweets.append(k.tweetStr)
				indx=list_tweets.index(k.tweetStr)
				score=fts[indx].score
				lexrank_scores.append(score)
				TWEET.append(k.tweetStr)
				sizes.append(size)
			maximum1=max(lexrank_scores)
			for k in cluster_tweets:
				if fts[list_tweets.index(k)].score==maximum1:
					TWEET_maxscores.append(k)
					cluster_size.append(size)
		#print len(TWEET_maxscores)
		#print len(cluster_size)
		while len(Summary)<l:
			maximum_t=[]
			for t in TWEET_maxscores:
				#print "index="+str(list_tweets.index(t))
				#print list_tweets
				temp=float((0.4*cluster_size[TWEET_maxscores.index(t)]/maximum)*fts[list_tweets.index(t)].score-0.6*avg_similarity(t,Summary))
				maximum_t.append(temp)
			temp_max=max(maximum_t)
			for  t in TWEET_maxscores:
				if maximum_t[TWEET_maxscores.index(t)]==temp_max:
					Summary.append(t)
			#ummary=list(set(Summary))
		#TWEET=set(TWEET)
		#print "size tweet"+str(len(TWEET))
		#TWEET=list(TWEET)
		Summary=list(set(Summary))
		#print "size tweet"+str(len(TWEET))+str(len(Summary))
		while len(Summary)<l:
			substracted_tweets,temp_size=list_substraction(Summary,TWEET,sizes)
			maximum_t=[]
			for  t in substracted_tweets:
				temp=float((0.4*temp_size[substracted_tweets.index(t)]/maximum)*fts[list_tweets.index(t)].score-0.6*avg_similarity(t,Summary))
				maximum_t.append(temp)
				#print temp
			temp_max=max(maximum_t)
			print "ma xis"+str(temp_max)
			for  t in substracted_tweets:
				if maximum_t[substracted_tweets.index(t)]==temp_max:
					Summary.append(t)
			Summary=list(set(Summary))
			if len(Summary)>l:
				Summary=Summary[0:5]
		#print "ft set"
		#print TWEET
		Summary=list(set(Summary))
		runtime=time.time() - start_time
		#print "#######"
		#print Summary
		#print len(Summary)
		#print "#######"
		if i==2825:
			summary_file.write("\nFinal Summary\n")
		summary_file.write("Number of Tweets : "+str(i)+"\n"),
		for line in Summary:
			summary_file.write(str(line)+"\n")
		#summary_file.writelines(Summary)
		summary_file.write(str(len(Summary)))
		summary_file.write("\n")
		summary_file.write("Running time ="+str(runtime)+"\n\n\n")
		summary_file.close()
	i += 1

#for tweet in dataStr.TWEET.tweetList:
#	print tweet.tweetWordDict

fOUT = open('ft_set','w')
#print dataStr.TCV.numOfCLuster
for cluster in dataStr.TCV.clusterList:
	#print cluster.clusterId
	#print cluster.n
	#print cluster.sum_v
	#print cluster.ft_set
	#for tweet in cluster.ft_set:
	#	print tweet.id
	#print cluster.wsum_v
	if len(cluster.ft_set) > 39:
		for tweet in cluster.ft_set:
			fOUT.write(tweet.tweetStr.encode('utf-8',errors='ignore'))
			fOUT.write('\n')
			#print tweet.id
		fOUT.write('\n\n\n\n\n')

fOUT.close()
###############################################################