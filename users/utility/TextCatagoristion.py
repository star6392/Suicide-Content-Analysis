import praw
from os import path
import wordcloud
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import csv
import requests
import json
from django.conf import settings
class ProcessTextCatagorisation:
    fieldsForAuthor = ["author", "created_utc"]
    fields = ["author", "brand_safe", "contest_mode", "created_utc", "full_link", "id", "is_self", "num_comments", "over_18", "retrieved_on", "score", "selftext", "subreddit", "subreddit_id", "title"]
    fieldsForCommonAuthor = ["author", "generalIssues_created_utc", "suicideWatch_created_utc"]

    def extractMentalHealth(self,reddit):
        print("Creating a text file containing all mentalhealth.")
        path = settings.MEDIA_ROOT + "\\" + "mentalHealth.txt"
        with open(path, 'w', encoding="utf-8") as writeReviews:
            mentalHealthSubreddit = reddit.subreddit('mentalhealth')
            top_subreddit = mentalHealthSubreddit.top()
            for entry in mentalHealthSubreddit.top(limit=2000):
                # print(entry.title)
                writeReviews.write(entry.title + "\n")
            pass
        writeReviews.close()
        print("Created a text file named mentalhealth.txt")

    def wordCloud(self,filename):
        print("Creating Word Cloud for Mental Health")
        d = path.dirname(__file__)
        mentalhealth = open(path.join(d, filename), encoding="utf-8").read()
        mentalhealthImage = np.array(Image.open(path.join(d, "mentalhealth.png")))
        stopwords = set(wordcloud.STOPWORDS)
        stopwords.add("english")  # to get rid of the most common words like "the", "it", "of" etc
        wc = wordcloud.WordCloud(background_color="white", max_words=2000, mask=mentalhealthImage, max_font_size=40,
                                 stopwords=stopwords, random_state=42)
        wc.generate(mentalhealth)
        print("Done generating words!")
        mental_health_colors = wordcloud.ImageColorGenerator(mentalhealthImage)
        plt.imshow(wc.recolor(color_func=mental_health_colors), interpolation="bilinear")
        plt.axis("off")
        print("Created Word Cloud for Mental Health")
        # plt.figure()
        plt.show()

    def wordCloud2(self,filename):
        print("Creating Word Cloud for Suicide Watch")
        d = path.dirname(__file__)
        suicidewatch = open(path.join(d, filename), encoding="utf-8").read()
        suicidewatchImage = np.array(Image.open(path.join(d, "suicide.png")))
        stopwords = set(wordcloud.STOPWORDS)
        stopwords.add("english")  # to get rid of the most common words like "the", "it", "of" etc
        wc = wordcloud.WordCloud(background_color="white", max_words=2000, mask=suicidewatchImage, max_font_size=40,
                                 stopwords=stopwords, random_state=42)
        wc.generate(suicidewatch)
        suicide_colors = wordcloud.ImageColorGenerator(suicidewatchImage)
        plt.imshow(wc.recolor(color_func=suicide_colors), interpolation="bilinear")
        plt.axis("off")
        print("Created Word Cloud for Suicide Watch")
        # plt.figure()
        plt.show()

    def extractSuicidalWatch(self,reddit):
        print("Creating a text file containing all SuicideWatch.")
        path = settings.MEDIA_ROOT + "\\" + "suicidewatch.txt"
        with open(path, 'w', encoding="utf-8") as writeReviews:
            suicideWatchSubreddit = reddit.subreddit('suicidewatch')
            top_subreddit = suicideWatchSubreddit.top()
            for entry in suicideWatchSubreddit.top(limit=2000):
                # print(entry.title)
                writeReviews.write(entry.title + "\n")
            pass
        writeReviews.close()
        print("Created a text file named suicidewatch.txt")

    def extractMentalHealthCSV(self,start, end):
        fields = ["author", "brand_safe", "contest_mode", "created_utc", "full_link", "id", "is_self", "num_comments",
                  "over_18", "retrieved_on", "score", "selftext", "subreddit", "subreddit_id", "title"]
        with open('mentalHealthTS1.csv', mode='a', encoding="utf-8") as fileObject:
            csvWriter = csv.writer(fileObject, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvWriter.writerow(fields)
            for delta in range(1, 365):
                start += datetime.timedelta(days=1)
                end += datetime.timedelta(days=1)
                epoch1 = int(time.mktime(start.timetuple()))
                epoch2 = int(time.mktime(end.timetuple()))
                mentalHealth = "https://api.pushshift.io/reddit/search/submission/?after={0}&before={1}&size={2}&subreddit={3}".format(
                    epoch1, epoch2, '1000', 'mentalhealth')
                generalIssues = "https://api.pushshift.io/reddit/search/submission/?after={0}&before={1}&size={2}&subreddit={3}".format(
                    epoch1, epoch2, '1000',
                    'mentalhealth,depression,traumatoolbox,bipolarreddit,BPD,ptsd,psychoticreddit,EatingDisorders,StopSelfHarm,survivorsofabuse,rapecounseling,hardshipmates,panicparty,socialanxiety')
                suicideWatch = "https://api.pushshift.io/reddit/search/submission/?after={0}&before={1}&size={2}&subreddit={3}".format(
                    epoch1, epoch2, '1000', 'suicidewatch')
                # url = "https://api.pushshift.io/reddit/search/submission/?after=1489208400&before=1502424000&size=40000&subreddit=mentalhealth"
                print("My Url is=", mentalHealth)

                data = requests.get(mentalHealth)
                data = data.json()
                count = 0
                for singlePost in data["data"]:
                    row = []
                    for field in fields:
                        row.append(singlePost.get(field, None))
                    count += 1
                    csvWriter.writerow(row)
                print(start, end, count)

    def extractAuthorsWithTimestamp(fromFile, toFile):
        fieldsForAuthor = ["author", "created_utc"]
        print("Extrating Authors from: ", fromFile, "to: ", toFile)
        tempSet = set()
        with open(fromFile, mode='r', encoding="utf-8") as fileReader:
            fileReader.readline()
            csvReader = csv.reader(fileReader, delimiter=',')
            for row in csvReader:
                if row[0] != "[deleted]":
                    tempSet.add(row[0])
        with open(toFile, mode='w', encoding="utf-8") as fileWriter:
            csvWriter = csv.writer(fileWriter, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvWriter.writerow([fieldsForAuthor[0]])
            for elem in tempSet:
                csvWriter.writerow([elem])
        print("Done Extracting!")

    def extractMHandSWcommonAuthors(generalIssuesFilename, suicideWatchFilename, commonAuthorsFilename):
        fieldsForCommonAuthor = ["author", "generalIssues_created_utc", "suicideWatch_created_utc"]
        print("Extracting Common Authors between: ", generalIssuesFilename, "and: ", suicideWatchFilename)
        swSet = set()
        with open(suicideWatchFilename, mode='r', encoding="utf-8") as swReader:
            swReader.readline()
            csvSWReader = csv.reader(swReader, delimiter=',')
            for row in csvSWReader:
                swSet.add(row[0])

        giSet = set()
        with open(generalIssuesFilename, mode='r', encoding="utf-8") as giReader:
            giReader.readline()
            csvGIReader = csv.reader(giReader, delimiter=',')
            for row in csvGIReader:
                giSet.add(row[0])

        common = swSet & giSet

        with open(commonAuthorsFilename, mode='w', encoding="utf-8") as commonWriter:
            csvWriter = csv.writer(commonWriter, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvWriter.writerow([fieldsForCommonAuthor[0]])
            for elem in common:
                csvWriter.writerow([elem])
        print("Done Extracting")

    def extractAllDataForCommonAuthors(postsFilename, commonAuthorsFilename, commonPostsFilename):
        fields = ["author", "brand_safe", "contest_mode", "created_utc", "full_link", "id", "is_self", "num_comments",
                  "over_18", "retrieved_on", "score", "selftext", "subreddit", "subreddit_id", "title"]
        commonAuthors = set()
        with open(commonAuthorsFilename, mode='r', encoding="utf-8") as commonReader:
            commonReader.readline()
            csvReader = csv.reader(commonReader, delimiter=',')
            for row in csvReader:
                commonAuthors.add(row[0])

        with open(commonPostsFilename, mode='w', encoding="utf-8") as commonWriter:
            csvWriter = csv.writer(commonWriter, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvWriter.writerow(fields)
            with open(postsFilename, mode='r', encoding="utf-8") as commonReader:
                commonReader.readline()
                csvReader = csv.reader(commonReader, delimiter=',')
                for row in csvReader:
                    if row[0] in commonAuthors:
                        csvWriter.writerow(row)

        print(len(commonAuthors))


