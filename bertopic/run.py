from bertopic import BERTopic

from sklearn.datasets import fetch_20newsgroups

# docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
# print(type(docs))
# print(len(docs))
# print(docs[0])

with open('../bertopic_fun/review_abstract.txt','r') as f:
    content = f.readlines()

model = BERTopic(language="english")
topics, probabilities = model.fit_transform(content)
# print(topics)
# print(probabilities)
print(model.get_topic_freq())
print(model.get_topic(1))

model.visualize_topics()
model.visualize_distribution(probabilities[0])

