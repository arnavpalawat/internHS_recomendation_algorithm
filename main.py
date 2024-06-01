import firebase_admin
from firebase_admin import credentials, firestore
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from jobs import Jobs

# Download the punkt tokenizer for NLTK
nltk.download('punkt')

# Initialize Firebase
cred = credentials.Certificate("intern-b54ae-firebase-adminsdk-f0340-e5dcc26685.json")
firebase_admin.initialize_app(cred)

# Initialize Firestore
db = firestore.client()

# Get FS data
jobs_ref = db.collection('jobs')
docs = jobs_ref.stream()

job_data = []

# Map FS data
for doc in docs:
    job_instance = Jobs.from_firebase(doc.to_dict())  # Create Jobs instance from Firebase data
    job_data.append(job_instance)  # Append instance to job_data list

for job in job_data:
    if not job.description:
        job.description = ''

# Convert to a Dataframe
job_dicts = [{'id': job.id, 'description': job.description, 'title': job.title} for job in job_data]

# Convert the list of dictionaries into a DataFrame
df = pd.DataFrame(job_dicts)
# Create a TfidfVectorizer
tfidf = TfidfVectorizer()
# Fit and transform the data to a tfidf matrix
tfidf_matrix = tfidf.fit_transform(df['description'])

cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df['title']).drop_duplicates()


def get_recommendations(title, cosine_sim=cosine_similarity, num_recommend=10):
    idx = indices[title]

    # Get the pairwsie similarity scores of all jobs with that job
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the jobs based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar jobs
    top_similar = sim_scores[1:num_recommend + 1]

    # Get the job indices
    movie_indices = [i[0] for i in top_similar]

    # Return the top 10 most similar jobs
    return df['id'].iloc[movie_indices]


if __name__ == "__main__":
    print(get_recommendations("Security Management Internship Summer 2024"))