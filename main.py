import falcon
import firebase_admin
from firebase_admin import firestore
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

from firebase_api_key import credential
from jobs import Jobs

# Download the punkt tokenizer for NLTK
nltk.download('punkt')

# Initialize Firebase
cred = credential
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

def get_recommendations(recommend_titles, avoid_titles, cosine_sim=cosine_similarity, num_recommend=10):
    # Get indices for recommendation and avoidance
    recommend_indices = indices[recommend_titles]
    avoid_indices = indices[avoid_titles]

    # Calculate similarity scores for recommended jobs
    sim_scores = []
    for idx in recommend_indices:
        sim_scores.extend(list(enumerate(cosine_sim[idx])))

    # Average the similarity scores
    sim_scores = pd.DataFrame(sim_scores, columns=['index', 'score'])
    sim_scores = sim_scores.groupby('index').mean().reset_index()

    # Sort the jobs based on the similarity scores
    sim_scores = sim_scores.sort_values(by='score', ascending=False)

    # Remove jobs that are similar to those in avoid_titles
    for idx in avoid_indices:
        sim_scores = sim_scores[sim_scores['index'] != idx]

    # Get the scores of the most similar jobs
    top_similar = sim_scores.head(num_recommend)

    # Get the job indices
    job_indices = top_similar['index'].values

    # Return the top most similar jobs
    return df['id'].iloc[job_indices]

if __name__ == "__main__":
    recommend_titles = ["Security Management Internship Summer 2024"]
    avoid_titles = ["Systems Integration Intern (Fall 2024)"]
    print(get_recommendations(recommend_titles, avoid_titles))
