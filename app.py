import streamlit as st
import pandas as pd
import pickle

from scipy.sparse import csr_matrix

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(

    page_title="Hybrid Recommendation System",

    layout="centered"

)

# ======================================================
# LOAD MODELS
# ======================================================

rf_model = pickle.load(
    open('rf_model.pkl', 'rb')
)

model_knn = pickle.load(
    open('knn_model.pkl', 'rb')
)

product_to_index = pickle.load(
    open('product_to_index.pkl', 'rb')
)

index_to_product = pickle.load(
    open('index_to_product.pkl', 'rb')
)

# ======================================================
# LOAD DATASET
# ======================================================

df = pd.read_csv('final_dataset.csv')

# ======================================================
# PRODUCT STATISTICS
# ======================================================

product_avg_rating = df.groupby(
    'prod_id'
)['ratings'].mean()

product_count = df.groupby(
    'prod_id'
)['ratings'].count()

user_avg_rating = df.groupby(
    'user_id'
)['ratings'].mean()

user_count = df.groupby(
    'user_id'
)['ratings'].count()

# ======================================================
# CREATE SPARSE MATRIX
# ======================================================

sparse_matrix = csr_matrix(

    (
        df['ratings'],

        (
            df['product_index'],
            df['user_index']
        )

    )

)

# ======================================================
# FIND SIMILAR PRODUCTS
# ======================================================

def get_similar_products(product_id, top_n=10):

    if product_id not in product_to_index:
        return pd.DataFrame()

    product_idx = product_to_index[
        product_id
    ]

    distances, indices = model_knn.kneighbors(

        sparse_matrix[product_idx],

        n_neighbors=top_n + 1

    )

    similar_products = []

    for i in range(1, len(indices.flatten())):

        idx = indices.flatten()[i]

        similar_product = index_to_product[idx]

        similarity_score = 1 - distances.flatten()[i]

        similar_products.append([

            similar_product,
            similarity_score

        ])

    return pd.DataFrame(

        similar_products,

        columns=[
            'prod_id',
            'similarity_score'
        ]

    )

# ======================================================
# USERS WHO LIKED PRODUCT
# ======================================================

def get_top_users(product_id):

    users = df[

        (df['prod_id'] == product_id) &
        (df['ratings'] >= 4)

    ]['user_id'].unique()

    return users

# ======================================================
# GENERATE CANDIDATES
# ======================================================

def generate_candidates(product_id):

    users = get_top_users(product_id)

    candidate_df = df[
        df['user_id'].isin(users)
    ]

    candidate_df = candidate_df[
        candidate_df['prod_id'] != product_id
    ]

    return candidate_df

# ======================================================
# HYBRID RECOMMENDATION
# ======================================================

def hybrid_recommendation(product_id, top_n=10):

    candidate_df = generate_candidates(
        product_id
    )

    if candidate_df.empty:
        return pd.DataFrame()

    similarity_df = get_similar_products(
        product_id,
        top_n=20
    )

    recommendation_rows = []

    for _, row in candidate_df.iterrows():

        user_id = row['user_id']

        candidate_product = row['prod_id']

        similarity_row = similarity_df[

            similarity_df['prod_id']
            == candidate_product

        ]

        if len(similarity_row) > 0:

            similarity_score = similarity_row[
                'similarity_score'
            ].values[0]

        else:

            similarity_score = 0

        avg_product_rating = product_avg_rating.get(
            candidate_product,
            0
        )

        total_product_interactions = product_count.get(
            candidate_product,
            0
        )

        avg_user_rating = user_avg_rating.get(
            user_id,
            0
        )

        total_user_interactions = user_count.get(
            user_id,
            0
        )

        features = pd.DataFrame([{

            'similarity_score': similarity_score,

            'avg_product_rating': avg_product_rating,

            'product_interactions': total_product_interactions,

            'avg_user_rating': avg_user_rating,

            'user_interactions': total_user_interactions

        }])

        score = rf_model.predict_proba(
            features
        )[0][1]

        recommendation_rows.append([

            candidate_product,
            score

        ])

    recommendation_df = pd.DataFrame(

        recommendation_rows,

        columns=[

            'recommended_product',
            'score'

        ]

    )

    recommendation_df = recommendation_df.groupby(

        'recommended_product'

    )['score'].mean().reset_index()

    recommendation_df = recommendation_df.sort_values(

        by='score',
        ascending=False

    )

    recommendation_df = recommendation_df.head(
        top_n
    )

    return recommendation_df

# ======================================================
# STREAMLIT UI
# ======================================================

st.title(
    "Hybrid Product Recommendation System"
)

st.write(
    "Enter Product ID and click the button"
)

# Product input
product_id = st.text_input(
    "Product ID"
)

# Button
if st.button("Get Recommendations"):

    if product_id == "":

        st.warning(
            "Please enter Product ID"
        )

    else:

        recommendations = hybrid_recommendation(

            product_id,

            top_n=10

        )

        if recommendations.empty:

            st.error(
                "No Recommendations Found"
            )

        else:

            st.success(
                "Top Recommendations"
            )

            st.dataframe(
                recommendations
            )