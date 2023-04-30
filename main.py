import pandas as pd
import numpy as np
import torch
import random
from sklearn.neighbors import NearestNeighbors

from flask import Flask, request, jsonify, abort

app = Flask(__name__)

seed = 777
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


@app.route('/country', methods=['POST'])
def get_country_data():
    data = request.get_json()
    country_name = data.get('country_name')
    attraction_names = data.get('attractions')
    num_days = data.get('days')

    if country_name == 'Spain':
        excel_file = '스페인(test).xlsx'
    elif country_name == 'Italy':
        excel_file = 'Italy(test).xlsx'
    elif country_name == 'British':
        excel_file = 'British.xlsx'
    else:
        error_message = {"message": "Country not supported."}
        return jsonify(error_message), 400

    seed = 777
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    rating = pd.read_excel(excel_file)
    rating_with_category = rating[['user', 'attraction', 'rating', 'feature']]
    user_item_matrix = rating_with_category.pivot_table("rating", "user", "attraction").fillna(0)
    categories = pd.get_dummies(rating_with_category['feature'], prefix='feature')

    knn = KNNModel(user_item_matrix)

    user_id = 1000
    user_id += 1

    new_user_ratings = get_user_input(user_id, attraction_names)

    knn_recommendations = knn.recommend(user_id, num_days, new_user_ratings=new_user_ratings)

    # 결과를 담을 빈 리스트를 생성합니다.
    recommendations = []

    # Replace the following loop
    # for attraction, prediction in knn_recommendations:
    # with the following loop
    for attraction in knn_recommendations:
        recommendations.append(attraction)

    # Return the recommendations wrapped in a JSON object with the key 'attractions'
    return jsonify({"attractions": recommendations})


# Create a class for KNN model
def add_new_user_ratings(user_item_matrix, new_user_ratings):
    new_user_df = pd.DataFrame(new_user_ratings, columns=["rating", "user", "attraction"])
    new_user_pivot = new_user_df.pivot_table("rating", "user", "attraction").reindex(columns=user_item_matrix.columns,
                                                                                     fill_value=0)
    updated_user_item_matrix = pd.concat([user_item_matrix, new_user_pivot.reindex(columns=user_item_matrix.columns)],
                                         ignore_index=True).fillna(0)
    return updated_user_item_matrix


class KNNModel:
    def __init__(self, user_item_matrix, k=8):
        self.user_item_matrix = user_item_matrix
        self.k = k
        self.model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=self.k + 1)
        self.model.fit(user_item_matrix)

    def recommend(self, user_id, num_days, new_user_ratings=None):
        # New user case
        if new_user_ratings is not None:
            updated_user_item_matrix = add_new_user_ratings(self.user_item_matrix, new_user_ratings)
            new_user_index = updated_user_item_matrix.shape[0] - 1
            self.model.fit(updated_user_item_matrix)
            distances, indices = self.model.kneighbors(
                updated_user_item_matrix.loc[new_user_index].values.reshape(1, -1), n_neighbors=self.k + 1)
            closest_users = updated_user_item_matrix.iloc[indices.flatten()[1:], :]

        # Existing user case
        else:
            distances, indices = self.model.kneighbors(self.user_item_matrix.loc[user_id].values.reshape(1, -1),
                                                       n_neighbors=self.k + 1)
            closest_users = self.user_item_matrix.iloc[indices.flatten()[1:], :]

        mean_ratings = closest_users.mean(axis=0)
        top_n = num_days * 4
        recommended_attractions = mean_ratings.nlargest(top_n).index.tolist()

        return recommended_attractions[:top_n]


def get_user_input(user_id, attraction_names):
    user_ratings = []
    for attraction in attraction_names:
        user_ratings.append((5, user_id, attraction))
    return user_ratings


if __name__ == '__main__':
    app.run(debug=True)
