import pandas as pd
import numpy as np
import torch
import random
from sklearn.neighbors import NearestNeighbors

from flask import Flask, request, jsonify, abort
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

import os

app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///recommendations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

seed = 777
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


class AttractionRecommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    attraction_name = db.Column(db.String(100), nullable=False)
    latitude = db.Column(db.String(50), nullable=False)
    longitude = db.Column(db.String(50), nullable=False)


class SpainAttraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    attraction_name = db.Column(db.String(100), nullable=False)
    latitude = db.Column(db.String(50), nullable=False)
    longitude = db.Column(db.String(50), nullable=False)


class ItalyAttraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    attraction_name = db.Column(db.String(100), nullable=False)
    latitude = db.Column(db.String(50), nullable=False)
    longitude = db.Column(db.String(50), nullable=False)


class BritishAttraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    attraction_name = db.Column(db.String(100), nullable=False)
    latitude = db.Column(db.String(50), nullable=False)
    longitude = db.Column(db.String(50), nullable=False)


class FranceAttraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    attraction_name = db.Column(db.String(100), nullable=False)
    latitude = db.Column(db.String(50), nullable=False)
    longitude = db.Column(db.String(50), nullable=False)


@app.route('/country', methods=['POST'])
def get_country_data():
    data = request.get_json()
    user_id = data.get('id')
    country_name = data.get('countryName')
    num_days = data.get('days')
    attraction_names = data.get('spot')

    data_directory = os.path.join(os.path.dirname(__file__), 'data')

    if country_name == 'Spain':
        recommendations = SpainAttraction.query.filter_by(user_id=user_id).all()
    elif country_name == 'Italy':
        recommendations = ItalyAttraction.query.filter_by(user_id=user_id).all()
    elif country_name == 'British':
        recommendations = BritishAttraction.query.filter_by(user_id=user_id).all()
    elif country_name == 'France':
        recommendations = FranceAttraction.query.filter_by(user_id=user_id).all()
    else:
        error_message = {"message": "Country not supported."}
        return jsonify(error_message), 400

    if not recommendations:
        if country_name == 'Spain':
            excel_file = './data/스페인(test).xlsx'

        elif country_name == 'Italy':
            excel_file = './data/이탈리아(test).xlsx'

        elif country_name == 'British':
            excel_file = os.path.join(data_directory, 'British.xlsx')
            # excel_file = './data/British.xlsx'

        elif country_name == "France":
            excel_file = os.path.join(data_directory, 'France.xlsx')
            # excel_file = './data/France.xlsx'

        else:
            error_message = {"message": "Country not supported."}
            return jsonify(error_message), 400

        seed = 777
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        rating = pd.read_excel(excel_file)
        rating_with_category = rating[['user', 'attraction', 'rating', 'feature', 'latitude', 'longitude']]
        user_item_matrix = rating_with_category.pivot_table("rating", "user", "attraction").fillna(0)
        categories = pd.get_dummies(rating_with_category['feature'], prefix='feature')

        knn = KNNModel(user_item_matrix, rating_with_category)

        new_user_ratings = get_user_input(user_id, attraction_names)

        knn_recommendations = knn.recommend(user_id, num_days, new_user_ratings=new_user_ratings)

        recommendations = []

        for attraction, latitude, longitude in knn_recommendations:
            recommendations.append({
                "spot": attraction,
                "latitude": latitude,
                "longitude": longitude
            })

        # Save recommendations to database
        save_recommendations(user_id, recommendations, country_name)

        # 각 추천에 대해 num_days * 4개의 장소 분할
        spots = [recommendations[i:i + num_days * 4] for i in range(0, len(recommendations), num_days * 4)]

        result = {"id": user_id}
        for i, spot in enumerate(spots):
            result["recommend" + str(i + 1)] = spot

        return jsonify(result)

    else:
        recommendation_list = []
        for recommendation in recommendations:
            recommendation_list.append({
                'spot': recommendation.attraction_name,
                'latitude': recommendation.latitude,
                'longitude': recommendation.longitude
            })

        # 각 추천에 대해 num_days * 4개의 장소 분할
        spots = [recommendation_list[i:i + num_days*4] for i in range(0, len(recommendation_list), num_days*4)]

        result = {"id": user_id}
        for i, spot in enumerate(spots):
            result["recommend" + str(i + 1)] = spot

        return jsonify(result)


@app.route('/getURI', methods=['POST'])
def get_category_data():
    data = request.get_json()
    country_names = data.get('countryName')
    features = data.get('features')

    data_directory = os.path.join(os.path.dirname(__file__), 'data')

    spot = []

    for country_name in country_names:
        if country_name == 'Spain':
            # excel_file = os.path.join(data_directory, 'Spain_Category.xlsx')
            excel_file = './data/Spain_Category.xlsx'

        elif country_name == 'Italy':
            excel_file = './data/이탈리아(test).xlsx'

        elif country_name == 'British':
            # excel_file = os.path.join(data_directory, 'British_Category.xlsx')
            excel_file = './data/British_Category.xlsx'

        elif country_name == 'Switzerland':
            excel_file = './data/Switzerland_Category.xlsx'

        elif country_name == 'France':
            # excel_file = os.path.join(data_directory, 'France_Category.xlsx')
            excel_file = './data/France_Category.xlsx'

        else:
            continue  # Skip to the next country if the current one is not supported

        for feature in features:  # Iterate through the list of features
            attractions_by_feature = get_attractions_by_feature(feature, excel_file)

            # Check if the feature exists in the excel file
            if attractions_by_feature.empty:
                continue  # Skip to the next feature if the current one is not found

            for index, row in attractions_by_feature.iterrows():
                spot.append({
                    "spot": row['attraction'],
                    "feature": row['feature'],
                    "URI": row['URI'],
                    "countryName": country_name
                })

    # Return an error message if no features are found
    if not spot:
        error_message = {"message": f"No attractions found for the given features."}
        return jsonify(error_message), 400

    return jsonify({"spot": spot})


def get_attractions_by_feature(feature, excel_file):
    data = pd.read_excel(excel_file)
    filtered_data = data[data['feature'] == feature]
    return filtered_data[['attraction', 'feature', 'URI']]


def save_recommendations(user_id, recommendations, country):
    # Determine which model to use based on the country
    if country == 'Spain':
        model = SpainAttraction
    elif country == 'Italy':
        model = ItalyAttraction
    elif country == 'British':
        model = BritishAttraction
    elif country == 'France':
        model = FranceAttraction
    else:
        raise ValueError(f"Unsupported country: {country}")

    # Save recommendations to the appropriate table
    for recommendation in recommendations:
        attraction_recommendation = model(
            user_id=user_id,
            attraction_name=recommendation['spot'],
            latitude=recommendation['latitude'],
            longitude=recommendation['longitude']
        )
        db.session.add(attraction_recommendation)
    db.session.commit()


# Create a class for KNN model
def add_new_user_ratings(user_item_matrix, new_user_ratings):
    new_user_df = pd.DataFrame(new_user_ratings, columns=["rating", "user", "attraction"])
    new_user_pivot = new_user_df.pivot_table("rating", "user", "attraction").reindex(columns=user_item_matrix.columns,
                                                                                     fill_value=0)
    updated_user_item_matrix = pd.concat([user_item_matrix, new_user_pivot.reindex(columns=user_item_matrix.columns)],
                                         ignore_index=True).fillna(0)
    return updated_user_item_matrix


class KNNModel:
    def __init__(self, user_item_matrix, rating_with_category, k=8):
        self.user_item_matrix = user_item_matrix
        self.rating_with_category = rating_with_category
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
        top_n = num_days * 4 * 3
        recommended_attractions = mean_ratings.nlargest(top_n).index.tolist()

        # 위도와 경도 값을 찾기 위해 데이터프레임으로 변경
        attraction_location = self.rating_with_category[['attraction', 'latitude', 'longitude']].drop_duplicates()
        recommended_locations = attraction_location[attraction_location['attraction'].isin(recommended_attractions)]

        return list(zip(recommended_attractions[:top_n], recommended_locations['latitude'], recommended_locations['longitude']))


def get_user_input(user_id, attraction_names):
    user_ratings = []
    for attraction in attraction_names:
        user_ratings.append((5, user_id, attraction))
    return user_ratings


if __name__ == "__main__":
    with app.app_context():
        db.drop_all()  # Add this line to drop all tables before creating them
        db.create_all()
        app.run(host="0.0.0.0", port=5000)
