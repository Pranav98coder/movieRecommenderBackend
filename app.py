from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np
import os
app = Flask(__name__)
CORS(app)  

movie_dict_path = os.path.join(BASE_DIR, 'movie_dict.pkl')

similarity_nz_path = os.path.join(BASE_DIR, 'similarity.npz')


movies_dict = pickle.load(open(movie_dict_path, 'rb'))
movies = pd.DataFrame(movies_dict)


with np.load(similarity_nz_path) as data:
    similarity = data['matrix']




@app.route('/api/movies', methods=['GET'])
def get_movie_list():
    """Returns a full list of available movie titles for the autocomplete dropdown."""
    titles = movies['title'].tolist()
    return jsonify(titles)

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """Handles recommendation calculation requests from the frontend."""
    data = request.json
    selected_movie = data.get('movie')

    if not selected_movie:
        return jsonify({'success': False, 'error': 'No movie selected'}), 400

    try:
        
        mov_ind = movies[movies['title'] == selected_movie].index[0]
        distance = similarity[mov_ind]
        
        
        lis = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
        
        recommendations = []
        for i in lis:
            recommendations.append({
                'id': int(movies.iloc[i[0]].movie_id),
                'title': movies.iloc[i[0]].title
            })
            
        return jsonify({'success': True, 'data': recommendations})
        
    except IndexError:
        return jsonify({'success': False, 'error': 'Movie not found in database'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    
    app.run(debug=True, port=5000)