from flask import Flask, render_template, request
import pickle
import pandas as pd


app = Flask(__name__)

# Load the trained pipeline/model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Dropdown options for teams and cities
teams = [
    "Australia", "India", "Bangladesh", "New Zealand", "South Africa",
    "England", "West Indies", "Afghanistan", "Pakistan", "Sri Lanka"
]

cities = [
    "Colombo", "Mirpur", "Johannesburg", "Dubai", "Auckland", "Cape Town",
    "London", "Pallekele", "Barbados", "Sydney", "Melbourne", "Durban",
    "St Lucia", "Wellington", "Lauderhill", "Hamilton", "Centurion",
    "Manchester", "Abu Dhabi", "Mumbai", "Nottingham", "Southampton",
    "Mount Maunganui", "Chittagong", "Kolkata", "Lahore", "Delhi",
    "Nagpur", "Chandigarh", "Adelaide", "Bangalore", "St Kitts", "Cardiff",
    "Christchurch", "Trinidad"
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Collect form inputs
        batting_team = request.form.get('batting_team')
        bowling_team = request.form.get('bowling_team')
        city = request.form.get('city')
        current_score = int(request.form.get('current_score'))
        overs = float(request.form.get('overs'))
        wickets = int(request.form.get('wickets'))
        last_five = int(request.form.get('last_five'))

        # Derived features (must match training schema)
        balls_left = 120 - int(overs * 6)
        wicket_left = 10 - wickets
        current_run_rate = current_score / overs if overs > 0 else 0

        # Build input DataFrame with exact column names
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city],
            'current_score': [current_score],
            'balls_left': [balls_left],
            'wicket_left': [wicket_left],
            'current_run_rate': [current_run_rate],
            'last_five': [last_five]
        })

        # Predict using pipeline
        result = pipe.predict(input_df)
        prediction = int(result[0])

    return render_template(
        'index.html',
        teams=teams,
        cities=cities,
        prediction=prediction
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

 