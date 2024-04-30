import streamlit as st
import plotly.graph_objects as go


def side_bar():

    slide_bar = slide_bar = {
        "Radius 📏": [
            ("Radius Mean", "radius_mean", 28.11, 6.981, 14.127291739894552),
            ("Radius SE", "radius_se", 2.873, 0.1115, 0.40517205623901575),
            ("Radius Worst", "radius_worst", 36.04, 7.93, 16.269189806678387),
        ],
        "Texture 🪶": [
            ("Texture Mean", "texture_mean", 39.28, 9.71, 19.289648506151142),
            ("Texture SE", "texture_se", 4.885, 0.3602, 1.2168534270650264),
            ("Texture Worst", "texture_worst", 49.54, 12.02, 25.677223198594024),
        ],
        "Perimeter 🔲": [
            ("Perimeter Mean", "perimeter_mean", 188.5, 43.79, 91.96903339191564),
            ("Perimeter SE", "perimeter_se", 21.98, 0.757, 2.8660592267135327),
            ("Perimeter Worst", "perimeter_worst", 251.2, 50.41, 107.26121265377857),
        ],
        "Area 🗺️": [
            ("Area Mean", "area_mean", 2501.0, 143.5, 654.8891036906855),
            ("Area SE", "area_se", 542.2, 6.802, 40.337079086116),
            ("Area Worst", "area_worst", 4254.0, 185.2, 880.5831282952548),
        ],
        "Smoothness 🧼": [
            ("Smoothness Mean", "smoothness_mean", 0.1634, 0.05263, 0.0963602811950791),
            ("Smoothness SE", "smoothness_se", 0.03113, 0.001713, 0.007040978910369069),
            (
                "Smoothness Worst",
                "smoothness_worst",
                0.2226,
                0.07117,
                0.13236859402460457,
            ),
        ],
        "Compactness 📦": [
            (
                "Compactness Mean",
                "compactness_mean",
                0.3454,
                0.01938,
                0.10434098418277679,
            ),
            (
                "Compactness SE",
                "compactness_se",
                0.1354,
                0.002252,
                0.025478138840070295,
            ),
            (
                "Compactness Worst",
                "compactness_worst",
                1.058,
                0.02729,
                0.25426504393673116,
            ),
        ],
        "Concavity 🕳️": [
            ("Concavity Mean", "concavity_mean", 0.4268, 0.0, 0.0887993158172232),
            ("Concavity SE", "concavity_se", 0.396, 0.0, 0.03189371634446397),
            ("Concavity Worst", "concavity_worst", 1.252, 0.0, 0.27218848330404216),
        ],
        "Concave Points🔘": [
            (
                "Concave Points Mean",
                "concave_points_mean",
                0.2012,
                0.0,
                0.04891914586994728,
            ),
            (
                "Concave Points SE",
                "concave_points_se",
                0.05279,
                0.0,
                0.011796137082601054,
            ),
            (
                "Concave Points Worst",
                "concave_points_worst",
                0.291,
                0.0,
                0.11460622319859401,
            ),
        ],
        "Symmetry ⚖️": [
            ("Symmetry Mean", "symmetry_mean", 0.304, 0.106, 0.18116186291739894),
            ("Symmetry SE", "symmetry_se", 0.07895, 0.007882, 0.02054229876977153),
            ("Symmetry Worst", "symmetry_worst", 0.6638, 0.1565, 0.2900755711775044),
        ],
        "Fractal Dimension 🌀": [
            (
                "Fractal Dimension Mean",
                "fractal_dimension_mean",
                0.09744,
                0.04996,
                0.06279760984182776,
            ),
            (
                "Fractal Dimension SE",
                "fractal_dimension_se",
                0.02984,
                0.0008948,
                0.0037949038664323374,
            ),
            (
                "Fractal Dimension Worst",
                "fractal_dimension_worst",
                0.2075,
                0.05504,
                0.0839458172231986,
            ),
        ],
    }

    input_dict = {}

    # Adding an expander in the sidebar for each category
    for category, values in slide_bar.items():
        with st.sidebar.expander(f"**{category}**"):
            for name, key, max_value, min_value, default_value in values:
                input_dict[key] = st.slider(
                    name, min_value=min_value, max_value=max_value, value=default_value
                )
                
    return input_dict


def get_redar_chart(input_data):

    input_data = get_scale_data(input_data)

    categories = [
        "Radius",
        "Texture",
        "Perimeter",
        "Area",
        "Smoothness",
        "Compactness",
        "Concavity",
        "Concave Points",
        "Symmetry",
        "Fractal Dimension",
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=[
                input_data["radius_mean"],
                input_data["texture_mean"],
                input_data["perimeter_mean"],
                input_data["area_mean"],
                input_data["smoothness_mean"],
                input_data["compactness_mean"],
                input_data["concavity_mean"],
                input_data["concave_points_mean"],
                input_data["symmetry_mean"],
                input_data["fractal_dimension_mean"],
            ],
            theta=categories,
            fill="toself",
            name="Mean",
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=[
                input_data["radius_se"],
                input_data["texture_se"],
                input_data["perimeter_se"],
                input_data["area_se"],
                input_data["smoothness_se"],
                input_data["compactness_se"],
                input_data["concavity_se"],
                input_data["concave_points_se"],
                input_data["symmetry_se"],
                input_data["fractal_dimension_se"],
            ],
            theta=categories,
            fill="toself",
            name="Standard Error",
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=[
                input_data["radius_worst"],
                input_data["texture_worst"],
                input_data["perimeter_worst"],
                input_data["area_worst"],
                input_data["smoothness_worst"],
                input_data["compactness_worst"],
                input_data["concavity_worst"],
                input_data["concave_points_worst"],
                input_data["symmetry_worst"],
                input_data["fractal_dimension_worst"],
            ],
            theta=categories,
            fill="toself",
            name="Worst",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
        ),
        showlegend=True,
    )

    return fig


def get_scale_data(input_data):

    min_max = {
        "radius_mean": (6.981, 28.11),
        "texture_mean": (9.71, 39.28),
        "perimeter_mean": (43.79, 188.5),
        "area_mean": (143.5, 2501.0),
        "smoothness_mean": (0.05263, 0.1634),
        "compactness_mean": (0.01938, 0.3454),
        "concavity_mean": (0.0, 0.4268),
        "concave_points_mean": (0.0, 0.2012),
        "symmetry_mean": (0.106, 0.304),
        "fractal_dimension_mean": (0.04996, 0.09744),
        "radius_se": (0.1115, 2.873),
        "texture_se": (0.3602, 4.885),
        "perimeter_se": (0.757, 21.98),
        "area_se": (6.802, 542.2),
        "smoothness_se": (0.001713, 0.03113),
        "compactness_se": (0.002252, 0.1354),
        "concavity_se": (0.0, 0.396),
        "concave_points_se": (0.0, 0.05279),
        "symmetry_se": (0.007882, 0.07895),
        "fractal_dimension_se": (0.0008948, 0.02984),
        "radius_worst": (7.93, 36.04),
        "texture_worst": (12.02, 49.54),
        "perimeter_worst": (50.41, 251.2),
        "area_worst": (185.2, 4254.0),
        "smoothness_worst": (0.07117, 0.2226),
        "compactness_worst": (0.02729, 1.058),
        "concavity_worst": (0.0, 1.252),
        "concave_points_worst": (0.0, 0.291),
        "symmetry_worst": (0.1565, 0.6638),
        "fractal_dimension_worst": (0.05504, 0.2075),
    }
    
    for key, value in input_data.items():
        input_data[key] = (value - min_max[key][0]) / (min_max[key][1] - min_max[key][0])
        
    return input_data


def main():
    st.set_page_config(
        page_title="Breast Cancer Detection",
        page_icon="🎗",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
   

    input_data = side_bar()

    with st.container():
        st.title("Breast Cancer Detection")
        st.markdown(
            "Welcome to the Breast Cancer Detection web application. This app leverages machine learning to assist in detecting breast cancer using a variety of medical features. By providing user-adjustable sliders for key metrics like radius, texture, perimeter, area, smoothness, compactness, concavity, and others, users can explore different data points related to breast cancer."
        )

    col1, col2 = st.columns([4, 1])

    with col1:
        redar_chart = get_redar_chart(input_data)
        st.plotly_chart(redar_chart)
    with col2:
        st.write("## About")


if __name__ == "__main__":
    main()
