import streamlit as st

def create_sidebar(slide_bar):
    input_dict = {}
    
    # Adding an expander in the sidebar
    with st.sidebar.expander("Features"):
        for name, key, min_value, max_value, default_value in slide_bar:
            # Add bold font and emojis to the feature names
            formatted_name = f"**{name} 😊**"
            input_dict[key] = st.slider(
                formatted_name, 
                min_value=min_value, 
                max_value=max_value, 
                value=default_value
            )
    
    return input_dict

# Example usage
slide_bar = [
    ("Feature A", "feature_a", 0, 100, 50), 
    ("Feature B", "feature_b", 0, 50, 25),
    ("Feature C", "feature_c", 0, 200, 100),
]

sidebar_values = create_sidebar(slide_bar)
# Display the sidebar values
st.write("Sidebar values:", sidebar_values)
