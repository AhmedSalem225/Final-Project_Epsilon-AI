import streamlit as st
import seaborn as sns
import time
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objs as go


url = "https://github.com/AhmedSalem225/Final-Project_Epsilon-AI/raw/f50d6074776bbc9a1792592029363a6d40c6a071/models/pages/Data/Sampled_finish_Clean_Data.csv"
df = pd.read_csv(url)


st.markdown("<h1 style='text-align: center; color: white;'>Choose Category</h1>", unsafe_allow_html=True)
tab1, tab2= st.tabs(["Univariant Analysis", "bivariant Analysis"])


with tab1:
#============================================================ Top 10 State with Accidents

    k= df.groupby("State")["ID"].count().reset_index()
    l= k.sort_values("ID", ascending=False)
    #=====================================

    st.markdown("<h1 style='text-align: center; color: white;'>Top 10 States with Accidents</h1>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x="State", y="ID", data=l.head(10), ax=ax)
    ax.set_xlabel("State")
    ax.set_ylabel("Number of Accidents")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Display the plot in Streamlit
    st.pyplot(fig)


#============================================================ lowest 10 States with Accidents

    st.markdown("<h1 style='text-align: center; color: white;'>lowest 10 States with Accidents</h1>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x="State", y="ID", data=l.tail(10), ax=ax)
    ax.set_xlabel("State")
    ax.set_ylabel("Number of Accidents")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Display the plot in Streamlit
    st.pyplot(fig)

#============================================================ Severity Distribution Analysis
    # Streamlit app
    st.markdown("<h1 style='text-align: center; color: white;'>Severity Distribution Analysis</h1>", unsafe_allow_html=True)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # Graph 1: Seaborn count plot
    sns.countplot(x="Severity", data=df, ax=axs[0])
    axs[0].set_title('Count of Accidents by Severity')
    axs[0].set_xlabel('Severity')
    axs[0].set_ylabel('Count')

    # Graph 2: Pie chart
    labels = df["Severity"].value_counts().index
    sizes = df["Severity"].value_counts().values
    colors = ["green", "red", "blue", "black"]

    # The pie chart must be created in a separate figure because it does not fit in the subplot
    axs[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.5))
    axs[1].legend(labels, loc="center right", bbox_to_anchor=(1, 0.5))
    axs[1].set_title('Severity Distribution')

    # Display the plots in Streamlit
    st.pyplot(fig)      # Display the count plot




    st.markdown("""
        - It appears that California has the highest numbers of accidents on the other hand Rhode Island has the lowest accidents
        - Sevrity graph shows that most of the accidents are second-degree accidents with a high percentage nearly 80% of the total accidents
        """)


#============================================================ Temperature Distribution Analysis

    st.markdown("<h1 style='text-align: center; color: white;'>Temperature Distribution Analysis</h1>", unsafe_allow_html=True)

    # Create a distribution plot with Seaborn
    plt.figure(figsize=(10, 9))
    sns.histplot(df['Temperature(F)'], color='g', bins=100, kde=True, alpha=0.4)

    # Set titles and labels
    plt.title('Temperature Distribution with Density Plot')
    plt.xlabel('Temperature (°F)')
    plt.ylabel('Frequency')

    # Display the plot in Streamlit
    st.pyplot(plt)

#============================================================ Count Plot of Accidents by Month

    st.markdown("<h1 style='text-align: center; color: white;'>Count Plot of Accidents by Month</h1>", unsafe_allow_html=True)
    # Create a count plot using Seaborn
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    sns.countplot(data=df, y='Month_of_accident', order=df['Month_of_accident'].value_counts().index)

    # Add titles and labels (optional)
    plt.xlabel('Count')
    plt.ylabel('Month')

    # Show the plot in Streamlit
    st.pyplot(plt)

#============================================================ Number of Accidents by Hour of the Day

    st.markdown("<h1 style='text-align: center; color: white;'>Number of Accidents by Hour of the Day</h1>", unsafe_allow_html=True)
    # Prepare the data
    hourly_counts = df['Hour_of_starting'].value_counts().reset_index()
    hourly_counts.columns = ['Hour', 'Count']
    hourly_counts['Hour'] = hourly_counts['Hour'].apply(lambda x: str(x) + ':00')

    # Create the figure
    fig = go.Figure()

    # Add the line plot
    fig.add_trace(go.Scatter(
        x=hourly_counts['Hour'],
        y=hourly_counts['Count'],
        mode='lines+markers',
        line=dict(shape='spline'),
    ))

    # Update layout
    fig.update_layout(
        xaxis_title="Hour of the Day",
        yaxis_title="Number of Accidents",
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)
    st.markdown("""
    - It appears that the probabilty of accidents decreases over the day time as it is highest at the begining of the day
    - It seems that most of the accidents occurred in January, with approximately 25,000 accidents, while the rest of the months had less than 25,000
    - As the temperature rises, the number of accidents increases , up to 80 Fahrenheit, the number of accidents decreases significantly
    """)




with tab2:
#============================================================ Correlation matrix
    # Filter columns that have float or integer data types
    numeric_columns = df.select_dtypes(include=['float', 'int']).columns
    
    # Create a new DataFrame with only numeric columns
    df_numeric = df[numeric_columns]
    
    # Compute the correlation matrix for numeric columns
    correlation_matrix = df_numeric.corr()

    # Create the heatmap
    st.markdown("<h1 style='text-align: center; color: white;'>Correlation Matrix Heatmap</h1>", unsafe_allow_html=True)

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(30, 20))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix', size=20)

    # Display the heatmap in Streamlit
    st.pyplot(fig)

#============================================================ Accident Data Visualization

    # Streamlit app
    st.markdown("<h1 style='text-align: center; color: white;'>Accident Data Visualization</h1>", unsafe_allow_html=True)

    # First graph: Histogram of Year_of_accedent colored by Season
    fig1 = px.histogram(df, x="Year_of_accedent", hover_data=df.columns, color="Season",
                        title="Histogram of Year of Accident by Season")
    # second graph: 
    fig = px.pie(df, 
                names='Year_of_accedent', 
                title='Distribution of Accidents by Year',
                color='Year_of_accedent', 
                color_discrete_sequence=px.colors.qualitative.Set1)




    # third graph: Histogram of Season
    fig2 = px.histogram(df, x="Season", hover_data=df.columns,
                        title="Histogram of Seasons")

    # Display the first histogram
    st.plotly_chart(fig1)

    # Show the pie chart in Streamlit
    st.plotly_chart(fig)

    # Display the second histogram
    st.plotly_chart(fig2)

#============================================================ Number of Accidents by State


    # Prepare the data for the choropleth map
    state_counts = df["State"].value_counts()

    # Create the choropleth map using Plotly
    fig = go.Figure(data=go.Choropleth(
        locations=state_counts.index, 
        z=state_counts.values.astype(float), 
        locationmode="USA-states", 
        colorscale="turbo",

    ))

    # Update layout
    fig.update_layout(
        title_text="Number of Accidents by State",
        geo_scope="usa",
        geo=dict(
            lakecolor='rgb(255, 255, 255)',  # Background color for lakes
            projection_scale=5,  # Zoom level
            showland=True,
            landcolor='rgb(255, 255, 255)',
            subunitcolor='rgb(217, 217, 217)',
            countrycolor='rgb(217, 217, 217)',
        )
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)


    #==============================
    st.markdown("""
    - Number of accidents increased significantlly over the year 2022 (more than 40% of the total accidents in this year)
    - The number of accidents was small until 2019, as they increased significantly during the years 2020 and 2021, then reached the maximum number of accidents in 2022, then declined again.
    - Winter has the most no of accidents
    - Heat map shows the distribution of accidents over the states of America 
    """)


