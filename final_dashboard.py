import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image


# Load cleaned data
data_cleaned = pd.read_csv('cleaned_data.csv')

# Sidebar menu
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", [
    "👥 About Us", 
    "📖 Introduction", 
    "📊 Exploratory Data Analysis", 
    "🔮 Prediction"
])

# About Us section
if option == "👥 About Us":
    st.title("👥 About Us")
    st.write("""
    Welcome to our Salary Analysis Dashboard. We are a team of data enthusiasts dedicated to helping students and professionals understand salary trends and make informed career decisions. This dashboard was developed to provide insights into the job market based on various factors like job roles, age, gender, years of experience, and educational background.

    This dashboard provides an in-depth analysis of salary data across various job roles and categories. Our goal is to help you understand the trends and patterns in the job market.
        
    This analysis includes:
    - **Exploratory Data Analysis (EDA)**: Visualizing and understanding the data.
    - **Predictive Modeling**: Building a model to predict salaries based on various factors.
        
    **Let's dive in!**

    Our goal is to equip you with the knowledge to navigate your career path confidently. Thank you for using our dashboard!
    """)

    
    st.header("Our Team")
    
    # Add team member photos and names
    team_members = [
        {"name": "Shubashenee Shashi Kumar", "photo": "shuba.jpeg"},
        {"name": "Enoch Leong Qi Cong", "photo": "enoch.jpg"},
    ]
    # Create columns to place images side-by-side
    cols = st.columns(len(team_members))

    for col, member in zip(cols, team_members):
        with col:
            member_image = Image.open(member['photo'])
            st.image(member_image, width=200)
            st.write(member['name'])

    # Centering the images and names
    st.markdown(
        """
        <style>
        .block-container {
            display: flex;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True

    )

# Introduction section
elif option == "📖 Introduction":
    st.title("📖 Introduction")
    st.write("""
    Ever noticed how some interviewers seem to squirm when the topic of salary comes up during an interview? 
    It’s almost like mentioning Voldemort in the wizarding world – a taboo subject! 
    Some interviewers might even tell you that discussing salary is against company policy, which can be frustrating.
    """)

    # Add the image
    st.image("salary.jpeg", caption="Discussing salary during the interview? Not always a comfortable topic!")

    # Add the description
    st.write("""
    But fear not! This dashboard is here to help you navigate the tricky waters of salary expectations. 
    By analyzing data on job roles, age, gender, years of experience, and education level, we can give you 
    a better idea of what you can expect to earn. So, next time you’re preparing for an interview, 
    use this tool to have a well-informed discussion about your potential salary, and avoid those awkward moments!
    """)

    # Displaying the dataset overview
    st.subheader("Dataset Overview")
    st.write(data_cleaned.head())
    

    # create the plot
    fig = px.scatter(data_cleaned, x='Age', y='Salary', color='Years of Experience',
                 size='Salary', hover_data=['Job Title', 'Industry'],
                 title='Salary by Age and Years of Experience<br><span style="font-weight: normal;">This scatter plot below depicts the relationship between age, salary, and years of experience.</span>',
                 labels={'Age': 'Age', 'Salary': 'Salary', 'Years of Experience': 'Years of Experience'})

    
    # Display the plot in Streamlit
    st.plotly_chart(fig)

    st.write("""
    ~ It is potrayed that in the above graph, as age and years of experience increase, salaries generally rise. However, this growth tends to plateau around the age of 50.
    """) 
    
    # Poll question
    st.subheader("Poll Question")
    st.markdown("**In our dataset, which industry do you think has the highest paid job?**")

    options = ["Computing", "Business", "Software Engineering", "Finance", "Human Resource", "Engineering", "Mass Media", "Scientist"]
    correct_answer = "Mass Media"
    
    # User selection
    user_answer = st.radio("Select an answer:", options)

    # Check answer
    if st.button("Submit"):
        if user_answer == correct_answer:
            st.success(f"Correct! The highest paid industry is {correct_answer}.")
        else:
            st.error(f"Incorrect. The highest paid industry is {correct_answer}.")

    # Interactive text with button
    st.subheader("Interesting Fact?")
    if st.button('Click to reveal an interesting fact'):
        st.markdown("""
        <div style="background-color:#000; padding: 10px; border-radius: 10px; position: relative; color: white;">
            <p style="color: #39ff14; font-size: 18px; font-family: 'Courier New', Courier, monospace; text-align: center; margin: 0;">
                &#10148; Did you know? Advanced degrees (Master's Degree, PhD) generally lead to higher salaries, reinforcing the value of higher education in career advancement.
            </p>
        </div>
        """, unsafe_allow_html=True)

      
# Exploratory Data Analysis (EDA) section
elif option == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("""
        Welcome to the Exploratory Data Analysis Dashboard! This dashboard will take you on an exploratory journey through the salary landscape across various job roles, industries, and demographics. Let's dive into the data and uncover some fascinating insights about salaries.
    """)

    # Chapter 1
    st.title("Discovering the Top Earners")

    # Calculate the average salary for each job title within each category
    avg_salary_by_job = data_cleaned.groupby(['Category', 'Job Title'])['Salary'].mean().reset_index()

    # Sort data by average salary within each category
    avg_salary_by_job_sorted = avg_salary_by_job.sort_values(['Category', 'Salary'], ascending=[True, False])

    # Get top 10 job roles by average salary within each category
    top_10_avg_salary_by_category = avg_salary_by_job_sorted.groupby('Category').head(10)

    # Create the bar plot
    fig1 = px.bar(top_10_avg_salary_by_category, x='Job Title', y='Salary', color='Category', title='Top 10 Job Roles by Average Salary within Each Category', labels={'Job Title': 'Job Title', 'Salary': 'Average Salary ($)'}, height=800)


    # Update layout for better visualization
    fig1.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),
        xaxis_title='Job Title',
        yaxis_title='Average Salary ($)',
        xaxis=dict(tickangle=45),
        yaxis=dict(tick0=0, dtick=5000)  # Adjusting y-axis increments by 5k
    )

    # Show the plot
    st.plotly_chart(fig1)
    st.write("""
    ~ The interactive chart above indicates that higher salary categories (t20) predominantly feature senior-level positions such as Chief Technical Officer and VP roles, while lower salary categories (b40) include roles like customer service and IT support.
    """) 

    
    # Chapter 2
    st.title("The Impact of Age on Earnings")
    # Create the interactive scatter plot with trend lines and custom colors
    fig2 = px.scatter(data_cleaned, x='Age', y='Salary', color='Gender', trendline='ols',
                 title='Salary vs Age with Trend Line by Gender',
                 labels={'Age': 'Age', 'Salary': 'Salary ($)', 'Gender': 'Gender'},
                 color_discrete_map={'female': 'deeppink', 'male': 'lightblue'})


    # Update layout for better visualization
    fig2.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),
        yaxis=dict(tick0=0, dtick=2000)  # Adjusting y-axis increments by 2k
    )

    # Show the plot
    st.plotly_chart(fig2)
    st.write("""
    ~ The scatter plot above illustrates that the older the age, the higher the salary tends to be, with men showing a slightly higher salary increase with age compared to women.
    """) 

    # Chapter 3
    st.title("Gender Dynamics in Career Experience")
    # Sort the data by 'Years of Experience' in descending order
    data_sorted = data_cleaned.sort_values(by='Years of Experience', ascending=False)

    # Create an interactive grouped bar chart using Plotly
    fig3 = px.bar(data_sorted, x='Category', y='Years of Experience', color='Gender', barmode='group',
                 title='Categories by Gender and Years of Experience',
                 labels={'Category': 'Category', 'Years of Experience': 'Years of Experience', 'Gender': 'Gender'},
                 template='plotly') 

    # Customize the layout to remove y-axis numbers
    fig3.update_layout(
        yaxis=dict(
            showticklabels=False  # Hide the tick labels on the y-axis
        )
    )

    # Update layout
    fig3.update_layout(
        yaxis=dict(showticklabels=False)
    )
    # Show the plot
    st.plotly_chart(fig3)
    st.write("""
    ~ From the box plot above, it is potrayed that as age and years of experience increase, salaries generally rise. However, this growth tends to plateau around the age of 50.
    """) 

    # Chapter 4
    st.title("The Role of Education")
    # Create an interactive scatter plot using Plotly with faceting
    fig4 = px.scatter(data_cleaned, x='Years of Experience', y='Salary', color='Education Level',
                     title='Salary Distribution by Education Level and Years of Experience',
                     labels={'Years of Experience': 'Years of Experience', 'Salary': 'Salary ($)', 'Education Level': 'Education Level'},
                     template='plotly', facet_col='Education Level', opacity=0.6, size_max=10)

    # Update layout for better visualization
    fig4.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),
        xaxis=dict(title='Years of Experience'),
        yaxis=dict(title='Salary ($)')
    )
    # Show the plot
    st.plotly_chart(fig4)
    st.write("""
    ~ Based on the faceted scatter plot, it is shown that men tend to have more years of experience in mid-career (m40) compared to women, while early-career (t20) and late-career (b40) experience is more evenly distributed between genders
    """) 

    # Chapter 5
    st.title("Industry Insights")
    # Calculate average salary by Education Level and Industry
    avg_salary = data_cleaned.groupby(['Education Level', 'Industry'])['Salary'].mean().reset_index()

    # Create an interactive heatmap
    fig5 = px.density_heatmap(avg_salary, x='Education Level', y='Industry', z='Salary',
                             title='Average Salary Heatmap by Education Level and Industry',
                             labels={'Education Level': 'Education Level', 'Industry': 'Industry', 'Salary': 'Average Salary ($)'})


    # Update layout for better visualization
    fig5.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),
        xaxis=dict(title='Education Level'),
        yaxis=dict(title='Industry'),
        coloraxis_colorbar=dict(title='Average Salary ($)')
    )
    # Show the plot
    st.plotly_chart(fig5)
    st.write("""
    ~ According to the heatmap, higher levels of education correlate with higher salaries and more years of experience, with PhD holders having the highest salaries and most years of experience, followed by those with a master's degree, and then bachelor's degree holders.
    """) 

    

# Prediction section
elif option == "🔮 Prediction":
    st.title("🔮 Salary Prediction")

    # Feature selection
    X = data_cleaned[['Age', 'Job Title', 'Years of Experience', 'Education Level', 'Gender']]
    y = data_cleaned['Salary']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define categorical and numerical columns
    categorical_cols = ['Job Title', 'Education Level', 'Gender']
    numerical_cols = ['Years of Experience', 'Age']

    # Preprocessing pipelines for both numerical and categorical data
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Define the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Create and evaluate the pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    #st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    #st.write(f"R-squared (R²): {r2:.2f}")

    st.header("Make a Prediction")
   # Select industry
    industry = st.selectbox('Select Industry', data_cleaned['Industry'].unique())

    # Filter job roles based on selected industry
    filtered_data = data_cleaned[data_cleaned['Industry'] == industry]
    job_title = st.selectbox('Select Job Title', filtered_data['Job Title'].unique())

    age = st.slider('Age', min_value=18, max_value=65, value=30)
    years_experience = st.slider('Years of Experience', min_value=0, max_value=40, value=5)
    education_level = st.selectbox('Education Level', data_cleaned['Education Level'].unique())
    gender = st.selectbox('Gender', data_cleaned['Gender'].unique())

    if st.button('Predict Salary'):
        input_data = pd.DataFrame([[age, job_title, years_experience, education_level, gender]], columns=X.columns)
        predicted_salary = pipeline.predict(input_data)[0]
        st.write(f'The predicted salary for a {job_title} in {industry} with {age} years old, {years_experience} years of experience, {education_level}, and {gender} is: RM {predicted_salary:.2f}')

      # Create data for the graph
        ages = list(range(age - years_experience, 51))
        experience = list(range(0, 51 - (age - years_experience)))
        predictions = []

        for exp in experience:
            temp_data = pd.DataFrame([[age - years_experience + exp, job_title, exp, education_level, gender]], columns=['Age', 'Job Title', 'Years of Experience', 'Education Level', 'Gender'])
            pred = pipeline.predict(temp_data)[0]
            predictions.append(pred)

        # Create a DataFrame for the graph
        graph_data = pd.DataFrame({
            'Age': ages,
            'Years of Experience': experience,
            'Predicted Salary': predictions
        })
        st.subheader("Come Have a Look At The Visual Representation of Your Prediction")

        # Create the interactive plot
        fig = px.line(graph_data, x='Age', y='Predicted Salary', title='Salary Growth Over Time',
                      labels={'Age': 'Age', 'Predicted Salary': 'Predicted Salary (RM)'},
                      hover_data={'Years of Experience': True, 'Age': True, 'Predicted Salary': ':.2f'},
                      markers=True)

        # Add "You are here" marker
        fig.add_scatter(x=[age], y=[predicted_salary], mode='markers+text', marker=dict(color='red', size=10),
                        text=['You are here'], textposition='top center')

        # Show the plot
        st.plotly_chart(fig)
