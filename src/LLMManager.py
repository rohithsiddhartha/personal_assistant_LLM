import openai
from openai import OpenAI


import os

class LLMManager:
    def __init__(self, model="gpt-3.5-turbo-0125"):
        """
        Initialize the LLMManager with the specified model.
        """
        self.client = OpenAI()
        self.model = model


    def process_profile(self, prompt):
        """
        Query OpenAI GPT with the given prompt and return the response - processed profile.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": """
                You are an assistant who is helping me extract all the profile information from the text passed. Extract the following profile information from the text:
                
                1. Personal Information
                Full Name:
                Date of Birth:
                Gender:
                Marital Status:
                Nationality:
                Languages Spoken:
                
                2. Contact Information
                Email Address:
                Phone Number:
                Mailing Address:
                Social Media Handles:
                
                3. Professional Information
                Current Job Title:
                Current Employer:
                Previous Job Titles:
                Work Experience:
                Education Background:
                Colleges or Universities:
                Skills and Expertise:
                Certifications and Licenses:
                Professional Interests:
                Career Goals:
                Achievements and Awards:
                Projects and Portfolios:
                
                4. Personal Interests and Hobbies
                Hobbies:
                Favorite Books:
                Favorite Movies:
                Favorite Music:
                Favorite Sports:
                Leisure Activities:
                
                5. Preferences and Dislikes
                Food Preferences:
                Travel Preferences:
                Pet Peeves:
                Allergies and Dietary Restrictions:
                Preferred Communication Style:
                
                6. Behavioral Attributes
                Personality Traits:
                Learning Style:
                Conflict Resolution Style:
                Decision-Making Style:
                Motivations and Drives:
                Stress Response:
                
                7. Health and Wellness
                Medical Conditions:
                Fitness Routine:
                Dietary Habits:
                Mental Health Information:
                
                8. Financial Information
                Income Range:
                Spending Habits:
                Investment Preferences:
                Financial Goals:
                
                9. Technology Usage
                Preferred Devices:
                Favorite Apps and Software:
                Tech Savviness:
                Privacy Concerns:
                
                10. Social Attributes
                Family Details:
                Friend Circle:
                Community Involvement:
                Volunteer Work:
                Social Causes Supported:
                
                11. Cultural Background
                Ethnicity:
                Religious Beliefs:
                Cultural Practices:
                Festivals Celebrated:
                
                12. Environmental Preferences
                Living Environment:
                Climate Preference:
                Home Layout and Design Preferences:
                
                13. Communication and Media Consumption
                Preferred News Sources:
                Social Media Engagement:
                Reading Habits:
                TV and Streaming Preferences:
                
                14. Travel and Mobility
                Travel History:
                Favorite Destinations:
                Mode of Transport:
                Travel Frequency:
                
                15. Shopping and Consumption
                Shopping Preferences:
                Brand Preferences:
                Frequency of Purchases:
                Loyalty Program Memberships:
                
                16. Ethical and Moral Values
                Core Beliefs:
                Ethical Stances:
                Philanthropic Interests:

                Make sure you give the final output which can be parsed into a JSON with these keys as their attributes as I'm storing user profile as a dictionary.
                """},
                {"role": "user", "content": prompt},
            ],
            response_format={ "type": "json_object" },
        )
        return response.choices[0].message.content


    def ask_question(self, context, query):
        """
        Ask a question based on the provided context and query.
        """
        combined_text = "User Information: "+ context + "\n\nQuery: " + query

        response = self.client.chat.completions.create( 
                    model=self.model,
                    messages=[
                        {"role": "system", "content": """You are a personal assistant. \
                        Your task is to answer user queries based on the personal preferences provided in the user's profile, which will be included in the prompt. \
                        Do not use any additional information other than what is mentioned in the prompt.

                          If the query cannot be answered based on the information in the user's profile, \
                          inform the user that you are unable to answer the question directly from the profile. \
                          Highlight that you are using your general knowledge as a personal assistant to provide the answer. \
                         Continue with the response that you'd like to give as it isn't in the users personal profile"""},
                        
                        {"role": "user", "content": combined_text}
                    ],
                    # response_format={ "type": "json_object" },
                    )

        return response.choices[0].message.content
    
    def ask_suggestion(self, context, query):
        """
        Provide a suggestion based on the provided context and query.
        """
        combined_text = "User Information: "+ context + "\n\nQuery: " + query

        response = self.client.chat.completions.create( 
        model=self.model,
        messages=[
            {"role": "system", "content": """

                You are a personal assistant. Your task is to answer user queries based on the personal preferences and profile details provided in the prompt. Do not use any additional information other than what is mentioned in the prompt.

                Here are some guidelines for handling different types of queries:

                1. **Emotional and Social Situations**:
                    - If the user expresses an emotion such as feeling sad, happy, or stressed, respond with appropriate suggestions based on their profile. For example, if they like music, suggest songs to uplift their mood. If they enjoy outdoor activities, suggest going for a walk.
                    - Example: "I'm feeling sad." Response: "I'm sorry to hear that. How about listening to your favorite songs to lift your spirits? Or maybe take a walk in the park?"

                2. **Cravings or Food Desires**:
                    - Determine if the user is health conscious or not.
                    - If the user expresses a craving or desire for specific food, check if the user is health-conscious based on their profile. If they are, suggest a healthy alternative or remind them of their health goals.
                    - If the user is not particularly health-conscious, offer to help them find a place to order the food or provide a recipe.
                    - If suggesting preparation at home, ask a follow-up question to offer a recipe.
                    - Example: "I'm craving chicken biryani." Response: "Are you looking to order chicken biryani or would you like a recipe to make it yourself?"

                Example responses:
                - For an emotional query: "I'm feeling stressed." Response: "I'm sorry you're feeling stressed. Would you like some tips on relaxation techniques or perhaps listen to some calming music?"
                - For a craving query: "I really want some ice cream." Response: "Would you like to find a nearby ice cream shop or a recipe to make it at home?"
                - For a general question: "What's the best way to learn Python?" Response: "I'm unable to answer your question directly from your profile. However, based on my general knowledge, I recommend starting with online courses like those on Coursera or Udemy."

                If the final output is multiple options that you are recommending, make it sound like a follow-up question instead of a response.
            """ },       
            {"role": "user", "content": combined_text}
        ],
        # response_format={ "type": "json_object" },
    )


        return response.choices[0].message.content
    

    def process_query(self, query):
        """
        Process a general query and provide a response.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": """You are a personal assistant. Your task is to answer user queries based on the personal preferences provided in the user's profile, which will be included in the prompt. 
                Do not use any additional information other than what is mentioned in the prompt.

                If the query cannot be answered based on the information in the user's profile, inform the user that you are unable to answer the question directly from the profile. Highlight that you are using your general knowledge as a personal assistant to provide the answer. Continue with the response that you'd like to give as it isn't in the user's personal profile.

                Additionally, if the user asks or says something to remind or save information, such as reminders for meetings, events, or tasks, process and save that information in a simple format. For example, if the user says, 'Please remind me that I have a meeting at 12:00 PM', save this information as 'You have a meeting at 12:00 PM'. When the user later asks about this information, such as 'When is my meeting?', you should be able to provide the saved reminder. 

                Your goal is to assist the user effectively by remembering important details they provide and using the profile information to give the most relevant responses."""},
                
                {"role": "user", "content": query}
            ]
        )

        return response.choices[0].message.content
