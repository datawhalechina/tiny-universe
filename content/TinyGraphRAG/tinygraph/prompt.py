GEN_NODES = """
## Goal
Please identify and extract triplet information from the provided article, focusing only on entities and relationships related to significant knowledge points. 
Each triplet should be in the form of (Subject, Predicate, Object). 
Follow these guidelines:

1. **Subject:** Concepts in Bayesian Optimization
2. **Predicate:** The action or relationship that links the subject to the object.
3. **Object:** Concepts in Bayesian Optimization that is affected by or related to the action of the subject.

## Example
For the sentence "Gaussian Processes are used to model the objective function in Bayesian Optimization" the triplet would be:

<triplet><subject>Gaussian Processes</subject><predicate>are used to model the objective function in</predicate><object>Bayesian Optimization</object></triplet>

For the sentence "John read a book on the weekend," which is not related to any knowledge points, no triplet should be extracted.

## Instructions
1. Read through the article carefully.
2. Think step by step. Try to find some useful knowledge points from the article. You need to reorganize the content of the sentence into corresponding knowledge points.
3. Identify key sentences that contain relevant triplet information related to significant knowledge points.
4. Extract and format the triplets as per the given example, excluding any information that is not relevant to significant knowledge points.

## Output Format
For each identified triplet, provide:
<triplet><subject>[Entity]</subject><predicate>The action or relationship</predicate><object>The entity</object></triplet>

## Article

{text}

## Your response
"""

GET_ENTITY = """
## Goal

You are an experienced machine learning teacher. 
You need to identify the key concepts related to machine learning that the article requires students to master. For each concept, provide a brief description that explains its relevance and importance in the context of the article.

## Example

article:
"In the latest study, we explored the potential of using machine learning algorithms for disease prediction. We used support vector machines (SVM) and random forest algorithms to analyze medical data. The results showed that these models performed well in predicting disease risk through feature selection and cross-validation. In particular, the random forest model showed better performance in dealing with overfitting problems. In addition, we discussed the application of deep learning in medical image analysis."

response:
<concept>
    <name>Support Vector Machine (SVM)</name>
    <description>A supervised learning model used for classification and regression tasks, particularly effective in high-dimensional spaces.</description>
</concept>
<concept>
    <name>Random Forest Algorithm</name>
    <description>An ensemble learning method that builds multiple decision trees and merges them together to get a more accurate and stable prediction, often used to reduce overfitting.</description>
</concept>
<concept>
    <name>Feature Selection</name>
    <description>The process of selecting a subset of relevant features for use in model construction, crucial for improving model performance and reducing complexity.</description>
</concept>
<concept>
    <name>Overfitting</name>
    <description>A common issue where a model learns the details and noise in the training data to the extent that it negatively impacts the model's performance on new data.</description>
</concept>
<concept>
    <name>Deep Learning</name>
    <description>A subset of machine learning that uses neural networks with many layers to model complex patterns in large datasets, often applied in image and speech recognition tasks.</description>
</concept>

## Format

Wrap each concept in the HTML tag <concept>, and include the name of the concept in the <name> tag and its description in the <description> tag.

## Article

{text}

## Your response
"""


ENTITY_DISAMBIGUATION = """
## Goal
Given multiple entities with the same name, determine if they can be merged into a single entity. If merging is possible, provide the transformation from entity id to entity id.

## Guidelines
1. **Entities:** A list of entities with the same name.
2. **Merge:** Determine if the entities can be merged into a single entity.
3. **Transformation:** If merging is possible, provide the transformation from entity id to entity id.

## Example
1. Entities:
   [
       {"name": "Entity A", "entity id": "entity-1"},
       {"name": "Entity A", "entity id": "entity-2"},
       {"name": "Entity A", "entity id": "entity-3"}
   ]
   
Your response should be:

<transformation>{"entity-2": "entity-1", "entity-3": "entity-1"}</transformation>


2. Entities:
   [
       {"name": "Entity B", "entity id": "entity-4"},
       {"name": "Entity C", "entity id": "entity-5"},
       {"name": "Entity B", "entity id": "entity-6"}
   ]

Your response should be:

<transformation>None</transformation>

## Output Format
Provide the following information:
- Transformation: A dictionary mapping entity ids to the final entity id after merging.

## Given Entities
{entities}

## Your response
"""

GET_TRIPLETS = """
## Goal
Identify and extract all the relationships between the given concepts from the provided text.
Identify as many relationships between the concepts as possible.
The relationship in the triple should accurately reflect the interaction or connection between the two concepts.

## Guidelines:
1. **Subject:** The first entity from the given entities.
2. **Predicate:** The action or relationship linking the subject to the object.
3. **Object:** The second entity from the given entities.

## Example:
1. Article :
    "Gaussian Processes are used to model the objective function in Bayesian Optimization" 
   Given entities: 
   [{{"name": "Gaussian Processes", "entity id": "entity-1"}}, {{"name": "Bayesian Optimization", "entity id": "entity-2"}}]
   Output:
   <triplet><subject>Gaussian Processes</subject><subject_id>entity-1</subject_id><predicate>are used to model the objective function in</predicate><object>Bayesian Optimization</object><object_id>entity-2</object_id></triplet>

2. Article :
    "Hydrogen is a colorless, odorless, non-toxic gas and is the lightest and most abundant element in the universe. Oxygen is a gas that supports combustion and is widely present in the Earth's atmosphere. Water is a compound made up of hydrogen and oxygen, with the chemical formula H2O."
    Given entities: 
    [{{"name": "Hydrogen", "entity id": "entity-3"}}, {{"name": "Oxygen", "entity id": "entity-4"}}, {{"name": "Water", "entity id": "entity-5"}}]
    Output:
    <triplet><subject>Hydrogen</subject><subject_id>entity-3</subject_id><predicate>is a component of</predicate><object>Water</object><object_id>entity-5</object_id></triplet>
3. Article :
    "John read a book on the weekend" 
    Given entities: 
    []
    Output:
    None

## Format:
For each identified triplet, provide:
**the entity should just from "Given Entities"**
<triplet><subject>[Entity]</subject><subject_id>[Entity ID]</subject_id><predicate>[The action or relationship]</predicate><object>[Entity]</object><object_id>[Entity ID]</object_id></triplet>

## Given Entities:
{entity}

### Article:
{text}

## Additional Instructions:
- Before giving your response, you should analyze and think about it sentence by sentence.
- Both the subject and object must be selected from the given entities and cannot change their content.
- If no relevant triplet involving both entities is found, no triplet should be extracted.
- If there are similar concepts, please rewrite them into a form that suits our requirements.

## Your response:
"""

TEST_PROMPT = """
## Foundation of students
{state}
## Gole
You will help students solve question through multiple rounds of dialogue.
Please follow the steps below to help students solve the question:
1. Explain the basic knowledge and principles behind the question and make sure the other party understands these basic concepts.
2. Don't give a complete answer directly, but guide the student to think about the key steps of the question.
3. After guiding the student to think, let them try to solve the question by themselves. Give appropriate hints and feedback to help them correct their mistakes and further improve their solutions.
4. Return to TERMINATE after solving the problem
"""

GEN_COMMUNITY_REPORT = """
## Role
You are an AI assistant that helps a human analyst to perform general information discovery. 
Information discovery is the process of identifying and assessing relevant information associated with certain entities (e.g., organizations and individuals) within a network.

## Goal
Write a comprehensive report of a community.
Given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact. 
The content of this report includes an overview of the community's key entities, their legal compliance, technical capabilities, reputation, and noteworthy claims.

## Report Structure

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
{{
"title": <report_title>,
"summary": <executive_summary>,
"findings": [
{{
"summary":<insight_1_summary>,
"explanation": <insight_1_explanation>
}},
{{
"summary":<insight_2_summary>,
"explanation": <insight_2_explanation>
}}
...
]
}}

## Grounding Rules
Do not include information where the supporting evidence for it is not provided.

## Example Input
-----------
Text:
```
Entities:
```csv
entity,description
VERDANT OASIS PLAZA,Verdant Oasis Plaza is the location of the Unity March
HARMONY ASSEMBLY,Harmony Assembly is an organization that is holding a march at Verdant Oasis Plaza
```
Relationships:
```csv
source,target,description
VERDANT OASIS PLAZA,UNITY MARCH,Verdant Oasis Plaza is the location of the Unity March
VERDANT OASIS PLAZA,HARMONY ASSEMBLY,Harmony Assembly is holding a march at Verdant Oasis Plaza
VERDANT OASIS PLAZA,UNITY MARCH,The Unity March is taking place at Verdant Oasis Plaza
VERDANT OASIS PLAZA,TRIBUNE SPOTLIGHT,Tribune Spotlight is reporting on the Unity march taking place at Verdant Oasis Plaza
VERDANT OASIS PLAZA,BAILEY ASADI,Bailey Asadi is speaking at Verdant Oasis Plaza about the march
HARMONY ASSEMBLY,UNITY MARCH,Harmony Assembly is organizing the Unity March
```
```
Output:
{{
"title": "Verdant Oasis Plaza and Unity March",
"summary": "The community revolves around the Verdant Oasis Plaza, which is the location of the Unity March. The plaza has relationships with the Harmony Assembly, Unity March, and Tribune Spotlight, all of which are associated with the march event.",
"findings": [
{{
"summary": "Verdant Oasis Plaza as the central location",
"explanation": "Verdant Oasis Plaza is the central entity in this community, serving as the location for the Unity March. This plaza is the common link between all other entities, suggesting its significance in the community. The plaza's association with the march could potentially lead to issues such as public disorder or conflict, depending on the nature of the march and the reactions it provokes."
}},
{{
"summary": "Harmony Assembly's role in the community",
"explanation": "Harmony Assembly is another key entity in this community, being the organizer of the march at Verdant Oasis Plaza. The nature of Harmony Assembly and its march could be a potential source of threat, depending on their objectives and the reactions they provoke. The relationship between Harmony Assembly and the plaza is crucial in understanding the dynamics of this community."
}},
{{
"summary": "Unity March as a significant event",
"explanation": "The Unity March is a significant event taking place at Verdant Oasis Plaza. This event is a key factor in the community's dynamics and could be a potential source of threat, depending on the nature of the march and the reactions it provokes. The relationship between the march and the plaza is crucial in understanding the dynamics of this community."
}},
{{
"summary": "Role of Tribune Spotlight",
"explanation": "Tribune Spotlight is reporting on the Unity March taking place in Verdant Oasis Plaza. This suggests that the event has attracted media attention, which could amplify its impact on the community. The role of Tribune Spotlight could be significant in shaping public perception of the event and the entities involved."
}}
]
}}

## Real Data
Use the following text for your answer. Do not make anything up in your answer.

Text:
```
{input_text}
```

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
{{
"title": <report_title>,
"summary": <executive_summary>,
"rating": <impact_severity_rating>,
"rating_explanation": <rating_explanation>,
"findings": [
{{
"summary":<insight_1_summary>,
"explanation": <insight_1_explanation>
}},
{{
"summary":<insight_2_summary>,
"explanation": <insight_2_explanation>
}}
...
]
}}

## Grounding Rules
Do not include information where the supporting evidence for it is not provided.

Output:
"""

GLOBAL_MAP_POINTS = """
You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response should be HTML formatted as follows:


<point><description>"Description of point 1..."</description><score>score_value</score></point>
<point><description>"Description of point 2..."</description><score>score_value</score></point>


The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".
Do not include information where the supporting evidence for it is not provided.


---Data tables---

{context_data}

---User query---

{query}

---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".
Do not include information where the supporting evidence for it is not provided.

The response should be HTML formatted as follows:

<point><description>"Description of point 1..."</description><score>score_value</score></point>
<point><description>"Description of point 2..."</description><score>score_value</score></point>


"""

LOCAL_QUERY = """
## User Query
{query}
## Context
{context}
## Task
Based on given context, please provide a response to the user query.
## Your Response
"""

GLOBAL_QUERY = """
## User Query
{query}
## Context
{context}
## Task
Based on given context, please provide a response to the user query.
## Your Response
"""
