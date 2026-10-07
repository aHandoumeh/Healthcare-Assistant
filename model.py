from openai import OpenAI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


load_dotenv()

# load database:
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings(),
)



# Messages:
conversation = ""
doctor_message = ""
patient_message = input("*********************\nQ:")

# query the database:
results = vector_store.similarity_search(patient_message, k=4)
context = "\n\n".join(
    f"{doc.page_content}"
    for doc in results
)



client = OpenAI()

flag = True

message_num = 1
while(True):

  # first message:
  if (flag):

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=patient_message,
    )

    flag = False

    # model resoponse:
    doctor_message = response.output_text
    temp = f"""

    patient: {patient_message}
    -------------------------------
    doctor: {doctor_message}

    """
    message2 = f"""
          You are a helpful healthcare assistant talking to a patient.
    
          Previous conversation:
          {conversation}
          --------------------------------------------
    
          Patient's new message:
          {patient_message}

          
          -------------------------------------------- 

          Relevant medical information retrieved from the database: 
          {context} 

          -------------------------------------------- 


          Use the previous conversation to understand the patient and use the retrieved medical information as supporting knowledge.

          Respond naturally and clearly like a healthcare professional.

          Instructions: 
          * Use the retrieved medical information when it is relevant to the patient's question. 
          * Explain possible reasons for the symptoms based on the available information. 
          * Give general recommendations. 
          * Mention warning signs when necessary.
          * Do not give a definite diagnosis.
          * Do not claim that the patient has a specific disease.
          * If the symptoms may indicate an emergency, tell the patient to seek immediate medical care.
          * If the retrieved information does not contain enough information to answer the question, say I don't know and the patient may indicate an emergency. 
          * Keep the response simple, clear, and concise. 
          """
    conversation += f"Message {message_num}:\n" + temp
    

  # other messages:
  else:
    # Update messages:
    patient_message = input("*********************\nQ:")
    
    # query the db according to the new message
    results = vector_store.similarity_search(patient_message, k=4)
    context = "\n\n".join(
        f"{doc.page_content}"
        for doc in results
    )


    message2 = f""" 
    You are a helpful healthcare assistant talking to a patient. 
    Previous conversation: 
    {conversation} 

    -------------------------------------------- 

    Patient's new message: 
    {patient_message} 

    -------------------------------------------- 

    Relevant medical information retrieved from the database: 
    {context} 

    --------------------------------------------
    
    Use the previous conversation to understand the patient and use the retrieved medical information as supporting knowledge.
    Respond naturally and clearly like a healthcare professional.
    Instructions: 
    * Use the retrieved medical information when it is relevant to the patient's question. 
    * Explain possible reasons for the symptoms based on the available information. 
    * Give general recommendations. 
    * Mention warning signs when necessary.
    * Do not give a definite diagnosis.
    * Do not claim that the patient has a specific disease.
    * If the symptoms may indicate an emergency, tell the patient to seek immediate medical care.
    * If the retrieved information does not contain enough information to answer the question, say I don't know and the patient may indicate an emergency. 
    * Keep the response simple, clear, and concise. 
    """

    response = client.responses.create(
            model="gpt-5.6-luna",
            input=message2,
    )

    doctor_message = response.output_text
    
    
    temp = f"""
    patient: {patient_message}
    -------------------------------
    doctor: {doctor_message}

    """


    conversation += f"Message {message_num}:\n" + temp
    

  
  message_num += 1

  print("\n\nAns: " + doctor_message)