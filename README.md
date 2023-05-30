# associated babyagi

# Objective
Attempt to add in an associated memory structure to babyagi
Associations are stored in firebase as relationship between 2 vectors in the pinecone vector store
Essentially making a knowledge graph of associations the system had learned.

# Image of ideal functionality


# How It Works
The script works in the same way as the original baby agi along with another step
![AGI_Associations](https://github.com/fjooord/associated_babyagi/assets/44952366/6ba927df-5d79-43ff-b305-590b480fd4a6)



This system is designed to utilize the capabilities of GPT, Pinecone, and a Firebase database to provide nuanced responses based on both the given task and the associations found in the existing knowledge.


## execution_agent() is the main function that performs tasks based on a given objective.

  - It first collects the context using the context_agent() function. The context is determined by querying an index in Pinecone and returning the metadata of the top matches.
  - It then uses the context and objective to generate a prompt for the OpenAI API.
  - It calls the OpenAI API with the prompt, and returns the response.

## context_agent() collects the context for a given query by querying Pinecone and returning metadata of top matches.

  - It first generates an embedding of the query using get_ada_embedding().
  - It then queries the Pinecone index using this embedding and returns the top n matches.
  - It sorts the matches by score in descending order.
  - It collects the metadata of the matches into a context.
  - It retrieves the associated information from a Firebase database, if any vector in the retrieved vectors is present in the associations in the database.
  - It finally returns the context concatenated with the associated information.

## find_new_associations() is a function that finds and stores new associations between a new vector and the existing knowledge in a given table.

  - It first gets possible related topics using the get_possible_related_topics() function.
  - For each topic, it generates an embedding and queries the Pinecone index for the top matches.
  - For each match, it checks if there are any connections between the new information and the existing knowledge using the OpenAI API.
  - If a connection is found, it describes the connection using the OpenAI API and appends the new association to the list of new associations.
  - It returns the new associations.

## get_possible_related_topics() is a function that uses the OpenAI API to suggest topics related to a given prompt.

  - It constructs a full prompt asking for cross-disciplinary connections related to the given knowledge.
  - It sends this prompt to the OpenAI API.
  - It parses the response to extract the suggestions and returns them.
 
