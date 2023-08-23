---
layout: distill
title: Semantic Embedding w/ ChatGPT
date: 2023-08-23
tags: machine-learning llm nlp search distill
giscus_comments: true

authors:
  - name: Stephen Lu
    url: "https://matrixmaster.me"
    affiliations:
      name: McGill University

bibliography: 2023-08-23-embedding-gpt.bib
---

## Motivation
Ever since the initial release of ChatGPT in November 2022, large language models have rapidly taken the spotlight of machine learning applications in industry. Leaders like Anthropic and Cohere have reached the unicorn status in the blink of an eye followed by a swarm of startups trying to apply **ChatGPT to X**. Like any other groundbreaking technology, large language models will take some time to fully integrate into our society, but it is undeniably something that is here to stay for the long term.

<div class="fake-img">
  {% include figure.html path="assets/img/blog/2023-08-23-embedding-gpt/funding.png" class="img-fluid rounded z-depth-1" zoomable=true %}
</div>
<div class="caption">
    Funding for generative AI has shot up by over 7 fold in the first half of 2023 (Image source: CB Insights, 2023)
</div>

Given that progress is being made at blinding speed, I wanted to try out some of the awesome new tools being developed and write a tutorial on how to harness gpt-style models to your own use cases. Before we dive into the code, I want to briefly do an overview of the large language model architecture then explain the two main approaches being used to customize these models: finetuning versus embedding.

## Transformers
At a very high level, large language models use a novel neural network architecture called transformers<d-cite key="vaswani2023attention"></d-cite> to generate the best responses to your prompt. You can think of transformers as extremely powerful sentence completion models that have **efficiently** learnt the underlying semantic patterns of written language to the extent that they can sample continuations to a textual prompt that are semantically meaningful. For an in depth understanding of the transformer architecture, I recommend this [blog post](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar. ChatGPT is an augmented version of the transformer model that was finetuned with reinforcement learning to better align with human instructions. Whether in the form of q&a or simply listening to instructions, ChatGPT seems **smart** because it has been trained very well to align with human intent.<d-cite key="ouyang2022training"></d-cite>

## Embedding versus Finetuning
Although large language models are trained on data from the entire internet, the goal of these models is to learn the semantic patterns in language, not to memorize all the information on the internet. When we ask ChatGPT to answer a factual question like *"Which country won the most gold medals at the 2012 olympics?"*, the model does not have this information embedded in its finite set of parameters. Instead, it will perform a semantic search <d-footnote>You can think of this like a Google search</d-footnote> and insert the raw information it needs to answer our question into its context. This technique is known as **semantic embedding** and we can also use it to help ChatGPT access information that is relevant to our application domain. For example, if we want to build a personal gpt assistant that can answer relevant questions about all my previous tax returns, then I can put these relevant text documents into a database and allow a large language model to query it before answering my questions pertaining to this topic. As you can see, semantic embeddings don't modify the underlying large language model at all. It just fetches the relevant information to better align the model's answer with the users prompt. 

Finetuning on the other hand does modify the weights of the model. We already gave an example of finetuning earlier when we described how ChatGPT was trained downstream with reinforcement learning to better align with human instructions. Thus, it can be understood that the weights of the model encode something related to the built-in biases and choices made by the model when instructions are open to interpretation. 

A better way to explain this is by drawing a parallel with human behaviour. Humans are extremely good at following instructions, especially when these instructions are very detailed and leave little to no room for interpretation. However, when we are given vague instructions or open-ended questions, the responses that we provide reflect our personal opinions and biases. Even in the case when the possible responses are relatively constrained, biases are still present in low-level constructs such as sentence structure, word choice, and semantic sentiment. For example, although two doctors might give you the same diagnosis and prescription, the way they convey that information, the amount of detail they provide, and the tone they convey could be drastically different. 

The same can be said for large language models when they are prompted with tasks that are open to interpretation. If we want a model that consistently and reliably provides responses in a particular tone with a particular set of predetermined biases, then it makes more sense to finetune the model so that it aligns its decisions with this set of priors. For example, Claude is the flagship llm by Anthropic that is finetuned using a reinforcement learning policy aligned with a set of [constitutional principles](https://www.anthropic.com/index/claudes-constitution) such as being *ethical*, *virtuous*, etc. In essense, finetuning can be thought of as introducing prior preferences into the llm that guide its responses on top of the fundamental prior that the model should be aligned with the user intent.

## Implementing a Conversational Retrieval Q&A agent in LangChain
In the following tutorial, I will implement a conversational Q&A agent such as ChatGPT that has access to the standard chat history context as well as a database of private documents that can be used for **semantic embedding** search. I will address model finetuning in a future blog post.

### Setting up a Zep server
Recall that semantic embedding search involves searching for relevant documents in a database so that we can inject relevant information into the context of the model before providing a response. In our implementation, we will use [Zep](https://github.com/getzep/zep) which is a long term RAM memory store that is optimized for storing text embeddings and performing various search algorithms over these embeddings.

You can follow these instructions on their [getting started](https://docs.getzep.com/deployment/quickstart/#starting-a-zep-server-locally-is-simple) page to start your Zep server. It is as easy as setting up some `API_KEY` environment variables and running a `docker-compose up` command.

### Adding documents to the Zep server
Now that the Zep database is up and running, we want to add our document embeddings into the database, so that we can perform retrieval against them. To do this, I wrote a python seeder script, but Zep also supports an SDK in native JavaScript as well as integration with 3rd party libraries such as LangChain and LlamaIndex which you can use to interact with the database as well.

#### Creating a collection
Now, the first step is to connect to the database and create a collection. Zep organizes its data in the following hierarchy: **collection > documents > embeddings**. Going back to the tax returns example, we could think of a document as individual tax related files, while a collection might group all such files for a particular year. Zep takes our documents and does the embedding for us, so we need to provide an embedding model. Here I choose the `text-embedding-ada-002` model by OpenAI since we will be using ChatGPT for our llm.

```python
from zep_python import ZepClient
from zep_python.document import Document

zep_api_url = "http://localhost:8000"

client = ZepClient(base_url=zep_api_url)

"""
Step 1: Create a new collection in which to store our documents for a given task class
        We will use chatgpt text-embedding-ada-002 model which has embedding size of 1536
"""

collection_name = "<name the collection>"

client.document.delete_collection(collection_name)

try:
    collection = client.document.add_collection(
        name=collection_name,  # required
        description="<some description>",  # optional
        embedding_dimensions=1536,  # this must match the model you've configured for 
        is_auto_embedded=True,  # use Zep's built-in embedder. Defaults to True
    )
except Exception as e:
    print(e)
```

#### Chunking up documents
Our documents can have a large variation in their length and content size, so it doesn't really make sense to squeeze each document into a fixed size embedding. Instead, we split up documents into fixed size chunks that have a predetermined token length, then embed each chunk into a vector embedding. Here I split my documents into chunks with a maximum of 1600 tokens per chunk, but this process will heavily depend on the nature and format of your documents. The code I provide below is just an example of how this chunking might be done, but you should write your own routine for this.

```python
DOC_DIR = "documents"
FILE_NAME = "documents/raw_convo.txt"

# Custom splitting for .txt file such that each entry in qa_data is a tuple of ([questions], answer)
# TODO: Add support for csv, json, and yml files
sections = split_into_sections(FILE_NAME)
qa_sections = split_into_qa_pairs(sections)
qa_data = []
for section, data in qa_sections:
    for questions, answer in data:
        qa_data.append((questions, answer))

# Split the qa pairs into chunks with a predefined max token length
MAX_TOKENS = 1600
qa_strings = []

for section in qa_data:
    qa_strings.extend(split_strings_from_subsection(section, max_tokens=MAX_TOKENS))
```

#### Embedding chunks into Zep
The last step is to embed the chunks into Zep using the `Document` class. Here we could choose to add metadata to each chunk to identify, for example, which original file it belongs to. These metadata can then serve as future *filters* when we search against the Zep database.

```python
"""
Step 3: Embed the document chunks and store them into the collection
"""
documents = [
    Document(
        content=chunk,
        document_id=f"{collection_name}-{i}",  # optional document ID
        metadata={"bar": i},  # optional metadata
    )
    for i, chunk in enumerate(qa_strings)
]

uuids = collection.add_documents(documents)
```

Finally, we can also spin up a busy waiting watcher process that waits for the documents to be embedded before exiting.

```python
"""
Step 4: Wait for the documents to be embedded and monitor the process
"""
while True:
    c = client.document.get_collection(collection_name)
    print(
        "Embedding status: "
        f"{c.document_embedded_count}/{c.document_count} documents embedded"
    )
    time.sleep(1)
    if c.status == "ready":
        break

print(f"Added {len(uuids)} documents to collection {collection_name}")
```

### Giving ChatGPT access to the Zep database
Now that we have our vector database ready to go, we just need to hook up a language model to query from it and add the relevant embeddings to its context before answering the user. I used [LangChain](https://github.com/hwchase17/langchainjs) which is an awesome framework that enables easy interaction with popular llms as well as integration with 3rd party plugins and databases such as Zep. Using the LangChain JavaScript SDK, I simply need to do the following steps:

1. Connect to the Zep database
2. Retrieve the user's chat history along with his current active prompt
3. Embed the chat history along with the current active prompt use as a prototype search vector
4. Use the search vector to find semantically related embeddings in the Zep database
5. Feed all the relevant embeddings from Zep and the chat context to an instance of ChatGPT model
6. Return the model response to the user

Using SvelteKit server module, this can be done in very few lines of code.

```typescript
import { OPENAI_API_KEY, OPENAI_ORGANIZATION } from "$env/static/private";
import { ChatOpenAI } from "langchain/chat_models/openai";

import { ConversationalRetrievalQAChain } from "langchain/chains";
import { ZepVectorStore } from "langchain/vectorstores/zep";
import { FakeEmbeddings } from "langchain/embeddings/fake";
import { BufferMemory } from "langchain/memory";

import { error } from '@sveltejs/kit';

export type MessageBody = { 
    question: string;
    settings: { temperature: number, relatedness: number };
}

const zepConfig = {
    apiUrl: "http://localhost:8000", // the URL of your Zep implementation
    collectionName: "<collection_name>",  // the name of your collection. alphanum values only
    embeddingDimensions: 1536,  // much match the embeddings you're using
    isAutoEmbedded: true,  // will automatically embed documents when they are added
};

const CUSTOM_QUESTION_GENERATOR_CHAIN_PROMPT = `
    Given the following conversation and a follow up question,
    return the conversation history excerpt that includes any relevant context to the question
    if it exists and rephrase the follow up question to be a standalone question.
    
    Chat History: {chat_history}
    Follow Up Input: {question}
    
    Your answer should follow the following format:

    \`\`\`
    <Here you can give some additional behavioural instructions to the model in the form of prompting. The result will not be as good as finetuning the model on a large amount of
    examples that properly introduce a set of behavioural guidelines for the model to respect.>
    ----------------
    <Relevant chat history excerpt as context here>
    Standalone question: <Rephrased question here>
    \`\`\`

    Your answer:
`;

const embeddings = new FakeEmbeddings();

export const POST = async ({ request }) => {
    const body: MessageBody = await request.json();

    if (!body) throw error(400, 'Missing Data');

    // Connect to the Zep vector store server
    const vectorStore = await new ZepVectorStore(embeddings, zepConfig);

    // Create a new readable stream of the chat response
    const readableStream = new ReadableStream({
        async start(controller) {
            // Create a new chat model
            const streamingModel = new ChatOpenAI({
                openAIApiKey: OPENAI_API_KEY,
                modelName: "gpt-4",
                streaming: true,
                temperature: body.settings.temperature,
                callbacks: [{
                    handleLLMNewToken: async (token: string) => controller.enqueue(token),
                }]
            }, {
                organization: OPENAI_ORGANIZATION,
            });

            const nonStreamingModel = new ChatOpenAI({
                openAIApiKey: OPENAI_API_KEY,
                modelName: "gpt-3.5-turbo",
                temperature: body.settings.temperature,
            }, {
                organization: OPENAI_ORGANIZATION,
            });

            const chain = ConversationalRetrievalQAChain.fromLLM(
                streamingModel,
                vectorStore.asRetriever(),
                {
                    memory: new BufferMemory({
                        memoryKey: "chat_history", // Must be set to "chat_history"
                        inputKey: "question", // The key for the input to the chain
                        outputKey: "text", // The key for the final conversational output of the chain
                        returnMessages: true, // If using with a chat model
                    }),
                    returnSourceDocuments: true,
                    questionGeneratorChainOptions: {
                        llm: nonStreamingModel,
                        template: CUSTOM_QUESTION_GENERATOR_CHAIN_PROMPT
                    },
                },
            );

            const question = body.question;
            if (!question) throw error(400, 'Missing Question');

            const resp = await chain.call({ question });
            controller.close();
        },
    });

    // Create and return a response of the readable stream
    return new Response(readableStream, {
        headers: {
            'Content-Type': 'text/plain',
        },
    });
}
```

For more information on this script, please visit the following [documentation](https://js.langchain.com/docs/modules/chains/popular/chat_vector_db). The full code for my walkthrough can be found [here](https://github.com/TheMatrixMaster/gpt-embeddings).
