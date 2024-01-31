# DTDL_findings

### 1. LLM size reduction and Similarity search for cache
Find the related slides here: 
https://docs.google.com/presentation/d/17Q7BPCxYXYrFRbr-FTYB-IAjB-nmSEbyKbbk6_51C_I/edit?usp=sharing

### 2. Langchain for Development Deeplearning.ai  Course Notes
Find the related slides here: 
https://docs.google.com/presentation/d/1qADZ5pMUx7X4qZ_cWaJ950kF1zWMY8Drne6z4KzrfSs/edit?usp=sharing

### 3. Personal Chatbot
This chatbot aims at chatting with your own data. It firstly creates the vector embeddings of all the data sources(pdf, ppt, docx) provided to it and stores it in chroma database. After that, the chatbot takes the user query as input, creates its embedding. The embedding is then sent to the LLM and the LLM returns the most appropriate answer. The problem here was that since the LLM size was too large, using the chatbot lead to my system gettting hanged. Will try to improve upon this in the future. Other future aspects are implementing a feedback mechanism for the chatbot.

### 4. Other tasks
Apart from this, I also helped the team with prompt size reduction as the maximum token limit exceeded for our LLM.
