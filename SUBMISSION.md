# Internship Assessment Submission - Yesha Jagad

## Candidate Information

- **Full Name:** Yesha Jagad
- **Email Address:** yeshajagad29@gmail.com
- **GitHub Profile URL:** https://github.com/Yeshajagad
- **LinkedIn Profile:** https://www.linkedin.com/in/yesha-jagad-2040ab296/
- **Submission Date:** 20 November 2026

---

## Task Deliverables

### Task 1: AI Document Assistant (RAG)

- **GitHub Repository URL:** https://github.com/Yeshajagad/ghanextech-technical-assessment/tree/main/task1-rag
- **Live Deployed Application URL:** https://yesha-rag-assistant.streamlit.app/
- **Primary Technologies Used:** Python, Streamlit, pypdf, FAISS, Ollama (bge-m3 embeddings and llama3.2), Groq, fastembed (MiniLM embeddings), pytest, GitHub Actions.

- **Project Summary:**

  Developed a document question-answering application that answers questions exclusively using uploaded PDF and text documents. The application extracts text from uploaded files, divides it into overlapping chunks, generates embeddings, and stores them in a FAISS vector index for efficient similarity-based retrieval.

  For every user question, the application retrieves the most relevant document chunks and provides them as context to the language model. The model is instructed to answer only from the retrieved content rather than relying on unsupported information. Answers are streamed into the chat interface, with citations such as [1] linking back to the source file, page number, and relevant text passage. If the uploaded documents do not contain sufficient information, the application clearly communicates that instead of generating an unsupported answer.

- **Key Challenges & Solutions:**

  1. **Local model hosting versus cloud deployment:** The local implementation uses Ollama, which runs on the development machine and cannot be accessed directly by a cloud-hosted application. I separated model and embedding integrations behind configurable adapters, allowing the application to use Ollama locally and Groq with fastembed-based embeddings in the deployed version.

  2. **Embedding dependency size:** Initially considered sentence-transformers for hosted embeddings, but its PyTorch dependency was too heavy for the free hosting environment. Switched to fastembed, which uses a smaller ONNX-based embedding implementation.

  3. **Scanned PDF handling:** Scanned PDFs may contain images instead of an extractable text layer. Added detection for documents without extractable text and display a clear message rather than silently indexing empty content.

---

### Task 2: AI Agent with Tool Calling

- **GitHub Repository URL:** https://github.com/Yeshajagad/ghanextech-technical-assessment/tree/main/task2-agent
- **Live Deployed Application URL:** https://yesha-ai-agent.streamlit.app/
- **Primary Technologies Used:** Python, Streamlit, httpx, Ollama (qwen3:4b), Groq (openai/gpt-oss-120b), Open-Meteo API, Wikipedia API, pytest.

- **Implemented Tools:**

  1. **calculator:** Safely evaluates mathematical expressions using Python's Abstract Syntax Tree (AST). Supports permitted arithmetic operations and functions while preventing arbitrary code execution.

  2. **get_weather:** Retrieves current weather information for a specified location using the Open-Meteo geocoding and weather APIs. Does not require an API key.

  3. **search_wikipedia:** Searches Wikipedia and retrieves introductory information about a specified topic.

- **Project Summary:**

  Developed a conversational AI agent that determines whether a user message can be answered directly or requires one or more external tools. The agent uses the language model's native tool-calling capability with JSON schemas to identify the appropriate tool and generate its arguments.

  The agent validates tool arguments, executes the selected tool, and sends the result back to the language model. This process continues until the model generates a final response or the maximum step limit is reached. It supports chained tool execution, such as retrieving weather information and then performing a calculation based on the result. Conversation history is maintained for the last 20 messages, and execution is limited to a maximum of 5 steps. The Streamlit interface also displays a "Tools Used" section showing each tool call, its arguments, and its result. Tool failures are returned as readable errors so the agent can explain the problem instead of crashing.

- **Key Challenges & Solutions:**

  1. **Different LLM provider formats:** Ollama and Groq represent tool calls and tool results differently. Implemented a provider-neutral message format and centralized the conversion logic in the LLM integration module.

  2. **Safe calculator implementation:** Using Python's `eval()` would allow unsafe code execution. Implemented an AST-based expression evaluator with an explicit allowlist of supported operations and functions, along with expression-length and exponent-size restrictions.

  3. **Incorrect tool arguments from smaller models:** Models can occasionally generate invalid arguments or fail to select the correct tool. Added schema-based argument validation and return clear tool errors to the model so it can correct its attempt.

  4. **Qwen3 reasoning output:** The local Qwen3 model could include `<think>` blocks in its responses. Disabled thinking for the Ollama integration where supported and added handling to remove leftover thinking blocks from the final response.

  5. **Model timeouts and application stability:** Slow local inference could produce timeout errors that interrupted the application. Added error handling around model calls to prevent raw exceptions from crashing the chat interface.

  6. **Model availability and configuration:** The original Groq model, `llama-3.3-70b-versatile`, became unavailable. Updated the Groq model configuration to `openai/gpt-oss-120b` so the application uses the replacement model.

---

### Task 3: GitHub Repository AI Analyzer & Open-Source Contribution

- **GitHub Repository URL:** https://github.com/Yeshajagad/ghanextech-technical-assessment/tree/main/task3-analyzer
- **Live Deployed Application URL:** https://yesha-repo-analyzer.streamlit.app/
- **Primary Technologies Used:** Python, Streamlit, httpx, GitHub REST API, Ollama (qwen3), Groq, pytest.

- **Project Summary:**

  Developed an AI-powered GitHub repository analyzer that accepts a repository in `owner/repo` format or a GitHub repository URL. The application retrieves repository information through the GitHub REST API, including repository details, programming languages, README content, file structure, package and dependency files, and good first issues.

  It presents repository facts, programming-language distribution, and a checklist covering important project components such as tests, CI configuration, license, contributing guidelines, documentation, and Dockerfile. The language model generates a structured JSON analysis containing a project summary, technology stack, strengths, architecture overview, at least three improvement recommendations, and potential contribution ideas.

  Repository facts and issue links are retrieved from GitHub rather than invented by the language model. The application also handles invalid repository links, private repositories, and API rate limits with readable notices. Importantly, repository information remains visible even if the language model fails to generate its analysis.

- **Open-Source Contribution:**

  - **Target Open-Source Repository:** https://github.com/ollama/ollama-python
  - **Contribution URL (Issue / Pull Request):** (https://github.com/ollama/ollama-python/issues/803)
  - **Contribution Summary:** Setting `timeout` on the Python client makes a slow server raise a raw
  `httpx.ReadTimeout`, which is inconsistent with how connection errors are converted into
  `ConnectionError`. I reproduced it with a local slow HTTP server (no model needed) across
  sync, async and both streaming paths, and found that streaming connection errors also
  leak `httpx.ConnectError`. I filed an issue with a runnable reproducer, code references,
  and a proposal to wrap `httpx.TimeoutException` in `TimeoutError`, asking maintainers
  about the exception type before sending a PR.

---

## AI Collaboration Disclosure

- **AI Tools Used:** ChatGPT (OpenAI).

- **How You Leveraged AI:**

  I used ChatGPT as a development and learning assistant throughout the assessment. It helped me plan implementation phases, understand architectural choices, work through Python concepts, debug errors, review code, develop test cases, and improve documentation.

  I also used local Ollama models within the applications, including bge-m3 and llama3.2 for Task 1 and qwen3:4b for Task 2. Groq was used as the hosted language-model provider for deployment. These models were part of the applications themselves and are separate from the AI tools I used during development.

- **Verification & Ownership:**

  I created and maintained the assessment repository, organized the implementation into phases, and managed the development branches, commits, merges, and version tags using Git. I ran the automated test suites, tested the applications locally, checked example prompts and error-handling scenarios, and worked through implementation and configuration issues during development.

  I also made and verified changes to address issues encountered during testing, including model configuration updates, compatibility between LLM providers, and application error handling. I used AI assistance for code development and debugging, and reviewed the implementation as part of the process.

  [Add an accurate description of your own open-source contribution and disclose whether AI assisted with its drafting or implementation.]

- **Confirmation:**

  I confirm that I have reviewed and tested the code included in this submission and can explain the implementation, architecture, design decisions, and debugging approaches during the technical interview.

---

## Evaluation Notice

I understand that the submitted repositories and deployed applications should be publicly accessible and that I may be invited to a technical interview to discuss my implementation details, architectural decisions, and debugging methodology.