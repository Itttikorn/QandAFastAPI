# Q & A using Text Vectorization and Fast API
There are two versions available:

### QandAwithLLAMA
Uses Llama 3.2 to summarize the top-k answers into a single response.

### QandAwithoutLLAMA
Does not use Llama 3.2 and simply returns the top-k answers.

## Prerequisites
- Docker Engine

## Getting Started
1. Navigate to the desired version's directory:
    ```sh
    cd QandAwithLLAMA   # or QandAwithoutLLAMA
    ```
2. Start the services with Docker Compose:
    ```sh
    docker compose up
    ```
3. Wait a few minutes for the API server to become fully functional.
4. Send a `GET` HTTP request to `http://localhost:8000/ask` with a JSON body.
    ```json
        {
            "query": "Your question",
            "k":5
        }
    ```
Visit http://localhost:8000/docs for API document