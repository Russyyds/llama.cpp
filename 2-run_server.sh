./build_cuda/bin/llama-server -m /root/workspace/models/google/gemma-3-4b-it/gemma-3-4B-it-BF16.gguf -ngl 34 -c 512 -fa -np 4

#  curl --request POST     --url http://localhost:8080/completion     --header "Content-Type: application/json"     --data '{"prompt": "What is the meaning of life?"}'