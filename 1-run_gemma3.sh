# ./build/bin/llama-gemma3-cli -m /root/workspace/models/google/gemma-3-4b-it-qat-q4_0-gguf/gemma-3-4b-it-q4_0.gguf \
#                                 --mmproj /root/workspace/models/google/gemma-3-4b-it-qat-q4_0-gguf/mmproj-model-f16-4B.gguf \
#                                 -p "Describe this image in detail" \
#                                 -ngl 34
                                # --image /home/jmtang/Workspace/Codes/Models/google/gemma-3-4b-it-qat-q4_0-gguf/surprise.png \
./build/bin/llama-gemma3-cli -m /root/workspace/models/OpenGVLab/InternVL3-2B/InternVL3-2B-F16.gguf \
                                --mmproj /root/workspace/models/OpenGVLab/InternVL3-2B/mmproj.gguf \
                                -p "Describe this image in detail" \
                                -ngl 34                           

# ./build_cuda/bin/llama-llava-cli -m /home/jmtang/Workspace/Codes/Models/mys/ggml_llava-v1.5-7b/ggml-model-q4_k.gguf \
#                                 --mmproj /home/jmtang/Workspace/Codes/Models/mys/ggml_llava-v1.5-7b/mmproj-model-f16.gguf \
#                                 -p "Describe this image in detail" \
#                                 --image /home/jmtang/Workspace/Codes/llama.cpp/examples/llava/test-1.jpeg \
#                                 -ngl 32
