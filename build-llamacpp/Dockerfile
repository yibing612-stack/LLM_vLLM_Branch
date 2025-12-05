FROM alpine:3.20 AS build
RUN apk add --no-cache git cmake make g++ musl-dev linux-headers
WORKDIR /src
RUN git clone https://github.com/ggerganov/llama.cpp.git .

# generic: disable AVX/AVX2; static link; ENABLE OpenMP for multi-threading
RUN cmake -S . -B build-generic \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_CURL=OFF \
  -DGGML_BLAS=OFF \
  -DGGML_CUDA=OFF \
  -DGGML_METAL=OFF \
  -DGGML_VULKAN=OFF \
  -DGGML_OPENMP=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_AVX=OFF \
  -DGGML_AVX2=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXE_LINKER_FLAGS="-static" \
  -DCMAKE_C_FLAGS="-O3 -pipe" \
  -DCMAKE_CXX_FLAGS="-O3 -pipe"
RUN cmake --build build-generic -j --target llama-cli llama-quantize && \
    strip build-generic/bin/llama-cli build-generic/bin/llama-quantize

# avx2: enable AVX2; static link; ENABLE OpenMP
RUN cmake -S . -B build-avx2 \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_CURL=OFF \
  -DGGML_BLAS=OFF \
  -DGGML_CUDA=OFF \
  -DGGML_METAL=OFF \
  -DGGML_VULKAN=OFF \
  -DGGML_OPENMP=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_AVX=ON \
  -DGGML_AVX2=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXE_LINKER_FLAGS="-static" \
  -DCMAKE_C_FLAGS="-O3 -pipe" \
  -DCMAKE_CXX_FLAGS="-O3 -pipe"
RUN cmake --build build-avx2 -j --target llama-cli llama-quantize && \
    strip build-avx2/bin/llama-cli build-avx2/bin/llama-quantize

FROM scratch AS out
COPY --from=build /src/build-generic/bin/llama-cli /llama-cli-generic
COPY --from=build /src/build-avx2/bin/llama-cli /llama-cli-avx2
COPY --from=build /src/build-generic/bin/llama-quantize /llama-quantize-generic
COPY --from=build /src/build-avx2/bin/llama-quantize /llama-quantize-avx2
