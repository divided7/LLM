# TODO
有一种说法是使用kvcache必须用python启动vllm, 用CLI启动vllm无法将长文本前缀提示词提前prefix

但是vllm也有`--enable-prefix-caching`参数 需要后续研究一下