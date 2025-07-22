from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama

############################################
################ Rate Limiters #############
############################################

slow_rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,
    check_every_n_seconds=0.5,
    max_bucket_size=1,
)

premium_rate_limiter = InMemoryRateLimiter(
    requests_per_second=4,
    check_every_n_seconds=0.5,
    max_bucket_size=1,
)

average_rate_limiter = InMemoryRateLimiter(
    requests_per_second=10,
    check_every_n_seconds=0.5,
    max_bucket_size=1,
)

###########################################
################ Models ###################
###########################################

slow_mistral = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0.3,
    max_retries=2,
    rate_limiter=slow_rate_limiter
)

# mistral7b =  ChatOllama(
#     model="mistral:7b",
#     temperature=0.7
# )

# qwen25_32b = ChatOllama(
#     model="qwen2.5:32b",
#     temperature=0.7
# )

# avg_dgx_qwq = ChatOllama(
#     model="qwq:32b",
#     temperature=0,
#     rate_limiter=average_rate_limiter
# )

# dgx_qwq = ChatOllama(
#     model="qwq:32b",
#     temperature=0
# )