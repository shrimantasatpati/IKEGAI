from dataclasses import dataclass,asdict

@dataclass
class OpenAIConfig:
    api_key:str="312ff50d6d954023b8748232617327b6"
    azure_endpoint:str="https://openai-lh.openai.azure.com/"
    azure_deployment:str="test"
    api_version:str="2024-02-15-preview"

