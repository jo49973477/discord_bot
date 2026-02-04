from google import genai
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from config import MainConfig


hydra.initialize(version_base=None, config_path="conf")
cfg = hydra.compose(config_name="config")
raw_config = OmegaConf.to_container(cfg, resolve=True)
main_cfg = MainConfig(**raw_config)
# API 키 설정 (직접 입력하거나 환경변수)
client = genai.Client(api_key=main_cfg.client)

print("=== 내 API 키로 사용 가능한 모델 목록 ===")
try:
    # 모델 목록을 다 가져와서 'generateContent' 기능이 있는 것만 보여줌
    for model in client.models.list():
        if "generateContent" in model.supported_actions:
            print(f"- {model.name}")
            # 별명 말고 resource_name(본명)도 확인
            print(f"  (ID: {model.name.split('/')[-1]})")
except Exception as e:
    print(f"에러 발생: {e}")
