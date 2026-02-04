from pydantic import BaseModel, Field
import hydra
from omegaconf import DictConfig, OmegaConf

from pydantic import BaseModel, Field
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import List

class MainConfig(BaseModel):
    client: str
    prompt: str
    discord_token: str
    model: str
    keywords: List[str]
    site: str


if __name__ == "__main__":
    # 1. Hydra 초기화 (config_path는 현재 파일 대비 상대 경로)
    with hydra.initialize(version_base=None, config_path="conf"):
        # 2. 설정을 하나로 합치기 (defaults에 정의된 파일들을 실제로 다 불러옴)
        cfg = hydra.compose(config_name="config")
        
        # 3. OmegaConf 객체를 일반 딕셔너리로 변환
        raw_config = OmegaConf.to_container(cfg, resolve=True)
        
        print(f"DEBUG: raw_config content -> {raw_config}")
        
        # 4. 이제 Pydantic 모델로 검증
        try:
            main_cfg = MainConfig(**raw_config)
            print("✅ Config validation successful!")
            print(f"main_cfg: {main_cfg}")
        except Exception as e:
            print(f"❌ Validation failed: {e}")