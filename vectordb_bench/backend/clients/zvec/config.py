from pydantic import BaseModel

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class ZvecConfig(DBConfig):
    """Zvec connection configuration."""

    db_label: str
    path: str

    def to_dict(self) -> dict:
        return {
            "path": self.path,
        }


class ZvecIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {}


class ZvecHNSWIndexConfig(ZvecIndexConfig):
    M: int | None = 50
    ef_construction: int | None = 500

    ef_search: int | None = 300

    quantize_type: str = ""

    is_using_refiner: bool = False


class ZvecOMEGAIndexConfig(ZvecIndexConfig):
    """OMEGA index configuration - ML-based adaptive early stopping for HNSW."""

    # HNSW base parameters
    M: int | None = 50
    ef_construction: int | None = 500

    # Query parameters
    ef_search: int | None = 300

    quantize_type: str = ""

    is_using_refiner: bool = False

    # OMEGA-specific parameters
    min_vector_threshold: int = 100000
    num_training_queries: int = 1000
    ef_training: int = 1000
    window_size: int = 100
    ef_groundtruth: int = 0  # 0 = brute force, >0 = HNSW with this ef (faster)

    # OMEGA query parameter
    target_recall: float = 0.95


# Dictionary mapping IndexType to config class
_zvec_case_config = {
    IndexType.HNSW: ZvecHNSWIndexConfig,
    IndexType.OMEGA: ZvecOMEGAIndexConfig,
}
