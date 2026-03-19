from typing import Annotated, Unpack

import click

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB


class ZvecTypedDict(CommonTypedDict):
    path: Annotated[
        str,
        click.option("--path", type=str, help="collection path", required=True),
    ]


class ZvecHNSWTypedDict(CommonTypedDict, ZvecTypedDict):
    m: Annotated[
        int,
        click.option("--m", type=int, default=50, help="HNSW index parameter m."),
    ]
    ef_construct: Annotated[
        int,
        click.option("--ef-construction", type=int, default=500, help="HNSW index parameter ef_construction"),
    ]
    ef_search: Annotated[
        int,
        click.option("--ef-search", type=int, default=300, help="HNSW index parameter ef for search"),
    ]
    quantize_type: Annotated[
        int,
        click.option("--quantize-type", type=str, default="", help="HNSW index quantize type, fp16/int8 supported"),
    ]
    is_using_refiner: Annotated[
        bool,
        click.option(
            "--is-using-refiner",
            is_flag=True,
            default=False,
            help="is using refiner, suitable for quantized index, "
            "recall `ef-search` results then refine with unquantized vector to `topk` results",
        ),
    ]


class ZvecOMEGATypedDict(ZvecHNSWTypedDict):
    """OMEGA index parameters - extends HNSW with ML-based adaptive early stopping."""
    min_vector_threshold: Annotated[
        int,
        click.option("--min-vector-threshold", type=int, default=100000,
                     help="Minimum vectors required to enable OMEGA optimization"),
    ]
    num_training_queries: Annotated[
        int,
        click.option("--num-training-queries", type=int, default=1000,
                     help="Number of training queries for OMEGA model training"),
    ]
    ef_training: Annotated[
        int,
        click.option("--ef-training", type=int, default=1000,
                     help="Candidate list size (ef) used during training searches"),
    ]
    window_size: Annotated[
        int,
        click.option("--window-size", type=int, default=100,
                     help="Sliding window size for distance statistics"),
    ]
    ef_groundtruth: Annotated[
        int,
        click.option("--ef-groundtruth", type=int, default=0,
                     help="ef for ground truth computation (0=brute force, >0=HNSW for faster training)"),
    ]
    target_recall: Annotated[
        float,
        click.option("--target-recall", type=float, default=0.95,
                     help="Target recall for OMEGA early stopping (0.0 to 1.0)"),
    ]


# default to hnsw
@cli.command()
@click_parameter_decorators_from_typed_dict(ZvecHNSWTypedDict)
def Zvec(**parameters: Unpack[ZvecHNSWTypedDict]):
    from .config import ZvecConfig, ZvecHNSWIndexConfig

    run(
        db=DB.Zvec,
        db_config=ZvecConfig(
            db_label=parameters["db_label"],
            path=parameters["path"],
        ),
        db_case_config=ZvecHNSWIndexConfig(
            M=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            quantize_type=parameters["quantize_type"],
            is_using_refiner=parameters["is_using_refiner"],
        ),
        **parameters,
    )


# OMEGA index command
@cli.command()
@click_parameter_decorators_from_typed_dict(ZvecOMEGATypedDict)
def ZvecOmega(**parameters: Unpack[ZvecOMEGATypedDict]):
    """Run benchmark with OMEGA index (ML-based adaptive early stopping for HNSW)."""
    from .config import ZvecConfig, ZvecOMEGAIndexConfig

    run(
        db=DB.Zvec,
        db_config=ZvecConfig(
            db_label=parameters["db_label"],
            path=parameters["path"],
        ),
        db_case_config=ZvecOMEGAIndexConfig(
            M=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            quantize_type=parameters["quantize_type"],
            is_using_refiner=parameters["is_using_refiner"],
            min_vector_threshold=parameters["min_vector_threshold"],
            num_training_queries=parameters["num_training_queries"],
            ef_training=parameters["ef_training"],
            window_size=parameters["window_size"],
            ef_groundtruth=parameters["ef_groundtruth"],
            target_recall=parameters["target_recall"],
        ),
        **parameters,
    )
