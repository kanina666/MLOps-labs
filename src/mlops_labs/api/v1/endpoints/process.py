from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

from mlops_labs.api.dependencies import extract_features, get_model
from mlops_labs.api.schemas import ProcessResponse

router = APIRouter()


@router.post("/process", response_model=ProcessResponse, status_code=status.HTTP_200_OK)
async def process(
    features_df: Annotated[pd.DataFrame, Depends(extract_features)],
    model: Annotated[Any, Depends(get_model)],
) -> ProcessResponse:
    try:
        prediction = await run_in_threadpool(model.predict, features_df)
        pred_value = float(prediction[0])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference error: {exc}",
        ) from exc

    return ProcessResponse(prediction=round(pred_value, 4))
