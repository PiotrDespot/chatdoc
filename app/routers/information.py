import os
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse


router = APIRouter(
    prefix="/info",
    tags=["info"]
)


@router.get("/models")
async def get_models():
    models_data = []

    model_paths = ["mistral:latest", "/path/to/model2"]

    for model_path in model_paths:
        model_name = model_path
        modified_at = datetime.now().isoformat()
        size = 10000

        model_info = {
            "name": model_name,
            "modified_at": modified_at,
            "size": size
        }

        models_data.append(model_info)

    response_data = {"models": models_data}
    return JSONResponse(content=response_data)


def get_model_details(model_name):
    # Example model path, replace this with your logic to fetch the actual model details
    model_path = f"/path/to/{model_name}"

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    modified_at = datetime.utcfromtimestamp(os.path.getmtime(model_path)).isoformat()
    size = os.path.getsize(model_path)

    # Replace the following placeholders with your logic to fetch specific model details
    license_content = "<contents of license block>"
    modelfile_content = "# Modelfile content"
    parameters_content = "stop                           [INST]\nstop                           [/INST]\nstop                           <<SYS>>\nstop                           <</SYS>>"
    template_content = "[INST] {{ if and .First .System }}<<SYS>>{{ .System }}<</SYS>>\n\n{{ end }}{{ .Prompt }} [/INST] "

    model_details = {
        "license": license_content,
        "modelfile": modelfile_content,
        "parameters": parameters_content,
        "template": template_content,
        "modified_at": modified_at,
        "size": size
    }

    return model_details


@router.post("/show")
async def show_model(name: str):
    try:
        model_details = get_model_details(name)
        return JSONResponse(content=model_details)
    except HTTPException as e:
        return JSONResponse(content={"error": str(e)}, status_code=e.status_code)

