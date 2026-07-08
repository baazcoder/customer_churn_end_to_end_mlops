from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.pipeline.prediction_pipeline import ChurnData, ChurnDataClassifier

app = FastAPI(
    title="Customer Churn Prediction",
    description="End-to-End MLOps Project",
    version="1.0"
)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# HTML Templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        "churn.html",
        {"request": request}
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,

    gender: str = Form(...),
    SeniorCitizen: int = Form(...),
    Partner: str = Form(...),
    Dependents: str = Form(...),
    tenure: int = Form(...),
    PhoneService: str = Form(...),
    MultipleLines: str = Form(...),
    InternetService: str = Form(...),
    OnlineSecurity: str = Form(...),
    OnlineBackup: str = Form(...),
    DeviceProtection: str = Form(...),
    TechSupport: str = Form(...),
    StreamingTV: str = Form(...),
    StreamingMovies: str = Form(...),
    Contract: str = Form(...),
    PaperlessBilling: str = Form(...),
    PaymentMethod: str = Form(...),
    MonthlyCharges: float = Form(...),
    TotalCharges: float = Form(...)
):

    customer = ChurnData(

        gender=gender,
        SeniorCitizen=SeniorCitizen,
        Partner=Partner,
        Dependents=Dependents,
        tenure=tenure,
        PhoneService=PhoneService,
        MultipleLines=MultipleLines,
        InternetService=InternetService,
        OnlineSecurity=OnlineSecurity,
        OnlineBackup=OnlineBackup,
        DeviceProtection=DeviceProtection,
        TechSupport=TechSupport,
        StreamingTV=StreamingTV,
        StreamingMovies=StreamingMovies,
        Contract=Contract,
        PaperlessBilling=PaperlessBilling,
        PaymentMethod=PaymentMethod,
        MonthlyCharges=MonthlyCharges,
        TotalCharges=TotalCharges

    )

    data = customer.get_churn_input_data_frame()

    classifier = ChurnDataClassifier()

    prediction = classifier.predict(data)[0]

    if prediction == 1:
        result = "⚠️ Customer is likely to Churn"
    else:
        result = "✅ Customer is likely to Stay"

    return templates.TemplateResponse(
        "churn.html",
        {
            "request": request,
            "prediction": result
        }
    )
# dj

@app.get("/health")
async def health():

    return {
        "status": "healthy"
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)