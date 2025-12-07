from inference import predict

samples = [
    {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": "1",
        "PhoneService": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": "29.85",
        "TotalCharges": "29.85",
        "InternetService": "Fiber optic"
    },
    {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": "36",
        "PhoneService": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": "56.95",
        "TotalCharges": "2000.00",
        "InternetService": "DSsrcL"
    }
]

for s in samples:
    try:
        print("INPUT:", s)
        print("OUTPUT:", predict(s))
    except Exception as e:
        print("Failed for sample:", e)
