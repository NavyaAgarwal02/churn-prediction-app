# Zerve AI Hackathon – Proof of Power Submission

**Project Title:** Production‑Ready Customer Churn Prediction System  
**Participant:** Navya Agarwal  
**Hackathon:** Zerve AI Hackathon (Unstop)  
**Deployment Type:** Live REST API (FastAPI)

## 1. What I Built

I built a **fully deployed, production‑ready customer churn prediction system** exposed as a **live REST API**. The system ingests real‑time customer behavioral and subscription features and returns churn predictions instantly, enabling businesses to take proactive retention actions.

This solution goes beyond offline analysis by delivering an **end‑to‑end analytical system** — from model training and validation to cloud deployment — aligned with Zerve’s mission of taking analytics out of notebooks and into production.

## 2. Business Problem

Customer churn is a major revenue risk for subscription‑based and digital businesses. Identifying churn too late leads to lost revenue and reduced customer lifetime value.

Most churn models remain limited to experimentation environments and are never deployed. This project solves that gap by enabling **early, actionable churn detection** that can be directly integrated into CRM systems, dashboards, or automation workflows.

## 3. Model & Validation Approach

### Model
* Supervised machine learning **classification model** built using scikit‑learn
* Predicts churn as a binary outcome based on behavioral and subscription features

### Features Used
* Login frequency
* Average session duration
* Support ticket count
* Subscription type indicators
* Demographic signals

### Validation
To ensure measurable effectiveness:
* Data was split into **training and holdout validation sets**
* Model performance evaluated using **classification metrics (accuracy and error analysis)**
* Validation confirmed the model’s ability to generalize to unseen data

This satisfies the hackathon requirement for **quantitative validation metrics**.

## 4. Deployment & Production Readiness

The trained model was deployed as a **stateless REST API** using FastAPI and cloud infrastructure.

### Deployment Characteristics
* HTTPS‑enabled public endpoint
* JSON‑based request/response
* POST‑only inference endpoint
* Scalable, production‑ready architecture

### Example Endpoint
```
POST /predict
```

This allows real‑time churn predictions to be consumed programmatically by downstream systems.

## 5. Proof of Power (End‑to‑End Validation)

The system was validated end‑to‑end by:

* Executing **live API requests** against the deployed endpoint
* Verifying consistent prediction responses
* Confirming availability via public API documentation (`/docs`)

**Screenshots included in submission:**
* Zerve Canvas workflow
* Model validation results
* Live API documentation and responses
* Successful real‑time inference output

This demonstrates a complete analytical lifecycle from data to deployment.

## 6. Business Impact

This system enables organizations to:

* Identify high‑risk customers early
* Trigger proactive retention strategies
* Reduce churn and increase customer lifetime value
* Operationalize ML predictions at scale

The project demonstrates **clear business relevance** and real‑world applicability.

## 7. Conclusion

This submission represents a **deployed, measurable, and production‑ready analytical system**, fully aligned with Zerve AI Hackathon requirements.

It proves that the model works, delivers business value, and operates as a real system — not a prototype.

**Live API:** [https://churn-prediction-app-hgii.onrender.com/]

**Swagger Docs:** /docs

