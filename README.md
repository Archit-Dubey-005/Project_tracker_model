# Intelligent Progress Tracking (IPT) - Render Deployment Package

This self-contained package contains everything needed to deploy your NLP matching model as a high-performance REST API on **Render** (or any Docker-compatible platform like Railway or Hugging Face Spaces).

---

## 📁 Directory Structure

```
render_deploy/
├── Dockerfile              # Production container build with CPU PyTorch & FAISS
├── requirements.txt        # Python dependencies
├── render.yaml             # Render 1-click blueprint configuration
├── main.py                 # FastAPI application with lifecycle model loader & endpoints
├── src/                    # Matcher logic (IPTMatcher)
│   ├── __init__.py
│   └── matcher.py
└── models/                 # Model artifacts
    ├── activity_index.faiss
    ├── activity_lookup.pkl
    ├── embeddings.npy
    ├── model.pkl
    └── progress_tracking_report.json
```

---

## 🚀 How to Deploy on Render (Step-by-Step)

### Step 1: Push this folder to a GitHub Repository
Open a terminal inside this `render_deploy` folder:

```bash
cd "c:\Users\98795\OneDrive\Desktop\Intelligent Progress Tracking (IPT)1\render_deploy"
git init
git add .
git commit -m "Initial commit for IPT model API"
git branch -M main
git remote add origin https://github.com/<your-github-username>/<your-repo-name>.git
git push -u origin main
```

*(Note: If you already have a repository, you can push this as a standalone repo or subfolder).*

### Step 2: Create a Web Service on Render
1. Log in to [Render.com](https://render.com).
2. Click **New +** -> **Web Service**.
3. Select **Build and deploy from a Git repository** and connect your GitHub repo.
4. Configure the service:
   - **Name**: `ipt-model-api`
   - **Region**: Closest to your users (e.g. Oregon, Frankfurt, Singapore)
   - **Branch**: `main`
   - **Runtime**: **Docker** (Render will automatically detect `Dockerfile`)
   - **Instance Type**: **Free** (or Starter $7/mo for 0 cold starts)
5. (Optional) In **Environment Variables**, add:
   - `API_SECRET_KEY`: `your_custom_secret_key` *(if you want to protect endpoints)*
6. Click **Create Web Service**.

Render will now build your Docker image, install PyTorch, and launch the service.
Once complete, you will receive a public HTTPS URL:
`https://ipt-model-api.onrender.com`

---

## 🔗 How to Connect to Your Vercel Website

1. Go to your **Vercel Dashboard** -> Your Project -> **Settings** -> **Environment Variables**.
2. Add:
   - `IPT_API_URL` = `https://ipt-model-api.onrender.com`
   - `IPT_API_KEY` = `your_custom_secret_key` (if configured)
3. In your Next.js / Node API route, make a `fetch()` call to `${process.env.IPT_API_URL}/api/v1/match`.

---

## 🧪 Local Testing

You can test this API locally before deploying:

```bash
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Visit:
- Interactive Swagger Docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
