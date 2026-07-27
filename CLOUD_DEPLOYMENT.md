# Free cloud deployment

The local Mac is no longer required once these services are configured:

1. Create a free Qdrant Cloud cluster and set GitHub Actions secrets
   `QDRANT_URL` and `QDRANT_API_KEY`.
2. Create a public Hugging Face Dataset repository and set `HF_TOKEN` and
   `HF_DATASET_REPO` secrets.
3. Push this repository to GitHub. The workflow in
   `.github/workflows/meme-sync.yml` runs every 15 minutes.
4. Deploy `streamlit_app.py` to Streamlit Community Cloud and add the Qdrant
   values as Streamlit secrets.

GitHub Actions is the scheduled worker; Streamlit only serves the UI. No local
Qdrant, FastAPI process, or Mac needs to remain online.
