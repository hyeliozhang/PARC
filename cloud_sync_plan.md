# Cloud Sync Plan

- Cloud root: `/data/ICDE2027/PARC`
- Repository checkout: `/data/ICDE2027/PARC/repo`
- Environment path: `/data/ICDE2027/PARC/env`
- Data path: `/data/ICDE2027/PARC/data`
- Run path: `/data/ICDE2027/PARC/runs`
- Artifact path: `/data/ICDE2027/PARC/artifacts`
- Scratch path: `/data/ICDE2027/PARC/scratch`

Suggested cloud checkout:

```bash
git clone https://github.com/hyeliozhang/PARC.git /data/ICDE2027/PARC/repo
cd /data/ICDE2027/PARC/repo
python -m venv /data/ICDE2027/PARC/env
source /data/ICDE2027/PARC/env/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Secrets policy: no API keys, GitHub tokens, cloud credentials, SSH keys, or `.env` files are committed or synced. Large rerun outputs should stay under the cloud run/artifact paths unless explicitly summarized into small public CSVs.
