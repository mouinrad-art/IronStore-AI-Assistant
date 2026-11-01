"""Run locally with OPENAI_API_KEY to generate the read-only index deployed on Vercel."""
import os,sys
from pathlib import Path
from dotenv import load_dotenv
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.rag import build_index
load_dotenv(ROOT/'.env')
if not os.getenv('OPENAI_API_KEY'):
    raise SystemExit('Missing OPENAI_API_KEY. Set it in .env before indexing.')
count=build_index(ROOT/'documents',ROOT/'index',os.environ['OPENAI_API_KEY'],print)
print(f'Indexed {count} chunks. Commit the index/ folder or include it in the deployment.')
