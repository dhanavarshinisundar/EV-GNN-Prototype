
# EV GNN Charging Recommendation Prototype

This is a presentation-ready student prototype based on the workflow in the supplied PPT:

Interaction graph → learned preference → Top-N → spatial cluster refinement → charging-type refinement → final ranking.

## Important
This demo does NOT claim to reproduce or retrain the research paper's NGCF/LightGCN/DGCF models.
The "learned preference" stage is simulated using historical driver-station usage so the prototype is easy to run and explain.

## Run on Windows

1. Install Python 3.10+.
2. Open Command Prompt in this folder.
3. Run:

    pip install -r requirements.txt

4. Start:

    streamlit run app.py

5. Open the local URL shown by Streamlit, usually:

    http://localhost:8501

## What to demonstrate

Select:
- Driver: D44
- Context: Workplace
- Charging preference: FAST
- Top-N: 5

Then explain:
1. Historical usage creates the driver-station interaction signal.
2. Preference score ranks candidates.
3. Spatial cluster keeps geographically relevant stations.
4. Charging type refines the list.
5. The app explains WHY each station was recommended.

## PPT mapping

- Slide 17: dataset + graph + Python UI
- Slide 18: input + Top-N recommendations + explainable result
- Slide 19: future scope — real-time station APIs, queue length, availability, malfunction status
