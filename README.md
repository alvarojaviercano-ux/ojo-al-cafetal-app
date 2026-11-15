# Eye on the Coffee Farm – coffee leaf inspection system

Information system for inspecting coffee farms. A grower registers farms and plots, uploads a photo of a leaf, and the system (Random Forest + K-Means + Q-Learning) says if the leaf looks healthy or sick, shows the damaged zone and suggests actions. Technicians review results; administrators manage users and watch the model.

- Live system: https://ojo-al-cafetal-app.onrender.com
- Data: RoCoLe (robusta), CC BY 4.0. Trained locally; only the trained models are deployed.

## Roles
| Role | Can do |
|---|---|
| Grower | Own farms, plots, inspections, actions, feedback |
| Technician | See all farms, review inspections, answer feedback, monitoring, reports |
| Administrator | Everything, plus users, event log, delete inspections |

Self-registration always creates a grower. Only an administrator creates other roles.

## Run locally
```
pip install -r requirements.txt
python app.py          # http://127.0.0.1:5000  (SQLite file sistema.db)
python selftest.py     # end-to-end test on a temporary database
```
Without `DATABASE_URL` the app uses SQLite. Copy `.env.example` to set accounts.

## Database
PostgreSQL in production. Tables (`users, farms, plots, inspections, recommendations, feedback, event_log, settings`) are created automatically at start. No manual SQL is needed.

## Deploy on Render
1. New → PostgreSQL (free or paid). Copy the **Internal Database URL**.
2. Web Service → this repository. Build: `pip install -r requirements.txt`. Start: `gunicorn app:app --timeout 120 --workers 1`.
3. Environment: `DATABASE_URL`, `SECRET_KEY`, and the `ADMIN_/TECHNICIAN_/GROWER_` user and password variables.
4. Deploy. Check `/health`.

The `scikit-learn` version on the server must be the same as the one used to train `models/modelo_app.joblib`. Pin it in `requirements.txt` (`pip freeze | findstr scikit` on your computer).

## Files
`app.py` routes and rules · `database.py` DB layer · `caracteristicas.py` leaf cut + 26 features · `models/` trained models and monitoring reference · `templates/` pages · `selftest.py` tests.

Test accounts are given privately to the instructor, not in this repository.
