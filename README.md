# FaceSense API

API de prediction d'attributs faciaux basee sur 3 strategies de deep learning :
- **Specialized** : un modele par attribut (age, genre, ethnie, emotion)
- **Multitask** : un seul modele multi-sorties
- **Transfer Learning** : fine-tuning d'un modele pre-entraine

## Prerequis

- Python 3.10+
- pip

## Installation

```bash
# Cloner le repo
git clone <repo-url>
cd whoamai-backend

# Setup automatique
chmod +x scripts/setup_dev.sh
./scripts/setup_dev.sh

# Ou manuellement :
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Configuration

Copier `.env.example` vers `.env` et ajuster les valeurs :

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_NAME` | Nom de l'application | FaceSense API |
| `DEBUG` | Mode debug | true |
| `PORT` | Port du serveur | 8000 |
| `DEVICE` | Device PyTorch (cpu/cuda) | cpu |
| `MODELS_DIR` | Repertoire des modeles | ./models |
| `MAX_IMAGE_SIZE_MB` | Taille max image | 10 |

## Lancement

```bash
# Developpement (hot-reload)
uvicorn app.main:app --reload

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Tests

```bash
# Lancer tous les tests
pytest

# Avec couverture
pytest --cov=app tests/

# Tests specifiques
pytest tests/test_health.py -v
```

## Endpoints API

| Methode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict/specialized` | Prediction modeles specialises |
| POST | `/predict/multitask` | Prediction modele multitache |
| POST | `/predict/transfer` | Prediction transfer learning |

### Exemple d'appel

```bash
curl -X POST http://localhost:8000/predict/specialized \
  -F "file=@photo.jpg"
```

## Structure du projet

```
whoamai-backend/
├── app/
│   ├── api/routes/        # Endpoints FastAPI
│   ├── core/              # Logique metier (detection, inference)
│   ├── schemas/           # Schemas Pydantic
│   ├── utils/             # Utilitaires
│   ├── config.py          # Configuration Settings
│   └── main.py            # Point d'entree FastAPI
├── models/                # Modeles PyTorch (.pth)
│   ├── specialized/
│   ├── multitask/
│   └── transfer/
├── tests/                 # Tests pytest
├── scripts/               # Scripts utilitaires
├── docs/                  # Documentation
├── .github/workflows/     # CI/CD
├── requirements.txt       # Dependencies production
├── requirements-dev.txt   # Dependencies dev
├── pyproject.toml         # Config outils (black, isort, pytest)
└── .env.example           # Template configuration
```

## Workflow Git

```
main        Production-ready, protege
  └── dev   Developpement en cours
       └── feature/xxx   Branches feature
```

### Conventions de branches

- `main` : code stable, deploye en production
- `dev` : integration des features en cours
- `feature/nom` : nouvelles fonctionnalites
- `fix/nom` : corrections de bugs
- `refactor/nom` : refactoring

### Conventions de commits

Format : `type: description`

Types : `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `style`

```bash
git checkout dev
git checkout -b feature/face-detection
# ... developpement ...
git add .
git commit -m "feat: implement MTCNN face detection"
git push origin feature/face-detection
# Creer Pull Request vers dev
```

## Contribution

1. Creer une branche depuis `dev`
2. Implementer la feature avec tests
3. Verifier : `black .`, `flake8 .`, `pytest`
4. Push et creer une Pull Request vers `dev`
5. Review par un pair avant merge

## TODO

- [ ] Implementer detection MTCNN
- [ ] Implementer chargement modeles PyTorch
- [ ] Implementer pipeline inference specialized
- [ ] Implementer pipeline inference multitask
- [ ] Implementer pipeline inference transfer
- [ ] Ajouter validation taille/format image
- [ ] Ajouter cache modeles
- [ ] Ajouter metriques Prometheus
- [ ] Dockeriser l'application
- [ ] Deployer sur cloud
