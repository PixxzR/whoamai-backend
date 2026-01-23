# FaceSense API - Etat du projet

> Derniere mise a jour : 23 janvier 2026

---

## Ce qui est FAIT et FONCTIONNEL

### Infrastructure
- [x] Structure projet complete (app/, models/, tests/, scripts/, docs/)
- [x] Configuration Git (main + dev branches, .gitignore)
- [x] CI/CD GitHub Actions (lint + tests sur push)
- [x] Pre-commit hooks (black, isort, flake8)
- [x] pyproject.toml (config outils)
- [x] requirements.txt + requirements-dev.txt
- [x] .env.example avec toutes les variables
- [x] Script setup_dev.sh
- [x] README.md complet

### Application
- [x] FastAPI app avec lifespan, CORS, middleware logging, exception handler
- [x] Configuration Pydantic (app/config.py) charge le .env
- [x] Swagger UI auto-genere sur /docs

### Detection de visages
- [x] MTCNN (facenet-pytorch) charge au demarrage
- [x] Detection temps reel avec bounding box + confidence
- [x] Extraction du visage aligne en tensor

### Endpoints API
| Endpoint | Status | Description |
|----------|--------|-------------|
| GET /health | OK | Status, version, nb modeles, demo_mode |
| POST /predict/specialized | OK (demo) | Detection reelle + predictions aleatoires |
| POST /predict/multitask | OK (demo) | Detection reelle + predictions aleatoires |
| POST /predict/transfer | OK (demo) | Detection reelle + predictions aleatoires |

### Validations
- [x] Taille image max (10MB) -> 413
- [x] Format image invalide -> 400
- [x] Aucun visage detecte -> 200 avec face_detected: false

### Tests
- [x] 7 tests passent (health + predict no-face + invalid + oversized)

---

## Ce qui reste a faire

### Phase 1 : Modeles et inference reelle

#### 1.1 Choix d'architecture reseau
> **Decision a prendre** : quelle architecture pour les classifieurs ?

Options possibles :
- **ResNet-18/34** : leger, bon pour du CPU, ~11M params
- **EfficientNet-B0** : meilleur ratio accuracy/performance, ~5M params
- **MobileNetV3** : optimise mobile/edge, tres leger
- **VGG-Face / ArcFace** : specialise visages

Criteres :
- Inference sur CPU (pas de GPU cote serveur pour l'instant)
- 4 attributs a predire : age, genre, ethnie, emotion
- Dataset d'entrainement disponible ?

#### 1.2 Strategie "Specialized" (un modele par attribut)
- [ ] Definir architecture du reseau (ex: ResNet-18 + FC layer)
- [ ] Entrainer modele age (ou recuperer pre-entraine)
- [ ] Entrainer modele gender
- [ ] Entrainer modele ethnicity
- [ ] Entrainer modele emotion
- [ ] Exporter en .pth dans models/specialized/
- [ ] Implementer inference reelle dans ModelManager.predict_specialized()

#### 1.3 Strategie "Multitask" (un modele, plusieurs tetes)
- [ ] Definir architecture multi-tetes (backbone partage + 4 heads)
- [ ] Entrainer le modele multitask
- [ ] Exporter en .pth dans models/multitask/
- [ ] Implementer inference reelle dans ModelManager.predict_multitask()

#### 1.4 Strategie "Transfer Learning"
- [ ] Choisir modele pre-entraine (VGGFace2, ArcFace, etc.)
- [ ] Fine-tuner sur les attributs cibles
- [ ] Exporter en .pth dans models/transfer/
- [ ] Implementer inference reelle dans ModelManager.predict_transfer()

#### 1.5 Preprocessing
- [ ] Adapter la taille d'entree selon l'architecture choisie
- [ ] Verifier la normalisation (ImageNet vs specifique visage)

---

### Phase 2 : Ameliorations API

- [ ] Endpoint POST /predict/compare : comparer les 3 strategies sur une image
- [ ] Temps d'inference dans la reponse (champ "inference_time_ms")
- [ ] Support multi-visages (keep_all=True dans MTCNN)
- [ ] Parametre optionnel "strategy" pour un endpoint unique /predict
- [ ] Pagination / batch prediction (plusieurs images)

---

### Phase 3 : Production-ready

- [ ] Dockerfile + docker-compose.yml
- [ ] Support GPU/MPS (Apple Silicon) avec detection auto device
- [ ] Cache inference (meme image = meme resultat)
- [ ] Rate limiting (slowapi ou middleware custom)
- [ ] Metriques Prometheus (/metrics endpoint)
- [ ] Logging structure JSON (pour ingestion ELK/Loki)
- [ ] Health check avance (verifier que MTCNN repond, RAM usage)
- [ ] Graceful shutdown (attendre fin des requetes en cours)

---

### Phase 4 : Tests et qualite

- [ ] Tests avec vraies images de visages (fixtures)
- [ ] Tests de performance (temps inference acceptable < 2s CPU)
- [ ] Tests de charge (locust ou k6)
- [ ] Coverage > 80%
- [ ] Typing complet (mypy strict)

---

## Architecture actuelle

```
Request (image)
    |
    v
[Validation] -- trop gros? -> 413
    |             invalide? -> 400
    v
[MTCNN Detection] -- pas de visage? -> {face_detected: false}
    |
    v
[Face Tensor extraction]
    |
    v
[ModelManager.predict_*()]  <-- DEMO MODE actuellement
    |
    v
[PredictionResponse JSON]
```

## Stack technique

| Composant | Techno | Version |
|-----------|--------|---------|
| Framework | FastAPI | >= 0.115 |
| Server | Uvicorn | >= 0.34 |
| Detection | MTCNN (facenet-pytorch) | >= 2.6 |
| ML | PyTorch | >= 2.5 |
| Validation | Pydantic v2 | >= 2.10 |
| Tests | Pytest | >= 7.4 |
| Python | | 3.10+ |

## Comment lancer

```bash
# Setup
./scripts/setup_dev.sh
# ou manuellement :
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Lancer
uvicorn app.main:app --reload

# Tester
curl -X POST http://localhost:8000/predict/specialized -F "file=@photo.jpg"

# Tests
pytest tests/ -v
```
