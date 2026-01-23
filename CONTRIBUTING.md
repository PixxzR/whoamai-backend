# Guide de contribution - FaceSense API

## Setup initial (une seule fois)

```bash
git clone https://github.com/PixxzR/whoamai-backend.git
cd whoamai-backend
./scripts/setup_dev.sh
```

Verifie que tout marche :
```bash
source venv/bin/activate
pytest tests/ -v
uvicorn app.main:app --reload
# -> http://localhost:8000/health doit repondre
```

---

## Workflow Git quotidien

### 1. Toujours partir de `dev` a jour

```bash
git checkout dev
git pull origin dev
```

### 2. Creer ta branche feature

```bash
git checkout -b feature/nom-de-ta-feature
# ou
git checkout -b fix/description-du-bug
```

Conventions de nommage :
- `feature/xxx` : nouvelle fonctionnalite
- `fix/xxx` : correction de bug
- `refactor/xxx` : refactoring sans changement fonctionnel
- `test/xxx` : ajout de tests
- `docs/xxx` : documentation

### 3. Developper

```bash
# Lancer le serveur en dev
uvicorn app.main:app --reload

# Tester ton endpoint
curl -X POST http://localhost:8000/predict/specialized -F "file=@photo.jpg"

# Lancer les tests regulierement
pytest tests/ -v
```

### 4. Commit tes changements

Format des commits : `type: description courte`

```bash
git add <fichiers modifies>
git commit -m "feat: implement age prediction model"
```

Types de commits :
| Type | Utilisation |
|------|------------|
| `feat` | Nouvelle feature |
| `fix` | Bug fix |
| `refactor` | Refactoring |
| `test` | Ajout/modif tests |
| `docs` | Documentation |
| `chore` | Config, CI, deps |
| `style` | Formatage (pas de changement logique) |

### 5. Push et Pull Request

```bash
git push origin feature/nom-de-ta-feature
```

Puis sur GitHub :
1. Creer une Pull Request vers `dev`
2. Remplir le template PR
3. Assigner un reviewer (quelqu'un d'autre du groupe)
4. Attendre que la CI passe (vert)
5. Attendre l'approbation du reviewer
6. Merge (Squash and merge recommande)

### 6. Apres le merge

```bash
git checkout dev
git pull origin dev
git branch -d feature/nom-de-ta-feature  # supprimer la branche locale
```

---

## Regles importantes

### A faire
- Toujours creer une PR (jamais push direct sur dev ou main)
- Toujours avoir au moins 1 review avant merge
- Toujours verifier que les tests passent avant push
- Mettre a jour PROJECT_STATUS.md quand une tache est terminee
- Ecrire des tests pour les nouvelles features

### A ne PAS faire
- Ne PAS push sur `main` directement
- Ne PAS merge sans review
- Ne PAS commit de fichiers .env, credentials, ou modeles .pth
- Ne PAS commit de code non formate (utiliser `black .` avant)
- Ne PAS modifier le travail des autres sans prevenir

---

## Style de code

Le formatage est automatique via pre-commit hooks :

```bash
# Formater manuellement si besoin
black app/ tests/
isort app/ tests/

# Verifier le lint
flake8 app/ tests/ --max-line-length=88 --extend-ignore=E203
```

Regles :
- Ligne max : 88 caracteres (config black)
- Imports tries par isort (profil black)
- Docstrings en francais ou anglais (rester coherent dans un fichier)
- Type hints encourages mais pas obligatoires pour l'instant

---

## Structure des fichiers

Ou mettre quoi :

| Je veux... | Je mets dans... |
|------------|----------------|
| Un nouvel endpoint | `app/api/routes/` |
| De la logique ML/detection | `app/core/` |
| Un schema Pydantic | `app/schemas/` |
| Une fonction utilitaire | `app/utils/` |
| Un test | `tests/test_*.py` |
| Un script one-shot | `scripts/` |

---

## Repartition du travail suggeree

Pour eviter les conflits, chaque personne travaille sur un module :

| Personne | Module | Fichiers principaux |
|----------|--------|-------------------|
| Dev 1 | Strategie Specialized | `app/core/inference.py` (specialized) |
| Dev 2 | Strategie Multitask | `app/core/inference.py` (multitask) |
| Dev 3 | Strategie Transfer | `app/core/inference.py` (transfer) |
| Dev 4 | API + Integration | `app/api/routes/`, tests, docs |

> Adaptez selon vos preferences. L'important est de ne pas travailler sur les memes fichiers en meme temps.

---

## En cas de conflit de merge

```bash
# Mettre a jour ta branche avec dev
git checkout ta-branche
git fetch origin
git rebase origin/dev

# Resoudre les conflits manuellement puis :
git add <fichiers resolus>
git rebase --continue

# Push force (seulement sur TA branche feature, jamais sur dev/main)
git push --force-with-lease origin ta-branche
```

---

## Communication

- Utiliser les **Issues GitHub** pour tracker les taches
- Utiliser les **PR comments** pour les discussions techniques
- Prevenir le groupe avant de modifier un fichier partage (main.py, config.py, schemas/)
