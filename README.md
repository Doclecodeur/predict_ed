# PREDICT-ED

**Early prediction of hospital admission at adult emergency department triage**

A pre-alert model estimating, at triage, the probability that an adult patient will be admitted to hospital at the end of their emergency department (ED) visit, using structured triage data together with the free-text chief complaint.

> ⚠️ **An organizational support tool, not clinical decision support.**
> The model is designed as an **organizational pre-alert** for bed management (anticipating downstream capacity needs). It is not intended to guide an individual clinical decision, and has undergone neither prospective validation nor regulatory clearance.

Work carried out as part of the **DU Data Analytics — Université Paris 1 Panthéon-Sorbonne**, 2025-2026 cohort.

---

## Background

ED crowding and boarding (waiting for an inpatient bed after the admission decision) are major organizational problems. Anticipating the probability of admission as early as triage would allow bed managers to prepare downstream capacity sooner, without substituting for clinical judgement.

Most published work on this task relies on structured data alone and on a random data split. This project adds two elements:

- use of the **free-text chief complaint** (TF-IDF), alongside structured variables;
- a **strict temporal validation**, closer to real deployment conditions than a random split.

---

## Key results

**Cohort** — MIMIC-IV-ED: 383,919 ED stays, 188,127 distinct patients, 38.2% admission rate.

**Operational model** — hybrid multilayer perceptron (MLP), structured + TF-IDF, uncalibrated.

| Metric | Value |
|---|---|
| AUROC — temporal validation (2017-2019 anchor year group, n = 67,963) | **0.888** (95% CI 0.886–0.891) |
| AUROC — internal test set | 0.862 |
| Brier score | 0.131 |
| Expected calibration error (ECE) | 0.009 |

**Operating thresholds** (2017-2019 held-out set)

| Threshold | Value | Sensitivity | Specificity |
|---|---|---|---|
| Pre-alert (bed management) | 0.241 | 89.1% | 68.6% |
| Standard | 0.500 | 70.9% | 87.6% |
| High (bed discussion) | 0.650 | 58.1% | 93.3% |

**Algorithm comparison** (Model 3, internal validation)

| Algorithm | AUROC |
|---|---|
| MLP | 0.861 |
| Logistic regression | 0.853 |
| XGBoost | 0.850 |
| Random forest | 0.849 |
| ESI comparator alone | 0.693 |

**Incremental contribution of variable blocks** (logistic regression): vital signs + age + pain 0.741 → + context, comorbidities, ESI 0.805 → + chief-complaint NLP 0.853.

**Sensitivity analysis** — after removing the token "transfer" from the text (without excluding the corresponding patients) and retraining on an unchanged population, AUROC moves from 0.888 to **0.884**: the prediction does not rest on this vocabulary artefact.

---

## Pipeline

| Notebook | Purpose |
|---|---|
| `01_Construction_Cohorte_EDA` | Cohort construction, comorbidities, chief-complaint NLP preprocessing, exploratory analysis |
| `02_Pretraitement_NLP` | Lexical validation of the chief complaint |
| `03_Modele_avec_transfer` | Algorithm comparison (M1/M2/M3), operational model, internal test |
| `04_Modele_sans_transfer` | Sensitivity analysis without the "transfer" token |
| `05_Comparaison_avec_sans_transfer` | Comparison of the two versions |
| `06_Tableaux_Synthese` | Summary tables |
| `07_Analyse_avec_transfer` | Temporal validation, thresholds, calibration, subgroups |
| `08_BERT_vs_TFIDF` | Exploratory secondary analysis: TF-IDF vs BioClinicalBERT |

**Features** — 32 encoded structured variables (vital signs, age at visit, ESI acuity, pain, mode of arrival, sex, prior admissions, comorbidity score) + 1,000 TF-IDF text features (unigrams and bigrams), for a total of 1,032 features.

---

## Data

**This repository contains no patient data.**

The data come from **MIMIC-IV-ED** (Beth Israel Deaconess Medical Center, Boston), distributed by PhysioNet. Access is **credentialed**: it requires human-subjects research training (CITI) and a signed data use agreement (DUA) that **prohibits any redistribution**, including of derived data.

> Johnson A., Bulgarelli L., Pollard T., Celi L.A., Mark R., Horng S. (2023). *MIMIC-IV-ED* (version 2.2). PhysioNet. https://doi.org/10.13026/5ntk-km72

Intermediate files (`data/`, `*.csv`), trained models and large outputs are excluded by `.gitignore`. Running the pipeline requires credentialed access and placing the source tables in `data/`.

---

## Installation

```bash
git clone https://github.com/Doclecodeur/predict_ed.git
cd predict_ed

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**PyTorch (notebook 08 only).** The CUDA build is not available on PyPI:

```bash
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

On a machine without a GPU, `torch==2.6.0` is sufficient — execution is simply slower. This notebook is not required to reproduce the operational model.

---

## Reproducibility

**Reference environment** — Python 3.13.13, Windows 11 (Anaconda), NVIDIA RTX 4060 GPU (CUDA 12.4, notebook 08 only). Exact versions are pinned in `requirements.txt`. The spaCy model version (`en_core_web_sm` 3.8.0) is stated because lemmatization determines the TF-IDF vocabulary.

**Leakage prevention**

- splits performed **at patient level** (`subject_id`): `GroupShuffleSplit` for train / validation / test, `StratifiedGroupKFold` for internal cross-validation;
- **temporal validation**: development on anchor year groups prior to 2017, final evaluation held out on the 2017-2019 group;
- median imputation, standardization and TF-IDF vectorization **fitted on the training set only**;
- comorbidity score built solely from diagnoses of hospital stays **completed before** ED arrival;
- model and threshold **frozen before** evaluation on the held-out set.

**Random seed** — `random_state = 42` throughout all notebooks.

**Comorbidity score** — a **simplified Charlson-type score**: twelve comorbidities out of the nineteen in the full index, with the original weights. Identification relies on ICD-9 and ICD-10 code prefixes inspired by the algorithm of Quan et al. (2005), but simplified; the score is therefore not directly comparable to a Charlson score computed with the full method. The exact patterns are given in notebook 01.

---

## Limitations

- **No external validation.** Results come from a single centre (BIDMC, Boston). Transportability to a different health system, in particular the French one, remains to be demonstrated.
- **Approximate temporal validation.** Because MIMIC-IV dates are shifted during de-identification, the split relies on anchor year groups: it is internal and single-centre.
- **No laboratory or imaging data** are available at triage, which sets an informational ceiling: some admissions are decided after investigations produced later.
- **Fairness analysis limited** to age and sex.
- **Exploratory BERT comparison**: on this corpus of short chief complaints, BioClinicalBERT shows no clear gain over TF-IDF (text only 0.801 vs 0.826; hybrid architecture 0.877 in both cases).
- The "transfer" token found in some chief complaints likely reflects an already-initiated care pathway as much as a clinical state; a dedicated sensitivity analysis addresses this.

---

## Citation

```bibtex
@mastersthesis{doisy2026predicted,
  author = {Doisy, Wilguy},
  title  = {Early prediction of hospital admission at adult
            emergency department triage},
  school = {Université Paris 1 Panthéon-Sorbonne — DU Data Analytics},
  year   = {2026}
}
```

---

## License

Code released under the MIT License (see `LICENSE`). This license covers **the code only**: MIMIC-IV-ED data remain subject to the PhysioNet data use agreement.

---

## Author

**Wilguy DOISY** — Chirurgien et Biostatisticien
DU Data Analytics, Université Sorbonne Panthéon, promotion 2025-2026

---

*[Version française](README.md)*
# PREDICT-ED

**Prédiction précoce de la probabilité d'hospitalisation à l'issue d'un passage aux urgences adultes**

Modèle de pré-alerte estimant, dès le triage, la probabilité qu'un patient adulte soit hospitalisé à l'issue de son passage aux urgences, à partir des données structurées de triage et du motif de recours en texte libre.

> ⚠️ **Outil d'aide à l'organisation, non d'aide à la décision médicale.**
> Le modèle est conçu comme une **pré-alerte organisationnelle** destinée à la gestion des lits (anticipation des besoins d'aval). Il n'a pas vocation à orienter une décision clinique individuelle et n'a fait l'objet d'aucune validation prospective ni d'aucun marquage réglementaire.

Travail réalisé dans le cadre du **DU Data Analytics — Sorbonne Panthéon**, promotion 2025-2026.

---

## Contexte

L'engorgement des services d'urgences et le *boarding* (attente d'un lit après décision d'hospitalisation) constituent un problème organisationnel majeur. Anticiper dès le triage la probabilité d'hospitalisation permettrait au gestionnaire de lits de préparer l'aval plus tôt, sans se substituer au jugement clinique.

La plupart des travaux publiés sur cette tâche reposent sur les seules données structurées et sur un découpage aléatoire des données. Ce projet ajoute deux éléments :

- l'exploitation du **motif de recours en texte libre** (TF-IDF), en complément des variables structurées ;
- une **validation temporelle stricte**, plus proche des conditions réelles de déploiement qu'un découpage aléatoire.

---

## Résultats principaux

**Cohorte** — MIMIC-IV-ED : 383 919 séjours, 188 127 patients distincts, taux d'hospitalisation 38,2 %.

**Modèle opérationnel** — perceptron multicouche (MLP) hybride, structuré + TF-IDF, non recalibré.

| Indicateur | Valeur |
|---|---|
| AUC — validation temporelle (groupe d'ancrage 2017-2019, n = 67 963) | **0,888** (IC95 % 0,886–0,891) |
| AUC — test interne | 0,862 |
| Score de Brier | 0,131 |
| Erreur de calibration attendue (ECE) | 0,009 |

**Seuils opérationnels** (jeu réservé 2017-2019)

| Seuil | Valeur | Sensibilité | Spécificité |
|---|---|---|---|
| Pré-alerte (gestion des lits) | 0,241 | 89,1 % | 68,6 % |
| Standard | 0,500 | 70,9 % | 87,6 % |
| Fort (discussion de lit) | 0,650 | 58,1 % | 93,3 % |

**Comparaison des algorithmes** (Modèle 3, validation interne)

| Algorithme | AUC |
|---|---|
| MLP | 0,861 |
| Régression logistique | 0,853 |
| XGBoost | 0,850 |
| Random Forest | 0,849 |
| Comparateur ESI seul | 0,693 |

**Apport progressif des blocs de variables** (régression logistique) : constantes + âge + douleur 0,741 → + contexte, comorbidités, ESI 0,805 → + NLP du motif 0,853.

**Analyse de sensibilité** — après retrait du terme « transfer » du texte (sans exclure les patients concernés) et réentraînement à population constante, l'AUC passe de 0,888 à **0,884** : la prédiction ne repose pas sur cet artefact de vocabulaire.

---

## Pipeline

| Notebook | Rôle |
|---|---|
| `01_Construction_Cohorte_EDA` | Construction de la cohorte, comorbidités, prétraitement NLP du motif, analyse exploratoire |
| `02_Pretraitement_NLP` | Validation lexicale du motif de recours |
| `03_Modele_avec_transfer` | Comparaison des algorithmes (M1/M2/M3), modèle opérationnel, test interne |
| `04_Modele_sans_transfer` | Analyse de sensibilité sans le terme « transfer » |
| `05_Comparaison_avec_sans_transfer` | Comparaison des deux versions |
| `06_Tableaux_Synthese` | Tableaux de synthèse |
| `07_Analyse_avec_transfer` | Validation temporelle, seuils, calibration, sous-groupes |
| `08_BERT_vs_TFIDF` | Analyse secondaire exploratoire : TF-IDF vs BioClinicalBERT |

**Variables** — 32 variables structurées encodées (constantes vitales, âge estimé au séjour, ESI, douleur, mode d'arrivée, sexe, antécédents, score de comorbidité) + 1 000 variables textuelles TF-IDF (unigrammes et bigrammes), soit 1 032 caractéristiques.

---

## Données

**Ce dépôt ne contient aucune donnée patient.**

Les données proviennent de **MIMIC-IV-ED** (Beth Israel Deaconess Medical Center, Boston), distribué par PhysioNet. Leur accès est **contrôlé** : il suppose une formation à la recherche sur sujets humains (CITI) et la signature d'un accord d'utilisation (DUA) qui **interdit toute redistribution**, y compris sous forme dérivée.

> Johnson A., Bulgarelli L., Pollard T., Celi L.A., Mark R., Horng S. (2023). *MIMIC-IV-ED* (version 2.2). PhysioNet. https://doi.org/10.13026/5ntk-km72

Les fichiers intermédiaires (`data/`, `*.csv`), les modèles entraînés et les sorties volumineuses sont exclus par le `.gitignore`. Pour exécuter le pipeline, il faut disposer d'un accès accrédité et placer les tables sources dans `data/`.

---

## Installation

```bash
git clone https://github.com/Doclecodeur/predict_ed.git
cd predict_ed

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**PyTorch (notebook 08 uniquement).** La build CUDA n'est pas disponible sur PyPI :

```bash
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

Sur une machine sans GPU, `torch==2.6.0` suffit — l'exécution est simplement plus lente. Ce notebook n'est pas nécessaire pour reproduire le modèle opérationnel.

---

## Reproductibilité

**Environnement de référence** — Python 3.13.13, Windows 11 (Anaconda), GPU NVIDIA RTX 4060 (CUDA 12.4, notebook 08 uniquement). Les versions exactes sont figées dans `requirements.txt`. La version du modèle spaCy (`en_core_web_sm` 3.8.0) est précisée car la lemmatisation conditionne le vocabulaire TF-IDF.

**Prévention des fuites d'information**

- séparation des jeux **au niveau du patient** (`subject_id`) : `GroupShuffleSplit` pour entraînement / validation / test, `StratifiedGroupKFold` pour la validation croisée interne ;
- **validation temporelle** : développement sur les groupes d'ancrage antérieurs à 2017, évaluation finale réservée au groupe 2017-2019 ;
- imputation médiane, standardisation et vectorisation TF-IDF **ajustées sur le seul jeu d'entraînement** ;
- score de comorbidité construit à partir des seuls diagnostics d'hospitalisations **terminées avant** l'arrivée aux urgences ;
- choix du modèle et du seuil **figés avant** l'évaluation sur le jeu réservé.

**Graine aléatoire** — `random_state = 42` dans l'ensemble des notebooks.

**Score de comorbidité** — score **de type Charlson simplifié** : douze comorbidités parmi les dix-neuf de l'index complet, avec les pondérations d'origine. L'identification repose sur des préfixes de codes ICD-9 et ICD-10 inspirés de l'algorithme de Quan et al. (2005), mais simplifiés ; le score n'est donc pas directement comparable à un score de Charlson calculé selon la méthode complète. Les motifs exacts figurent dans le notebook 01.

---

## Limites

- **Aucune validation externe.** Les résultats proviennent d'un centre unique (BIDMC, Boston). La transposabilité à un système de santé différent, en particulier français, reste à démontrer.
- **Validation temporelle approximative.** Les dates de MIMIC-IV étant décalées lors de l'anonymisation, la séparation repose sur les groupes d'ancrage : elle est interne et monocentrique.
- **Aucune donnée biologique ni d'imagerie** n'est disponible au triage, ce qui constitue un plafond informationnel : certaines hospitalisations sont décidées après des examens produits plus tard.
- **Analyse d'équité limitée** à l'âge et au sexe.
- **Comparaison BERT exploratoire** : sur ce corpus de motifs courts, BioClinicalBERT n'apporte pas de gain clair par rapport à TF-IDF (texte seul 0,801 vs 0,826 ; architecture hybride 0,877 dans les deux cas).
- Le terme « transfer » présent dans certains motifs traduit vraisemblablement une orientation déjà engagée autant qu'un état clinique ; une analyse de sensibilité y est consacrée.

---

## Citation

```bibtex
@mastersthesis{doisy2026predicted,
  author = {Doisy, Wilguy},
  title  = {Prédiction précoce de la probabilité d'hospitalisation
            à l'issue d'un passage aux urgences adultes},
  school = {Université Sorbonne Panthéon — DU Data Analytics},
  year   = {2026}
}
```

---

## Licence

Code distribué sous licence MIT (voir `LICENSE`). Cette licence couvre **le code uniquement** : les données MIMIC-IV-ED restent soumises à l'accord d'utilisation de PhysioNet.

---

## Auteur

**Wilguy DOISY** — Chirurgien et Biostatisticien
DU Data Analytics, Université Sorbonne Panthéon, promotion 2025-2026
