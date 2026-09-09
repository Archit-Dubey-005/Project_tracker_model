"""High-precision activity matching for Intelligent Progress Tracking."""

import os
import re

import faiss
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class IPTMatcher:
    """Semantic activity matcher with metadata-aware reranking.

    ``project_id`` is optional, but callers should supply it whenever it is
    known. This prevents identical activities in different projects from
    competing with each other.
    """

    _ACTIVITY_ID_PATTERN = re.compile(r"\b(P\d{5})[\s_-]*A(\d{4})\b", re.IGNORECASE)
    _PROJECT_ID_PATTERN = re.compile(r"\b(P\d{5})(?=\b|[_-])", re.IGNORECASE)
    _TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
    _LOCATION_PATTERN = re.compile(
        r"\bzone\s*[- ]?\s*(\d+)\s*,?\s*floor\s*[- ]?\s*(\d+)\b", re.IGNORECASE
    )
    _STOP_WORDS = frozenset(
        {
            "a", "an", "and", "are", "as", "at", "been", "by", "during",
            "for", "from", "in", "inspection", "is", "it", "of", "on",
            "progress", "quality", "site", "status", "that", "the", "this",
            "to", "today", "under", "was", "were", "with", "work", "works",
            "completed", "complete", "check", "execution", "observed", "underway",
        }
    )
    _ALIASES = {
        "pcc": ("plain", "cement", "concrete"),
        "rcc": ("reinforced", "cement", "concrete"),
        "rebar": ("reinforcement",),
        "shuttering": ("formwork",),
        "mep": ("mechanical", "electrical", "plumbing"),
    }

    def __init__(self, models_dir="models", model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.models_dir = models_dir
        self.model_name = model_name

        index_path = os.path.join(models_dir, "activity_index.faiss")
        lookup_path = os.path.join(models_dir, "activity_lookup.pkl")
        embeddings_path = os.path.join(models_dir, "embeddings.npy")
        model_path = os.path.join(models_dir, "model.pkl")

        missing = [path for path in (index_path, lookup_path) if not os.path.exists(path)]
        if missing:
            names = ", ".join(os.path.basename(path) for path in missing)
            raise FileNotFoundError(f"Model artifacts are missing in {models_dir}: {names}.")

        # Prefer the packaged model, avoiding a version mismatch or network
        # download during application startup.
        if os.path.exists(model_path):
            print(f"Loading packaged sentence model from {model_path}...")
            self.model = joblib.load(model_path)
        else:
            print(f"Loading Sentence Transformer model: {model_name}...")
            self.model = SentenceTransformer(model_name)

        print(f"Loading FAISS vector index from {index_path}...")
        self.index = faiss.read_index(index_path)
        print(f"Loading activity lookup metadata from {lookup_path}...")
        self.lookup = joblib.load(lookup_path).reset_index(drop=True)
        self._validate_lookup()

        progress_report_path = os.path.join(models_dir, "progress_tracking_report.json")
        self.progress_metrics = pd.read_json(progress_report_path) if os.path.exists(progress_report_path) else pd.DataFrame()

        # Project-scoped search must examine every activity in the project,
        # rather than filtering an already-truncated global top-k result.
        self.embeddings = None
        if os.path.exists(embeddings_path):
            embeddings = np.load(embeddings_path, mmap_mode="r")
            if embeddings.ndim != 2 or embeddings.shape != (len(self.lookup), self.index.d):
                raise ValueError(
                    "embeddings.npy is not aligned with the activity lookup or FAISS index. "
                    "Rebuild the model artifacts together."
                )
            self.embeddings = embeddings

        self._activity_ids = self.lookup["activity_id"].astype(str).str.upper().to_numpy()
        self._id_to_position = {activity_id: pos for pos, activity_id in enumerate(self._activity_ids)}
        self._project_to_positions = self._build_project_positions()
        self._field_tokens = {
            field: [self._tokenize(value) for value in self.lookup[field].fillna("")]
            for field in ("activity_name", "keywords", "discipline")
        }
        self._locations = [self._parse_location(value) for value in self.lookup["location"].fillna("")]
        print(f"IPTMatcher initialized successfully with {self.index.ntotal} indexed activities.")

    def _validate_lookup(self):
        required = {"activity_id", "activity_name", "discipline", "location", "keywords"}
        missing_columns = required - set(self.lookup.columns)
        if missing_columns:
            raise ValueError(f"Activity metadata is missing required columns: {', '.join(sorted(missing_columns))}.")
        if len(self.lookup) != self.index.ntotal:
            raise ValueError("FAISS index and activity lookup have different row counts. Rebuild the artifacts together.")
        if self.lookup["activity_id"].astype(str).str.upper().duplicated().any():
            raise ValueError("activity_lookup.pkl contains duplicate activity IDs.")

    def _build_project_positions(self):
        projects = {}
        for position, activity_id in enumerate(self._activity_ids):
            project_id = self._extract_project_id(activity_id)
            if project_id:
                projects.setdefault(project_id, []).append(position)
        return {project_id: np.asarray(positions, dtype=np.int64) for project_id, positions in projects.items()}

    @staticmethod
    def clean_text(text):
        if pd.isna(text) or not str(text).strip():
            return ""
        value = re.sub(r"\s+", " ", str(text).strip())
        return re.sub(r"[^\w\s\-\.,:/]", "", value)

    @classmethod
    def _tokenize(cls, text):
        tokens = set(cls._TOKEN_PATTERN.findall(str(text).lower())) - cls._STOP_WORDS
        expanded = set(tokens)
        for token in tokens:
            expanded.update(cls._ALIASES.get(token, ()))
        return expanded

    @classmethod
    def _parse_location(cls, text):
        match = cls._LOCATION_PATTERN.search(str(text))
        return (match.group(1), match.group(2)) if match else None

    @classmethod
    def _extract_activity_id(cls, text):
        match = cls._ACTIVITY_ID_PATTERN.search(text)
        return f"{match.group(1).upper()}_A{match.group(2)}" if match else None

    @classmethod
    def _extract_project_id(cls, text):
        match = cls._PROJECT_ID_PATTERN.search(str(text))
        return match.group(1).upper() if match else None

    @classmethod
    def _normalise_project_id(cls, project_id):
        if project_id is None or pd.isna(project_id):
            return None
        normalised = cls._extract_project_id(project_id)
        if normalised is None:
            raise ValueError("project_id must contain a project identifier such as 'P00085'.")
        return normalised

    @staticmethod
    def _coverage_score(query_tokens, candidate_tokens):
        if not query_tokens or not candidate_tokens:
            return 0.0
        overlap = len(query_tokens & candidate_tokens)
        # F1 rewards a complete activity-name match without treating a long
        # progress sentence as a weak match.
        return 2.0 * overlap / (len(query_tokens) + len(candidate_tokens))

    @classmethod
    def keyword_score(cls, query, text):
        """Backward-compatible lexical score using improved tokenisation."""
        return cls._coverage_score(cls._tokenize(query), cls._tokenize(text))

    @staticmethod
    def calculate_final_score(semantic_score, name_score, keyword_score, location_score=0.0, discipline_score=0.0):
        return (
            0.60 * float(semantic_score)
            + 0.22 * float(name_score)
            + 0.08 * float(keyword_score)
            + 0.08 * float(location_score)
            + 0.02 * float(discipline_score)
        )

    def _candidate_positions(self, query_embedding, top_k, scoped_project_id):
        if scoped_project_id in self._project_to_positions:
            positions = self._project_to_positions[scoped_project_id]
            if self.embeddings is not None:
                semantic_scores = np.asarray(self.embeddings[positions] @ query_embedding[0])
            else:
                vectors = np.vstack([self.index.reconstruct(int(position)) for position in positions])
                semantic_scores = vectors @ query_embedding[0]
            order = np.argsort(-semantic_scores)
            return positions[order], semantic_scores[order]

        # A wider pool lets structured evidence fix close semantic matches.
        search_k = min(max(top_k * 12, 80), self.index.ntotal)
        scores, positions = self.index.search(query_embedding, search_k)
        valid = positions[0] >= 0
        return positions[0][valid], scores[0][valid]

    def _rerank(self, positions, semantic_scores, query_tokens, query_location):
        candidates = self.lookup.iloc[positions].copy().reset_index(drop=True)
        candidates["semantic_score"] = np.asarray(semantic_scores, dtype=float)
        name_scores, keyword_scores, location_scores, discipline_scores, final_scores = [], [], [], [], []

        for position, semantic_score in zip(positions, semantic_scores):
            name_score = self._coverage_score(query_tokens, self._field_tokens["activity_name"][position])
            keyword_score = self._coverage_score(query_tokens, self._field_tokens["keywords"][position])
            discipline_score = self._coverage_score(query_tokens, self._field_tokens["discipline"][position])
            candidate_location = self._locations[position]
            location_score = float(query_location == candidate_location) if query_location and candidate_location else 0.0
            name_scores.append(name_score)
            keyword_scores.append(keyword_score)
            location_scores.append(location_score)
            discipline_scores.append(discipline_score)
            final_scores.append(self.calculate_final_score(
                semantic_score, name_score, keyword_score, location_score, discipline_score
            ))

        candidates["name_score"] = name_scores
        candidates["keyword_score"] = keyword_scores
        candidates["location_score"] = location_scores
        candidates["discipline_score"] = discipline_scores
        candidates["final_score"] = final_scores
        return candidates.sort_values("final_score", ascending=False, kind="stable").reset_index(drop=True)

    def match(self, query_text, top_k=5, project_id=None):
        """Match a free-text report to a scheduled activity.

        An activity ID contained in the report is authoritative. Otherwise,
        ``project_id`` (or a project ID found in the report) scopes retrieval
        before semantic ranking and metadata reranking.
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        clean_query = self.clean_text(query_text)
        if not clean_query:
            raise ValueError("query_text must contain a non-empty site report.")

        requested_project_id = self._normalise_project_id(project_id)
        query_activity_id = self._extract_activity_id(clean_query)
        inferred_project_id = self._extract_project_id(query_activity_id or clean_query)
        scoped_project_id = requested_project_id or inferred_project_id
        exact_position = self._id_to_position.get(query_activity_id) if query_activity_id else None

        query_embedding = self.model.encode([clean_query], normalize_embeddings=True).astype("float32")
        positions, semantic_scores = self._candidate_positions(query_embedding, top_k, scoped_project_id)
        candidates = self._rerank(
            positions, semantic_scores, self._tokenize(clean_query), self._parse_location(clean_query)
        )

        match_method = "metadata_reranked_semantic_search"
        if exact_position is not None:
            exact_id = self._activity_ids[exact_position]
            exact_candidates = candidates[candidates["activity_id"].astype(str).str.upper() == exact_id]
            if exact_candidates.empty:
                exact_row = self.lookup.iloc[[exact_position]].copy()
                exact_semantic = float(query_embedding[0] @ self.embeddings[exact_position]) if self.embeddings is not None else 0.0
                exact_row["semantic_score"] = exact_semantic
                exact_row["name_score"] = self.keyword_score(clean_query, exact_row.iloc[0]["activity_name"])
                exact_row["keyword_score"] = self.keyword_score(clean_query, exact_row.iloc[0]["keywords"])
                exact_row["location_score"] = 0.0
                exact_row["discipline_score"] = self.keyword_score(clean_query, exact_row.iloc[0]["discipline"])
                exact_row["final_score"] = 1.0
                candidates = pd.concat([exact_row, candidates], ignore_index=True)
            else:
                candidates.loc[exact_candidates.index, "final_score"] = 1.0
            candidates = candidates.sort_values("final_score", ascending=False, kind="stable").reset_index(drop=True)
            match_method = "explicit_activity_id"

        candidates = candidates.head(top_k).reset_index(drop=True)
        best_match = candidates.iloc[0]

        act_id = str(best_match["activity_id"])
        proj_id = act_id.split("_")[0]
        conf_score = round(float(best_match["final_score"]), 4)

        status_info = {
            "Query": query_text,
            "Activity ID": act_id,
            "Project ID": proj_id,
            "Activity Name": str(best_match["activity_name"]),
            "Schedule Status": "N/A",
            "Delay/Early Duration": "0 days",
            "Activity Completion Percentage": 0.0,
            "Project Completion Percentage": 0.0,
            "AI Confidence Score": conf_score
        }

        if not self.progress_metrics.empty and "Activity ID" in self.progress_metrics.columns:
            m_row = self.progress_metrics[self.progress_metrics["Activity ID"] == act_id]
            if not m_row.empty:
                info = m_row.iloc[0]
                status_info["Schedule Status"] = str(info.get("Schedule Status", "N/A"))
                status_info["Delay/Early Duration"] = str(info.get("Delay/Early Duration", "0 days"))
                status_info["Activity Completion Percentage"] = float(info.get("Activity Completion Percentage", 0.0))
                status_info["Project Completion Percentage"] = float(info.get("Project Completion Percentage", 0.0))

        return {
            "query": query_text,
            "scope_project_id": scoped_project_id if scoped_project_id in self._project_to_positions else None,
            "match_method": match_method,
            "top_match": {
                "activity_id": act_id,
                "activity_name": str(best_match["activity_name"]),
                "score": conf_score,
                **status_info
            },
            "candidates": candidates.to_dict(orient="records"),
        }

    def batch_process_file(self, input_file_path, output_file_path, query_col="raw_text", project_id_col="project_id"):
        """Read queries from a CSV or JSON file, run matcher, and export results to a CSV or JSON file."""
        if input_file_path.endswith(".json"):
            df = pd.read_json(input_file_path)
        else:
            df = pd.read_csv(input_file_path)

        results = []
        for _, row in df.iterrows():
            q_text = str(row.get(query_col, ""))
            p_id = str(row.get(project_id_col, "")) if project_id_col in row and pd.notna(row[project_id_col]) else None
            match_res = self.match(query_text=q_text, top_k=1, project_id=p_id)
            results.append(match_res["top_match"])

        res_df = pd.DataFrame(results)

        if output_file_path.endswith(".json"):
            res_df.to_json(output_file_path, orient="records", indent=2)
        else:
            res_df.to_csv(output_file_path, index=False)

        print(f"Exported {len(res_df)} processed results to {output_file_path}")
        return res_df


if __name__ == "__main__":
    matcher = IPTMatcher()
    result = matcher.match("RCC footing reinforcement completed at Block A", top_k=3, project_id="P00010")
    print("\nStandalone Test Results:")
    print(result["top_match"])
