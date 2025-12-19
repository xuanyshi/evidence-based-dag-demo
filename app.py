"""Streamlit demo to explore the DAG edges produced in `notebooks/for_dag.ipynb`.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import ast

import pandas as pd
import numpy as np
import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components
from sentence_transformers import SentenceTransformer, util
import re
import os
import sys

# Add notebooks directory to path to import triangulation
sys.path.append(str(Path(__file__).parent / "notebooks"))
from triangulation import analyze_tri_df


# Base directory setup - all paths are relative to project root
# This ensures the code works on any machine (local, GitHub, HuggingFace, etc.)
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Data file paths (relative to project root)
# Required files in data/ directory:
#   - final_causal_master_list_with_sets_1.2.xlsx
#   - labeled_parents_1.2.xlsx
#   - labeled_mediators_1.2.xlsx
#   - labeled_children_1.2.xlsx
#   - final_strategy_evidence_full.xlsx
#   - umls_df_v3.csv
# Required files in data/extracted_results/ directory:
#   - final_dag_edges_clean.csv
#   - final_dag_evidence.csv
#   - all_df.csv
DATA_PATH_NETWORK = DATA_DIR / "extracted_results" / "final_dag_edges_clean.csv"
DATA_PATH_FINAL = DATA_DIR / "final_causal_master_list_with_sets_1.2.xlsx"
DATA_PATH_PARENTS = DATA_DIR / "labeled_parents_1.2.xlsx"
DATA_PATH_MEDIATORS = DATA_DIR / "labeled_mediators_1.2.xlsx"
DATA_PATH_CHILDREN = DATA_DIR / "labeled_children_1.2.xlsx"
DATA_PATH_STRATEGY = DATA_DIR / "final_strategy_evidence_full.xlsx"
DATA_PATH_UMLS = DATA_DIR / "umls_df_v3.csv"
DATA_PATH_ALL_DF = DATA_DIR / "extracted_results" / "all_df.csv"
X_NODE = "Influenza Vaccine"
Y_NODE = "Stroke"


def drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that start with 'Unnamed:'."""
    if df.empty:
        return df
    cols_to_drop = [col for col in df.columns if str(col).startswith('Unnamed:')]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


@st.cache_data
def load_final_list() -> pd.DataFrame:
    """Load final causal list and keep only relevant categories."""
    df = pd.read_excel(DATA_PATH_FINAL)
    df = drop_unnamed_columns(df)
    df = df[df["final_category"].str.upper() != "IRRELEVANT"].copy()
    df["variable_clean"] = df["variable"].astype(str).str.strip().str.title()
    df["final_category"] = df["final_category"].str.upper()
    return df


@st.cache_data
def load_domain_mappings() -> dict[str, str]:
    """Load domain mappings from labeled parents, mediators, and children files."""
    domain_map = {}
    
    for path in [DATA_PATH_PARENTS, DATA_PATH_MEDIATORS, DATA_PATH_CHILDREN]:
        if path.exists():
            df = pd.read_excel(path)
            df = drop_unnamed_columns(df)
            for _, row in df.iterrows():
                var_name = str(row["var_name"]).strip().title()
                domain = str(row["domain"]).strip()
                domain_map[var_name] = domain
    
    return domain_map


@st.cache_data
def load_strategy_evidence() -> pd.DataFrame:
    """Load strategy evidence file and parse PMID lists."""
    if not DATA_PATH_STRATEGY.exists():
        return pd.DataFrame()
    
    df = pd.read_excel(DATA_PATH_STRATEGY)
    df = drop_unnamed_columns(df)
    
    # Parse stringified lists in PMID columns
    pmid_columns = ["pmids_var_to_flu", "pmids_var_to_stroke", "pmids_flu_to_var", "pmids_stroke_to_var"]
    for col in pmid_columns:
        if col in df.columns:
            def parse_pmid_list(x):
                if pd.isna(x):
                    return []
                if isinstance(x, list):
                    return x
                if isinstance(x, str) and x.strip().startswith('['):
                    try:
                        return ast.literal_eval(x)
                    except (ValueError, SyntaxError):
                        return []
                return []
            df[col] = df[col].apply(parse_pmid_list)
    
    # Clean variable names
    if "variable" in df.columns:
        df["variable_clean"] = df["variable"].astype(str).str.strip().str.title()
    
    return df


@st.cache_data
def load_umls_raw() -> pd.DataFrame:
    """Load raw UMLS evidence dataset."""
    if not DATA_PATH_UMLS.exists():
        return pd.DataFrame()
    
    # Read in chunks if file is very large
    try:
        df = pd.read_csv(DATA_PATH_UMLS, low_memory=False)
        df = drop_unnamed_columns(df)
    except Exception as e:
        st.warning(f"Error loading UMLS data: {e}")
        return pd.DataFrame()
    
    return df


@st.cache_data
def load_all_df() -> pd.DataFrame:
    """Load all_df for evidence retrieval."""
    if not DATA_PATH_ALL_DF.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(DATA_PATH_ALL_DF, low_memory=False)
        df = drop_unnamed_columns(df)
    except Exception as e:
        st.warning(f"Error loading all_df: {e}")
        return pd.DataFrame()
    
    return df


class EvidenceRetriever:
    """Evidence retriever using semantic similarity."""
    def __init__(self, df: pd.DataFrame, 
                 model_name='UMCU/SapBERT-from-PubMedBERT-fulltext_bf16',
                 cache_path=None,
                 force_refresh=False):
        self.df = df.copy()
        self.model_name = model_name
        self.cache_path = cache_path or str(Path(__file__).parent / "notebooks" / "sapbert_embeddings_cache.npz")
        
        # Sort and unique for consistent indexing
        self.unique_exps = np.sort(self.df['exposure_umls_name'].dropna().unique().astype(str))
        self.unique_outs = np.sort(self.df['outcome_umls_name'].dropna().unique().astype(str))
        self.outcome_blocklist_tuis = ['T061xxx', 'T059xxx', 'T060xxx'] 

        # Cache loading logic
        if not force_refresh and os.path.exists(self.cache_path):
            if self._load_cache():
                self.model = SentenceTransformer(model_name)
                return

        self.model = SentenceTransformer(model_name)
        self.exp_embeddings = self.model.encode(self.unique_exps, batch_size=64, show_progress_bar=True)
        self.out_embeddings = self.model.encode(self.unique_outs, batch_size=64, show_progress_bar=True)
        self._save_cache()

    def _save_cache(self):
        np.savez_compressed(self.cache_path, exp_names=self.unique_exps, exp_vecs=self.exp_embeddings, out_names=self.unique_outs, out_vecs=self.out_embeddings)

    def _load_cache(self):
        try:
            data = np.load(self.cache_path)
            if not np.array_equal(self.unique_exps, data['exp_names']) or not np.array_equal(self.unique_outs, data['out_names']): 
                return False
            self.exp_embeddings = data['exp_vecs']
            self.out_embeddings = data['out_vecs']
            return True
        except: 
            return False

    def _parse_exposure_direction_str(self, exp_name):
        """Derives exposure direction string from concept name."""
        exp_name = str(exp_name).lower()
        neg_patterns = [r'\blow\b', r'\bdecrease', r'\bdeficien', r'\brestrict', r'\bhypo', r'\black of\b', r'\bless\b']
        
        for pat in neg_patterns:
            if re.search(pat, exp_name):
                return 'decreased'
        
        return 'increased'

    def _calculate_final_relation(self, row):
        """Calculates Excitatory/Inhibitory based on string directions."""
        exp_dir = row['exposure_direction']
        raw_out = str(row['direction']).lower()
        
        out_dir = 'unknown'
        if 'inc' in raw_out or raw_out == '1': 
            out_dir = 'increase'
        elif 'dec' in raw_out or raw_out == '-1': 
            out_dir = 'decrease'
        
        if out_dir == 'unknown': 
            return 'No Change'

        e_val = 1 if exp_dir == 'increased' else -1
        o_val = 1 if out_dir == 'increase' else -1
        
        return 'Excitatory' if e_val * o_val > 0 else 'Inhibitory'

    def retrieve(self, target_exposure: str, target_outcome: str, threshold: float = 0.9, strict_tui: bool = True):
        """Retrieve evidence for exposure-outcome pair."""
        # 1. Vector Search
        q_exp_vec = self.model.encode(target_exposure)
        q_out_vec = self.model.encode(target_outcome)
        
        sim_exps = util.cos_sim(q_exp_vec, self.exp_embeddings)[0].cpu().numpy()
        sim_outs = util.cos_sim(q_out_vec, self.out_embeddings)[0].cpu().numpy()
        
        valid_exp_names = self.unique_exps[sim_exps > threshold]
        valid_out_names = self.unique_outs[sim_outs > threshold]
        
        if len(valid_exp_names) == 0 or len(valid_out_names) == 0:
            return pd.DataFrame() 
            
        # 2. Database Filter
        mask = (
            self.df['exposure_umls_name'].astype(str).isin(valid_exp_names) &
            self.df['outcome_umls_name'].astype(str).isin(valid_out_names)
        )
        result_df = self.df[mask].copy()
        
        if strict_tui and not result_df.empty:
            result_df = result_df[~result_df['outcome_tui'].isin(self.outcome_blocklist_tuis)]
        
        if result_df.empty:
            return result_df

        # 3. Add 'exposure_direction' column
        result_df['exposure_direction'] = result_df['exposure_umls_name'].apply(self._parse_exposure_direction_str)
        
        # Calculate final relation using the new column
        result_df['final_relation'] = result_df.apply(self._calculate_final_relation, axis=1)
        
        return result_df


@st.cache_data
def load_network_edges() -> pd.DataFrame:
    """Load network edges produced in for_dag.ipynb."""
    df = pd.read_csv(DATA_PATH_NETWORK)
    df = drop_unnamed_columns(df)
    df["Type"] = df["Type"].fillna("Unknown")
    df["biggest"] = df["biggest"].fillna("no change").str.lower()
    # Keep only relation directions we care about
    df.loc[~df["biggest"].isin(["excitatory", "inhibitory", "no change"]), "biggest"] = "no change"
    df["Source"] = df["Source"].astype(str).str.strip().str.title()
    df["Target"] = df["Target"].astype(str).str.strip().str.title()
    return df


def build_base_edges(final_df: pd.DataFrame) -> pd.DataFrame:
    """Construct base edges from final list according to category rules."""
    rows = []
    for _, row in final_df.iterrows():
        z = row["variable_clean"]
        cat = row["final_category"]
        if cat == "FINAL_CONFOUNDER":
            rows.append((z, X_NODE, "Confounder"))
            rows.append((z, Y_NODE, "Confounder"))
        elif cat == "FINAL_MEDIATOR":
            rows.append((X_NODE, z, "Mediator"))
            rows.append((z, Y_NODE, "Mediator"))
        elif cat == "FINAL_COLLIDER":
            rows.append((X_NODE, z, "Collider"))
            rows.append((Y_NODE, z, "Collider"))
        elif cat == "FINAL_PROGNOSTIC":
            rows.append((z, Y_NODE, "Prognostic"))
        else:
            continue

    base = pd.DataFrame(rows, columns=["Source", "Target", "Type"])
    if base.empty:
        return base
    # For base edges, treat relation direction as "no change" to keep filter clean.
    base["biggest"] = "no change"
    base["loe"] = 1.0  # ensure they are visible with any slider threshold and sorted last
    base["count"] = 0
    base["p_excitatory"] = 0.0
    base["p_no_change"] = 0.0
    base["p_inhibitory"] = 0.0
    return base


def build_network(
    edges: pd.DataFrame,
    highlight_nodes: set[str],
    node_category: dict[str, str],
    node_domain: dict[str, str],
    focus_node: str | None = None,
    focus_edge: tuple[str, str] | None = None,
) -> Network:
    """Create a PyVis Network from the filtered edges."""
    net = Network(
        height="750px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#0f172a",
    )
    net.barnes_hut()

    nodes = set(edges["Source"]).union(edges["Target"])

    # Pre-compute adjacency for focus highlighting
    neighbors = {n: set() for n in nodes}
    for _, r in edges.iterrows():
        neighbors[r["Source"]].add(r["Target"])
        neighbors[r["Target"]].add(r["Source"])

    dim_node_color = "rgba(148,163,184,0.35)"  # gray with alpha
    dim_edge_color = "rgba(0,0,0,0.15)"

    for node in nodes:
        cat = node_category.get(node, "Unknown")
        domain = node_domain.get(node, "Unknown")
        cat_color_map = {
            "FINAL_CONFOUNDER": "#f97316",
            "FINAL_MEDIATOR": "#8b5cf6",
            "FINAL_COLLIDER": "#0d9488",
            "FINAL_PROGNOSTIC": "#3b82f6",
            "EXPOSURE": "#2563eb",
            "OUTCOME": "#dc2626",
            "Unknown": "#94a3b8",
        }
        base_color = cat_color_map.get(cat, "#94a3b8")
        if node in highlight_nodes:
            base_color = "#facc15"

        # Dimming logic
        color = base_color
        if focus_edge:
            src, tgt = focus_edge
            if node not in {src, tgt}:
                color = dim_node_color
        elif focus_node:
            if node != focus_node and node not in neighbors.get(focus_node, set()):
                color = dim_node_color

        # Fixed positions for X and Y nodes, others are draggable
        fixed = False
        x = y = None
        physics = True
        if node == X_NODE:
            fixed = True
            physics = False
            x, y = -800, 0
        elif node == Y_NODE:
            fixed = True
            physics = False
            x, y = 800, 0

        label_size = 56 if node in {X_NODE, Y_NODE} else 48
        node_size = 48 if node in {X_NODE, Y_NODE} else 36

        net.add_node(
            node,
            label=node,
            title=f"{node}\nCategory: {cat}\nDomain: {domain}",
            color=color,
            size=node_size,
            x=x,
            y=y,
            fixed={"x": fixed, "y": fixed} if fixed else False,
            physics=physics,
            font={"size": label_size},
        )

    # Colors
    peer_color = "#34d399"  # light green for peer interaction edges
    default_edge_color = "#60a5fa"  # light blue for base edges

    for _, row in edges.iterrows():
        edge_color = peer_color if row["Type"] == "Peer Interaction" else default_edge_color
        # Apply dimming based on focus
        if focus_edge:
            src, tgt = focus_edge
            if not ((row["Source"] == src and row["Target"] == tgt) or (row["Source"] == tgt and row["Target"] == src)):
                edge_color = dim_edge_color
        elif focus_node:
            if focus_node not in {row["Source"], row["Target"]}:
                edge_color = dim_edge_color
        tooltip = (
            f"{row['Source']} → {row['Target']}<br>"
            f"Type: {row['Type']}<br>"
            f"Relation: {row['biggest']}<br>"
            f"LOE: {row['loe']:.3f} | Studies: {row['count']}<br>"
            f"p_excitatory: {row['p_excitatory']:.3f} | "
            f"p_no_change: {row['p_no_change']:.3f} | "
            f"p_inhibitory: {row['p_inhibitory']:.3f}"
        )
        net.add_edge(
            row["Source"],
            row["Target"],
            value=0.2,  # uniform control strength
            title=tooltip,
            color=edge_color,
            arrowStrikethrough=False,
            width=0.05,  # uniform thinner edge
        )

    net.set_options(
        """
        {
            "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -120,
                "centralGravity": 0.002,
                "springLength": 200,
                "springConstant": 0.12
            },
            "minVelocity": 0.5,
            "solver": "forceAtlas2Based",
            "timestep": 0.35
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 120,
                "multiselect": true
            }
        }
        """
    )
    return net


def render_strategy_evidence_viewer() -> None:
    """Render the Strategy Evidence Viewer tab."""
    st.header("Strategy Evidence Viewer")
    st.caption("Explore raw evidence for causal paths based on triangulation strategy.")
    
    # Load data
    strategy_df = load_strategy_evidence()
    umls_df = load_umls_raw()
    
    if strategy_df.empty:
        st.error(f"Strategy evidence file not found: {DATA_PATH_STRATEGY}")
        return
    
    if umls_df.empty:
        st.error(f"UMLS raw data file not found: {DATA_PATH_UMLS}")
        return
    
    # Variable selector
    if "variable_clean" not in strategy_df.columns:
        st.error("Strategy file missing 'variable' column.")
        return
    
    variables = sorted(strategy_df["variable_clean"].dropna().unique().tolist())
    if not variables:
        st.error("No variables found in strategy file.")
        return
    
    selected_var = st.selectbox("Select Variable (Z)", variables)
    
    # Get row for selected variable
    var_row = strategy_df[strategy_df["variable_clean"] == selected_var].iloc[0]
    final_category = var_row.get("final_category", "Unknown")
    
    st.subheader(f"Variable: {selected_var}")
    st.write(f"**Category:** {final_category}")
    
    # Extract PMID lists
    pmids_var_to_flu = var_row.get("pmids_var_to_flu", [])
    pmids_var_to_stroke = var_row.get("pmids_var_to_stroke", [])
    pmids_flu_to_var = var_row.get("pmids_flu_to_var", [])
    pmids_stroke_to_var = var_row.get("pmids_stroke_to_var", [])
    
    # Ensure they are lists
    if not isinstance(pmids_var_to_flu, list):
        pmids_var_to_flu = []
    if not isinstance(pmids_var_to_stroke, list):
        pmids_var_to_stroke = []
    if not isinstance(pmids_flu_to_var, list):
        pmids_flu_to_var = []
    if not isinstance(pmids_stroke_to_var, list):
        pmids_stroke_to_var = []
    
    # Summary table
    summary_data = {
        "Path": [
            f"{selected_var} → {X_NODE}",
            f"{selected_var} → {Y_NODE}",
            f"{X_NODE} → {selected_var}",
            f"{Y_NODE} → {selected_var}",
        ],
        "PMID Count": [
            len(pmids_var_to_flu),
            len(pmids_var_to_stroke),
            len(pmids_flu_to_var),
            len(pmids_stroke_to_var),
        ],
        "PMIDs": [
            pmids_var_to_flu,
            pmids_var_to_stroke,
            pmids_flu_to_var,
            pmids_stroke_to_var,
        ],
    }
    summary_df = pd.DataFrame(summary_data)
    
    st.subheader("Path Summary")
    st.dataframe(summary_df[["Path", "PMID Count"]], use_container_width=True)
    
    # Path selector for drill down
    st.subheader("Drill Down: View Raw Evidence")
    path_options = summary_df["Path"].tolist()
    selected_path = st.selectbox("Select a path to view evidence", path_options)
    
    if selected_path:
        # Get PMIDs for selected path
        path_idx = path_options.index(selected_path)
        selected_pmids = summary_df.iloc[path_idx]["PMIDs"]
        
        if not selected_pmids:
            st.info(f"No PMIDs found for path: {selected_path}")
        else:
            st.write(f"**Found {len(selected_pmids)} PMIDs for this path**")
            
            # Convert PMIDs to strings for matching (handle both string and int PMIDs)
            selected_pmids_str = [str(pmid) for pmid in selected_pmids]
            
            # Filter UMLS dataframe
            # Try different possible column names for PMID
            pmid_col = None
            for col in ["pmid", "PMID", "pmid_clean", "pubmed_id"]:
                if col in umls_df.columns:
                    pmid_col = col
                    break
            
            if not pmid_col:
                st.error("Could not find PMID column in UMLS data. Available columns: " + ", ".join(umls_df.columns.tolist()))
                return
            
            # Convert PMID column to string for matching
            umls_df[pmid_col] = umls_df[pmid_col].astype(str)
            evidence_filtered = umls_df[umls_df[pmid_col].isin(selected_pmids_str)].copy()
            
            if evidence_filtered.empty:
                st.warning(f"No evidence rows found in UMLS data for PMIDs: {selected_pmids_str[:5]}...")
            else:
                st.write(f"**Showing {len(evidence_filtered)} evidence rows**")
                
                # Display all columns
                st.dataframe(evidence_filtered.reset_index(drop=True), use_container_width=True)
                
                # Download button
                st.download_button(
                    "Download evidence (CSV)",
                    data=evidence_filtered.to_csv(index=False),
                    file_name=f"evidence_{selected_var.replace(' ', '_')}_{selected_path.replace(' ', '_').replace('→', 'to')}.csv",
                    mime="text/csv",
                )


@st.cache_resource
def get_retriever(_all_df: pd.DataFrame) -> EvidenceRetriever:
    """Get or create EvidenceRetriever instance (cached)."""
    if _all_df.empty:
        return None
    return EvidenceRetriever(_all_df)


def render_questionnaire() -> None:
    """Render the scientific evaluation questionnaire."""
    st.header("System Usability and Impact Evaluation")
    st.caption(
        "This questionnaire evaluates the usability, design, and scientific impact of the Evidence-Based DAG Explorer. "
        "Your responses will contribute to academic research and system improvement."
    )
    
    # Initialize session state for responses
    if "questionnaire_responses" not in st.session_state:
        st.session_state.questionnaire_responses = None
    
    # Likert scale options (1-5, where 5 is best)
    likert_5 = ["Strongly Agree (5)", "Agree (4)", "Neutral (3)", "Disagree (2)", "Strongly Disagree (1)"]
    quality_5 = ["Excellent (5)", "Good (4)", "Adequate (3)", "Poor (2)", "Very Poor (1)"]
    frequency_5 = ["Always (5)", "Often (4)", "Sometimes (3)", "Rarely (2)", "Never (1)"]
    
    def extract_score(answer: str) -> int:
        """Extract numeric score from answer string."""
        if not answer:
            return None
        match = re.search(r'\((\d+)\)', answer)
        return int(match.group(1)) if match else None
    
    def extract_text(answer: str) -> str:
        """Extract text part from answer string."""
        if not answer:
            return ""
        return re.sub(r'\s*\(\d+\)$', '', answer).strip()
    
    with st.form("evaluation_form"):
        st.markdown("""
        **Instructions**: Please rate each statement based on your experience using the system. 
        For short-answer questions, provide concise responses (2-3 sentences). 
        We appreciate your positive feedback and constructive suggestions.
        """)
        
        st.divider()
        
        # Q1: System Usability Scale (SUS-inspired)
        q1 = st.radio(
            "**Q1. System Usability**: The system is easy to learn and use for exploring causal evidence networks.",
            likert_5,
            index=None,
            help="Rate your agreement with the statement about overall system usability."
        )
        
        # Q2: Design and Interface
        q2 = st.radio(
            "**Q2. Interface Design**: The visual design and layout facilitate efficient navigation and information discovery.",
            likert_5,
            index=None,
            help="Evaluate the quality of the user interface design."
        )
        
        # Q3: Functionality
        q3 = st.radio(
            "**Q3. Functional Completeness**: The system provides useful features for evidence-based DAG exploration and triangulation analysis.",
            likert_5,
            index=None,
            help="Assess the usefulness of available features."
        )
        
        # Q4: Information Architecture
        q4 = st.radio(
            "**Q4. Information Clarity**: The evidence behind each edge, variable classification systems, and relationship directions are clearly presented and understandable.",
            likert_5,
            index=None,
            help="Evaluate clarity of information presentation, including evidence transparency and variable classifications."
        )
        
        # Q5: Scientific Rigor
        q5 = st.radio(
            "**Q5. Methodological Transparency**: The DAG generation methodology and evidence integration approach are adequately documented and transparent.",
            likert_5,
            index=None,
            help="Assess transparency of scientific methodology."
        )
        
        # Q6: Short answer - Design strengths
        st.markdown("**Q6. Design Strengths** (Short answer): What aspects of the system's design do you find most effective?")
        q6_text = st.text_area(
            "Please describe 2-3 specific design elements that enhance usability or scientific rigor:",
            height=80,
            key="q6_design",
            help="Examples: color coding, filtering options, interactive visualizations, etc."
        )
        
        # Q7: Short answer - Usability improvements
        st.markdown("**Q7. Usability Improvements** (Short answer): What enhancements would make the system even more user-friendly?")
        q7_text = st.text_area(
            "Please suggest improvements for navigation, feature discovery, or workflow (if any):",
            height=80,
            key="q7_usability",
            help="Optional: Share ideas for making the system more intuitive or efficient."
        )
        
        # Q8: Scientific Impact
        q8 = st.radio(
            "**Q8. Research Utility**: This system would be valuable for my research in causal inference or evidence synthesis.",
            likert_5,
            index=None,
            help="Assess potential research applications."
        )
        
        # Q9: Comparative Advantage
        q9 = st.radio(
            "**Q9. Innovation**: The integration of DAG structure with evidence triangulation represents a novel and useful approach.",
            likert_5,
            index=None,
            help="Evaluate the innovative aspects of the methodology."
        )
        
        # Q10: Short answer - Overall assessment
        st.markdown("**Q10. Overall Assessment** (Short answer): Provide a brief evaluation highlighting the system's strengths and contributions.")
        q10_text = st.text_area(
            "Please provide a concise assessment (2-3 sentences) covering the system's scientific contribution, usability strengths, and potential for future enhancement:",
            height=100,
            key="q10_overall",
            help="Focus on strengths and positive contributions, with optional suggestions for enhancement."
        )
        
        # Optional demographics
        st.divider()
        st.markdown("### Optional Demographics")
        
        role = st.selectbox(
            "Primary role:",
            ["", "Researcher/Academic", "Clinician", "Graduate Student", "Data Scientist", "Other"],
            index=0
        )
        
        experience = st.selectbox(
            "Experience with causal inference methods:",
            ["", "Expert", "Advanced", "Intermediate", "Beginner", "None"],
            index=0
        )
        
        # Submit button
        submitted = st.form_submit_button("Submit Evaluation", type="primary")
        
        if submitted:
            # Validate required fields
            required_quantitative = [q1, q2, q3, q4, q5, q8, q9]
            required_qualitative = [q6_text, q10_text]  # Q7 is optional
            
            if None in required_quantitative:
                st.error("Please answer all quantitative questions (Q1-Q5, Q8-Q9).")
            elif not all(required_qualitative):
                st.error("Please provide responses to required short-answer questions (Q6, Q10). Q7 is optional.")
            else:
                # Create response list
                responses_list = []
                
                # Quantitative questions with scores
                quantitative_questions = [
                    ("Q1. System Usability", q1),
                    ("Q2. Interface Design", q2),
                    ("Q3. Functional Completeness", q3),
                    ("Q4. Information Clarity", q4),
                    ("Q5. Methodological Transparency", q5),
                    ("Q8. Research Utility", q8),
                    ("Q9. Innovation", q9),
                ]
                
                for question, answer in quantitative_questions:
                    score = extract_score(answer)
                    answer_text = extract_text(answer)
                    responses_list.append({
                        "Question": question,
                        "Answer": answer_text,
                        "Score": score,
                        "Type": "Quantitative"
                    })
                
                # Qualitative questions
                qualitative_questions = [
                    ("Q6. Design Strengths", q6_text),
                    ("Q7. Usability Improvements", q7_text if q7_text else "No suggestions provided"),
                    ("Q10. Overall Assessment", q10_text),
                ]
                
                for question, answer in qualitative_questions:
                    responses_list.append({
                        "Question": question,
                        "Answer": answer,
                        "Score": "",
                        "Type": "Qualitative"
                    })
                
                # Demographics
                if role:
                    responses_list.append({"Question": "Role", "Answer": role, "Score": "", "Type": "Demographic"})
                if experience:
                    responses_list.append({"Question": "Experience Level", "Answer": experience, "Score": "", "Type": "Demographic"})
                
                # Store in session state
                st.session_state.questionnaire_responses = responses_list
                st.success("Thank you for your evaluation! Your responses have been recorded.")
                st.rerun()
    
    # Display results and download button outside the form
    if st.session_state.questionnaire_responses:
        st.divider()
        st.subheader("Evaluation Summary")
        
        # Create DataFrame
        response_df = pd.DataFrame(st.session_state.questionnaire_responses)
        
        # Separate quantitative and qualitative
        quantitative_df = response_df[response_df["Type"] == "Quantitative"]
        qualitative_df = response_df[response_df["Type"] == "Qualitative"]
        
        # Display quantitative scores
        if not quantitative_df.empty:
            st.markdown("**Quantitative Ratings:**")
            st.dataframe(
                quantitative_df[["Question", "Answer", "Score"]],
                use_container_width=True,
                hide_index=True
            )
            
            # Calculate average score
            scores = [r["Score"] for r in st.session_state.questionnaire_responses 
                     if r.get("Score") and isinstance(r["Score"], int)]
            if scores:
                avg_score = sum(scores) / len(scores)
                st.metric("Average Score (Q1-Q5, Q8-Q9)", f"{avg_score:.2f} / 5.00")
        
        # Display qualitative responses
        if not qualitative_df.empty:
            st.markdown("**Qualitative Feedback:**")
            for _, row in qualitative_df.iterrows():
                st.markdown(f"**{row['Question']}:**")
                st.write(row['Answer'])
                st.markdown("---")
        
        # Download button
        csv_data = response_df[["Question", "Answer", "Score", "Type"]].to_csv(index=False)
        st.download_button(
            "Download Evaluation Responses (CSV)",
            data=csv_data,
            file_name=f"evaluation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        
        st.info("""
        **Data Usage**: Your responses will be used for:
        - Academic research and publication
        - System usability analysis
        - Methodological validation
        - Future system improvements
        """)
        
        # Reset button
        if st.button("Submit New Evaluation"):
            st.session_state.questionnaire_responses = None
            st.rerun()


def render_dag_generation() -> None:
    """Render the DAG Generation Methodology page."""
    st.header("DAG Generation Methodology")
    st.caption("How the Evidence-Based DAG is constructed from raw literature evidence")
    
    st.markdown("""
    This page explains the step-by-step process of generating the evidence-based directed acyclic graph (DAG) 
    that represents causal relationships between variables in the Influenza Vaccine → Stroke research question.
    """)
    
    st.divider()
    
    # Step 1
    st.subheader("Step 1: Query Evidence from Literature")
    st.markdown("""
    **Objective**: Retrieve relevant biomedical literature from multiple databases.
    
    **Process**:
    - Query PubMed, Embase, Scopus, and Web of Science databases
    - Search for studies related to influenza vaccine, stroke, and potential confounding/mediating variables
    - Retrieve abstracts, titles, and metadata (PMID, publication year, study design, etc.)
    - Collect evidence across different study designs: RCTs, observational studies, Mendelian randomization
    
    **Output**: Raw evidence database containing thousands of publications with their abstracts and metadata.
    """)
    
    # Step 2
    st.subheader("Step 2: Extract Entities and Relations Using Large Language Models")
    st.markdown("""
    **Objective**: Automatically extract structured information from unstructured text.
    
    **Process**:
    - Use large language models (LLMs) to parse abstracts and extract:
      - **Exposure entities**: Variables that may cause or influence outcomes
      - **Outcome entities**: Variables that may be affected by exposures
      - **Relations**: Causal relationships between exposure-outcome pairs
      - **Effect directions**: Excitatory (positive), inhibitory (negative), or no change
      - **Study design**: RCT, observational study, Mendelian randomization, etc.
      - **Statistical significance**: p-values, confidence intervals, effect sizes
    
    **Output**: Structured evidence table with exposure, outcome, relation type, study design, and effect measures.
    """)
    
    # Step 3
    st.subheader("Step 3: Validate Exposure-Outcome Relevance Using LLMs")
    st.markdown("""
    **Objective**: Ensure extracted entities are relevant to the research question.
    
    **Process**:
    - Use LLMs to check if extracted exposures and outcomes are:
      - Related to **Influenza Vaccine** (X) or **Stroke** (Y)
      - Relevant to the causal pathway of interest
      - Not spurious or irrelevant associations
    
    **Filtering**:
    - Remove studies where exposure/outcome pairs are not meaningfully related to fluvac-stroke relationship
    - Validate that entities are biomedical concepts (not artifacts or parsing errors)
    - Ensure semantic relevance to the research domain
    
    **Output**: Validated evidence with confirmed relevance to Influenza Vaccine → Stroke research question.
    """)
    
    # Step 4
    st.subheader("Step 4: Map to UMLS Concepts Using SciSpaCy and UMLS API")
    st.markdown("""
    **Objective**: Standardize entity names using biomedical ontologies.
    
    **Process**:
    - Use **SciSpaCy** (biomedical NLP library) for named entity recognition
    - Map extracted entities to **UMLS (Unified Medical Language System)** concepts:
      - Resolve synonyms and variations (e.g., "flu vaccine" = "influenza vaccine")
      - Link to standardized concept unique identifiers (CUIs)
      - Leverage UMLS semantic types and relationships
    - Use UMLS API to retrieve:
      - Preferred concept names
      - Semantic types (e.g., "Pharmacologic Substance", "Disease or Syndrome")
      - Related concepts and hierarchical relationships
    
    **Benefits**:
    - Standardizes terminology across different studies
    - Enables semantic matching and similarity calculations
    - Links to broader biomedical knowledge graphs
    
    **Output**: Evidence table with UMLS-mapped exposure and outcome concepts.
    """)
    
    # Step 5
    st.subheader("Step 5: Transform Relations to Discover Indication Variables")
    st.markdown("""
    **Objective**: Identify variables that may indicate or contraindicate influenza vaccination.
    
    **Process**:
    - **Relation Transformation**: Reverse causal directions to discover indication variables
      - Original: `Influenza → Disease` (vaccine prevents disease)
      - Transformed: `Disease → Influenza Vaccine` (disease indicates need for vaccine)
    - Identify variables that:
      - Are **indications** for vaccination (e.g., chronic conditions, age groups)
      - Are **contraindications** for vaccination (e.g., allergies, immunocompromised states)
      - May **modify** vaccine effectiveness or uptake
    
    **Rationale**:
    - Diseases/conditions that increase vaccine uptake are potential confounders
    - Conditions that affect vaccine response are potential mediators
    - This bidirectional exploration captures the full causal landscape
    
    **Output**: Expanded set of variables with bidirectional relationships to influenza vaccine.
    """)
    
    # Step 6
    st.subheader("Step 6: Identify Parents and Children of Key Variables")
    st.markdown("""
    **Objective**: Map causal ancestors (causes) and descendants (consequences) of Influenza Vaccine and Stroke.
    
    **Process**:
    - **For Influenza Vaccine (X)**:
      - **Parents (Causes)**: Variables that influence vaccine uptake or administration
        - Demographics (age, gender, socioeconomic status)
        - Health conditions (chronic diseases, comorbidities)
        - Healthcare access and policies
      - **Children (Consequences)**: Variables affected by vaccination
        - Immune response markers
        - Infection rates
        - Health outcomes
    
    - **For Stroke (Y)**:
      - **Parents (Causes)**: Risk factors for stroke
        - Cardiovascular risk factors (hypertension, diabetes, etc.)
        - Lifestyle factors
        - Genetic factors
      - **Children (Consequences)**: Outcomes of stroke
        - Disability, mortality
        - Healthcare utilization
    
    **Method**:
    - Query evidence database for all relationships where X or Y appears as exposure or outcome
    - Aggregate by direction to identify causal parents and children
    - Count evidence strength (number of studies, LOE) for each relationship
    
    **Output**: Comprehensive lists of variables that are causes or consequences of Influenza Vaccine and Stroke.
    """)
    
    # Step 7
    st.subheader("Step 7: Cluster Variables Using Semantic Similarity (SapBERT)")
    st.markdown("""
    **Objective**: Group similar variables to reduce redundancy and identify variable categories.
    
    **Process**:
    - Use **SapBERT** (Semantically-aware BERT) embeddings:
      - Model: `UMCU/SapBERT-from-PubMedBERT-fulltext_bf16`
      - Encodes biomedical concepts into high-dimensional vector representations
      - Captures semantic similarity between concepts
    
    - **Clustering Steps**:
      1. Generate embeddings for all unique variables
      2. Calculate pairwise cosine similarity between embeddings
      3. Apply clustering algorithms (e.g., hierarchical clustering, k-means)
      4. Group variables with high semantic similarity
    
    - **Domain Identification**:
      - Variables are grouped into domains such as:
        - **Cardiovascular Risk**: Hypertension, diabetes, cholesterol
        - **Infection/Immunity**: Immune disorders, vaccination history
        - **Neurological**: Cognitive function, neurological conditions
        - **Demographics**: Age, gender, socioeconomic factors
    
    **Benefits**:
    - Reduces variable space by grouping synonyms and related concepts
    - Enables domain-based filtering and analysis
    - Identifies variable categories for DAG structure
    
    **Output**: Clustered variables with domain assignments and semantic similarity scores.
    """)
    
    # Step 8
    st.subheader("Step 8: Classify Variables by Causal Role")
    st.markdown("""
    **Objective**: Determine whether each variable is a confounder, mediator, collider, or prognostic factor.
    
    **Classification Logic**:
    
    **Confounders (Z)**: Variables that affect both X and Y
    - **Pattern**: Z → X AND Z → Y
    - **Identification**: Variables that are:
      - Common parents of both Influenza Vaccine and Stroke
      - Have evidence supporting relationships to both X and Y
    - **Example**: Age (affects vaccine uptake and stroke risk)
    
    **Mediators (Z)**: Variables on the causal path from X to Y
    - **Pattern**: X → Z → Y
    - **Identification**: Variables that are:
      - Children of Influenza Vaccine (X → Z)
      - Parents of Stroke (Z → Y)
      - On the causal pathway
    - **Example**: Immune response (vaccine affects immune response, which affects stroke risk)
    
    **Colliders (Z)**: Variables affected by both X and Y
    - **Pattern**: X → Z ← Y
    - **Identification**: Variables that are:
      - Children of both Influenza Vaccine and Stroke
      - Affected by both exposure and outcome
    - **Example**: Hospital admission (affected by both vaccine status and stroke occurrence)
    
    **Prognostic Factors**: Variables that affect Y but not X
    - **Pattern**: Z → Y (but not Z → X)
    - **Identification**: Variables that are:
      - Parents of Stroke
      - Not related to Influenza Vaccine
    - **Example**: Pre-existing cardiovascular conditions (affect stroke risk independently)
    
    **Evidence-Based Classification**:
    - Each classification requires evidence from the literature
    - Variables are classified based on the pattern of relationships found in evidence
    - Classification is stored in `final_causal_master_list_with_sets_1.2.xlsx`
    
    **Output**: Final variable classification with causal roles (Confounder, Mediator, Collider, Prognostic).
    """)
    
    st.divider()
    
    st.subheader("DAG Construction")
    st.markdown("""
    **Final DAG Structure**:
    
    1. **Base Edges**: Constructed from variable classifications
       - Confounders: Z → X, Z → Y
       - Mediators: X → Z → Y
       - Colliders: X → Z ← Y
       - Prognostic: Z → Y
    
    2. **Peer Interactions**: Additional edges discovered through evidence retrieval
       - Relationships between variables (Z_i → Z_j)
       - Only included if evidence strength (LOE) meets threshold criteria
    
    3. **Evidence Integration**:
       - Each edge is supported by evidence from multiple study designs
       - Level of Evidence (LOE) calculated based on:
         - Study design hierarchy (RCT > OS > MR)
         - Number of supporting studies
         - Consistency of effect directions
    
    **Result**: A comprehensive, evidence-based DAG representing the causal structure around 
    Influenza Vaccine → Stroke relationship, accounting for confounders, mediators, and colliders.
    """)
    
    st.divider()
    
    st.subheader("Key Technologies")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **NLP & LLMs**:
        - Large Language Models for entity extraction
        - SciSpaCy for biomedical NER
        - Semantic similarity for clustering
        
        **Biomedical Ontologies**:
        - UMLS for concept standardization
        - Semantic type classification
        """)
    
    with col2:
        st.markdown("""
        **Evidence Synthesis**:
        - Triangulation across study designs
        - Level of Evidence calculation
        - Probability-based relationship inference
        
        **Visualization**:
        - PyVis for interactive network graphs
        - Force-directed layout algorithms
        """)


def render_about() -> None:
    """Render the About page."""
    st.header("About Evidence-Based DAG Explorer")
    
    st.markdown("""
    ## Project Overview
    
    This interactive web application is designed for exploring and analyzing **evidence-based directed acyclic graphs (DAGs)** 
    that represent causal relationships between variables in biomedical research. The DAG structure accounts for different 
    causal roles of variables—confounders, mediators, colliders, and prognostic factors—and integrates evidence from 
    multiple study designs to support robust causal inference.
    
    The application focuses on a specific research question: **How does Influenza Vaccine affect Stroke risk?** 
    The DAG includes variables (Z) that may confound, mediate, or interact with this relationship.
    
    ---
    """)
    
    st.subheader("📊 DAG Structure & Variable Classification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Base DAG Structure
        
        Variables are classified by their causal role:
        
        - **Confounders (Z)**: Affect both exposure (X) and outcome (Y)
          - Pattern: **Z → X, Z → Y**
          - Example: Age affects both vaccine uptake and stroke risk
        
        - **Mediators (Z)**: On the causal path from X to Y
          - Pattern: **X → Z → Y**
          - Example: Vaccine → Immune Response → Stroke Protection
        
        - **Colliders (Z)**: Affected by both X and Y
          - Pattern: **X → Z ← Y**
          - Example: Hospital Admission (affected by both vaccine status and stroke)
        
        - **Prognostic Factors**: Affect outcome but not exposure
          - Pattern: **Z → Y**
          - Example: Pre-existing conditions affecting stroke risk
        """)
    
    with col2:
        st.markdown("""
        ### Evidence Integration
        
        **Base Edges**: Derived from variable classification in the final causal master list
        - Each variable's category determines its connections to X (Influenza Vaccine) and Y (Stroke)
        
        **Peer Interactions**: Additional edges discovered through evidence retrieval
        - Relationships between variables (Z_i → Z_j) found in the literature
        - Only shown if evidence strength (LOE) meets threshold criteria
        
        **Level of Evidence (LOE)**: Quantifies evidence strength (0-1 scale)
        - Based on study design hierarchy (RCT > OS > MR)
        - Number of supporting studies
        - Consistency of effect directions
        """)
    
    st.subheader("🔧 How to Use Each Module")
    
    tab_features = st.tabs(["Network Explorer", "Strategy Evidence Viewer", "Evidence Triangulation"])
    
    with tab_features[0]:
        st.markdown("""
        ### Network Explorer Module
        
        **Purpose**: Visualize and interactively explore the causal DAG structure.
        
        **Step-by-Step Guide**:
        
        1. **Understanding the Network**:
           - **Blue nodes**: Exposure (Influenza Vaccine) and Outcome (Stroke) - fixed positions
           - **Colored nodes**: Variables (Z) color-coded by category:
             - 🟠 Orange: Confounders
             - 🟣 Purple: Mediators
             - 🟢 Green: Colliders
             - 🔵 Blue: Prognostic factors
           - **Light blue edges**: Base edges from variable classification
           - **Light green edges**: Peer interaction edges from evidence
        
        2. **Using Sidebar Filters**:
           - **Minimum LOE**: Filter edges by evidence strength (default: 0.0 shows all)
           - **Minimum studies**: Filter peer interactions by number of supporting studies
           - **Show base edges**: Toggle base DAG structure (from variable classification)
           - **Show peer interactions**: Toggle additional evidence-based edges
           - **Node types**: Filter by variable category (Confounder, Mediator, Collider, Prognostic)
           - **Relationship direction**: Filter by effect type (Excitatory, Inhibitory, No Change)
           - **Domain filters**: Show only variables from specific domains (e.g., CV Risk, Infection, Neuro)
           - **Peer Interaction Filters**: Show peer interactions only between specific variable categories
        
        3. **Interactive Features**:
           - **Drag nodes**: Click and drag variable nodes (Z) to reposition them
           - **Focus on node**: Select a node to highlight it and its connections (others dimmed)
           - **Focus on edge**: Select an edge to highlight it and its endpoints
           - **Hover**: Hover over nodes/edges to see details
           - **Search**: Type in node name to filter edges
        
        4. **Viewing Evidence**:
           - **Edge Table**: Shows all filtered edges with LOE, study counts, and probabilities
           - **Evidence Viewer**: Select source and target nodes to view raw evidence from `final_dag_evidence.csv`
           - **Download**: Export filtered edges or evidence as CSV
        
        5. **Tips**:
           - Start with all filters enabled to see the full network
           - Use domain filters to focus on specific research areas
           - Increase LOE threshold to see only high-confidence relationships
           - Use focus mode to reduce visual clutter when exploring specific relationships
        """)
    
    with tab_features[1]:
        st.markdown("""
        ### Strategy Evidence Viewer Module
        
        **Purpose**: Explore the raw evidence supporting each variable's role in the DAG strategy.
        
        **Step-by-Step Guide**:
        
        1. **Select a Variable**:
           - Choose any variable (Z) from the dropdown list
           - Variables come from the final causal master list
        
        2. **Review Path Summary**:
           - The summary table shows PMID counts for four potential causal paths:
             - **Z → Influenza Vaccine**: Does this variable affect vaccine uptake?
             - **Z → Stroke**: Does this variable affect stroke risk?
             - **Influenza Vaccine → Z**: Does vaccine affect this variable?
             - **Stroke → Z**: Does stroke affect this variable?
           - Higher PMID counts indicate more evidence for that path
        
        3. **Drill Down to Evidence**:
           - Select one of the four paths from the dropdown
           - View all raw evidence rows from `umls_df_v3.csv` that support this path
           - The table shows all available columns including:
             - PMID, exposure, outcome, study design
             - Effect measures, confidence intervals
             - UMLS mappings, and more
        
        4. **Download Evidence**:
           - Click "Download evidence (CSV)" to export the filtered evidence table
           - File name includes variable name and path for easy identification
        
        5. **Use Cases**:
           - Verify evidence quality for specific variables
           - Understand why a variable was classified as confounder/mediator/collider
           - Explore evidence gaps (paths with few PMIDs)
           - Extract evidence for further analysis or reporting
        """)
    
    with tab_features[2]:
        st.markdown("""
        ### Evidence Triangulation Module
        
        **Purpose**: Query any exposure-outcome pair and perform triangulation analysis using evidence from the database.
        
        **Step-by-Step Guide**:
        
        1. **Enter Query**:
           - **Exposure**: Enter any exposure term (e.g., "Salt", "Diabetes", "High Blood Pressure")
           - **Outcome**: Enter any outcome term (e.g., "Stroke", "Hypertension", "Cardiovascular Disease")
           - Terms don't need to match exactly - semantic similarity is used
        
        2. **Set Similarity Threshold**:
           - **Lower threshold (0.5-0.7)**: More results, broader matching
           - **Higher threshold (0.8-0.9)**: Fewer, more precise results
           - **Default (0.7)**: Balanced between recall and precision
           - Adjust based on your needs: start low, increase if too many irrelevant results
        
        3. **Run Analysis**:
           - Click "Run Triangulation Analysis"
           - System retrieves evidence using semantic similarity (SapBERT embeddings)
           - Triangulation algorithm analyzes evidence across study designs
        
        4. **Interpret Results**:
           - **Primary Relation**: The dominant relationship type (Excitatory/Inhibitory/No Change)
           - **Level of Evidence (LOE)**: Strength of evidence (0-1, higher = stronger)
           - **Total Studies**: Number of evidence rows found
           - **Unique PMIDs**: Number of unique publications
           - **Probability Breakdown**: Distribution showing:
             - p_excitatory: Probability of excitatory relationship
             - p_inhibitory: Probability of inhibitory relationship
             - p_no_change: Probability of no effect
           - **Study Design Distribution**: Breakdown by RCT, OS, MR, etc.
        
        5. **Explore Evidence**:
           - Review the full evidence table with all retrieved rows
           - Check study designs, effect measures, and confidence intervals
           - Download results for further analysis
        
        6. **Tips**:
           - Use specific medical terms for better matching
           - If no results, try lowering the threshold
           - Compare results across different thresholds to assess robustness
           - Use UMLS concept names if you know them for most precise matching
        """)
    
    st.subheader("📁 Data Sources & Structure")
    st.markdown("""
    The application integrates data from multiple sources:
    
    | Source File | Description | Used In |
    |-------------|-------------|---------|
    | `final_causal_master_list_with_sets_1.2.xlsx` | Master list of variables with final categories (Confounder/Mediator/Collider/Prognostic) | Network Explorer (base edges) |
    | `labeled_parents/mediators/children_1.2.xlsx` | Domain classifications for variables (e.g., CV Risk, Infection, Neuro) | Network Explorer (domain filters) |
    | `final_dag_edges_clean.csv` | Network edges with LOE scores, study counts, and probabilities | Network Explorer (peer interactions) |
    | `final_dag_evidence.csv` | Raw evidence supporting each edge (PMID, exposure, outcome, etc.) | Network Explorer (evidence viewer) |
    | `final_strategy_evidence_full.xlsx` | Strategy evidence with PMID lists for each variable's paths | Strategy Evidence Viewer |
    | `umls_df_v3.csv` | Complete UMLS-mapped evidence database with full text/sentences | Strategy Evidence Viewer, Evidence Triangulation |
    | `all_df.csv` | Processed evidence database with exposure direction mappings | Evidence Triangulation |
    """)
    
    st.subheader("🔬 Technical Implementation")
    
    st.markdown("""
    ### Evidence Retrieval System
    
    **Semantic Similarity Search**:
    - Uses **SapBERT** (Semantically-aware BERT) model: `UMCU/SapBERT-from-PubMedBERT-fulltext_bf16`
    - Pre-computes embeddings for all unique exposure/outcome concepts in the database
    - Encodes query terms and matches against database using cosine similarity
    - Threshold-based filtering ensures relevance
    
    ### Triangulation Algorithm
    
    The `analyze_tri_df()` function processes evidence to compute:
    1. **Study Design Separation**: Handles MR studies differently from observational/RCT
    2. **Direction Filtering**: Validates exposure and outcome directions
    3. **Weighting** (optional): Applies participant-based weights for large datasets
    4. **Aggregation**: Groups by study design, exposure direction, outcome direction
    5. **Probability Calculation**: Computes p_excitatory, p_inhibitory, p_no_change
    6. **LOE Calculation**: Quantifies evidence strength based on convergence patterns
    
    ### Network Visualization
    
    **PyVis Network**:
    - Force-directed physics simulation for natural node positioning
    - Fixed positions for X (Influenza Vaccine) and Y (Stroke) nodes
    - Draggable intermediate nodes (Z) for custom layouts
    - Color-coding by variable category
    - Edge colors distinguish base edges (light blue) from peer interactions (light green)
    - Focus mode with transparency for highlighting specific relationships
    """)
    
    st.subheader("📚 Key Concepts")
    
    st.markdown("""
    ### Variable Categories in DAG
    
    - **Confounder**: Creates spurious association between X and Y
      - Must be controlled in analysis
      - Example: Age confounding vaccine-stroke relationship
    
    - **Mediator**: Explains how X affects Y
      - On the causal pathway
      - Example: Immune response mediating vaccine-stroke protection
    
    - **Collider**: Creates selection bias if conditioned on
      - Should NOT be controlled
      - Example: Hospital admission (collider of vaccine and stroke)
    
    - **Prognostic Factor**: Affects outcome independently
      - May be useful for stratification
      - Example: Pre-existing conditions affecting stroke risk
    
    ### Evidence Metrics
    
    - **LOE (Level of Evidence)**: 0-1 scale quantifying evidence strength
      - Higher = more studies, better designs, greater consistency
    
    - **Relationship Directions**:
      - **Excitatory**: Exposure increases outcome (positive association)
      - **Inhibitory**: Exposure decreases outcome (negative association)
      - **No Change**: No significant effect
    
    ### Study Designs
    
    - **RCT**: Randomized controlled trials (gold standard)
    - **OS**: Observational studies (cohort, case-control, cross-sectional)
    - **MR**: Mendelian randomization (genetic instrumental variables)
    """)
    
    st.subheader("🚀 Quick Start Guide")
    
    st.markdown("""
    1. **Start with Network Explorer**:
       - View the overall DAG structure
       - Understand variable classifications
       - Explore base edges and peer interactions
    
    2. **Use Strategy Evidence Viewer**:
       - Select variables of interest
       - Review evidence supporting their classification
       - Drill down to raw evidence for specific paths
    
    3. **Try Evidence Triangulation**:
       - Query custom exposure-outcome pairs
       - Adjust threshold to balance precision/recall
       - Analyze triangulation results
    
    4. **Export and Analyze**:
       - Download filtered edges and evidence tables
       - Use exported data for further statistical analysis
       - Generate reports and visualizations
    """)


def render_evidence_triangulation() -> None:
    """Render the Evidence Triangulation tab."""
    st.header("Evidence Triangulation")
    st.caption("Query any exposure-outcome pair and get triangulation analysis results.")
    
    # Load all_df
    all_df = load_all_df()
    if all_df.empty:
        st.error(f"all_df file not found: {DATA_PATH_ALL_DF}")
        return
    
    # Initialize retriever (cached)
    with st.spinner("Initializing evidence retriever (this may take a moment on first run)..."):
        retriever = get_retriever(all_df)
        if retriever is None:
            st.error("Failed to initialize evidence retriever.")
            return
    
    # Input section
    st.subheader("Query Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        exposure_input = st.text_input("Exposure", placeholder="e.g., Salt, High Blood Pressure, Diabetes")
    
    with col2:
        outcome_input = st.text_input("Outcome", placeholder="e.g., Stroke, Hypertension, Cardiovascular Disease")
    
    threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Higher threshold means stricter matching (default: 0.7)"
    )
    
    # Query button
    if st.button("Run Triangulation Analysis", type="primary"):
        if not exposure_input.strip() or not outcome_input.strip():
            st.warning("Please enter both exposure and outcome.")
            return
        
        with st.spinner("Retrieving evidence..."):
            # Retrieve evidence
            evidence_df = retriever.retrieve(
                target_exposure=exposure_input.strip(),
                target_outcome=outcome_input.strip(),
                threshold=threshold,
                strict_tui=True
            )
        
        if evidence_df.empty:
            st.warning(f"No evidence found for '{exposure_input}' → '{outcome_input}' with threshold {threshold}.")
            st.info("Try lowering the threshold or checking the spelling of your terms.")
            return
        
        # Display evidence summary
        st.success(f"Found **{len(evidence_df)} evidence rows** from **{evidence_df['pmid'].nunique()} unique studies**")
        
        # Run triangulation analysis
        with st.spinner("Running triangulation analysis..."):
            try:
                tri_results = analyze_tri_df(evidence_df, detail_info=False, use_participant_weights=True)
            except Exception as e:
                st.error(f"Error in triangulation analysis: {e}")
                st.exception(e)
                return
        
        # Display results
        st.subheader("Triangulation Results")
        
        # Results summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Primary Relation", tri_results.get('biggest', 'Unknown').title())
        with col2:
            st.metric("Level of Evidence (LOE)", f"{tri_results.get('loe', 0):.3f}")
        with col3:
            st.metric("Total Studies", tri_results.get('count', 0))
        with col4:
            st.metric("Unique PMIDs", evidence_df['pmid'].nunique())
        
        # Probability breakdown
        st.subheader("Probability Breakdown")
        prob_cols = ['p_excitatory', 'p_no_change', 'p_inhibitory']
        prob_data = {
            'Relation Type': ['Excitatory', 'No Change', 'Inhibitory'],
            'Probability': [
                tri_results.get('p_excitatory', 0),
                tri_results.get('p_no_change', 0),
                tri_results.get('p_inhibitory', 0)
            ]
        }
        prob_df = pd.DataFrame(prob_data)
        st.dataframe(prob_df, use_container_width=True)
        
        # Visualize probabilities
        if any(prob_df['Probability'] > 0):
            st.bar_chart(prob_df.set_index('Relation Type')['Probability'])
        
        # Evidence table
        st.subheader("Retrieved Evidence")
        st.dataframe(evidence_df.reset_index(drop=True), use_container_width=True)
        
        # Download button
        st.download_button(
            "Download Evidence (CSV)",
            data=evidence_df.to_csv(index=False),
            file_name=f"evidence_{exposure_input.replace(' ', '_')}_to_{outcome_input.replace(' ', '_')}.csv",
            mime="text/csv",
        )
        
        # Study design breakdown
        if 'study_design' in evidence_df.columns:
            st.subheader("Study Design Distribution")
            design_counts = evidence_df['study_design'].value_counts()
            st.dataframe(design_counts.reset_index().rename(columns={'index': 'Study Design', 'study_design': 'Count'}))


def render_network_explorer() -> None:
    """Render the Network Explorer tab."""
    st.header("Network Explorer")
    st.caption("Interactively explore DAG edges based on the final causal list and network outputs.")

    if not DATA_PATH_FINAL.exists():
        st.error(f"Final list not found: {DATA_PATH_FINAL}")
        return
    if not DATA_PATH_NETWORK.exists():
        st.error(f"Network edge file not found: {DATA_PATH_NETWORK}")
        return

    final_df = load_final_list()
    base_edges = build_base_edges(final_df)
    network_df = load_network_edges()
    domain_map = load_domain_mappings()

    # Allowed nodes = X, Y, and all Z from final list. No new nodes permitted.
    allowed_nodes = {X_NODE, Y_NODE}.union(set(final_df["variable_clean"].tolist()))

    # Build node category mapping
    node_category = {}
    node_category[X_NODE] = "EXPOSURE"
    node_category[Y_NODE] = "OUTCOME"
    for _, row in final_df.iterrows():
        node_category[row["variable_clean"]] = row["final_category"]
    
    # Domain mappings
    node_domain = {}
    node_domain[X_NODE] = "Exposure"
    node_domain[Y_NODE] = "Outcome"
    for node in allowed_nodes:
        if node in domain_map:
            node_domain[node] = domain_map[node]

    # Keep only network edges that do not involve X or Y and stay within allowed nodes.
    network_filtered = network_df[
        (~network_df["Source"].isin({X_NODE, Y_NODE}))
        & (~network_df["Target"].isin({X_NODE, Y_NODE}))
        & (network_df["Source"].isin(allowed_nodes))
        & (network_df["Target"].isin(allowed_nodes))
    ].copy()

    st.sidebar.header("Filters")
    loe_min = st.sidebar.slider(
        "Minimum LOE",
        0.0,
        float(max(network_filtered["loe"].max() if not network_filtered.empty else 1.0, 1.0)),
        0.0,
        0.01,
    )
    study_min = st.sidebar.slider(
        "Minimum number of studies (network edges only)",
        0,
        int(network_filtered["count"].max() if not network_filtered.empty else 10),
        0,
        1,
    )

    include_base = st.sidebar.checkbox("Show base edges (node types from final list)", value=True)
    include_network = st.sidebar.checkbox("Show peer interaction edges", value=True)
    
    # Filter peer interactions by node category
    st.sidebar.subheader("Peer Interaction Filters")
    category_options = ["FINAL_CONFOUNDER", "FINAL_MEDIATOR", "FINAL_COLLIDER", "FINAL_PROGNOSTIC"]
    selected_categories = st.sidebar.multiselect(
        "Show peer interactions only between nodes of these categories (leave empty for all)",
        category_options,
        default=[],
    )
    
    # Filter by domain
    st.sidebar.subheader("Domain Filters")
    all_domains = sorted(set(domain_map.values()))
    selected_domains = st.sidebar.multiselect(
        "Show only nodes from these domains (leave empty for all)",
        all_domains,
        default=[],
    )

    # Combine to derive available types/relations for filters.
    combined_df = pd.concat([base_edges, network_filtered], ignore_index=True)
    node_type_options = sorted(base_edges["Type"].unique().tolist())
    selected_types = st.sidebar.multiselect("Node types (from final category)", node_type_options, default=node_type_options)
    relation_opts = ["excitatory", "inhibitory", "no change"]
    selected_relations = st.sidebar.multiselect("Relationship direction", relation_opts, default=relation_opts)

    all_nodes = sorted(set(combined_df["Source"]).union(combined_df["Target"]))
    highlight_choice = st.sidebar.multiselect("Highlight nodes", all_nodes)

    search_text = st.sidebar.text_input("Node name contains", "")

    pieces = []
    if include_base and not base_edges.empty:
        base_part = base_edges[
            (base_edges["Type"].isin(selected_types)) & (base_edges["biggest"].isin(selected_relations))
        ].copy()
        
        # Apply domain filter to base edges
        if selected_domains:
            base_part["source_domain"] = base_part["Source"].map(node_domain)
            base_part["target_domain"] = base_part["Target"].map(node_domain)
            base_part = base_part[
                (base_part["Source"].isin({X_NODE, Y_NODE})) | (base_part["source_domain"].isin(selected_domains))
            ].copy()
            base_part = base_part[
                (base_part["Target"].isin({X_NODE, Y_NODE})) | (base_part["target_domain"].isin(selected_domains))
            ].copy()
            base_part = base_part.drop(columns=["source_domain", "target_domain"])
        
        pieces.append(base_part)

    if include_network and not network_filtered.empty:
        net_part = network_filtered[
            (network_filtered["loe"] >= loe_min)
            & (network_filtered["count"] >= study_min)
            & (network_filtered["biggest"].isin(selected_relations))
        ].copy()
        
        # Apply category filter for peer interactions if specified
        if selected_categories:
            net_part["source_category"] = net_part["Source"].map(node_category)
            net_part["target_category"] = net_part["Target"].map(node_category)
            net_part = net_part[
                (net_part["source_category"].isin(selected_categories))
                & (net_part["target_category"].isin(selected_categories))
            ].copy()
            net_part = net_part.drop(columns=["source_category", "target_category"])
        
        # Apply domain filter to network edges
        if selected_domains:
            net_part["source_domain"] = net_part["Source"].map(node_domain)
            net_part["target_domain"] = net_part["Target"].map(node_domain)
            net_part = net_part[
                (net_part["source_domain"].isin(selected_domains))
                & (net_part["target_domain"].isin(selected_domains))
            ].copy()
            net_part = net_part.drop(columns=["source_domain", "target_domain"])
        
        pieces.append(net_part)

    filtered = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    # Sort by LOE from high to low
    if not filtered.empty:
        filtered = filtered.sort_values(by=["loe"], ascending=[False]).reset_index(drop=True)

    if search_text.strip():
        mask = filtered["Source"].str.contains(search_text, case=False, na=False) | filtered["Target"].str.contains(
            search_text, case=False, na=False
        )
        filtered = filtered[mask]

    st.write(
        f"Showing **{len(filtered)} edges** between **{filtered['Source'].nunique()} sources** and "
        f"**{filtered['Target'].nunique()} targets** after filtering."
    )

    if filtered.empty:
        st.info("No edges match the current filters.")
        return

    # Focus selections
    focus_node = st.selectbox("Focus on node (others dim)", [""] + all_nodes)
    focus_node_val = focus_node if focus_node else None

    edge_labels = [f"{r.Source} → {r.Target}" for _, r in filtered.iterrows()]
    focus_edge_label = st.selectbox("Focus on edge (others dim)", [""] + edge_labels)
    focus_edge_val = None
    if focus_edge_label:
        parts = focus_edge_label.split(" → ")
        if len(parts) == 2:
            focus_edge_val = (parts[0], parts[1])

    net = build_network(filtered, set(highlight_choice), node_category, node_domain, focus_node_val, focus_edge_val)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        html = Path(tmp.name).read_text()

    components.html(html, height=780, scrolling=True)

    # Legend for node colors
    st.markdown(
        """
**Node color legend**
- 🟠 Confounder (FINAL_CONFOUNDER)
- 🟣 Mediator (FINAL_MEDIATOR)
- 🟢 Collider (FINAL_COLLIDER)
- 🔵 Prognostic (FINAL_PROGNOSTIC)
- 🔷 Exposure (Influenza Vaccine)
- 🔴 Outcome (Stroke)
- 🟡 Highlighted nodes (user selection)
**Edge color legend**
- 🔵 Base edges (light blue)
- 🟢 Peer interaction edges (light green)
        """
    )

    st.subheader("Edge Table")
    filtered_reset = filtered.reset_index(drop=True)
    st.dataframe(filtered_reset)
    st.download_button(
        "Download filtered edges (CSV)",
        data=filtered_reset.to_csv(index=False),
        file_name="filtered_edges.csv",
        mime="text/csv",
    )

    # Evidence viewer (from final_dag_evidence.csv)
    evidence_path = DATA_DIR / "extracted_results" / "final_dag_evidence.csv"
    if evidence_path.exists():
        evidence_df = pd.read_csv(evidence_path)
        evidence_df = drop_unnamed_columns(evidence_df)
        st.subheader("Evidence (from final_dag_evidence.csv)")
        
        # Get unique sources and targets from filtered edges
        unique_sources = sorted(filtered_reset["Source"].unique().tolist())
        unique_targets = sorted(filtered_reset["Target"].unique().tolist())
        
        if unique_sources and unique_targets:
            col1, col2 = st.columns(2)
            with col1:
                selected_source = st.selectbox("Select source node", [""] + unique_sources)
            with col2:
                # Filter targets based on selected source
                if selected_source:
                    available_targets = sorted(
                        filtered_reset[filtered_reset["Source"] == selected_source]["Target"].unique().tolist()
                    )
                else:
                    available_targets = unique_targets
                selected_target = st.selectbox("Select target node", [""] + available_targets)
            
            if selected_source and selected_target:
                ev = evidence_df[
                    (evidence_df.get("Edge_Source") == selected_source) 
                    & (evidence_df.get("Edge_Target") == selected_target)
                ]
                if ev.empty:
                    st.info(f"No evidence rows for edge: {selected_source} → {selected_target}")
                else:
                    st.dataframe(ev.reset_index(drop=True))
        else:
            st.info("No edges to view evidence.")
    else:
        st.info("Evidence file not found: final_dag_evidence.csv")


def main() -> None:
    """Main entry point with tabbed interface."""
    st.set_page_config(page_title="Evidence DAG Explorer", layout="wide")
    st.title("Evidence DAG Explorer")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Network Explorer", "Strategy Evidence Viewer", "Evidence Triangulation", "DAG Generation", "About", "Evaluation"])
    
    with tab1:
        render_network_explorer()
    
    with tab2:
        render_strategy_evidence_viewer()
    
    with tab3:
        render_evidence_triangulation()
    
    with tab4:
        render_dag_generation()
    
    with tab5:
        render_about()
    
    with tab6:
        render_questionnaire()


if __name__ == "__main__":
    main()

