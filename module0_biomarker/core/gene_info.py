"""
Gene clinical explanations and ODE parameter lookup.
Doctor-friendly language for each PD biomarker gene.
"""
from typing import Dict, List

# ── Clinical explanations (doctor-friendly) ───────────────────────────
GENE_EXPLANATIONS: Dict[str, Dict] = {
    "SNCA": {
        "full_name":    "Alpha-Synuclein",
        "role":         "Primary hub gene. Produces the alpha-synuclein protein "
                        "that accumulates in Lewy bodies — the pathological hallmark "
                        "of Parkinson's disease. Controls dopamine vesicle release.",
        "in_pd":        "Mutations (A53T, A30P, E46K) and gene duplication cause "
                        "toxic protein aggregation, damaging dopaminergic neurons.",
        "clinical_tip": "Elevated CSF alpha-synuclein levels are a key biomarker. "
                        "Target for immunotherapy trials.",
        "color":        "#4f8ef7",
    },
    "LRRK2": {
        "full_name":    "Leucine-Rich Repeat Kinase 2",
        "role":         "Multifunctional kinase regulating vesicle trafficking, "
                        "cytoskeletal dynamics, and inflammatory signalling.",
        "in_pd":        "G2019S mutation is the most common genetic cause of PD "
                        "worldwide (~1-2% of all PD cases). Causes increased kinase "
                        "activity leading to neuronal dysfunction.",
        "clinical_tip": "LRRK2 inhibitors are in active clinical trials. "
                        "Test for G2019S in all new PD patients.",
        "color":        "#f59e0b",
    },
    "PINK1": {
        "full_name":    "PTEN-Induced Kinase 1",
        "role":         "Mitochondrial kinase that monitors mitochondrial health. "
                        "First responder to mitochondrial damage — recruits Parkin.",
        "in_pd":        "Loss-of-function mutations cause autosomal recessive "
                        "early-onset PD through impaired mitophagy and accumulation "
                        "of damaged mitochondria.",
        "clinical_tip": "PINK1 deficiency leads to oxidative stress. "
                        "Consider antioxidant supplementation in PINK1 mutation carriers.",
        "color":        "#a78bfa",
    },
    "PRKN": {
        "full_name":    "Parkin RBR E3 Ubiquitin Ligase (PARK2)",
        "role":         "E3 ubiquitin ligase that tags damaged proteins and "
                        "mitochondria for degradation. Works downstream of PINK1.",
        "in_pd":        "Most common gene in autosomal recessive juvenile-onset PD. "
                        "Exon deletions are the most frequent mutation type. "
                        "Loss impairs proteasomal clearance of toxic proteins.",
        "clinical_tip": "PARKIN mutation carriers often have slow progression "
                        "and good levodopa response. Genetic testing recommended "
                        "in patients under 50.",
        "color":        "#10b981",
    },
    "PARKIN": {
        "full_name":    "Parkin RBR E3 Ubiquitin Ligase (PARK2)",
        "role":         "E3 ubiquitin ligase — tags damaged proteins for proteasomal degradation.",
        "in_pd":        "Loss-of-function causes recessive juvenile PD through impaired protein quality control.",
        "clinical_tip": "Good levodopa response. Test in early-onset cases.",
        "color":        "#10b981",
    },
    "PARK7": {
        "full_name":    "DJ-1 Protein (PARK7)",
        "role":         "Multifunctional protein acting as an oxidative stress sensor, "
                        "transcriptional co-activator, and neuroprotective chaperone.",
        "in_pd":        "Rare cause of autosomal recessive early-onset PD. "
                        "DJ-1 oxidation in response to cellular stress is an "
                        "early indicator of PD pathology.",
        "clinical_tip": "Plasma DJ-1 levels may serve as an early biomarker. "
                        "Oxidised DJ-1 found in CSF of sporadic PD patients.",
        "color":        "#ef4444",
    },
    "GBA": {
        "full_name":    "Glucocerebrosidase",
        "role":         "Lysosomal enzyme involved in glucocerebroside metabolism. "
                        "Critical for normal lysosomal function and autophagy.",
        "in_pd":        "GBA mutations are the most common genetic risk factor "
                        "for PD, increasing risk 5-fold. Found in ~5-15% of PD patients.",
        "clinical_tip": "GBA mutation carriers may have faster progression and "
                        "higher cognitive risk. Monitor cognition carefully.",
        "color":        "#06b6d4",
    },
    "TH": {
        "full_name":    "Tyrosine Hydroxylase",
        "role":         "Rate-limiting enzyme in dopamine synthesis. "
                        "Converts tyrosine to L-DOPA.",
        "in_pd":        "TH-positive neuron loss in substantia nigra is the "
                        "defining pathological feature of PD. "
                        "TH activity decreases by 80% in advanced PD.",
        "clinical_tip": "TH staining is used to quantify dopaminergic neuron "
                        "loss in post-mortem and animal model studies.",
        "color":        "#f97316",
    },
}

# ── ODE parameters per gene ───────────────────────────────────────────
ODE_PARAMETERS: Dict[str, List[Dict]] = {
    "SNCA": [
        {"name": "k_3p",   "value": 0.42,  "unit": "μM/h",   "source": "Abeliovich et al. 2000"},
        {"name": "k_fs",   "value": 0.18,  "unit": "h⁻¹",    "source": "Volpicelli-Daley et al. 2014"},
        {"name": "k_ieg",  "value": 0.035, "unit": "h⁻¹",    "source": "Lashuel et al. 2013"},
        {"name": "m",      "value": 2.0,   "unit": "—",       "source": "Cremades et al. 2012"},
        {"name": "k_iSa",  "value": 0.062, "unit": "h⁻¹",    "source": "Lashuel et al. 2013"},
        {"name": "k_Sas",  "value": 0.12,  "unit": "h⁻¹",    "source": "Parra-Rojas et al. 2013"},
        {"name": "K_act",  "value": 2.15,  "unit": "μM",      "source": "Bhatt et al. 2020"},
        {"name": "S_0",    "value": 0.85,  "unit": "μM",      "source": "Shi et al. 2014 (CSF)"},
    ],
    "LRRK2": [
        {"name": "k_L2tax","value": 0.38,  "unit": "h⁻¹",    "source": "Paisan-Ruiz et al. 2004"},
        {"name": "k_ToCG", "value": 0.21,  "unit": "h⁻¹",    "source": "Zimprich et al. 2004"},
        {"name": "k_ans",  "value": 0.13,  "unit": "h⁻¹",    "source": "Paisan-Ruiz et al. 2013"},
        {"name": "K_dg0",  "value": 2.80,  "unit": "μM",      "source": "Greggio et al. 2006"},
        {"name": "k_ing",  "value": 0.07,  "unit": "h⁻¹",    "source": "Anand et al. 2009"},
        {"name": "L2_0",   "value": 1.00,  "unit": "norm.",   "source": "Paisan-Ruiz et al. 2004"},
    ],
    "PINK1": [
        {"name": "k_P1pmd","value": 0.28,  "unit": "h⁻¹",    "source": "Narendra et al. 2008"},
        {"name": "k_p1ant","value": 0.19,  "unit": "h⁻¹",    "source": "Matsuda et al. 2010"},
        {"name": "K_sp",   "value": 1.85,  "unit": "nM",      "source": "Matsuda et al. 2010"},
        {"name": "k_pt1g", "value": 0.22,  "unit": "h⁻¹",    "source": "Narendra et al. 2010"},
        {"name": "P1_0",   "value": 1.00,  "unit": "norm.",   "source": "Narendra et al. 2008"},
    ],
    "PRKN": [
        {"name": "k_P2pmd","value": 0.31,  "unit": "h⁻¹",    "source": "Kitada et al. 1998"},
        {"name": "k_gn2",  "value": 0.17,  "unit": "h⁻¹",    "source": "Narendra et al. 2008"},
        {"name": "K_p1",   "value": 0.90,  "unit": "—",       "source": "Trempe et al. 2013"},
        {"name": "k_inh",  "value": 0.08,  "unit": "h⁻¹",    "source": "Spratt et al. 2013"},
        {"name": "P7_0",   "value": 1.00,  "unit": "norm.",   "source": "Kitada et al. 1998"},
    ],
    "PARKIN": [
        {"name": "k_P2pmd","value": 0.31,  "unit": "h⁻¹",    "source": "Kitada et al. 1998"},
        {"name": "k_inh",  "value": 0.08,  "unit": "h⁻¹",    "source": "Spratt et al. 2013"},
        {"name": "P7_0",   "value": 1.00,  "unit": "norm.",   "source": "Kitada et al. 1998"},
    ],
    "PARK7": [
        {"name": "k_D2pmd","value": 0.25,  "unit": "μM/h",   "source": "Taira et al. 2004"},
        {"name": "k_DJ2",  "value": 0.16,  "unit": "h⁻¹",    "source": "Canet-Aviles et al. 2004"},
        {"name": "k_αs",   "value": 0.09,  "unit": "h⁻¹",    "source": "Blackinton et al. 2009"},
        {"name": "K_αx",   "value": 1.45,  "unit": "μM",      "source": "Bonello et al. 2019"},
        {"name": "D7_0",   "value": 0.78,  "unit": "μM",      "source": "Taira et al. 2004"},
    ],
}


def get_explanation(gene: str) -> str:
    """Get doctor-friendly explanation for a gene."""
    info = GENE_EXPLANATIONS.get(gene, {})
    if not info:
        return f"{gene}: Known Parkinson's disease associated gene."
    return (
        f"{info['full_name']}: {info['role']} "
        f"In PD: {info['in_pd']} "
        f"Clinical note: {info.get('clinical_tip', '')}"
    )


def get_ode_params(genes: List[str]) -> List[Dict]:
    """Get ODE parameters for a list of genes."""
    result = []
    for gene in genes:
        params = ODE_PARAMETERS.get(gene, [])
        for p in params:
            result.append({
                "gene":  gene,
                "name":  p["name"],
                "value": p["value"],
                "unit":  p["unit"],
                "source":p["source"],
            })
    return result
