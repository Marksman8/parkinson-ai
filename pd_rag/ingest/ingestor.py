"""
Document Ingestor
Handles ingestion of all 4 knowledge sources into ChromaDB:
  1. PD research papers (your PDFs)
  2. Clinical guidelines (text/PDF)
  3. Gene databases (ClinVar/PharmGKB CSV/JSON)
  4. Module 0 biomarker results (auto-synced from SQLite)
"""
import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# ── Built-in clinical guidelines text ────────────────────────────────
# Embedded so system works even before you add your own documents
BUILTIN_GUIDELINES = [
    {
        "id":   "mds_diagnosis_2015",
        "text": "MDS Clinical Diagnostic Criteria for Parkinson's Disease (2015): "
                "Diagnosis requires bradykinesia plus at least one of resting tremor "
                "or rigidity. Supportive criteria include clear beneficial response "
                "to dopaminergic therapy, presence of levodopa-induced dyskinesia, "
                "rest tremor of a limb, and anosmia.",
        "source": "MDS Diagnostic Criteria 2015",
    },
    {
        "id":   "levodopa_protocol",
        "text": "Levodopa-carbidopa is the most effective symptomatic treatment for "
                "Parkinson's disease. Initial dosing typically starts at 100/25mg "
                "three times daily with food. Dose titration based on response and "
                "tolerability. Long-term use associated with motor complications "
                "including wearing-off and dyskinesia in majority of patients "
                "after 5 years.",
        "source": "Clinical Treatment Guidelines",
    },
    {
        "id":   "genetic_testing_guideline",
        "text": "Genetic testing recommendations for Parkinson's disease: LRRK2 G2019S "
                "testing recommended in Ashkenazi Jewish and North African Berber "
                "populations. PARKIN and PINK1 testing in early-onset PD (under 50). "
                "GBA testing increasingly recommended as it is the most common "
                "genetic risk factor. Genetic counselling required before and after testing.",
        "source": "Genetic Testing Guidelines PD",
    },
    {
        "id":   "hoehn_yahr_staging",
        "text": "Hoehn and Yahr Staging for Parkinson's Disease: "
                "Stage 1: Unilateral involvement only. "
                "Stage 2: Bilateral involvement without postural instability. "
                "Stage 3: Mild to moderate bilateral disease with postural instability. "
                "Stage 4: Severe disability but still able to walk or stand unassisted. "
                "Stage 5: Wheelchair bound or bedridden unless aided.",
        "source": "Hoehn Yahr Scale",
    },
    {
        "id":   "dbs_criteria",
        "text": "Deep Brain Stimulation (DBS) criteria for Parkinson's disease: "
                "Confirmed PD diagnosis, adequate response to levodopa, disabling "
                "motor fluctuations or dyskinesias despite optimal medical therapy, "
                "no significant cognitive impairment, no major psychiatric comorbidity. "
                "STN and GPi are the main targets. Typical improvement 40-60% in "
                "motor UPDRS scores.",
        "source": "DBS Clinical Guidelines",
    },
    {
        "id":   "pd_biomarker_consensus",
        "text": "Consensus biomarkers for Parkinson's disease research: "
                "Alpha-synuclein seed amplification assay (SAA) in CSF is highly "
                "sensitive and specific for PD. Neuroimaging biomarkers include "
                "DAT-SPECT, neuromelanin MRI, and cardiac MIBG scintigraphy. "
                "Blood-based biomarkers under investigation include plasma "
                "alpha-synuclein, neurofilament light chain, and GFAP.",
        "source": "PD Biomarker Consensus Statement",
    },
    {
        "id":   "pink1_parkin_pathway",
        "text": "PINK1-Parkin mitophagy pathway in Parkinson's disease: "
                "Upon mitochondrial depolarisation, PINK1 accumulates on the outer "
                "mitochondrial membrane and phosphorylates ubiquitin at Ser65. "
                "This recruits and activates Parkin, which ubiquitinates mitochondrial "
                "surface proteins, signalling autophagosome engulfment. Loss of either "
                "PINK1 or Parkin function results in accumulation of dysfunctional "
                "mitochondria and subsequent neuronal death.",
        "source": "PINK1-Parkin Pathway Review",
    },
    {
        "id":   "lrrk2_kinase_activity",
        "text": "LRRK2 kinase activity in Parkinson's disease: "
                "LRRK2 G2019S mutation increases kinase activity approximately 2-fold. "
                "Hyperactive LRRK2 phosphorylates multiple Rab GTPases, disrupting "
                "vesicle trafficking and lysosomal function. LRRK2 kinase inhibitors "
                "currently in Phase II clinical trials show promising safety profiles. "
                "Urinary Rab10 phosphorylation is a pharmacodynamic biomarker for "
                "LRRK2 inhibition.",
        "source": "LRRK2 Biology Review",
    },
]

# ── Built-in gene database ────────────────────────────────────────────
BUILTIN_GENE_DB = [
    {
        "id":   "snca_variants",
        "text": "SNCA pathogenic variants in Parkinson's disease: "
                "A53T (rs104893877) — autosomal dominant, high penetrance, "
                "early-onset PD, rapid progression. "
                "A30P (rs104893875) — autosomal dominant, variable penetrance. "
                "E46K (rs104893878) — severe phenotype, dementia common. "
                "Gene duplication — PD with typical age of onset. "
                "Gene triplication — young onset, rapid progression, dementia. "
                "All variants lead to alpha-synuclein aggregation and Lewy body formation.",
        "source": "ClinVar SNCA variants",
    },
    {
        "id":   "lrrk2_variants",
        "text": "LRRK2 pathogenic variants in Parkinson's disease: "
                "G2019S (rs34637584) — most common, 1-2% sporadic PD, "
                "4% familial PD, 20% in Ashkenazi Jewish, 40% in North African Berber. "
                "R1441C/G/H — kinase domain mutations, high penetrance. "
                "Y1699C — COR domain, reduced GTPase activity. "
                "I2020T — kinase domain, Japanese families. "
                "All increase LRRK2 kinase activity leading to neurodegeneration.",
        "source": "ClinVar LRRK2 variants",
    },
    {
        "id":   "gba_variants",
        "text": "GBA variants as Parkinson's disease risk factors: "
                "N370S — most common Gaucher variant, 5-fold PD risk increase. "
                "L444P — severe Gaucher phenotype, high PD risk. "
                "E326K — mild risk variant, prevalent in general population. "
                "GBA mutations found in 5-15% of PD patients depending on population. "
                "Mechanism involves lysosomal dysfunction and impaired "
                "alpha-synuclein degradation. "
                "GBA mutation carriers have faster cognitive decline in PD.",
        "source": "PharmGKB GBA variants",
    },
    {
        "id":   "parkin_variants",
        "text": "PRKN (Parkin) variants in juvenile Parkinson's disease: "
                "Exon deletions most common mutation type (especially exons 3-4). "
                "Point mutations include R42P, K161N, T240M, C289G. "
                "Autosomal recessive inheritance — biallelic mutations required. "
                "Onset typically before age 40. Slow progression. "
                "Good and sustained levodopa response. "
                "Dyskinesias develop early at low levodopa doses.",
        "source": "ClinVar PRKN variants",
    },
    {
        "id":   "pink1_variants",
        "text": "PINK1 variants in early-onset Parkinson's disease: "
                "Q456X — loss of function, common in European families. "
                "V170G — kinase domain impairment. "
                "Autosomal recessive, onset 30-40 years. "
                "Clinical features similar to PARKIN mutations: "
                "slow progression, good levodopa response, early dyskinesia, "
                "hyperreflexia, psychiatric features. "
                "Mitochondrial dysfunction is primary mechanism.",
        "source": "ClinVar PINK1 variants",
    },
]


class DocumentIngestor:
    """
    Ingests all knowledge sources into ChromaDB collections.
    Run once to build the knowledge base.
    """

    def __init__(
        self,
        vector_db_path:  str = "data/vector_db",
        papers_dir:      str = "data/papers",
        guidelines_dir:  str = "data/guidelines",
        gene_db_dir:     str = "data/gene_db",
        module0_db_path: str = "../module0_biomarker/data/pd_memory.db",
    ):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.papers_dir      = Path(papers_dir)
        self.guidelines_dir  = Path(guidelines_dir)
        self.gene_db_dir     = Path(gene_db_dir)
        self.module0_db_path = module0_db_path

        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.client = chromadb.PersistentClient(path=vector_db_path)

        # Four separate collections
        self.collections = {
            "pd_papers":     self.client.get_or_create_collection(
                "pd_papers",     metadata={"hnsw:space": "cosine"}
            ),
            "pd_guidelines": self.client.get_or_create_collection(
                "pd_guidelines", metadata={"hnsw:space": "cosine"}
            ),
            "pd_gene_db":    self.client.get_or_create_collection(
                "pd_gene_db",    metadata={"hnsw:space": "cosine"}
            ),
            "pd_module0":    self.client.get_or_create_collection(
                "pd_module0",    metadata={"hnsw:space": "cosine"}
            ),
        }

    # ── INGEST ALL ────────────────────────────────────────────────────
    def ingest_all(self) -> Dict[str, int]:
        """Run full ingestion pipeline. Returns count per collection."""
        counts = {}
        counts["pd_papers"]     = self.ingest_papers()
        counts["pd_guidelines"] = self.ingest_guidelines()
        counts["pd_gene_db"]    = self.ingest_gene_database()
        counts["pd_module0"]    = self.sync_module0()
        logger.info(f"Ingestion complete: {counts}")
        return counts

    # ── 1. PD RESEARCH PAPERS ────────────────────────────────────────
    def ingest_papers(self) -> int:
        """Chunk and embed all PDFs from papers directory."""
        col   = self.collections["pd_papers"]
        count = 0
        self.papers_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = list(self.papers_dir.glob("*.pdf"))
        txt_files = list(self.papers_dir.glob("*.txt"))

        if not pdf_files and not txt_files:
            logger.info("No papers found — add PDFs to data/papers/")
            return 0

        for pdf_path in pdf_files:
            try:
                from pdfminer.high_level import extract_text
                text = extract_text(str(pdf_path))
                chunks = self._chunk(text, source=pdf_path.name)
                self._upsert(col, chunks, prefix="paper")
                count += len(chunks)
                logger.info(f"Ingested {pdf_path.name}: {len(chunks)} chunks")
            except Exception as e:
                logger.warning(f"Failed to ingest {pdf_path.name}: {e}")

        for txt_path in txt_files:
            try:
                text   = txt_path.read_text(encoding="utf-8", errors="ignore")
                chunks = self._chunk(text, source=txt_path.name)
                self._upsert(col, chunks, prefix="paper")
                count += len(chunks)
            except Exception as e:
                logger.warning(f"Failed to ingest {txt_path.name}: {e}")

        return count

    # ── 2. CLINICAL GUIDELINES ───────────────────────────────────────
    def ingest_guidelines(self) -> int:
        """Ingest built-in guidelines + any custom guideline files."""
        col   = self.collections["pd_guidelines"]
        count = 0

        # Built-in guidelines
        for g in BUILTIN_GUIDELINES:
            embedding = self.embedder.encode(g["text"]).tolist()
            col.upsert(
                ids=[g["id"]],
                embeddings=[embedding],
                documents=[g["text"]],
                metadatas=[{"source": g["source"], "type": "guideline"}],
            )
            count += 1

        # Custom guideline files
        self.guidelines_dir.mkdir(parents=True, exist_ok=True)
        for f in list(self.guidelines_dir.glob("*.pdf")) + \
                 list(self.guidelines_dir.glob("*.txt")):
            try:
                text = (
                    self._extract_pdf(str(f))
                    if f.suffix == ".pdf"
                    else f.read_text(encoding="utf-8", errors="ignore")
                )
                chunks = self._chunk(text, source=f.name)
                self._upsert(col, chunks, prefix="guideline")
                count += len(chunks)
            except Exception as e:
                logger.warning(f"Guideline ingestion failed {f.name}: {e}")

        logger.info(f"Guidelines ingested: {count} entries")
        return count

    # ── 3. GENE DATABASE ─────────────────────────────────────────────
    def ingest_gene_database(self) -> int:
        """Ingest built-in gene DB + ClinVar/PharmGKB files."""
        col   = self.collections["pd_gene_db"]
        count = 0

        # Built-in gene DB
        for g in BUILTIN_GENE_DB:
            embedding = self.embedder.encode(g["text"]).tolist()
            col.upsert(
                ids=[g["id"]],
                embeddings=[embedding],
                documents=[g["text"]],
                metadatas=[{"source": g["source"], "type": "gene_variant"}],
            )
            count += 1

        # Custom gene files (ClinVar TSV, PharmGKB TSV, JSON)
        self.gene_db_dir.mkdir(parents=True, exist_ok=True)
        for f in self.gene_db_dir.glob("*.tsv"):
            try:
                count += self._ingest_clinvar_tsv(str(f), col)
            except Exception as e:
                logger.warning(f"ClinVar TSV failed {f.name}: {e}")

        for f in self.gene_db_dir.glob("*.json"):
            try:
                count += self._ingest_pharmgkb_json(str(f), col)
            except Exception as e:
                logger.warning(f"PharmGKB JSON failed {f.name}: {e}")

        logger.info(f"Gene database ingested: {count} entries")
        return count

    def _ingest_clinvar_tsv(self, path: str, col) -> int:
        """Parse ClinVar TSV export and ingest PD-related variants."""
        import csv
        count = 0
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for i, row in enumerate(reader):
                gene = row.get("GeneSymbol", "")
                cond = row.get("PhenotypeList", "")
                if "Parkinson" not in cond and "parkinson" not in cond:
                    continue
                text = (
                    f"Gene: {gene}. "
                    f"Variant: {row.get('Name', '')}. "
                    f"Clinical significance: {row.get('ClinicalSignificance', '')}. "
                    f"Condition: {cond}. "
                    f"Review status: {row.get('ReviewStatus', '')}."
                )
                emb = self.embedder.encode(text).tolist()
                col.upsert(
                    ids=[f"clinvar_{i}"],
                    embeddings=[emb],
                    documents=[text],
                    metadatas=[{"source": "ClinVar", "gene": gene, "type": "variant"}],
                )
                count += 1
        return count

    def _ingest_pharmgkb_json(self, path: str, col) -> int:
        """Parse PharmGKB JSON export."""
        count = 0
        data  = json.loads(Path(path).read_text())
        items = data if isinstance(data, list) else data.get("data", [])
        for i, item in enumerate(items[:200]):
            gene = item.get("gene", item.get("symbol", ""))
            drug = item.get("drug", item.get("chemical", ""))
            ann  = item.get("annotation", item.get("description", ""))
            if not ann:
                continue
            text = f"Gene: {gene}. Drug: {drug}. Annotation: {ann}"
            emb  = self.embedder.encode(text).tolist()
            col.upsert(
                ids=[f"pharmgkb_{i}"],
                embeddings=[emb],
                documents=[text],
                metadatas=[{"source": "PharmGKB", "gene": gene, "type": "drug_gene"}],
            )
            count += 1
        return count

    # ── 4. MODULE 0 SYNC ─────────────────────────────────────────────
    def sync_module0(self) -> int:
        """Sync latest Module 0 biomarker results into ChromaDB."""
        col   = self.collections["pd_module0"]
        count = 0

        if not Path(self.module0_db_path).exists():
            logger.info("Module 0 DB not found — skipping sync")
            return 0

        try:
            conn = sqlite3.connect(self.module0_db_path)
            conn.row_factory = sqlite3.Row

            sessions = conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC LIMIT 20"
            ).fetchall()

            for s in sessions:
                sid   = s["session_id"]
                genes = conn.execute(
                    "SELECT * FROM top_genes WHERE session_id = ? ORDER BY rank",
                    (sid,),
                ).fetchall()

                if not genes:
                    continue

                gene_text = ", ".join(
                    f"{g['gene']} (C_D={g['degree_c']:.3f})"
                    for g in genes
                )
                text = (
                    f"Biomarker discovery session {sid[:20]} "
                    f"({s['created_at'][:10]}). "
                    f"Clinical context: {s['doctor_input'][:200]}. "
                    f"Top hub genes identified: {gene_text}."
                )
                emb = self.embedder.encode(text).tolist()
                col.upsert(
                    ids=[f"m0_{sid[:20]}"],
                    embeddings=[emb],
                    documents=[text],
                    metadatas=[{
                        "source":     "Module0",
                        "session_id": sid,
                        "type":       "biomarker_session",
                    }],
                )
                count += 1

            conn.close()
            logger.info(f"Module 0 sync: {count} sessions ingested")
        except Exception as e:
            logger.warning(f"Module 0 sync failed: {e}")

        return count

    # ── HELPERS ───────────────────────────────────────────────────────
    def _chunk(
        self,
        text:       str,
        source:     str,
        chunk_size: int = 400,
        overlap:    int = 50,
    ) -> List[Dict]:
        text   = re.sub(r"\s+", " ", text).strip()
        words  = text.split()
        chunks = []
        step   = chunk_size - overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i: i + chunk_size])
            if len(chunk) > 60:
                chunks.append({"text": chunk, "source": source})
        return chunks

    def _upsert(self, col, chunks: List[Dict], prefix: str = "doc"):
        existing = col.count()
        for i, chunk in enumerate(chunks):
            emb = self.embedder.encode(chunk["text"]).tolist()
            col.upsert(
                ids=[f"{prefix}_{existing + i}"],
                embeddings=[emb],
                documents=[chunk["text"]],
                metadatas=[{"source": chunk["source"], "type": prefix}],
            )

    def _extract_pdf(self, path: str) -> str:
        from pdfminer.high_level import extract_text
        return extract_text(path)

    def collection_stats(self) -> Dict[str, int]:
        return {name: col.count() for name, col in self.collections.items()}
