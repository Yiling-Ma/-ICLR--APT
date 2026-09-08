PYTHON ?= python3
SCALING_DIR ?= outputs/patient_cell_scaling_fair
PDF_TMP ?= tmp/pdfs

.PHONY: composition-audit fair-scaling-run fair-scaling-aggregate fair-scaling-summarize paper revision

composition-audit:
	$(PYTHON) analysis/generate_composition_protocol_audit.py

fair-scaling-run:
	$(PYTHON) analysis/patient_cell_scaling.py --output-dir $(SCALING_DIR) plan
	$(PYTHON) analysis/patient_cell_scaling.py --output-dir $(SCALING_DIR) run --models logistic_regression,xgboost

fair-scaling-aggregate:
	$(PYTHON) analysis/patient_cell_scaling.py --output-dir $(SCALING_DIR) aggregate
	$(MAKE) fair-scaling-summarize PYTHON=$(PYTHON) SCALING_DIR=$(SCALING_DIR)

fair-scaling-summarize:
	$(PYTHON) analysis/summarize_fair_patient_cell_scaling.py --output-dir $(SCALING_DIR)
	cp $(SCALING_DIR)/fixed_total_scaling_table.tex tables/fixed_total_scaling.tex
	cp $(SCALING_DIR)/fixed_total_scaling.pdf figures/fixed_total_scaling.pdf

paper:
	mkdir -p $(PDF_TMP) output/pdf
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=$(PDF_TMP) iclr2027_conference.tex
	cp $(PDF_TMP)/iclr2027_conference.pdf output/pdf/APT-Bench_ICLR2027_revised.pdf

revision: composition-audit fair-scaling-summarize paper
