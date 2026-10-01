# Forced-hotspot homopolymer add-on experiment

## Question

Can an XGBoost add-on recover AOHC hotspot homopolymer indels rejected by the
Ion Torrent/Genexus Oncomine caller without materially degrading PPV?

## Design

- Four AOHC and four NA24385 BAM/VCF pairs.
- Every normalized Oncomine `HS` indel with reference or alternate HP length
  ≥4 was measured, including `0/0` and `NOCALL` records.
- AOHC labels used AOHC synthetic plus native GIAB truth. NA24385 labels used
  GIAB truth only. A negative label required complete containment in the GIAB
  benchmark BED; all other alleles were `UNKNOWN`.
- The model was an add-on: `final = original Oncomine OR ML rescue`.
- Five outer folds were grouped by normalized allele. Each outer fold held out
  one positive HP locus, and inner grouped folds selected the rescue threshold
  under a zero-incremental-FP training budget.
- Predefined success gate: rescue at least 8 of 18 original AOHC HP false
  negatives with at most 1 added AOHC false positive.

## Validated dataset

| Item | Result |
|---|---:|
| Forced HP hotspot observations | 616 |
| Labeled observations used for modeling | 592 |
| Positive observations | 20 |
| Confident-reference observations | 572 |
| UNKNOWN observations excluded | 24 |
| Unique forced HP hotspot alleles | 77 |
| Unique positive AOHC HP hotspot alleles | 5 |
| Rows with usable ZM flow measurements | 616/616 |
| AOHC all-hotspot truth observations | 64: 16 loci × 4 runs |
| Unit tests | 23 passed |

## Results

| System | TP | FP | FN | Sensitivity | PPV | Rescued original FNs | Added FPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original Oncomine | 37 | 1 | 27 | 57.81% | 97.37% | 0 | 0 |
| VCF + context add-on | 37 | 1 | 27 | 57.81% | 97.37% | 0 | 0 |
| VCF + BAM add-on | 37 | 1 | 27 | 57.81% | 97.37% | 0 | 0 |
| VCF + BAM + flow add-on | 39 | 2 | 25 | 60.94% | 95.12% | 2 | 1 |
| Flow + paired NA deltas | 38 | 2 | 26 | 59.38% | 95.00% | 1 | 1 |

### HP and non-HP strata

| System | Stratum | TP | FP | FN | Sensitivity | PPV |
|---|---|---:|---:|---:|---:|---:|
| Original Oncomine | HP | 2 | 0 | 18 | 10.00% | 100.00% |
| VCF + BAM + flow add-on | HP | 4 | 1 | 16 | 20.00% | 80.00% |
| Original Oncomine | non-HP | 35 | 1 | 9 | 79.55% | 97.22% |
| VCF + BAM + flow add-on | non-HP | 35 | 1 | 9 | 79.55% | 97.22% |

The two rescued observations were the same PTEN A6-to-A5 deletion
(`10:89717769 TA>T`) in AOHC2 and AOHC4. The incremental false positive was an
A8-to-A7 deletion (`14:45645954 CA>C`) in AOHC4. Allowing one inner-training FP
instead of zero produced the same held-out result.

## Interpretation

Forced hotspot measurement solved the principal candidate-ascertainment issue
in the earlier blind discovery experiment: every target HP truth allele entered
evaluation even when CIGAR support was absent or TVC emitted only a reference
or NOCALL record. Flow features contained some discriminating signal, but the
gain was two run observations from one locus and cost one false positive. The
predefined success gate was not met.

This is evidence for a small proof-of-concept effect, not a generalizable
variant caller. More independent positive HP loci are needed before additional
model tuning is scientifically meaningful.
