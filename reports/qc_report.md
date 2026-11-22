# Audio Labelling Quality Control (QC) Report
**Total Segments:** 62  
**Total Audio Duration:** 9.79 minutes  

## 1. Class Distribution & Balance
| Class | Segment Count | Duration (min) | Percentage (%) |
| :--- | :--- | :--- | :--- |
| `mixed` | 1 | 0.09 min | 0.9% |
| `silence` | 4 | 0.48 min | 5.0% |
| `speech` | 57 | 9.22 min | 94.1% |

## 2. Review Urgency & Certainty Breakdown
- **LOW:** 44 segments (71.0%)
- **MEDIUM:** 12 segments (19.4%)
- **URGENT:** 6 segments (9.7%)

## 3. Tiered Cut Engine Metrics
| Cut Type | Count | Percentage (%) |
| :--- | :--- | :--- |
| `safe_pause` | 47 | 75.8% |
| `energy_valley_cut` | 4 | 6.5% |
| `remainder_valid` | 4 | 6.5% |
| `short_pause_safe` | 4 | 6.5% |
| `extended_window_safe` | 3 | 4.8% |

**Acoustic Margins:**
- Mean Trailing Margin: `164.0 ms`
- Mean Leading Margin: `164.0 ms`
- Segments with standard 100-200ms margins: `85.5%`

## 5. Integrity Checks
- Monotonic timestamps: PASS
- Zero overlap between cut segments: PASS
- All speech cuts occur at validated pauses or energy valleys: PASS