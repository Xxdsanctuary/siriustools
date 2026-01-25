# Code Review & Self-Assessment

**Team Sirius | Cargill-SMU Datathon 2026**

This document provides a self-assessment of our codebase, identifying strengths, areas for improvement, and lessons learned.

---

## Code Quality Assessment

### Strengths ✅

| Area | Assessment |
|------|------------|
| **Modularity** | Code is organized into separate modules (`data_loader.py`, `optimization.py`, `app.py`) |
| **Single Source of Truth** | All calculations flow from `optimization.py` - no duplicate formulas |
| **Reproducibility** | Notebook runs end-to-end without manual intervention |
| **Documentation** | Functions have docstrings explaining purpose and parameters |
| **Error Handling** | Graceful fallbacks when data is missing or calculations fail |

### Areas for Improvement 🔧

| Area | Current State | Ideal State | Priority |
|------|---------------|-------------|----------|
| **Type Hints** | Partial | Full type annotations | Medium |
| **Unit Tests** | None | pytest coverage >80% | High |
| **Data Classes** | Using dicts | Use `@dataclass` for Vessel/Cargo | Medium |
| **Config Management** | Hardcoded values | External config file | Low |
| **Logging** | Print statements | Python `logging` module | Low |

---

## Architecture Review

### Current Architecture

```
siriustools/
├── src/
│   ├── data_loader.py    # Data ingestion & preprocessing
│   ├── optimization.py   # Core calculation engine
│   └── __init__.py
├── chatbot/
│   └── app.py            # Streamlit UI (imports from src/)
├── notebooks/
│   └── Sirius_Tools_Datathon_Submission.ipynb
└── README.md
```

### Design Decisions

1. **Why separate `src/` from `chatbot/`?**
   - Allows notebook and chatbot to share the same calculation logic
   - Prevents code duplication and inconsistencies
   - Makes testing easier (can test `src/` independently)

2. **Why hardcode vessel/cargo data?**
   - Competition data is fixed and small (4 vessels, 3 cargoes)
   - Avoids file path issues across different environments
   - Faster iteration during development

3. **Why greedy optimization instead of LP/MIP?**
   - Problem size is small (4×3 = 12 combinations)
   - Greedy with laycan constraints gives optimal solution
   - Easier to explain to business stakeholders

---

## Calculation Verification

### Excel Formula Mapping

| Excel Cell | Python Implementation | Verified? |
|------------|----------------------|-----------|
| Load Time = Qty/30000 + 0.5 | `load_time = quantity / load_rate + turn_time` | ✅ |
| Ballast Days = Dist/Speed/24 | `ballast_days = distance / speed / 24` | ✅ |
| Fuel Cost = Fuel × Price | `fuel_cost = fuel_mt * vlsfo_price` | ✅ |
| TCE = Profit / Days | `tce = voyage_profit / total_days` | ✅ |

### Sanity Checks Performed

1. **TCE Range**: All TCE values between -$10,000 and +$50,000/day ✅
2. **Voyage Duration**: All voyages between 20-80 days ✅
3. **Fuel Consumption**: 30-50 MT/day at sea is realistic ✅
4. **Profit Margins**: 15-40% of gross freight is typical ✅

---

## Known Limitations

1. **No weather routing** - Assumes fixed distances from port database
2. **No bunker optimization** - Uses single price, doesn't consider bunkering locations
3. **Static laycan check** - Doesn't account for delays or early arrival
4. **No market cargo optimization** - Only optimizes Cargill cargoes

---

## Previous Review Notes (Jan 24)

### Data Values Corrected ✅

Vessel speeds now use eco speeds as per PPTX:

| Vessel | Ballast Speed | Laden Speed |
|--------|---------------|-------------|
| Ann Bell | 12.5 kn | 12.0 kn |
| Ocean Horizon | 12.8 kn | 12.3 kn |
| Pacific Glory | 12.7 kn | 12.2 kn |
| Golden Ascent | 12.3 kn | 11.8 kn |

### Laycan Validation Added ✅

Optimization now checks if vessel can arrive before laycan window closes.

### Port Name Matching Fixed ✅

`data_loader.py` handles port aliases (e.g., "Kamsar" → "KAMSAR ANCHORAGE").

---

## Future Improvements (Post-Datathon)

1. **Add Monte Carlo simulation** for risk analysis
2. **Implement proper LP solver** (PuLP or OR-Tools) for larger fleets
3. **Add API integration** for live bunker prices
4. **Build vessel tracking** with AIS data
5. **Create mobile-responsive dashboard**

---

## Lessons Learned

1. **Start with the Excel** - Understanding the spreadsheet logic first saved hours of debugging
2. **Test with toy examples** - Small numbers (100 MT cargo, 100 NM distance) catch formula errors fast
3. **Single source of truth** - Having one `optimization.py` prevented chatbot/notebook inconsistencies
4. **Laycan constraints matter** - Our initial model ignored them and gave infeasible solutions

---

*Last updated: January 25, 2026*
*Team Sirius*
