# Task Briefing Package

This package contains all necessary information and strategic guidance for the Coder Agent.

---

## 1. Current Task Details

This is the full specification of the task you must complete.

```json
{
  "task_id": "I2.T1",
  "iteration_id": "I2",
  "iteration_goal": "Enhance test coverage for all 6 bar generator types from 27-50% to 70%+",
  "description": "Enhance test_dollar_bars.py to achieve 70%+ coverage",
  "agent_type_hint": "BackendAgent",
  "inputs": "bars/dollar_bars.py source, existing tests",
  "target_files": [
    "tests/test_bars/test_dollar_bars.py"
  ],
  "input_files": [
    "simple_futures_backtester/bars/dollar_bars.py",
    "tests/test_bars/test_dollar_bars.py"
  ],
  "deliverables": "- Tests for dollar value accumulation\n- Tests for volume tracking per bar\n- Edge cases (zero volume, high prices)\n- Performance benchmark test\n- bar_indices mapping verification",
  "acceptance_criteria": "- Coverage for dollar_bars.py reaches 70%+\n- All accumulation scenarios tested\n- Performance target met (1M+ rows/sec)\n- Edge cases handled gracefully",
  "dependencies": [],
  "parallelizable": true,
  "done": false
}
```

---

## 2. Architectural & Planning Context

The following are the relevant sections from the architecture and plan documents.

### Context: Bar Generation Architecture Principles

**Performance Target:** All bar generators must achieve 1M+ rows/sec throughput via Numba JIT compilation.

**Bar Generation Pattern:**
1. **JIT-compiled core function** (`_generate_*_bars_nb`) - Low-level algorithm with `@_jit` decorator
2. **High-level wrapper** (`generate_*_bars_series`) - Validation, volume aggregation, BarSeries creation
3. **Factory registration** - Auto-register on module import with `register_bar_type()`

**Technology Stack Requirements:**
- NumPy for array operations (float64 for prices, int64 for indices/volume)
- Numba JIT compilation for performance-critical loops
- C-contiguous array layouts for optimal performance
- Deterministic, reproducible computation

### Context: Iteration 2 Testing Strategy

**Goal:** Enhance test coverage for all 6 bar generator types from 27-50% to 70%+

**Testing Requirements for Dollar Bars (I2.T1):**
- Tests for dollar threshold generation
- Tests for dollar volume calculation (price * volume)
- Edge cases (zero price, zero volume)
- Performance benchmark
- Coverage for dollar_bars.py reaches 70%+
- Dollar volume calculation verified
- Edge cases prevent invalid bars
- Performance benchmark passes (1M+ rows/sec)

**Parallelizable:** Yes (can run independently of other bar type tests)

### Context: Dollar Bars Design (from bars/dollar_bars.py)

**Purpose:** Dollar bars aggregate source bars until cumulative dollar volume reaches a threshold. Dollar volume is calculated as `close[i] * volume[i]` for each source bar.

**Key Algorithm (Lines 37-133):**
```
_generate_dollar_bars_nb():
  1. Initialize state with first bar's OHLC values
  2. Calculate initial dollar volume: close[0] * volume[0]
  3. For each subsequent bar:
     - Update high/low if exceeded
     - Accumulate dollar volume: close[i] * volume[i]
     - If cumulative_dollars >= dollar_threshold:
       * Close bar with current OHLC values
       * Reset for next bar starting from close[i]
       * Reset cumulative_dollars to 0.0
  4. Trim arrays to actual bar count
```

**High-Level Wrapper (Lines 136-242):**
```
generate_dollar_bars_series():
  1. Validate OHLCV arrays (validate_ohlcv_arrays)
  2. Handle edge cases:
     - n < 2: Return empty BarSeries
     - dollar_threshold <= 0: Raise ValueError
  3. Call JIT-compiled core algorithm
  4. Aggregate raw volume per bar (sum volume_arr[start:end])
  5. Return BarSeries with type="dollar"
```

**Edge Cases:**
- Empty input (n=0): Returns empty BarSeries
- Single row (n=1): Returns empty BarSeries (need at least 2)
- dollar_threshold <= 0: Raises ValueError
- Zero price: Valid (creates zero dollar volume, may never trigger bar)
- Zero volume: Valid (contributes zero to dollar volume)
- High prices: Should create bars faster at same volume
- Low prices: Should create bars slower at same volume

**Performance:** Targets 1M+ rows/sec via Numba JIT compilation.

---

## 3. Codebase Analysis & Strategic Guidance

The following analysis is based on my direct review of the current codebase. Use these notes and tips to guide your implementation.

### Relevant Existing Code

*   **File:** `simple_futures_backtester/bars/dollar_bars.py`
    *   **Summary:** Dollar bar generator with JIT-compiled core algorithm and high-level wrapper.
    *   **Current State:**
        - Lines 37-133: `_generate_dollar_bars_nb()` - JIT function (**UNCOVERED** by tests)
        - Lines 136-242: `generate_dollar_bars_series()` - High-level wrapper (PARTIALLY COVERED)
        - Lines 245-246: Auto-registration with bar factory
    *   **Key Algorithm Details:**
        - Dollar volume calculation: `close[i] * np.float64(volume[i])` (lines 93, 103)
        - Threshold check: `if cumulative_dollars >= dollar_threshold` (line 107)
        - Bar closing: Lines 109-115 (store OHLC, indices, increment bar_count)
        - State reset: Lines 117-123 (start from close[i], reset cumulative_dollars)
        - Volume aggregation: Lines 228-231 (sum raw volume for each bar)
    *   **Recommendation:** You MUST create test scenarios that trigger the JIT algorithm's internal loops and branches. The core algorithm (lines 74-126) contains the critical logic that needs coverage.

*   **File:** `tests/test_bars/test_dollar_bars.py`
    *   **Summary:** Existing test suite with 20 tests (19 passing, 1 skipped).
    *   **Current Coverage: 44%** (NOT meeting 70% target - need +26% coverage)
    *   **Existing Test Classes:**
        - `TestDollarRegistration`: 2 tests (factory registration) ✅
        - `TestDollarBasicFunctionality`: 5 tests (simple bars, high/low volume, volume aggregation, index map) ✅
        - `TestDollarAlgorithmCorrectness`: 4 tests (dollar volume calc, bar reset, OHLC tracking, price sensitivity) ✅
        - `TestDollarEdgeCases`: 5 tests (empty, single row, invalid threshold, inconsistent lengths) ✅
        - `TestDollarOutputValidation`: 3 tests (dtypes, contiguity, lengths) ✅
        - `TestDollarPerformance`: 1 test (**SKIPPED** - needs pytest-benchmark fixture)
    *   **Current Test Quality:** Tests are well-structured and comprehensive for the HIGH-LEVEL API, but don't exercise all code paths in the JIT function.
    *   **Recommendation:** Add tests that specifically target the uncovered JIT algorithm paths. Focus on:
        - Multiple bar generation scenarios (trigger loop iterations)
        - Dollar volume accumulation across multiple bars
        - State reset verification after bar closes
        - Edge cases (zero volume contribution, high/low price extremes)

*   **File:** `tests/test_bars/test_range_bars.py` (REFERENCE - Similar structure, 70%+ coverage achieved)
    *   **Summary:** Comprehensive test coverage for Range bars (completed in I2.T2).
    *   **Patterns to Follow:**
        - Multiple test classes organized by concern
        - Algorithm correctness tests with detailed OHLC verification
        - Edge case tests for boundary conditions
        - Performance benchmark with 1M row fixture using manual timing
        - Volume aggregation verification across multiple bars
        - Specific test classes: `TestRangeVolumeAggregation`, `TestRangeBarReset`, `TestRangeBoundaryConditions`
    *   **Recommendation:** Study the Range bar test patterns (especially volume aggregation tests and boundary condition tests) and apply similar comprehensive coverage to Dollar bars.

*   **File:** `simple_futures_backtester/bars/__init__.py`
    *   **Summary:** Bar factory registration system with `register_bar_type()` and `get_bar_generator()`.
    *   **Recommendation:** You do NOT need to modify the factory. Dollar bars are already registered on module import (line 246 in dollar_bars.py).

*   **File:** `simple_futures_backtester/utils/jit_utils.py`
    *   **Summary:** JIT compilation utilities providing `get_njit_decorator()` and `validate_ohlcv_arrays()`.
    *   **Key Functions:**
        - `get_njit_decorator(cache=True, parallel=False)`: Returns configured `@njit` decorator
        - `validate_ohlcv_arrays()`: Validates input arrays for correct dtypes, lengths, contiguity
    *   **Recommendation:** You do NOT need to modify or test these utilities. They are already tested and used by all bar generators.

### Implementation Gaps Analysis

**Why Current Coverage is Only 44%:**

1. **JIT Function Uncovered:** The JIT function `_generate_dollar_bars_nb()` contains 52 lines (74-126) of core algorithm logic that is NOT being fully exercised by current tests.

2. **Limited Test Scenarios:** Current tests mostly verify the high-level API works, but don't create scenarios that exercise all branches:
   - Tests don't verify behavior when cumulative_dollars EXACTLY equals threshold
   - Tests don't verify state reset logic after bar closes (verification of continuity between bars)
   - Tests don't verify high/low aggregation across multiple iterations with varying patterns
   - Tests don't verify array trimming logic

3. **Skipped Performance Test:** The performance benchmark is skipped, so the JIT warmup path is never executed.

**To Reach 70%+ Coverage (Need +26%), You Must:**

1. **Add tests that exercise the JIT function's internal logic:**
   - Create scenarios with precise dollar volume calculations to verify threshold triggering
   - Test multiple bar generation (e.g., 5-10 bars) to exercise loop iterations
   - Verify OHLC aggregation logic with designed patterns
   - Test state reset after each bar closes

2. **Add performance benchmark test:**
   - Replace the skipped `test_1m_rows_performance` with working benchmark
   - Use manual `time.perf_counter()` timing (similar to range_bars.py approach)
   - Verify throughput meets 1M+ rows/sec target
   - This will add ~5-10% coverage by exercising the JIT warmup path

3. **Add stress/edge case tests:**
   - Zero volume scenario (verify dollar volume = 0, no bars generated if threshold never met)
   - Zero price scenario (verify dollar volume = 0)
   - Extremely high prices (verify no overflow, correct bar generation)
   - Extremely low prices (verify bars generated slowly)
   - Exact threshold match (verify bar closes when cumulative_dollars == threshold)

### Implementation Strategy

**Step 1: Add Advanced Algorithm Correctness Tests (Target +10-15% coverage)**

Create new test class `TestDollarAdvancedScenarios`:
```python
class TestDollarAdvancedScenarios:
    """Advanced tests for dollar bar algorithm implementation."""

    def test_exact_threshold_match(self) -> None:
        """Should close bar when cumulative dollar volume exactly matches threshold."""
        # Design OHLCV to hit threshold exactly

    def test_multiple_bar_generation_with_known_values(self) -> None:
        """Should generate multiple bars with predictable dollar volumes."""
        # Create 10 bars with known dollar volumes, verify each bar closes correctly

    def test_high_low_aggregation_across_bars(self) -> None:
        """Should correctly track high/low across accumulated source bars."""
        # Design OHLC pattern: increasing highs, decreasing lows within each bar

    def test_state_reset_after_bar_close(self) -> None:
        """After closing a bar, should reset state from close price of triggering bar."""
        # Verify open of next bar equals close of previous bar

    def test_zero_volume_contribution(self) -> None:
        """Zero volume should contribute zero to dollar volume accumulation."""
        # Mix zero and non-zero volumes, verify bars close correctly

    def test_zero_price_contribution(self) -> None:
        """Zero price should contribute zero to dollar volume accumulation."""
        # Mix zero and non-zero prices, verify bars close correctly
```

**Step 2: Add Performance Benchmark (Target +5-10% coverage)**

Replace `test_1m_rows_performance` with working implementation:
```python
def test_1m_rows_performance(self) -> None:
    """Should achieve 1M+ rows/sec throughput."""
    import time

    # Generate 1M rows of realistic price and volume data
    n = 1_000_000
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
    # ... generate high, low, volume

    # Warmup JIT (small dataset)
    _ = generate_dollar_bars_series(...[:1000], dollar_threshold=50000.0)

    # Measure throughput
    start = time.perf_counter()
    result = generate_dollar_bars_series(..., dollar_threshold=5000000.0)
    elapsed = time.perf_counter() - start

    throughput = n / elapsed
    assert throughput >= 1_000_000, f"Throughput {throughput:,.0f} < 1M rows/sec"
    assert isinstance(result, BarSeries)
    assert len(result) > 0
```

**Step 3: Add Volume Aggregation Tests (Target +5% coverage)**

Create new test class `TestDollarVolumeAggregation` (similar to Range bars pattern):
```python
class TestDollarVolumeAggregation:
    """Tests for exact volume aggregation logic."""

    def test_volume_first_bar_exact(self) -> None:
        """First bar should aggregate volume from 0 to bar_indices[0]."""

    def test_volume_subsequent_bars_exact(self) -> None:
        """Subsequent bars should aggregate from previous+1 to current index."""

    def test_volume_total_conservation(self) -> None:
        """Total volume across all bars should equal total input volume."""
```

**Step 4: Add Boundary Condition Tests (Target +5% coverage)**

Create new test class `TestDollarBoundaryConditions`:
```python
class TestDollarBoundaryConditions:
    """Tests for dollar_threshold boundary conditions."""

    def test_exact_threshold_closes_bar(self) -> None:
        """Dollar volume exactly at threshold should close bar."""

    def test_below_threshold_no_close(self) -> None:
        """Dollar volume below threshold should not close bar."""

    def test_above_threshold_closes_immediately(self) -> None:
        """Dollar volume exceeding threshold should close bar immediately."""

    def test_large_gap_single_bar(self) -> None:
        """Large dollar volume spike should create single bar, not multiple."""
```

**Step 5: Verify Coverage Reached 70%+**

Run coverage check:
```bash
cd /home/buckstrdr/simple_futures_backtester
pytest tests/test_bars/test_dollar_bars.py \
  --cov=simple_futures_backtester.bars.dollar_bars \
  --cov-report=term-missing \
  --cov-fail-under=70
```

Expected result: Coverage ≥ 70%, all tests passing

---

## 4. Implementation Tips & Notes

*   **Tip 1 - JIT Coverage Limitation:** Numba JIT functions may not show accurate line-by-line coverage even when exercised. Focus on creating diverse test scenarios that logically exercise all code paths, even if coverage report doesn't reflect it perfectly.

*   **Tip 2 - Dollar Volume Calculation:** Remember the formula is `close[i] * volume[i]`. Design test data where you can manually calculate expected dollar volumes:
    ```python
    # Example: Close prices 100, 110, 120 with volumes 500, 600, 700
    # Dollar volumes: 50000, 66000, 84000
    # Cumulative: 50000, 116000 -> bar closes at index 1 (116000 >= 100000)
    ```

*   **Tip 3 - Performance Benchmark Pattern (from test_range_bars.py:637-680):**
    ```python
    import time
    import numpy as np

    def test_performance(self) -> None:
        n = 1_000_000
        np.random.seed(42)  # Reproducible
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        volume = np.abs(np.random.randn(n) * 500 + 1000).astype(np.int64)

        # JIT warmup (not measured)
        _ = generate_dollar_bars_series(
            open_arr=close[:1000], high_arr=high[:1000],
            low_arr=low[:1000], close_arr=close[:1000],
            volume_arr=volume[:1000], dollar_threshold=50000.0
        )

        # Measure (actual benchmark)
        start = time.perf_counter()
        result = generate_dollar_bars_series(
            open_arr=close, high_arr=high,
            low_arr=low, close_arr=close,
            volume_arr=volume, dollar_threshold=5000000.0
        )
        elapsed = time.perf_counter() - start

        throughput = n / elapsed
        print(f"\nThroughput: {throughput:,.0f} rows/sec ({elapsed:.3f}s for {n:,} rows)")
        assert throughput >= 1_000_000, f"Performance below target: {throughput:,.0f} rows/sec < 1M rows/sec"
    ```

*   **Tip 4 - Volume Aggregation Verification (from test_range_bars.py:341-364):**
    ```python
    # Verify raw volume (not dollar volume) is aggregated correctly
    volume = np.array([100, 200, 300, 400, 500], dtype=np.int64)
    bars = generate_dollar_bars_series(..., volume_arr=volume, dollar_threshold=X)

    # First bar aggregates from index 0 to bar_indices[0]
    first_bar_end_idx = bars.index_map[0]
    expected_volume = np.sum(volume[: first_bar_end_idx + 1])
    assert bars.volume[0] == expected_volume
    ```

*   **Tip 5 - Index Map Verification:**
    ```python
    # index_map should point to source row where each bar completed
    bars = generate_dollar_bars_series(..., dollar_threshold=50000.0)

    # Verify indices are valid and monotonically increasing
    n = len(source_data)
    assert all(0 <= idx < n for idx in bars.index_map)
    if len(bars) > 1:
        assert all(bars.index_map[i] < bars.index_map[i+1] for i in range(len(bars)-1))
    ```

*   **Tip 6 - State Reset Verification (from test_range_bars.py:446-468):**
    ```python
    # After a bar closes, next bar's open should equal close price where previous bar closed
    bars = generate_dollar_bars_series(..., dollar_threshold=50000.0)

    if len(bars) >= 2:
        # Second bar should start from close price at first bar's completion
        first_close_idx = bars.index_map[0]
        assert bars.open[1] == close[first_close_idx]
    ```

*   **Tip 7 - Boundary Condition Testing (from test_range_bars.py:522-544):**
    ```python
    # Test exact threshold match (cumulative_dollars == dollar_threshold)
    # Design data so cumulative dollar volume reaches exactly threshold
    close = np.array([100.0, 100.0, 100.0])
    volume = np.array([500, 500, 0], dtype=np.int64)
    # Dollar volumes: 50000, 50000, 0
    # Cumulative at index 1: 100000 (exactly threshold)

    bars = generate_dollar_bars_series(
        ..., volume_arr=volume, dollar_threshold=100000.0
    )

    # Should close bar at index 1 (>= condition)
    assert len(bars) == 1
    assert bars.index_map[0] == 1
    ```

*   **Note:** Your existing test structure is excellent and follows project conventions. You just need to add more test cases that exercise different code paths and fix the performance benchmark.

*   **Warning:** DO NOT modify dollar_bars.py source code. This task is ONLY about enhancing test coverage, not changing implementation.

---

## 5. Acceptance Criteria Checklist

Before marking this task complete, verify:

- [ ] Coverage for dollar_bars.py ≥ 70% (run pytest --cov)
- [ ] All existing 19 tests still pass
- [ ] New tests added for algorithm correctness (exact threshold, multiple bars, state reset)
- [ ] Performance benchmark test implemented and passing (1M+ rows/sec)
- [ ] Edge cases tested (zero volume, zero price, extreme values)
- [ ] Volume aggregation tested across multiple scenarios
- [ ] Index map correctness verified
- [ ] State reset after bar close verified
- [ ] No regressions in existing functionality
- [ ] Test code follows project conventions (class-based organization, clear test names)

---

**Recommended Implementation Order:**

1. **Remove @pytest.mark.skip from test_1m_rows_performance** and implement with manual timing (follow range_bars.py pattern)
2. **Add TestDollarAdvancedScenarios class** with 6-8 new tests for algorithm correctness
3. **Add TestDollarVolumeAggregation class** with 3-4 tests for volume aggregation verification
4. **Add TestDollarBoundaryConditions class** with 4-5 tests for threshold boundary conditions
5. **Run coverage check** to measure progress toward 70%
6. **Final verification** with full test suite run

**Estimated Coverage Gain:**
- Current: 44%
- After performance benchmark fix: ~50-55%
- After advanced scenarios: ~60-65%
- After volume aggregation tests: ~65-70%
- After boundary condition tests: ~70-75%

**Target: 70%+ coverage with all tests passing**
