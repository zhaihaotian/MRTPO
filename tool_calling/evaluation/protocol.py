"""BFCL V4 single-turn categories and the aggregation used in the paper."""
import statistics

V4_COMMIT = '6ea57973c7a6097fd7c5915698c54c17c5b1b6c8'
SIMPLE = ['simple_python', 'simple_java', 'simple_javascript']
CALLS = ['multiple', 'parallel', 'parallel_multiple']
LIVE = ['live_simple', 'live_multiple', 'live_parallel', 'live_parallel_multiple']
COLUMNS = ['simple', 'multiple', 'parallel', 'parallel_multiple', 'non_live',
           'live_simple', 'live_multiple', 'live_parallel', 'live_parallel_multiple', 'live']
SUMMARY = COLUMNS + ['single_turn']


def categories():
    return SIMPLE + CALLS + LIVE


def expected_counts():
    """2,501 cases: Non-Live 1,150 and Live 1,351."""
    return dict(zip(categories(), [400, 100, 50, 200, 200, 200, 258, 1053, 16, 24]))


def aggregate(values, counts):
    """Category values (fractions; accuracy or format rate) -> paper columns. Simple = mean of the three languages;
    Non-Live = mean of Simple, Multiple, Parallel, Parallel Multiple; Live = case-weighted mean of the four Live
    categories; single_turn = Non-Live and Live weighted by their case counts (1,150 : 1,351)."""
    simple = statistics.mean(values[k] for k in SIMPLE)
    non_live = statistics.mean([simple] + [values[k] for k in CALLS])
    live = sum(values[k] * counts[k] for k in LIVE) / sum(counts[k] for k in LIVE)
    n_non_live = sum(counts[k] for k in SIMPLE + CALLS)
    n_live = sum(counts[k] for k in LIVE)
    single_turn = (n_non_live * non_live + n_live * live) / (n_non_live + n_live)  # 1,150 : 1,351
    out = dict(simple=simple, non_live=non_live, live=live, single_turn=single_turn)
    out.update({k: values[k] for k in CALLS + LIVE})
    return {k: out[k] for k in SUMMARY}
