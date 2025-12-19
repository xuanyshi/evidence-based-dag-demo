import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

__all__ = ['analyze_tri_df', 'plot_cumulative_trends','analyze_tri_df_binary','plot_cumulative_trends_binary']

import pandas as pd

import pandas as pd
import numpy as np

def analyze_tri_df(input_df, detail_info=False, use_participant_weights=True):
    # 1. Separate MR from non-MR
    df_non_mr = input_df[input_df['study_design'] != 'MR']
    df_mr     = input_df[input_df['study_design'] == 'MR']

    # 2. Filter valid directions
    df_non_mr = df_non_mr[df_non_mr['exposure_direction'].isin(['increased','decreased'])]
    df_non_mr = df_non_mr[df_non_mr['direction'].isin(['increase','decrease','no_change'])]
    
    # 3. Percentile Filtering (Optional)
    if use_participant_weights and df_non_mr['pmid'].nunique() >= 100:
        lower_bound = df_non_mr['number_of_participants'].quantile(0.05)
        upper_bound = df_non_mr['number_of_participants'].quantile(0.95)

        df_non_mr_filtered = df_non_mr[
            (df_non_mr['number_of_participants'] >= lower_bound) &
            (df_non_mr['number_of_participants'] <= upper_bound)
        ]
    else:
        df_non_mr_filtered = df_non_mr.copy()

    # 4. Combine MR and Filtered Non-MR
    input_df = pd.concat([df_non_mr_filtered, df_mr], ignore_index=True)

    # 5. Groupby Aggregation
    tri_df = (
        input_df
        .groupby(['study_design', 'exposure_direction', 'direction'], as_index=False)
        .agg(
            count=('direction', 'size'),
            participants_sum=('number_of_participants', 'sum')
        )
    )

    # --- Debug Info (Conditional) ---
    if detail_info:
        print(f"Input Shape: {input_df.shape}, Unique PMIDs: {input_df['pmid'].nunique()}")
        print(tri_df)

    # Helper function
    def handle_zero_and_sum(*args):
        # Add small epsilon (0.1) to zeros to avoid division by zero issues later or zero sums
        adjusted_values = [val + 0.1 if val == 0 else val for val in args]
        total_sum = sum(adjusted_values)
        return adjusted_values, total_sum

    # ==============================================================================
    # RCT Calculations
    # ==============================================================================
    
    # --- RCT Increased Exposure ---
    rct_i_i = tri_df.loc[(tri_df['study_design'] == 'RCT') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'increase'), 'count'].sum()
    rct_i_i_ppl = tri_df.loc[(tri_df['study_design'] == 'RCT') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'increase'), 'participants_sum'].sum()

    rct_i_n = tri_df.loc[(tri_df['study_design'] == 'RCT') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'no_change'), 'count'].sum()
    rct_i_n_ppl = tri_df.loc[(tri_df['study_design'] == 'RCT') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'no_change'), 'participants_sum'].sum()

    rct_i_d = tri_df.loc[(tri_df['study_design'] == 'RCT') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'decrease'), 'count'].sum()
    rct_i_d_ppl = tri_df.loc[(tri_df['study_design'] == 'RCT') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'decrease'), 'participants_sum'].sum()

    if use_participant_weights:
        denominator_rct_i = rct_i_i_ppl + rct_i_n_ppl + rct_i_d_ppl
        rct_i_i = rct_i_i * (rct_i_i_ppl / denominator_rct_i) if denominator_rct_i != 0 else 0
        rct_i_n = rct_i_n * (rct_i_n_ppl / denominator_rct_i) if denominator_rct_i != 0 else 0
        rct_i_d = rct_i_d * (rct_i_d_ppl / denominator_rct_i) if denominator_rct_i != 0 else 0
        
        if detail_info and denominator_rct_i != 0:
            print(f"RCT Inc Ratio (Inc/Total): {rct_i_i_ppl / denominator_rct_i:.2f}")
            print(f"RCT Inc Ratio (NoCh/Total): {rct_i_n_ppl / denominator_rct_i:.2f}")
            print(f"RCT Inc Ratio (Dec/Total): {rct_i_d_ppl / denominator_rct_i:.2f}")
    
    (rct_i_i, rct_i_n, rct_i_d), rct_i_sum = handle_zero_and_sum(rct_i_i, rct_i_n, rct_i_d)

    # --- RCT Decreased Exposure ---
    rct_d_i = tri_df.loc[(tri_df['study_design'] == 'RCT') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'increase'), 'count'].sum()
    rct_d_i_ppl = tri_df.loc[(tri_df['study_design'] == 'RCT') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'increase'), 'participants_sum'].sum()

    rct_d_n = tri_df.loc[(tri_df['study_design'] == 'RCT') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'no_change'), 'count'].sum()
    rct_d_n_ppl = tri_df.loc[(tri_df['study_design'] == 'RCT') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'no_change'), 'participants_sum'].sum()

    rct_d_d = tri_df.loc[(tri_df['study_design'] == 'RCT') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'decrease'), 'count'].sum()
    rct_d_d_ppl = tri_df.loc[(tri_df['study_design'] == 'RCT') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'decrease'), 'participants_sum'].sum()

    if use_participant_weights:
        denominator_rct_d = rct_d_i_ppl + rct_d_n_ppl + rct_d_d_ppl
        rct_d_i = rct_d_i * (rct_d_i_ppl / denominator_rct_d) if denominator_rct_d != 0 else 0
        rct_d_n = rct_d_n * (rct_d_n_ppl / denominator_rct_d) if denominator_rct_d != 0 else 0
        rct_d_d = rct_d_d * (rct_d_d_ppl / denominator_rct_d) if denominator_rct_d != 0 else 0
        
        if detail_info and denominator_rct_d != 0:
            print(f"RCT Dec Ratio (Inc/Total): {rct_d_i_ppl / denominator_rct_d:.2f}")
            print(f"RCT Dec Ratio (NoCh/Total): {rct_d_n_ppl / denominator_rct_d:.2f}")
            print(f"RCT Dec Ratio (Dec/Total): {rct_d_d_ppl / denominator_rct_d:.2f}")

    (rct_d_i, rct_d_n, rct_d_d), rct_d_sum = handle_zero_and_sum(rct_d_i, rct_d_n, rct_d_d)


    # ==============================================================================
    # MR Calculations
    # ==============================================================================
    # Increased
    mr_i_i = tri_df.loc[(tri_df['study_design'] == 'MR') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'increase'), 'count'].sum()
    mr_i_n = tri_df.loc[(tri_df['study_design'] == 'MR') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'no_change'), 'count'].sum()
    mr_i_d = tri_df.loc[(tri_df['study_design'] == 'MR') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'decrease'), 'count'].sum()
    (mr_i_i, mr_i_n, mr_i_d), mr_i_sum = handle_zero_and_sum(mr_i_i, mr_i_n, mr_i_d)

    # Decreased
    mr_d_i = tri_df.loc[(tri_df['study_design'] == 'MR') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'increase'), 'count'].sum()
    mr_d_n = tri_df.loc[(tri_df['study_design'] == 'MR') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'no_change'), 'count'].sum()
    mr_d_d = tri_df.loc[(tri_df['study_design'] == 'MR') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'decrease'), 'count'].sum()
    (mr_d_i, mr_d_n, mr_d_d), mr_d_sum = handle_zero_and_sum(mr_d_i, mr_d_n, mr_d_d)


    # ==============================================================================
    # OS Calculations
    # ==============================================================================
    
    # --- OS Increased Exposure ---
    os_i_i = tri_df.loc[(tri_df['study_design'] == 'OS') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'increase'), 'count'].sum()
    os_i_i_ppl = tri_df.loc[(tri_df['study_design'] == 'OS') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'increase'), 'participants_sum'].sum()

    os_i_n = tri_df.loc[(tri_df['study_design'] == 'OS') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'no_change'), 'count'].sum()
    os_i_n_ppl = tri_df.loc[(tri_df['study_design'] == 'OS') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'no_change'), 'participants_sum'].sum()

    os_i_d = tri_df.loc[(tri_df['study_design'] == 'OS') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'decrease'), 'count'].sum()
    os_i_d_ppl = tri_df.loc[(tri_df['study_design'] == 'OS') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'decrease'), 'participants_sum'].sum()

    if use_participant_weights:
        denominator_os_i = os_i_i_ppl + os_i_n_ppl + os_i_d_ppl
        os_i_i = os_i_i * (os_i_i_ppl / denominator_os_i) if denominator_os_i != 0 else 0
        os_i_n = os_i_n * (os_i_n_ppl / denominator_os_i) if denominator_os_i != 0 else 0
        os_i_d = os_i_d * (os_i_d_ppl / denominator_os_i) if denominator_os_i != 0 else 0
        
        if detail_info and denominator_os_i != 0:
            print(f"OS Inc Ratio (Inc/Total): {os_i_i_ppl / denominator_os_i:.2f}")
            print(f"OS Inc Ratio (NoCh/Total): {os_i_n_ppl / denominator_os_i:.2f}")
            print(f"OS Inc Ratio (Dec/Total): {os_i_d_ppl / denominator_os_i:.2f}")

    (os_i_i, os_i_n, os_i_d), os_i_sum = handle_zero_and_sum(os_i_i, os_i_n, os_i_d)

    # --- OS Decreased Exposure ---
    os_d_i = tri_df.loc[(tri_df['study_design'] == 'OS') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'increase'), 'count'].sum()
    os_d_i_ppl = tri_df.loc[(tri_df['study_design'] == 'OS') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'increase'), 'participants_sum'].sum()

    os_d_n = tri_df.loc[(tri_df['study_design'] == 'OS') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'no_change'), 'count'].sum()
    os_d_n_ppl = tri_df.loc[(tri_df['study_design'] == 'OS') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'no_change'), 'participants_sum'].sum()

    os_d_d = tri_df.loc[(tri_df['study_design'] == 'OS') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'decrease'), 'count'].sum()
    os_d_d_ppl = tri_df.loc[(tri_df['study_design'] == 'OS') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'decrease'), 'participants_sum'].sum()

    if use_participant_weights:
        denominator_os_d = os_d_i_ppl + os_d_n_ppl + os_d_d_ppl
        os_d_i = os_d_i * (os_d_i_ppl / denominator_os_d) if denominator_os_d != 0 else 0
        os_d_n = os_d_n * (os_d_n_ppl / denominator_os_d) if denominator_os_d != 0 else 0
        os_d_d = os_d_d * (os_d_d_ppl / denominator_os_d) if denominator_os_d != 0 else 0
        
        if detail_info and denominator_os_d != 0:
            print(f"OS Dec Ratio (Inc/Total): {os_d_i_ppl / denominator_os_d:.2f}")
            print(f"OS Dec Ratio (NoCh/Total): {os_d_n_ppl / denominator_os_d:.2f}")
            print(f"OS Dec Ratio (Dec/Total): {os_d_d_ppl / denominator_os_d:.2f}")

    (os_d_i, os_d_n, os_d_d), os_d_sum = handle_zero_and_sum(os_d_i, os_d_n, os_d_d)

    # ==============================================================================
    # Final Probability Calculation
    # ==============================================================================
    # 1/6 weight for each of the 6 components:
    # (RCT Inc, MR Inc, OS Inc, RCT Dec, MR Dec, OS Dec)
    # Note: For Decreased exposure, the logic for 'Excitatory' is outcome 'Decrease'
    # Wait, the logic below follows standard: 
    # Excitatory = (Inc->Inc) + (Dec->Dec)
    # Inhibitory = (Inc->Dec) + (Dec->Inc)
    
    p_excitatory = 1/6 * ((rct_i_i/rct_i_sum) + (mr_i_i/mr_i_sum) + (os_i_i/os_i_sum) + 
                          (rct_d_d/rct_d_sum) + (mr_d_d/mr_d_sum) + (os_d_d/os_d_sum))
    
    p_no_change = 1/6 * ((rct_i_n/rct_i_sum) + (mr_i_n/mr_i_sum) + (os_i_n/os_i_sum) + 
                         (rct_d_n/rct_d_sum) + (mr_d_n/mr_d_sum) + (os_d_n/os_d_sum))
    
    p_inhibitory = 1/6 * ((rct_i_d/rct_i_sum) + (mr_i_d/mr_i_sum) + (os_i_d/os_i_sum) + 
                          (rct_d_i/rct_d_sum) + (mr_d_i/mr_d_sum) + (os_d_i/os_d_sum))

    # Calculate LOE
    biggest = max(p_excitatory, p_no_change, p_inhibitory)
    # Normalize LOE: (max - 1/3) / (2/3)
    loe = (biggest - (1/3)) / (1 - (1/3))

    # Determine relationship
    if biggest == p_excitatory:
        biggest_relation = 'excitatory'
    elif biggest == p_no_change:
        biggest_relation = 'no_change'
    else:
        biggest_relation = 'inhibitory'

    return {
        "p_excitatory": round(p_excitatory, 3),
        "p_no_change": round(p_no_change, 3),
        "p_inhibitory": round(p_inhibitory, 3),
        "loe": round(loe, 3),
        "biggest": biggest_relation
    }

def plot_cumulative_trends(score_df_yearly_dfs,
                           start_display=1980,
                           end_display=2020,
                           save_path=None,
                           focus_year=None,
                           source_data_path=None,
                           base_fontsize=16):
    """
    Plots cumulative trends for p_excitatory, p_no_change, p_inhibitory,
    and cumulative evidence counts (no rolling average).

    Args
    ----
    score_df_yearly_dfs : list[pd.DataFrame]
        One DataFrame per publication year.
    start_display : int
        First year shown on the x-axis.
    end_display : int
        Last year shown on the x-axis.
    save_path : str | None
        Optional path to save the PNG (dpi=600).
    focus_year : int | None
        If provided, draw a vertical dashed line at this year.
    source_data_path : str | None
        If provided, write the subsetted results to CSV.
    base_fontsize : int
        Base font size for every text element.
    """

    # ──────────────────────────────── style ────────────────────────────
    plt.rcParams.update({'font.size': base_fontsize,
                         'font.family':'sans-serif',
                         'font.sans-serif':['DejaVu Sans']})

    # ──────────────────────── accumulate yearly stats ──────────────────
    results = []
    cumulative_counts = dict(p_excitatory=0, p_no_change=0, p_inhibitory=0)

    all_years = sorted({int(y[0]) for df in score_df_yearly_dfs
                        for y in [df['Publication Year'].unique()] if len(y) > 0})

    prev_loc, prev_biggest = None, None

    for end_year in all_years:
        cumulative_dfs = [df for df in score_df_yearly_dfs
                          if (not df.empty
                              and int(df['Publication Year'].unique()[0]) <= end_year)]

        if not cumulative_dfs:
            continue

        combined_df   = pd.concat(cumulative_dfs)
        current_year_df = combined_df[combined_df['Publication Year'] == end_year]
        current_year_df = current_year_df[current_year_df['study_design'] != 'MR']

        # cumulative counts
        cumulative_counts['p_excitatory'] = (combined_df['relationship'] == 'excitatory').sum()
        cumulative_counts['p_no_change']  = (combined_df['relationship'] == 'no_change').sum()
        cumulative_counts['p_inhibitory'] = (combined_df['relationship'] == 'inhibitory').sum()

        # custom analysis (user-provided)
        analysis_result = analyze_tri_df(combined_df, False)
        curr_loc      = analysis_result.pop('loe')
        curr_biggest  = analysis_result.pop('biggest')

        prev_loc, prev_biggest = curr_loc, curr_biggest

        biggest_p = max(analysis_result.values())
        biggest_p_label = max(analysis_result, key=analysis_result.get)

        results.append({
            'end_year': end_year,
            'loc':      curr_loc,
            **analysis_result,
            'cumulative_counts_p_excitatory': cumulative_counts['p_excitatory'],
            'cumulative_counts_p_no_change':  cumulative_counts['p_no_change'],
            'cumulative_counts_p_inhibitory': cumulative_counts['p_inhibitory'],
            'biggest_p':       biggest_p,
            'biggest_p_label': biggest_p_label
        })

    # ─────────────────────────── prepare plotting df ───────────────────
    results_df = pd.DataFrame(results).sort_values('end_year')
    results_df_disp = results_df[(results_df['end_year'] >= start_display) &
                                 (results_df['end_year'] <= end_display)]

    if source_data_path:
        results_df_disp.to_csv(source_data_path, index=False)

    # ─────────────────────────────── plotting ──────────────────────────
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()

    # right-axis: cumulative counts
    ax2.plot(results_df_disp['end_year'],
             results_df_disp['cumulative_counts_p_excitatory'],
             linestyle='dotted', linewidth=2, color='green',
             label='Excitatory')
    ax2.plot(results_df_disp['end_year'],
             results_df_disp['cumulative_counts_p_no_change'],
             linestyle='dotted', linewidth=2, color='orange',
             label='No Change')
    ax2.plot(results_df_disp['end_year'],
             results_df_disp['cumulative_counts_p_inhibitory'],
             linestyle='dotted', linewidth=2, color='red',
             label='Inhibitory')

    # left-axis: probabilities
    ax1.plot(results_df_disp['end_year'], results_df_disp['p_excitatory'],
             marker='o', color='green',  label='p_excitatory', zorder=3)
    ax1.plot(results_df_disp['end_year'], results_df_disp['p_no_change'],
             marker='o', color='orange', label='p_no_change',  zorder=3)
    ax1.plot(results_df_disp['end_year'], results_df_disp['p_inhibitory'],
             marker='o', color='red',    label='p_inhibitory', zorder=3)

    # axis labels
    ax1.set_xlabel('Year', fontsize=base_fontsize + 2)
    ax1.set_ylabel('Probability of Relation', fontsize=base_fontsize + 2)
    ax2.set_ylabel('Number of Studies', fontsize=base_fontsize + 2)  # ← added label

    ax1.set_ylim(0, 1)

    # 5-year ticks
    start_tick = (results_df_disp['end_year'].min() // 5) * 5
    end_tick   = (results_df_disp['end_year'].max() // 5) * 5
    xticks = np.arange(start_tick, end_tick + 1, 5)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks, fontsize=base_fontsize)
    ax1.tick_params(axis='y', labelsize=base_fontsize)
    ax2.tick_params(axis='y', labelsize=base_fontsize)
    ax1.set_xlim(results_df_disp['end_year'].min(), results_df_disp['end_year'].max())
    ax1.grid(False)

    # focus year marker
    if focus_year and focus_year in results_df_disp['end_year'].values:
        ax1.axvline(focus_year, linestyle='dashed', linewidth=1.5, color='blue')
        ax1.text(focus_year, 0.53, str(focus_year),
                 color='blue', fontsize=base_fontsize,
                 ha='center', fontweight='bold')

    # # legends
    # ax1.legend(loc='upper left',  fontsize=base_fontsize)
    #ax2.legend(loc='upper center', fontsize=base_fontsize)
    legend_handles = [
    Patch(facecolor='green',  edgecolor='green',  label='Excitatory'),
    Patch(facecolor='orange', edgecolor='orange', label='No Change'),
    Patch(facecolor='red',    edgecolor='red',    label='Inhibitory')]

# Add it once, on whichever axis you prefer (ax1 here)
    ax1.legend(handles=legend_handles,
              loc='upper left',     # pick a position you like
              fontsize=base_fontsize)

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.show()


import pandas as pd

def analyze_tri_df_bias(input_df, detail_info=False, if_weight=True):
    """
    Triangulation analysis by bias_group (quasi / cohort / otherweak).

    - 使用 bias_group 中的三类: 'quasi', 'cohort', 'otherweak'
    - 所有三类（quasi/cohort/otherweak）在 increased / decreased 下都按 number_of_participants 加权
    - 不再做 MR vs non-MR 过滤，直接用传入的 input_df
    """

    # 1) 直接使用原始 df
    input_df2 = input_df.copy()

    # 2) 按 bias_group × exposure_direction × direction 聚合
    tri_df = (
        input_df2
        .groupby(['bias_group', 'exposure_direction', 'direction'], as_index=False)
        .agg(
            count=('direction', 'size'),
            participants_sum=('number_of_participants', 'sum')
        )
    )

    print(input_df2.shape, input_df2['pmid'].nunique())
    if detail_info:
        print(tri_df)

    # helper：防止三格都是 0 的情况
    def handle_zero_and_sum(*args):
        adjusted_values = [val + 0.1 if val == 0 else val for val in args]
        total_sum = sum(adjusted_values)
        return adjusted_values, total_sum

    # =====================================================
    # quasi 组
    # =====================================================
    # increased
    quasi_i_i = tri_df.loc[
        (tri_df['bias_group'] == 'quasi') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'increase'),
        'count'
    ].sum()
    quasi_i_i_ppl = tri_df.loc[
        (tri_df['bias_group'] == 'quasi') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'increase'),
        'participants_sum'
    ].sum()

    quasi_i_n = tri_df.loc[
        (tri_df['bias_group'] == 'quasi') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'no_change'),
        'count'
    ].sum()
    quasi_i_n_ppl = tri_df.loc[
        (tri_df['bias_group'] == 'quasi') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'no_change'),
        'participants_sum'
    ].sum()

    quasi_i_d = tri_df.loc[
        (tri_df['bias_group'] == 'quasi') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'decrease'),
        'count'
    ].sum()
    quasi_i_d_ppl = tri_df.loc[
        (tri_df['bias_group'] == 'quasi') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'decrease'),
        'participants_sum'
    ].sum()

    denominator_quasi_i = quasi_i_i_ppl + quasi_i_n_ppl + quasi_i_d_ppl
    if denominator_quasi_i != 0:
        quasi_i_i = quasi_i_i * (quasi_i_i_ppl / denominator_quasi_i)
        quasi_i_n = quasi_i_n * (quasi_i_n_ppl / denominator_quasi_i)
        quasi_i_d = quasi_i_d * (quasi_i_d_ppl / denominator_quasi_i)
        print(f"Ratio (quasi_i_i_ppl / total): {quasi_i_i_ppl / denominator_quasi_i:.2f}")
        print(f"Ratio (quasi_i_n_ppl / total): {quasi_i_n_ppl / denominator_quasi_i:.2f}")
        print(f"Ratio (quasi_i_d_ppl / total): {quasi_i_d_ppl / denominator_quasi_i:.2f}")
    else:
        print("Ratio (quasi_i_i_ppl / total): 0.00")
        print("Ratio (quasi_i_n_ppl / total): 0.00")
        print("Ratio (quasi_i_d_ppl / total): 0.00")

    (quasi_i_i, quasi_i_n, quasi_i_d), quasi_i_sum = handle_zero_and_sum(quasi_i_i, quasi_i_n, quasi_i_d)

    # decreased
    quasi_d_i = tri_df.loc[
        (tri_df['bias_group'] == 'quasi') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'increase'),
        'count'
    ].sum()
    quasi_d_i_ppl = tri_df.loc[
        (tri_df['bias_group'] == 'quasi') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'increase'),
        'participants_sum'
    ].sum()

    quasi_d_n = tri_df.loc[
        (tri_df['bias_group'] == 'quasi') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'no_change'),
        'count'
    ].sum()
    quasi_d_n_ppl = tri_df.loc[
        (tri_df['bias_group'] == 'quasi') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'no_change'),
        'participants_sum'
    ].sum()

    quasi_d_d = tri_df.loc[
        (tri_df['bias_group'] == 'quasi') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'decrease'),
        'count'
    ].sum()
    quasi_d_d_ppl = tri_df.loc[
        (tri_df['bias_group'] == 'quasi') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'decrease'),
        'participants_sum'
    ].sum()

    denominator_quasi_d = quasi_d_i_ppl + quasi_d_n_ppl + quasi_d_d_ppl
    if denominator_quasi_d != 0:
        quasi_d_i = quasi_d_i * (quasi_d_i_ppl / denominator_quasi_d)
        quasi_d_n = quasi_d_n * (quasi_d_n_ppl / denominator_quasi_d)
        quasi_d_d = quasi_d_d * (quasi_d_d_ppl / denominator_quasi_d)
        print(f"Ratio (quasi_d_i_ppl / total): {quasi_d_i_ppl / denominator_quasi_d:.2f}")
        print(f"Ratio (quasi_d_n_ppl / total): {quasi_d_n_ppl / denominator_quasi_d:.2f}")
        print(f"Ratio (quasi_d_d_ppl / total): {quasi_d_d_ppl / denominator_quasi_d:.2f}")
    else:
        print("Ratio (quasi_d_i_ppl / total): 0.00")
        print("Ratio (quasi_d_n_ppl / total): 0.00")
        print("Ratio (quasi_d_d_ppl / total): 0.00")

    (quasi_d_i, quasi_d_n, quasi_d_d), quasi_d_sum = handle_zero_and_sum(quasi_d_i, quasi_d_n, quasi_d_d)

    # =====================================================
    # cohort 组
    # =====================================================
    cohort_i_i = tri_df.loc[
        (tri_df['bias_group'] == 'cohort') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'increase'),
        'count'
    ].sum()
    cohort_i_i_ppl = tri_df.loc[
        (tri_df['bias_group'] == 'cohort') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'increase'),
        'participants_sum'
    ].sum()

    cohort_i_n = tri_df.loc[
        (tri_df['bias_group'] == 'cohort') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'no_change'),
        'count'
    ].sum()
    cohort_i_n_ppl = tri_df.loc[
        (tri_df['bias_group'] == 'cohort') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'no_change'),
        'participants_sum'
    ].sum()

    cohort_i_d = tri_df.loc[
        (tri_df['bias_group'] == 'cohort') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'decrease'),
        'count'
    ].sum()
    cohort_i_d_ppl = tri_df.loc[
        (tri_df['bias_group'] == 'cohort') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'decrease'),
        'participants_sum'
    ].sum()

    denominator_cohort_i = cohort_i_i_ppl + cohort_i_n_ppl + cohort_i_d_ppl
    if denominator_cohort_i != 0:
        cohort_i_i = cohort_i_i * (cohort_i_i_ppl / denominator_cohort_i)
        cohort_i_n = cohort_i_n * (cohort_i_n_ppl / denominator_cohort_i)
        cohort_i_d = cohort_i_d * (cohort_i_d_ppl / denominator_cohort_i)
        print(f"Ratio (cohort_i_i_ppl / total): {cohort_i_i_ppl / denominator_cohort_i:.2f}")
        print(f"Ratio (cohort_i_n_ppl / total): {cohort_i_n_ppl / denominator_cohort_i:.2f}")
        print(f"Ratio (cohort_i_d_ppl / total): {cohort_i_d_ppl / denominator_cohort_i:.2f}")
    else:
        print("Ratio (cohort_i_i_ppl / total): 0.00")
        print("Ratio (cohort_i_n_ppl / total): 0.00")
        print("Ratio (cohort_i_d_ppl / total): 0.00")

    (cohort_i_i, cohort_i_n, cohort_i_d), cohort_i_sum = handle_zero_and_sum(cohort_i_i, cohort_i_n, cohort_i_d)

    cohort_d_i = tri_df.loc[
        (tri_df['bias_group'] == 'cohort') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'increase'),
        'count'
    ].sum()
    cohort_d_i_ppl = tri_df.loc[
        (tri_df['bias_group'] == 'cohort') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'increase'),
        'participants_sum'
    ].sum()

    cohort_d_n = tri_df.loc[
        (tri_df['bias_group'] == 'cohort') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'no_change'),
        'count'
    ].sum()
    cohort_d_n_ppl = tri_df.loc[
        (tri_df['bias_group'] == 'cohort') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'no_change'),
        'participants_sum'
    ].sum()

    cohort_d_d = tri_df.loc[
        (tri_df['bias_group'] == 'cohort') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'decrease'),
        'count'
    ].sum()
    cohort_d_d_ppl = tri_df.loc[
        (tri_df['bias_group'] == 'cohort') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'decrease'),
        'participants_sum'
    ].sum()

    denominator_cohort_d = cohort_d_i_ppl + cohort_d_n_ppl + cohort_d_d_ppl
    if denominator_cohort_d != 0:
        cohort_d_i = cohort_d_i * (cohort_d_i_ppl / denominator_cohort_d)
        cohort_d_n = cohort_d_n * (cohort_d_n_ppl / denominator_cohort_d)
        cohort_d_d = cohort_d_d * (cohort_d_d_ppl / denominator_cohort_d)
        print(f"Ratio (cohort_d_i_ppl / total): {cohort_d_i_ppl / denominator_cohort_d:.2f}")
        print(f"Ratio (cohort_d_n_ppl / total): {cohort_d_n_ppl / denominator_cohort_d:.2f}")
        print(f"Ratio (cohort_d_d_ppl / total): {cohort_d_d_ppl / denominator_cohort_d:.2f}")
    else:
        print("Ratio (cohort_d_i_ppl / total): 0.00")
        print("Ratio (cohort_d_n_ppl / total): 0.00")
        print("Ratio (cohort_d_d_ppl / total): 0.00")

    (cohort_d_i, cohort_d_n, cohort_d_d), cohort_d_sum = handle_zero_and_sum(cohort_d_i, cohort_d_n, cohort_d_d)

    # =====================================================
    # otherweak 组
    # =====================================================
    oth_i_i = tri_df.loc[
        (tri_df['bias_group'] == 'otherweak') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'increase'),
        'count'
    ].sum()
    oth_i_i_ppl = tri_df.loc[
        (tri_df['bias_group'] == 'otherweak') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'increase'),
        'participants_sum'
    ].sum()

    oth_i_n = tri_df.loc[
        (tri_df['bias_group'] == 'otherweak') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'no_change'),
        'count'
    ].sum()
    oth_i_n_ppl = tri_df.loc[
        (tri_df['bias_group'] == 'otherweak') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'no_change'),
        'participants_sum'
    ].sum()

    oth_i_d = tri_df.loc[
        (tri_df['bias_group'] == 'otherweak') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'decrease'),
        'count'
    ].sum()
    oth_i_d_ppl = tri_df.loc[
        (tri_df['bias_group'] == 'otherweak') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'decrease'),
        'participants_sum'
    ].sum()

    denominator_oth_i = oth_i_i_ppl + oth_i_n_ppl + oth_i_d_ppl
    if denominator_oth_i != 0:
        oth_i_i = oth_i_i * (oth_i_i_ppl / denominator_oth_i)
        oth_i_n = oth_i_n * (oth_i_n_ppl / denominator_oth_i)
        oth_i_d = oth_i_d * (oth_i_d_ppl / denominator_oth_i)
        print(f"Ratio (oth_i_i_ppl / total): {oth_i_i_ppl / denominator_oth_i:.2f}")
        print(f"Ratio (oth_i_n_ppl / total): {oth_i_n_ppl / denominator_oth_i:.2f}")
        print(f"Ratio (oth_i_d_ppl / total): {oth_i_d_ppl / denominator_oth_i:.2f}")
    else:
        print("Ratio (oth_i_i_ppl / total): 0.00")
        print("Ratio (oth_i_n_ppl / total): 0.00")
        print("Ratio (oth_i_d_ppl / total): 0.00")

    (oth_i_i, oth_i_n, oth_i_d), oth_i_sum = handle_zero_and_sum(oth_i_i, oth_i_n, oth_i_d)

    oth_d_i = tri_df.loc[
        (tri_df['bias_group'] == 'otherweak') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'increase'),
        'count'
    ].sum()
    oth_d_i_ppl = tri_df.loc[
        (tri_df['bias_group'] == 'otherweak') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'increase'),
        'participants_sum'
    ].sum()

    oth_d_n = tri_df.loc[
        (tri_df['bias_group'] == 'otherweak') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'no_change'),
        'count'
    ].sum()
    oth_d_n_ppl = tri_df.loc[
        (tri_df['bias_group'] == 'otherweak') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'no_change'),
        'participants_sum'
    ].sum()

    oth_d_d = tri_df.loc[
        (tri_df['bias_group'] == 'otherweak') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'decrease'),
        'count'
    ].sum()
    oth_d_d_ppl = tri_df.loc[
        (tri_df['bias_group'] == 'otherweak') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'decrease'),
        'participants_sum'
    ].sum()

    denominator_oth_d = oth_d_i_ppl + oth_d_n_ppl + oth_d_d_ppl
    if denominator_oth_d != 0:
        oth_d_i = oth_d_i * (oth_d_i_ppl / denominator_oth_d)
        oth_d_n = oth_d_n * (oth_d_n_ppl / denominator_oth_d)
        oth_d_d = oth_d_d * (oth_d_d_ppl / denominator_oth_d)
        print(f"Ratio (oth_d_i_ppl / total): {oth_d_i_ppl / denominator_oth_d:.2f}")
        print(f"Ratio (oth_d_n_ppl / total): {oth_d_n_ppl / denominator_oth_d:.2f}")
        print(f"Ratio (oth_d_d_ppl / total): {oth_d_d_ppl / denominator_oth_d:.2f}")
    else:
        print("Ratio (oth_d_i_ppl / total): 0.00")
        print("Ratio (oth_d_n_ppl / total): 0.00")
        print("Ratio (oth_d_d_ppl / total): 0.00")

    (oth_d_i, oth_d_n, oth_d_d), oth_d_sum = handle_zero_and_sum(oth_d_i, oth_d_n, oth_d_d)

    # -------- group weights --------
    if if_weight:
        w_quasi = 3.0   # quasi（MR / sibling / negative control / etc.）
        w_cohort = 2.0  # cohort
        w_oth = 1.0     # otherweak
    else:
        w_quasi = 1.0
        w_cohort = 1.0
        w_oth = 1.0

    # total weight: 每组有 increased + decreased 两个方向
    W = 2 * (w_quasi + w_cohort + w_oth)

    # -------- weighted probabilities --------
    p_excitatory = (
        w_quasi * (quasi_i_i / quasi_i_sum + quasi_d_d / quasi_d_sum) +
        w_cohort * (cohort_i_i / cohort_i_sum + cohort_d_d / cohort_d_sum) +
        w_oth * (oth_i_i / oth_i_sum + oth_d_d / oth_d_sum)
    ) / W

    p_no_change = (
        w_quasi * (quasi_i_n / quasi_i_sum + quasi_d_n / quasi_d_sum) +
        w_cohort * (cohort_i_n / cohort_i_sum + cohort_d_n / cohort_d_sum) +
        w_oth * (oth_i_n / oth_i_sum + oth_d_n / oth_d_sum)
    ) / W

    p_inhibitory = (
        w_quasi * (quasi_i_d / quasi_i_sum + quasi_d_i / quasi_d_sum) +
        w_cohort * (cohort_i_d / cohort_i_sum + cohort_d_i / cohort_d_sum) +
        w_oth * (oth_i_d / oth_i_sum + oth_d_i / oth_d_sum)
    ) / W

    biggest = max(p_excitatory, p_no_change, p_inhibitory)
    loe = (biggest - (1/3)) / (1 - (1/3))

    if biggest == p_excitatory:
        biggest_relation = 'excitatory'
    elif biggest == p_no_change:
        biggest_relation = 'no_change'
    else:
        biggest_relation = 'inhibitory'

    return {
        "p_excitatory": round(p_excitatory, 3),
        "p_no_change":  round(p_no_change, 3),
        "p_inhibitory": round(p_inhibitory, 3),
        "loe":          round(loe, 3),
        "biggest":      biggest_relation
    }

def analyze_tri_df_binary(input_df, detail_info):
    # Separate MR from non-MR
    df_non_mr = input_df[input_df['study_design'] != 'MR']
    df_mr     = input_df[input_df['study_design'] == 'MR']

    df_non_mr = df_non_mr[df_non_mr['exposure_direction'].isin(['increased','decreased'])]
    df_non_mr = df_non_mr[df_non_mr['direction'].isin(['increase','decrease','no_change'])]
    # Calculate 5th and 95th percentiles for the non-MR group
    if df_non_mr['pmid'].nunique() >= 100:
        lower_bound = df_non_mr['number_of_participants'].quantile(0.05)
        upper_bound = df_non_mr['number_of_participants'].quantile(0.95)

        df_non_mr_filtered = df_non_mr[
            (df_non_mr['number_of_participants'] >= lower_bound) &
            (df_non_mr['number_of_participants'] <= upper_bound)
        ]
    else:
        # If fewer than 100 unique PMIDs, skip quantile filtering
        df_non_mr_filtered = df_non_mr.copy()

    # Combine MR (unfiltered) and filtered Non-MR
    input_df = pd.concat([df_non_mr_filtered, df_mr], ignore_index=True)

    tri_df = (
        input_df
        .groupby(['study_design', 'exposure_direction', 'direction'], as_index=False)
        .agg(
            count=('direction', 'size'),
            participants_sum=('number_of_participants', 'sum')
        )
    )

    print(input_df.shape, input_df['pmid'].nunique())
    if detail_info:
        print(tri_df)

    # helper: avoid zero counts
    def handle_zero_and_sum(*args):
        adjusted_values = [val + 0.1 if val == 0 else val for val in args]
        total_sum = sum(adjusted_values)
        return adjusted_values, total_sum

    # ---------------- RCT increased ----------------
    rct_i_i = tri_df.loc[
        (tri_df['study_design'] == 'RCT') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'increase'),
        'count'
    ].sum()
    rct_i_i_ppl = tri_df.loc[
        (tri_df['study_design'] == 'RCT') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'increase'),
        'participants_sum'
    ].sum()

    rct_i_n = tri_df.loc[
        (tri_df['study_design'] == 'RCT') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'no_change'),
        'count'
    ].sum()
    rct_i_n_ppl = tri_df.loc[
        (tri_df['study_design'] == 'RCT') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'no_change'),
        'participants_sum'
    ].sum()

    rct_i_d = tri_df.loc[
        (tri_df['study_design'] == 'RCT') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'decrease'),
        'count'
    ].sum()
    rct_i_d_ppl = tri_df.loc[
        (tri_df['study_design'] == 'RCT') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'decrease'),
        'participants_sum'
    ].sum()

    denominator_rct_i = rct_i_i_ppl + rct_i_n_ppl + rct_i_d_ppl
    rct_i_i = rct_i_i * (rct_i_i_ppl / denominator_rct_i) if denominator_rct_i != 0 else 0
    print(f"Ratio (rct_i_i_ppl / total): {rct_i_i_ppl / denominator_rct_i:.2f}" if denominator_rct_i != 0 else "Ratio (rct_i_i_ppl / total): 0.00")
    rct_i_n = rct_i_n * (rct_i_n_ppl / denominator_rct_i) if denominator_rct_i != 0 else 0
    print(f"Ratio (rct_i_n_ppl / total): {rct_i_n_ppl / denominator_rct_i:.2f}" if denominator_rct_i != 0 else "Ratio (rct_i_n_ppl / total): 0.00")
    rct_i_d = rct_i_d * (rct_i_d_ppl / denominator_rct_i) if denominator_rct_i != 0 else 0
    print(f"Ratio (rct_i_d_ppl / total): {rct_i_d_ppl / denominator_rct_i:.2f}" if denominator_rct_i != 0 else "Ratio (rct_i_d_ppl / total): 0.00")

    (rct_i_i, rct_i_n, rct_i_d), rct_i_sum = handle_zero_and_sum(rct_i_i, rct_i_n, rct_i_d)

    # ---------------- RCT decreased ----------------
    rct_d_i = tri_df.loc[
        (tri_df['study_design'] == 'RCT') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'increase'),
        'count'
    ].sum()
    rct_d_i_ppl = tri_df.loc[
        (tri_df['study_design'] == 'RCT') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'increase'),
        'participants_sum'
    ].sum()

    rct_d_n = tri_df.loc[
        (tri_df['study_design'] == 'RCT') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'no_change'),
        'count'
    ].sum()
    rct_d_n_ppl = tri_df.loc[
        (tri_df['study_design'] == 'RCT') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'no_change'),
        'participants_sum'
    ].sum()

    rct_d_d = tri_df.loc[
        (tri_df['study_design'] == 'RCT') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'decrease'),
        'count'
    ].sum()
    rct_d_d_ppl = tri_df.loc[
        (tri_df['study_design'] == 'RCT') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'decrease'),
        'participants_sum'
    ].sum()

    denominator_rct_d = rct_d_i_ppl + rct_d_n_ppl + rct_d_d_ppl
    rct_d_i = rct_d_i * (rct_d_i_ppl / denominator_rct_d) if denominator_rct_d != 0 else 0
    print(f"Ratio (rct_d_i_ppl / total): {rct_d_i_ppl / denominator_rct_d:.2f}" if denominator_rct_d != 0 else "Ratio (rct_d_i_ppl / total): 0.00")
    rct_d_n = rct_d_n * (rct_d_n_ppl / denominator_rct_d) if denominator_rct_d != 0 else 0
    print(f"Ratio (rct_d_n_ppl / total): {rct_d_n_ppl / denominator_rct_d:.2f}" if denominator_rct_d != 0 else "Ratio (rct_d_n_ppl / total): 0.00")
    rct_d_d = rct_d_d * (rct_d_d_ppl / denominator_rct_d) if denominator_rct_d != 0 else 0
    print(f"Ratio (rct_d_d_ppl / total): {rct_d_d_ppl / denominator_rct_d:.2f}" if denominator_rct_d != 0 else "Ratio (rct_d_d_ppl / total): 0.00")

    (rct_d_i, rct_d_n, rct_d_d), rct_d_sum = handle_zero_and_sum(rct_d_i, rct_d_n, rct_d_d)

    # ---------------- MR increased/decreased ----------------
    mr_i_i = tri_df.loc[
        (tri_df['study_design'] == 'MR') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'increase'),
        'count'
    ].sum()
    mr_i_n = tri_df.loc[
        (tri_df['study_design'] == 'MR') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'no_change'),
        'count'
    ].sum()
    mr_i_d = tri_df.loc[
        (tri_df['study_design'] == 'MR') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'decrease'),
        'count'
    ].sum()
    (mr_i_i, mr_i_n, mr_i_d), mr_i_sum = handle_zero_and_sum(mr_i_i, mr_i_n, mr_i_d)

    mr_d_i = tri_df.loc[
        (tri_df['study_design'] == 'MR') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'increase'),
        'count'
    ].sum()
    mr_d_n = tri_df.loc[
        (tri_df['study_design'] == 'MR') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'no_change'),
        'count'
    ].sum()
    mr_d_d = tri_df.loc[
        (tri_df['study_design'] == 'MR') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'decrease'),
        'count'
    ].sum()
    (mr_d_i, mr_d_n, mr_d_d), mr_d_sum = handle_zero_and_sum(mr_d_i, mr_d_n, mr_d_d)

    # ---------------- OS increased ----------------
    os_i_i = tri_df.loc[
        (tri_df['study_design'] == 'OS') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'increase'),
        'count'
    ].sum()
    os_i_i_ppl = tri_df.loc[
        (tri_df['study_design'] == 'OS') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'increase'),
        'participants_sum'
    ].sum()

    os_i_n = tri_df.loc[
        (tri_df['study_design'] == 'OS') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'no_change'),
        'count'
    ].sum()
    os_i_n_ppl = tri_df.loc[
        (tri_df['study_design'] == 'OS') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'no_change'),
        'participants_sum'
    ].sum()

    os_i_d = tri_df.loc[
        (tri_df['study_design'] == 'OS') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'decrease'),
        'count'
    ].sum()
    os_i_d_ppl = tri_df.loc[
        (tri_df['study_design'] == 'OS') &
        (tri_df['exposure_direction'] == 'increased') &
        (tri_df['direction'] == 'decrease'),
        'participants_sum'
    ].sum()

    denominator_os_i = os_i_i_ppl + os_i_n_ppl + os_i_d_ppl
    os_i_i = os_i_i * (os_i_i_ppl / denominator_os_i) if denominator_os_i != 0 else 0
    print(f"Ratio (os_i_i_ppl / total): {os_i_i_ppl / denominator_os_i:.2f}" if denominator_os_i != 0 else "Ratio (os_i_i_ppl / total): 0.00")
    os_i_n = os_i_n * (os_i_n_ppl / denominator_os_i) if denominator_os_i != 0 else 0
    print(f"Ratio (os_i_n_ppl / total): {os_i_n_ppl / denominator_os_i:.2f}" if denominator_os_i != 0 else "Ratio (os_i_n_ppl / total): 0.00")
    os_i_d = os_i_d * (os_i_d_ppl / denominator_os_i) if denominator_os_i != 0 else 0
    print(f"Ratio (os_i_d_ppl / total): {os_i_d_ppl / denominator_os_i:.2f}" if denominator_os_i != 0 else "Ratio (os_i_d_ppl / total): 0.00")

    (os_i_i, os_i_n, os_i_d), os_i_sum = handle_zero_and_sum(os_i_i, os_i_n, os_i_d)

    # ---------------- OS decreased ----------------
    os_d_i = tri_df.loc[
        (tri_df['study_design'] == 'OS') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'increase'),
        'count'
    ].sum()
    os_d_i_ppl = tri_df.loc[
        (tri_df['study_design'] == 'OS') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'increase'),
        'participants_sum'
    ].sum()

    os_d_n = tri_df.loc[
        (tri_df['study_design'] == 'OS') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'no_change'),
        'count'
    ].sum()
    os_d_n_ppl = tri_df.loc[
        (tri_df['study_design'] == 'OS') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'no_change'),
        'participants_sum'
    ].sum()

    os_d_d = tri_df.loc[
        (tri_df['study_design'] == 'OS') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'decrease'),
        'count'
    ].sum()
    os_d_d_ppl = tri_df.loc[
        (tri_df['study_design'] == 'OS') &
        (tri_df['exposure_direction'] == 'decreased') &
        (tri_df['direction'] == 'decrease'),
        'participants_sum'
    ].sum()

    denominator_os_d = os_d_i_ppl + os_d_n_ppl + os_d_d_ppl
    os_d_i = os_d_i * (os_d_i_ppl / denominator_os_d) if denominator_os_d != 0 else 0
    print(f"Ratio (os_d_i_ppl / total): {os_d_i_ppl / denominator_os_d:.2f}" if denominator_os_d != 0 else "Ratio (os_d_i_ppl / total): 0.00")
    os_d_n = os_d_n * (os_d_n_ppl / denominator_os_d) if denominator_os_d != 0 else 0
    print(f"Ratio (os_d_n_ppl / total): {os_d_n_ppl / denominator_os_d:.2f}" if denominator_os_d != 0 else "Ratio (os_d_n_ppl / total): 0.00")
    os_d_d = os_d_d * (os_d_d_ppl / denominator_os_d) if denominator_os_d != 0 else 0
    print(f"Ratio (os_d_d_ppl / total): {os_d_d_ppl / denominator_os_d:.2f}" if denominator_os_d != 0 else "Ratio (os_d_d_ppl / total): 0.00")

    (os_d_i, os_d_n, os_d_d), os_d_sum = handle_zero_and_sum(os_d_i, os_d_n, os_d_d)

    # ---------------- 二分类：excitatory vs non_excitatory ----------------
    # excitatory:
    #   increased exposure  -> outcome increase
    #   decreased exposure  -> outcome decrease
    p_excitatory = (1/6) * (
        (rct_i_i / rct_i_sum) + (mr_i_i / mr_i_sum) + (os_i_i / os_i_sum) +
        (rct_d_d / rct_d_sum) + (mr_d_d / mr_d_sum) + (os_d_d / os_d_sum)
    )

    # non_excitatory = no_change + inhibitory
    # inhibitory:
    #   increased exposure  -> outcome decrease
    #   decreased exposure  -> outcome increase
    p_non_excitatory = (1/6) * (
        ((rct_i_n + rct_i_d) / rct_i_sum) +
        ((mr_i_n + mr_i_d) / mr_i_sum) +
        ((os_i_n + os_i_d) / os_i_sum) +
        ((rct_d_n + rct_d_i) / rct_d_sum) +
        ((mr_d_n + mr_d_i) / mr_d_sum) +
        ((os_d_n + os_d_i) / os_d_sum)
    )

    # LOE: 二分类 baseline = 0.5
    biggest = max(p_excitatory, p_non_excitatory)
    loe = (biggest - 0.5) / (1 - 0.5)   # = 2*biggest - 1

    biggest_relation = 'excitatory' if biggest == p_excitatory else 'non_excitatory'

    return {
        "p_excitatory": round(p_excitatory, 3),
        "p_non_excitatory": round(p_non_excitatory, 3),
        "loe": round(loe, 3),
        "biggest": biggest_relation
    }